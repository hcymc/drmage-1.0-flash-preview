#!/usr/bin/env python
"""02_clean.py — 阶段二之一：多源合并、去重、分层切分（**纯 CSV 运算，不碰像素**）。

数据来源（都被上游摄取成同构的宽表）
------------------------------------
* ``data/mineskin_manifest.csv``        —— 02a（自采 MineSkin 通道）
* ``data/hf_manifest.csv``              —— 02b（HF 1.1M 去重包）
* ``data/hf_captioned_manifest.csv``    —— 02c（HF 带描述集合）

流水线
------
1. 三份宽表合并（上游已经做过 UV 校验 + 骨架判别 + 质量打分，这里不重复）；
2. **质量闸门**：``quality_tier == reject`` 一律剔除；``--strict`` 连 ``simple`` 也剔；
3. **精确去重**：sha256 相同只留一条（跨源也会命中，先来的源优先）；
4. **近似去重**：64 位 dHash + multi-index LSH（切 4 段 16 位，任一段相同进候选，
   Hamming ≤ 8 判重）。dHash 是上游算好写进 CSV 的，所以**这步也不需要读像素**；
5. **按 (骨架类型 × 新旧格式) 分层** 切 train/val/test = 8:1:1。

为什么这步能完全不碰像素
------------------------
上游摄取那一趟已经把图解码在内存里了，顺手把 dHash / 调色板 / HSV / 面部统计
全算进了 CSV。这里要用的每一个数字（sha256、dHash、质量档、骨架类型）
都能从 CSV 直接拿到。少遍历一遍上亿像素，就是少几十分钟。

产出
----
* ``data/clean_manifest.csv`` —— 最终清单（含 ``sid`` / ``split``）
* ``data/splits.json``
* ``logs/clean_report.json`` / ``logs/clean_report.md``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from manifest import CLEAN_EXTRA, read_rows, write_rows  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
CLEAN_MANIFEST = os.path.join(DATA, "clean_manifest.csv")

HAMMING_TOL = 8
GROUP_CAP = 3000        # 单个 LSH 段组内最多两两比对多少样本（防 O(N²) 爆炸）

#: 合并顺序即优先级（同 sha256 时先来的赢）
SOURCES = [
    ("mineskin", os.path.join(DATA, "mineskin_manifest.csv")),
    ("hf_dedup", os.path.join(DATA, "hf_manifest.csv")),
    ("hf_captioned", os.path.join(DATA, "hf_captioned_manifest.csv")),
]


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "clean.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _num_stats(xs: list) -> dict:
    if not xs:
        return {}
    a = np.array(xs, dtype=float)
    return {"mean": round(float(a.mean()), 3), "p10": round(float(np.percentile(a, 10)), 3),
            "p50": round(float(np.percentile(a, 50)), 3),
            "p90": round(float(np.percentile(a, 90)), 3), "max": round(float(a.max()), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="连 quality_tier=simple（颜色简陋）也剔除")
    ap.add_argument("--no-near-dup", action="store_true", help="跳过近似去重（调试用）")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    gate = "simple" if args.strict else "reject"
    stats = Counter()
    cand: list[dict] = []

    # ---- 0) 三份宽表合并 ---------------------------------------------
    for name, mp in SOURCES:
        if not os.path.isfile(mp):
            log(f"  (跳过 {name}：无清单 {os.path.relpath(mp, ROOT)})")
            continue
        rows = read_rows(mp)
        log(f"读入 {name}：{len(rows)} 条")
        before = len(cand)
        for r in rows:
            tier = r.get("quality_tier") or "reject"
            if tier == "reject" or (gate == "simple" and tier == "simple"):
                stats[f"quality_{tier}"] += 1
                stats["rejected"] += 1
                continue
            r["source"] = name
            cand.append(r)
        log(f"  合格 {len(cand) - before} / {len(rows)}")

    log(f"候选合计 {len(cand)}（剔除 {stats['rejected']}）")
    if not cand:
        log("没有候选样本，退出")
        return 1

    # ---- 1) 精确去重（跨源，按 sha256）-------------------------------
    seen: set[str] = set()
    kept: list[dict] = []
    for c in cand:
        h = (c.get("sha256") or "").strip()
        if not h:
            stats["no_sha"] += 1
            continue
        if h in seen:
            stats["dup_exact"] += 1
            continue
        seen.add(h)
        kept.append(c)
    log(f"精确去重后 {len(kept)}（剔除 {stats['dup_exact']}）")

    # ---- 2) 近似去重（dHash，同样不读像素）----------------------------
    n_compared, n_groups, n_capped = 0, 0, 0
    if not args.no_near_dup:
        fps: list[int] = []
        valid: list[dict] = []
        for c in kept:
            s = (c.get("dhash") or "").strip()
            try:
                fp = int(s, 16)
            except (TypeError, ValueError):
                stats["no_dhash"] += 1
                continue
            fps.append(fp)
            valid.append(c)
        kept = valid

        n = len(fps)
        SHIFT = max(1, (n - 1).bit_length())    # 把 (i,j) 对压成一个整数，比 tuple 省内存
        # multi-index LSH：64 位 dHash 切 4 段 16 位，任一段完全相同即进候选。
        # 早期版本用「量化键前 2 字节」单键分桶，全空间只有 ~16 个桶，
        # 15 万样本被塞进十几个大桶里再被 GROUP_CAP 截断 —— 既慢又大量漏检。
        groups: dict[int, list[int]] = defaultdict(list)
        for i, fp in enumerate(fps):
            for b in range(4):
                groups[(fp >> (16 * b)) & 0xFFFF].append(i)
        n_groups = len(groups)

        seen_pair: set[int] = set()
        drop: set[int] = set()
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            if len(idxs) > GROUP_CAP:
                n_capped += 1
                idxs = idxs[:GROUP_CAP]
            for a in range(len(idxs)):
                ia = idxs[a]
                if ia in drop:
                    continue
                fa = fps[ia]
                for b in range(a + 1, len(idxs)):
                    ib = idxs[b]
                    if ib in drop:
                        continue
                    lo, hi = (ia, ib) if ia < ib else (ib, ia)
                    pk = (lo << SHIFT) | hi
                    if pk in seen_pair:
                        continue
                    seen_pair.add(pk)
                    n_compared += 1
                    if (fa ^ fps[ib]).bit_count() <= HAMMING_TOL:
                        drop.add(ib)
        kept = [c for i, c in enumerate(kept) if i not in drop]
        stats["dup_near"] = len(drop)
        log(f"近似去重后 {len(kept)}（段桶 {n_groups} 个，比对 {n_compared} 对，"
            f"剔除 {len(drop)}，超限分组 {n_capped}）")

    # ---- 3) 分层切分 --------------------------------------------------
    kept.sort(key=lambda c: c["sha256"])
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for c in kept:
        strata[(c.get("model_type") or "unknown", int(c.get("is_legacy") or 0))].append(c)

    for _key, items in sorted(strata.items(), key=lambda x: str(x[0])):
        m = len(items)
        v = max(1, round(m * 0.1)) if m >= 10 else 0
        t = max(1, round(m * 0.1)) if m >= 10 else 0
        for i, c in enumerate(items):
            c["split"] = "val" if i < v else ("test" if i < v + t else "train")

    for i, c in enumerate(kept):
        c["sid"] = f"{'L' if int(c.get('is_legacy') or 0) else 'M'}{i:06d}"
    splits = Counter(c["split"] for c in kept)

    write_rows(CLEAN_MANIFEST, kept, extra=CLEAN_EXTRA, mode="w")
    with open(os.path.join(DATA, "splits.json"), "w", encoding="utf-8") as fh:
        json.dump({"total": len(kept), "splits": dict(splits),
                   "stratified_by": ["model_type", "is_legacy"],
                   "assign": {c["sid"]: c["split"] for c in kept}},
                  fh, ensure_ascii=False)
    log(f"最终清单 {CLEAN_MANIFEST}：{len(kept)} 条，切分 {dict(splits)}")

    # ---- 4) 报告 -------------------------------------------------------
    modern = [c for c in kept if not int(c.get("is_legacy") or 0)]
    legacy = [c for c in kept if int(c.get("is_legacy") or 0)]

    def _f(c, k):
        try:
            return float(c.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "quality_gate": gate,
        "final": len(kept),
        "modern_64x64": len(modern),
        "legacy_64x32": len(legacy),
        "dup_exact_removed": stats["dup_exact"],
        "dup_near_removed": stats.get("dup_near", 0),
        "near_dup_compared_pairs": n_compared,
        "lsh_groups": n_groups,
        "lsh_groups_over_cap": n_capped,
        "rejected_by_quality": stats["rejected"],
        "reject_breakdown": {k: v for k, v in stats.items() if k.startswith("quality_")},
        "source_dist": dict(Counter(c["source"] for c in kept)),
        "model_type_dist_modern": dict(Counter(c.get("model_type") for c in modern)),
        "quality_tier_dist": dict(Counter(c.get("quality_tier") for c in kept)),
        "splits": dict(splits),
        "n_colors_stats": _num_stats([_f(c, "n_colors") for c in modern]),
        "edge_density_stats": _num_stats([_f(c, "edge_density") for c in modern]),
        "dom_ratio_stats": _num_stats([_f(c, "dom_ratio") for c in modern]),
    }
    with open(os.path.join(LOGS, "clean_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    write_markdown(report)
    log(f"完成：最终 {len(kept)}（现代 {len(modern)} / 旧版 {len(legacy)}）"
        f" 骨架分布 {report['model_type_dist_modern']}")
    return 0


def write_markdown(rep: dict) -> None:
    md = [
        "# 阶段二 · 清洗与去重报告", "",
        f"生成时间：{rep['generated_at']}", "",
        f"质量闸门：`{rep['quality_gate']}`"
        "（`reject` = 只丢最差档；`simple` = 连「颜色简陋」也丢）", "",
        "## 数量", "", "| 指标 | 数值 |", "|---|---|",
        f"| **最终合格** | **{rep['final']}** |",
        f"| 现代 64×64 | {rep['modern_64x64']} |",
        f"| 旧版 64×32（单独标记，不参与训练） | {rep['legacy_64x32']} |",
        f"| 精确重复剔除 | {rep['dup_exact_removed']} |",
        f"| 近似重复剔除 | {rep['dup_near_removed']} |",
        f"| 质量不合格剔除 | {rep['rejected_by_quality']} |",
        "",
        "## 来源分布", "", "| 来源 | 数量 |", "|---|---|",
    ]
    for k, v in sorted(rep["source_dist"].items(), key=lambda x: -x[1]):
        md.append(f"| {k} | {v} |")
    md += ["", "## 骨架类型分布（仅现代 64×64）", "", "| 类型 | 数量 | 占比 |", "|---|---|---|"]
    tot = max(rep["modern_64x64"], 1)
    for k, v in sorted(rep["model_type_dist_modern"].items(), key=lambda x: -x[1]):
        md.append(f"| {k} | {v} | {v/tot:.1%} |")
    md += ["", "> `classic` = Steve 型（手臂 4px），`slim` = Alex 型（手臂 3px），"
           "`unknown` = 手臂未绘制无法判定。切分按该字段分层。", ""]
    md += ["## 质量分档", "", "| tier | 数量 |", "|---|---|"]
    for k, v in sorted(rep["quality_tier_dist"].items(), key=lambda x: -x[1]):
        md.append(f"| {k} | {v} |")
    md += ["", "## 切分", "", "| split | 数量 |", "|---|---|"]
    for k, v in rep["splits"].items():
        md.append(f"| {k} | {v} |")
    md += ["", "## 剔除构成", "", "| 原因 | 数量 |", "|---|---|"]
    for k, v in sorted(rep["reject_breakdown"].items(), key=lambda x: -x[1]):
        md.append(f"| {k} | {v} |")
    md += ["", "## 近似去重方法说明", "",
           f"- multi-index LSH：64 位 dHash 切 4 段 16 位，段桶 {rep['lsh_groups']} 个，"
           f"实际比对 {rep['near_dup_compared_pairs']} 对，超限截断分组 "
           f"{rep['lsh_groups_over_cap']} 个（上限 3000）。",
           "- 这是 **近似去重**：任一段 16 位相同才进候选，Hamming ≤ 8 判重。"
           "两图相差 8 位时至少一段零差异的概率约 81%，因此存在理论漏检。",
           "- **这步完全不读像素**：dHash 是摄取那一趟算好写进 CSV 的。",
           "- 精确重复（sha256 相同）已 100% 剔除；HF 的 1.1M 源本身也是 BLAKE3 逐像素去重过的。",
           "", "## 现代皮肤画像统计", "",
           "| 指标 | mean | p10 | p50 | p90 | max |", "|---|---|---|---|---|---|"]
    for name, key in [("可见颜色数", "n_colors_stats"),
                      ("边缘密度", "edge_density_stats"),
                      ("最高频颜色占比", "dom_ratio_stats")]:
        s = rep.get(key) or {}
        md.append(f"| {name} | {s.get('mean', '')} | {s.get('p10', '')} | "
                  f"{s.get('p50', '')} | {s.get('p90', '')} | {s.get('max', '')} |")
    with open(os.path.join(LOGS, "clean_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
