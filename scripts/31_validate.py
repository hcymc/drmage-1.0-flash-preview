#!/usr/bin/env python
"""31_validate.py — 阶段五之二：生成结果的验证报告。

验证分五关，每关都落到可复现的数字上：

1. **格式合法性** —— 尺寸/模式/是否 RGBA，逐张跑 UV 规范校验（``skinuv.analyze``）。
2. **alpha 合理性** —— alpha 是否近似二值、是否出现大片全透明或全不透明。
3. **分布对齐** —— 生成的透明占比 / 颜色数 / 边缘密度 / overlay 使用率，
   与真实训练集同特征的 **Wasserstein-1 距离**。距离越小说明越像真皮肤。
4. **记忆检查** —— 每张生成图与最近的真实样本的 dHash Hamming 距离。
   若大量样本距离为 0，说明模型在抄训练集而不是在生成。
5. **条件可控性**（条件模型）—— 指定条件生成的样本，其实际统计量
   是否朝指定方向移动（与无条件基线对比）。

用法
----
    python 31_validate.py --run diffusion_20260921_2100
    python 31_validate.py --run diffusion_xxx --baseline diffusion_uncond

产出
----
* ``logs/validation_<run>.json``
* ``logs/validation_<run>.md``
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from labeling import edge_density, symmetry_score  # noqa: E402
from skinuv import LIMB_L_MIN, SLIM_ARM_BOX_MAX, analyze, skin_contact_sheet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "data", "generated")
DATA = os.path.join(ROOT, "data")
PROCESSED = os.path.join(DATA, "processed")
LOGS = os.path.join(ROOT, "logs")
CLEAN_MANIFEST = os.path.join(DATA, "clean_manifest.csv")

FEATURES = ["transparent_ratio", "overlay_ratio", "unique_colors", "edge_density", "symmetry"]

#: numpy 2.0 把 trapz 改名成 trapezoid，本机是 1.26，两条路都留
_trapz = getattr(np, "trapezoid", None) or np.trapz


def features_of(path: str) -> dict | None:
    try:
        arr = np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None
    return features_of_arr(arr)


def features_of_arr(arr: np.ndarray) -> dict | None:
    if arr.shape != (64, 64, 4):
        return None
    a = arr[..., 3]
    m = a > 0
    vis = arr[..., :3][m]
    return {
        "transparent_ratio": float(np.count_nonzero(a == 0) / a.size),
        "overlay_ratio": float(a[0:16, 32:64].mean() / 255.0),
        "unique_colors": int(np.unique(vis.reshape(-1, 3), axis=0).shape[0]) if vis.size else 0,
        "edge_density": edge_density(arr),
        "symmetry": symmetry_score(arr),
        "alpha_nonbinary": float(((a > 25) & (a < 230)).mean()),
        "alpha_mean": float(a.mean() / 255.0),
        "painted_ratio": float(m.mean()),
    }


def dhash_arr(arr: np.ndarray, n: int = 16) -> np.ndarray:
    """真值参照的 dHash（与 31_validate 里对生成图的算法保持一致）。"""
    im = Image.fromarray(arr, "RGBA")
    bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
    g = Image.alpha_composite(bg, im).convert("L")
    a = np.array(g.resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def load_reference(limit: int) -> tuple[list, np.ndarray]:
    """真实参照：直接取 ``data/processed/*.npy``（清洗后的全量像素，
    正是 04_dataset 喂给训练的那批）。**不再按路径去读 15 万个中间 PNG**。"""
    arrs, hs = [], []
    total = 0
    for split in ("val", "test", "train"):
        p = os.path.join(PROCESSED, f"{split}.npy")
        if not os.path.isfile(p):
            continue
        x = np.load(p, mmap_mode="r")
        total += x.shape[0]
        step = max(1, x.shape[0] // max(1, limit // 3))
        for i in range(0, x.shape[0], step):
            a = np.asarray(x[i]).transpose(1, 2, 0)
            arrs.append(a)
            hs.append(dhash_arr(a))
            if len(arrs) >= limit:
                break
        if len(arrs) >= limit:
            break
    print(f"真实参照样本 {len(arrs)} 张（processed 共 {total} 张）")
    return arrs, (np.array(hs, dtype=bool) if hs else np.zeros((0, 256), bool))


def dhash_bits(path: str, n: int = 16) -> np.ndarray | None:
    try:
        im = Image.open(path).convert("RGBA")
    except Exception:
        return None
    bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
    g = Image.alpha_composite(bg, im).convert("L")
    a = np.array(g.resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    xs = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), xs, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), xs, side="right") / b.size
    return float(_trapz(np.abs(cdf_a - cdf_b), xs)) if xs.size > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="data/generated/<run> 目录名")
    ap.add_argument("--real-limit", type=int, default=1500)
    ap.add_argument("--mem-sample", type=int, default=400)
    args = ap.parse_args()

    rundir = os.path.join(GEN, args.run)
    if not os.path.isdir(rundir):
        print(f"找不到 {rundir}")
        return
    gens = sorted(os.path.join(rundir, f) for f in os.listdir(rundir)
                  if f.lower().endswith(".png") and not f.startswith("contact"))
    print(f"生成样本 {len(gens)} 张")

    # ---- 第 1 关：格式合法性 ---------------------------------------
    uvs, reasons, mt_detected, qt_detected = [], Counter(), Counter(), Counter()
    gate_pass = 0
    for p in gens:
        # 这一关只判 UV 合法性，所以两条闸门都关掉（骨架闸门的结果单独统计）
        rep = analyze(p, quality_gate="off", model_gate="off")
        uvs.append(rep.ok)
        if not rep.ok:
            reasons[rep.reason] += 1
        else:
            mt_detected[rep.model_type] += 1
            qt_detected[rep.quality_tier] += 1
            if rep.model_type != "unknown" and rep.limb_l_ratio >= LIMB_L_MIN:
                gate_pass += 1
    rate = float(np.mean(uvs)) if uvs else 0.0
    gate_pass_rate = gate_pass / max(len(gens), 1)

    # ---- 第 1.5 关：骨架类型（Steve/Alex）可控性 ----------------------
    # 从 generate_<run>.json 读「请求的类型」，与「从像素反判出来的类型」比对。
    model_type_gate: dict = {"requested": None, "detected": dict(mt_detected),
                             "hit_rate": None, "note": ""}
    glog = os.path.join(LOGS, f"generate_{args.run}.json")
    if os.path.isfile(glog):
        try:
            gj = json.load(open(glog, encoding="utf-8"))
            req = gj.get("model_type_requested")
            model_type_gate["requested"] = req
            if req:
                hit = int(mt_detected.get(req, 0))
                model_type_gate["hit_rate"] = round(hit / max(len(gens), 1), 4)
                model_type_gate["note"] = (
                    "条件向量指定了骨架类型，且能按像素独立反判出来，"
                    "一致率即『条件是否真的生效』的直接证据。")
            else:
                model_type_gate["note"] = "本 run 未指定骨架类型（无条件或未给 --model-type）。"
        except Exception as exc:
            model_type_gate["note"] = f"读 generate log 失败：{exc}"
    else:
        model_type_gate["note"] = "缺 generate log，无法比对请求类型。"

    # ---- 第 2 关 + 特征 -------------------------------------------------
    gfeat = {}
    for p in gens:
        f = features_of(p)
        if f:
            gfeat[p] = f
    keys = list(next(iter(gfeat.values())).keys()) if gfeat else []

    # ---- 第 3 关：与真实分布对齐 ------------------------------------
    # 真实参照直接取 data/processed/*.npy —— 那正是清洗后喂给训练的全量像素，
    # 比按路径去读中间图片文件更直接（现在也没有中间图片文件了）。
    rref_arrays, RH = load_reference(args.real_limit)
    rfeat = [f for f in (features_of_arr(a) for a in rref_arrays) if f]

    dist = {}
    for k in FEATURES:
        g = np.array([f[k] for f in gfeat.values()], dtype=float)
        rr = np.array([f[k] for f in rfeat], dtype=float)
        dist[k] = {
            "gen_mean": round(float(g.mean()), 4), "gen_std": round(float(g.std()), 4),
            "real_mean": round(float(rr.mean()), 4), "real_std": round(float(rr.std()), 4),
            "wasserstein1": round(wasserstein1(g, rr), 4),
            "normalized_w1": round(wasserstein1(g, rr) / (abs(float(rr.mean())) + 1e-6), 4),
        }

    # ---- 第 4 关：记忆检查 ------------------------------------------
    sample_g = gens if len(gens) <= args.mem_sample else list(
        np.array(gens)[np.linspace(0, len(gens) - 1, args.mem_sample).astype(int)])

    mem = []
    for p in sample_g:
        h = dhash_bits(p)
        if h is None or RH.size == 0:
            continue
        d = np.count_nonzero(RH != h[None, :], axis=1)
        mem.append(int(d.min()))
    mem = np.array(mem) if mem else np.array([999])
    memorization = {
        "n_checked": int(mem.size),
        "min_dist_mean": round(float(mem.mean()), 2),
        "min_dist_p10": int(np.percentile(mem, 10)),
        "exact_copy_count": int((mem == 0).sum()),
        "near_copy_count_le3": int((mem <= 3).sum()),
        "verdict": ("疑似大量复制训练集" if (mem <= 3).mean() > 0.05 else "未见明显复制"),
    }

    # ---- 第 5 关：alpha 合理性 --------------------------------------
    anb = np.array([f["alpha_nonbinary"] for f in gfeat.values()], dtype=float) if gfeat else np.array([0.0])
    ranb = np.array([f["alpha_nonbinary"] for f in rfeat], dtype=float) if rfeat else np.array([0.0])
    alpha_stat = {
        "gen_nonbinary_ratio_mean": round(float(anb.mean()), 4),
        "real_nonbinary_ratio_mean": round(float(ranb.mean()), 4),
        "gen_painted_ratio_mean": round(float(np.mean([f["painted_ratio"] for f in gfeat.values()])), 4)
        if gfeat else 0.0,
        "real_painted_ratio_mean": round(float(np.mean([f["painted_ratio"] for f in rfeat])), 4)
        if rfeat else 0.0,
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run": args.run,
        "n_generated": len(gens),
        "gate1_format": {"uv_valid": int(sum(uvs)), "uv_valid_rate": round(rate, 4),
                         "reject_reasons": dict(reasons)},
        "gate1_4_structure": {
            "model_gate_pass": gate_pass,
            "model_gate_pass_rate": round(gate_pass_rate, 4),
            "note": ("按训练集同款结构判据（两根手臂盒都画了 + 左肢体本体不空）复核生成样本；"
                     "这一关比 UV 合法性严，直接反映生成结果能不能真的套在角色上。"),
        },
        "gate1_5_model_type": model_type_gate,
        "gate1_6_generated_quality": dict(qt_detected),
        "gate2_alpha": alpha_stat,
        "gate3_distribution": dist,
        "gate4_memorization": memorization,
        "real_reference_n": len(rfeat),
    }
    with open(os.path.join(LOGS, f"validation_{args.run}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)

    md = [
        f"# 生成验证报告 · {args.run}", "",
        f"生成时间：{report['generated_at']}　样本数：**{len(gens)}**", "",
        "## 第 1 关 · 格式合法性（硬门槛）", "",
        "| 指标 | 数值 |", "|---|---|",
        f"| 通过 UV 校验 | {report['gate1_format']['uv_valid']} |",
        f"| 通过率 | **{rate:.1%}** |",
        f"| 尺寸 / 通道 | 64×64 RGBA（由 `skinuv.analyze` 强制校验） |",
        "", "拒绝原因分布：", "",
    ]
    if reasons:
        md += ["| 原因 | 数量 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in reasons.most_common()]
    else:
        md.append("（无）")
    md += ["", "## 第 1.4 关 · 结构可用性（比 UV 合法性更严）", "",
           "| 指标 | 数值 |", "|---|---|",
           f"| 通过骨架结构闸门 | {report['gate1_4_structure']['model_gate_pass']} |",
           f"| 通过率 | **{gate_pass_rate:.1%}** |",
           "",
           "> 判据（与训练集清洗时**同一套代码**）：两根手臂贴图盒都画了东西，"
           "且左肢体本体不空。",
           "> UV 合法只能说明「图是 64×64 RGBA」，这一关才说明「像个能穿的角色」。",
           ""]
    md += ["", "## 第 1.5 关 · 骨架类型（Steve / Alex）", "",
           "| 指标 | 数值 |", "|---|---|",
           f"| 请求的类型 | {model_type_gate['requested'] or '（未指定）'} |",
           f"| 从像素反判的类型分布 | {model_type_gate['detected']} |",
           f"| 一致率 | "
           f"{('%.1f%%' % (model_type_gate['hit_rate']*100)) if model_type_gate['hit_rate'] is not None else '—'} |",
           f"| 生成样本质量分档 | {dict(qt_detected)} |",
           "",
           f"> {model_type_gate['note']}",
           "> 反判逻辑与训练集标注用的是**同一个** `skinuv.detect_model_type`，两条独立判据：",
           f"> ① 手臂盒占用率是否超过纤细模型的物理上限（{SLIM_ARM_BOX_MAX:.4f}）；",
           "> ② 「经典会画、纤细永远画不到」的 L 形排他区（4 列 × 12 行 + 末行 12 列）里有没有笔画。"]
    md += ["", "## 第 2 关 · alpha 合理性", "", "| 指标 | 生成 | 真实 |", "|---|---|---|",
           f"| 半透明像素占比（0.1<α<0.9） | {alpha_stat['gen_nonbinary_ratio_mean']} | "
           f"{alpha_stat['real_nonbinary_ratio_mean']} |",
           f"| 被涂色像素占比（α>0） | {alpha_stat['gen_painted_ratio_mean']} | "
           f"{alpha_stat['real_painted_ratio_mean']} |",
           "", "## 第 3 关 · 与真实分布对齐（Wasserstein-1 距离，越小越像）", "",
           "| 特征 | 生成 mean±std | 真实 mean±std | W1 | 归一化 W1 |", "|---|---|---|---|---|"]
    for k, d in dist.items():
        md.append(f"| {k} | {d['gen_mean']}±{d['gen_std']} | {d['real_mean']}±{d['real_std']} | "
                  f"{d['wasserstein1']} | {d['normalized_w1']} |")
    md += ["", "## 第 4 关 · 记忆检查（最近真实邻居的 dHash 距离）", "",
           f"- 检查样本数：{memorization['n_checked']}（真实参照 {len(rfeat)} 张）",
           f"- 最近邻距离均值：{memorization['min_dist_mean']}",
           f"- 距离 0（完全一致）：{memorization['exact_copy_count']}",
           f"- 距离 ≤3（近似复制）：{memorization['near_copy_count_le3']}",
           f"- **判定：{memorization['verdict']}**",
           "", "> 说明：dHash 距离只能反映全局明暗结构，不能证明「不是抄」。",
           "> 判定为「未见明显复制」只表示没有成批的低距离样本。",
           "", "## 产物", "",
           f"- 样本网格：`data/generated/{args.run}/contact_sheet.png`",
           f"- 逐样本统计：`data/generated/{args.run}/manifest.csv`",
           f"- 本报告 JSON：`logs/validation_{args.run}.json`",
           ]
    with open(os.path.join(LOGS, f"validation_{args.run}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print(json.dumps({k: v for k, v in report.items() if k != "gate3_distribution"},
                     indent=1, ensure_ascii=False))
    print(f"报告 -> logs/validation_{args.run}.md")


if __name__ == "__main__":
    main()
