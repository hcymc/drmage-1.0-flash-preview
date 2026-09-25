#!/usr/bin/env python
"""02b_ingest_hf.py — 阶段一/二（HF 通道）：一趟读遍去重 zip，只落一份宽表 CSV。

做什么
------
``data/raw/hf_dedup/minecraft_skins_64x64.zip`` 里是 1,107,411 张 PNG
（成员名 ``skin_XXXXXXXX.png``）。本脚本：

1. **主线程顺序读** zip 成员（不解压整包到磁盘，边读边判）；
2. 交给 worker 线程池：UV 规范校验 + Steve/Alex 判别 + 质量三维打分
   **+ 顺带算完下游要用的全部特征**（dHash / 调色板 / HSV / 面部 / 对称度 /
   复杂度档）——图已经在内存里解码好了，多算这几个便宜统计量几乎不要钱；
3. 每个合格样本写一行宽表到 ``data/hf_manifest.csv``；
4. 合并 ``tags.jsonl`` 里的规则标签（color / detailed / cape / human …）。

**不落任何中间图片文件。** 这一点是实测逼出来的：

    从 zip 顺序读         10,959 /s
    UV 校验+判别+质量       687 /s
    创建小文件             13 ~ 50 /s     ← 瓶颈，占总耗时 97.8%

早期版本给每张合格皮肤写一个 PNG，9 万张要 30~115 分钟的纯等待；改成写
一个大 memmap 又只是把冗余搬了个地方（下游完全可以按 ``(source, key)``
从原始容器里再取回来）。现在的口径是：**摄取这趟把能算的都算完，
清洗和标注只做 CSV 运算，只有构建训练集时才第二次碰像素。**

三个必须踩过的坑（都已在代码里规避）
------------------------------------
* **``zipfile.ZipFile`` 不是线程安全的**。多个线程同时 ``read()`` 会互相破坏共享的
  文件偏移，读回来的是错位的字节（表现为 ``BadZipFile`` 或静默的坏数据）。
  现在**只有主线程碰 ZipFile**，worker 只吃已经读出来的 ``bytes``。
* **随机访问大 zip 很慢**。包内顺序（``skin_XXXXXXXX`` 的编号来自原始 20M 集合）
  已经打乱，对质量没有系统性偏置，因此**默认顺序读**。
* **「是皮肤」不等于「是能用的皮肤」**。实测 5 万样本里有 7.1% 的两根手臂盒
  完全没有像素——肉眼复核确认是品牌 logo、平铺 2D 插画、近空白模板，
  内容根本不按 UV 布局摆放。这类垃圾靠「面部有没有画」拦不住（它们的脸恰好
  画在脸上），必须用**骨架类型闸门**（``--model-gate strict``）剔除。

用法
----
    # 先看分布（不写清单）
    python 02b_ingest_hf.py --stats 60000

    # 正式摄取
    python 02b_ingest_hf.py --target 90000 --quality simple --model-gate strict --workers 6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import queue
import sys
import threading
import time
import zipfile
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from manifest import FIELDS, features_for, write_rows  # noqa: E402
from skinuv import analyze_image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
LOGS = os.path.join(ROOT, "logs")
STATE = os.path.join(ROOT, "data", "state")

ZIP_PATH = os.path.join(RAW, "hf_dedup", "minecraft_skins_64x64.zip")
TAGS_PATH = os.path.join(RAW, "hf_dedup", "tags.jsonl")
SOURCE = "hf_dedup"
MANIFEST = os.path.join(ROOT, "data", "hf_manifest.csv")

_lock = threading.Lock()
_log_fh = None


def log(msg: str) -> None:
    global _log_fh
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    if _log_fh is None:
        _log_fh = open(os.path.join(LOGS, "ingest_hf.log"), "a", encoding="utf-8")
    with _lock:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def load_tags() -> dict[str, list[str]]:
    """``skin_XXXXXXXX.png`` → [tags]。80MB JSONL，几秒。"""
    if not os.path.exists(TAGS_PATH):
        log("  (tags.jsonl 不存在，跳过标签合并)")
        return {}
    out: dict[str, list[str]] = {}
    with open(TAGS_PATH, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
                out[d["filename"]] = d.get("tags") or []
            except Exception:
                continue
    log(f"  已载入规则标签 {len(out)} 条")
    return out


def load_done() -> set[str]:
    """已摄取过的 zip 成员名（从清单反查，断点续跑用）。"""
    done: set[str] = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("source") == SOURCE and r.get("key"):
                    done.add(r["key"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=90000, help="写进清单的目标数量")
    ap.add_argument("--stats", type=int, default=0, help="只看 N 条的分布，不写清单")
    ap.add_argument("--quality", choices=["off", "reject", "simple"], default="reject")
    ap.add_argument("--model-gate", choices=["strict", "off"], default="strict",
                    help="strict=丢掉骨架类型判不出/左肢体缺画的破损皮肤（默认）")
    ap.add_argument("--order", choices=["seq", "shuffle"], default="seq",
                    help="seq=按包内顺序（默认，快且无质量偏置）；shuffle=随机取（慢）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--zip", default=ZIP_PATH)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    if not os.path.exists(args.zip):
        log(f"[x] 找不到 zip: {args.zip}")
        return 1

    os.makedirs(STATE, exist_ok=True)
    tags = load_tags()

    log(f"=== HF 摄取启动 | zip={os.path.basename(args.zip)} | quality={args.quality} "
        f"| model_gate={args.model_gate} | order={args.order} "
        f"| target={args.target or args.stats} ===")
    t0 = time.time()

    zf = zipfile.ZipFile(args.zip)
    names = [n for n in zf.namelist() if n.lower().endswith(".png")]
    log(f"  zip 内 PNG 成员: {len(names)}")

    done: set[str] = set()
    if not args.stats:
        done = load_done()
        if done:
            log(f"  清单里已有 {len(done)} 条，将跳过")

    pool = [n for n in names if os.path.basename(n) not in done]
    if args.order == "shuffle":
        import random
        random.Random(args.seed).shuffle(pool)
    if args.stats:
        pool = pool[: args.stats]
    elif args.target:
        pool = pool[: max(args.target * 4, 2000)]
    log(f"  本次扫描 {len(pool)} 个成员")

    q: "queue.Queue[tuple[str, bytes] | None]" = queue.Queue(maxsize=256)
    counter = {"seen": 0, "ok": 0}
    reasons, model_dist, tier_dist = Counter(), Counter(), Counter()
    rows: list[dict] = []
    stop = threading.Event()

    def worker() -> None:
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                return
            name, data = item
            key = os.path.basename(name)
            try:
                try:
                    im = Image.open(io.BytesIO(data))
                    im.load()
                except Exception as exc:
                    with _lock:
                        counter["seen"] += 1
                        reasons[f"decode:{type(exc).__name__}"] += 1
                    continue
                rep = analyze_image(im, key, quality_gate=args.quality,
                                    model_gate=args.model_gate)
                with _lock:
                    counter["seen"] += 1
                    model_dist[rep.model_type or "-"] += 1
                    if not rep.ok:
                        reasons[rep.reason.split(":")[0]] += 1
                        continue
                    tier_dist[rep.quality_tier] += 1
                    if args.stats or stop.is_set():
                        continue
                    if counter["ok"] >= args.target:
                        stop.set()
                        continue
                    counter["ok"] += 1            # 先占坑，避免多线程超发

                arr = np.asarray(im, dtype=np.uint8)
                row = features_for(arr, rep)      # 一次性算完下游要的所有特征
                row["key"] = key
                row["sha256"] = hashlib.sha256(data).hexdigest()
                row["hf_tags"] = "|".join(tags.get(key, []))
                rows.append(row)
                if counter["ok"] >= args.target:
                    stop.set()
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()

    try:
        for i, name in enumerate(pool):
            if stop.is_set():
                break
            try:
                data = zf.read(name)        # 只有主线程碰 ZipFile
            except Exception as exc:
                with _lock:
                    reasons[f"zipread:{type(exc).__name__}"] += 1
                continue
            q.put((name, data))
            if i and i % 20000 == 0:
                el = time.time() - t0
                log(f"  进度 投入{i}/{len(pool)} 处理{counter['seen']} 入库{counter['ok']} "
                    f"| {counter['seen']/max(el,1e-6):.0f} it/s | {el:.0f}s")
    finally:
        q.join()
        for _ in threads:
            q.put(None)
        for t in threads:
            t.join(timeout=5)
        zf.close()

    n = counter["seen"]
    log(f"--- 扫描 {n} 条，合格 {counter['ok']}"
        f"（通过率 {counter['ok']/max(n,1):.1%}），用时 {time.time()-t0:.0f}s ---")
    log(f"  骨架类型分布: {dict(model_dist)}")
    log(f"  质量分档(仅通过者): {dict(tier_dist)}")
    log(f"  淘汰原因 top: {reasons.most_common(12)}")

    if rows:
        write_rows(MANIFEST, rows)
        log(f"  宽表写入 {MANIFEST} (+{len(rows)} 行 / {len(FIELDS)} 列)")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "zip": os.path.basename(args.zip), "order": args.order,
        "quality_gate": args.quality, "model_gate": args.model_gate,
        "scanned": n, "passed": counter["ok"], "stored": len(rows),
        "pass_rate": round(counter["ok"] / max(n, 1), 4),
        "model_dist": dict(model_dist), "tier_dist": dict(tier_dist),
        "reject_reasons": dict(reasons.most_common(30)),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(STATE, "hf_ingest_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
