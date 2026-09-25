#!/usr/bin/env python
"""02a_ingest_mineskin.py — 把自采（MineSkin 通道）的松散 PNG 也并进同一套宽表。

为什么需要这一步
----------------
``01_collect.py`` 产出的 ``data/manifest.csv`` 是**采集台账**：只记了
size/mode/几个结构统计，没有 dHash、调色板、HSV、面部这些下游标注要用的特征。
而 02_clean / 10_label 现在统一吃宽表 CSV（口径见 ``lib/manifest.py``）。

所以这里做一次**本地**重扫：1.1 万张 PNG 全在磁盘上，一趟约 20 秒，
把它们补成与 02b / 02c 完全同构的 ``data/mineskin_manifest.csv``。

产出
----
* ``data/mineskin_manifest.csv`` —— 宽表，列与 ``manifest.FIELDS`` 一致
* 日志 ``logs/ingest_mineskin.log``

用法
----
    python 02a_ingest_mineskin.py
    python 02a_ingest_mineskin.py --quality simple --model-gate strict
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from manifest import read_rows, write_rows  # noqa: E402
from skinuv import analyze_image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
SRC_MANIFEST = os.path.join(DATA, "manifest.csv")            # 01_collect 的台账
OUT_MANIFEST = os.path.join(DATA, "mineskin_manifest.csv")   # 本脚本产出的宽表
SOURCE = "mineskin"


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "ingest_mineskin.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", choices=["off", "reject", "simple"], default="reject")
    ap.add_argument("--model-gate", choices=["strict", "off"], default="strict")
    args = ap.parse_args()

    from manifest import features_for

    if not os.path.isfile(SRC_MANIFEST):
        log(f"[x] 找不到 {SRC_MANIFEST}，先跑 01_collect.py mineskin")
        return 1
    rows_in = read_rows(SRC_MANIFEST)
    log(f"=== mineskin 并入启动 | 台账 {len(rows_in)} 条 | quality={args.quality} "
        f"| model_gate={args.model_gate} ===")
    t0 = time.time()

    stats = Counter()
    out: list[dict] = []
    for i, r in enumerate(rows_in, 1):
        rel = r.get("path") or ""
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not rel or not os.path.isfile(p):
            stats["missing_file"] += 1
            continue
        try:
            im = Image.open(p)
            im.load()
        except Exception as exc:
            stats[f"decode:{type(exc).__name__}"] += 1
            continue
        rep = analyze_image(im, rel, quality_gate=args.quality, model_gate=args.model_gate)
        if not rep.ok:
            stats["reject:" + rep.reason.split(":")[0]] += 1
            continue
        with open(p, "rb") as fh:
            data = fh.read()
        row = features_for(np.asarray(im.convert("RGBA"), dtype=np.uint8), rep)
        row["key"] = os.path.basename(p)
        row["file_path"] = rel
        row["sha256"] = hashlib.sha256(data).hexdigest()
        out.append(row)
        if i % 2000 == 0:
            log(f"  {i}/{len(rows_in)}  入库 {len(out)}")

    if out:
        write_rows(OUT_MANIFEST, out, mode="w")

    log(f"--- 台账 {len(rows_in)} 条，入库 {len(out)}，用时 {time.time()-t0:.1f}s ---")
    log(f"    骨架类型分布: {dict(Counter(r['model_type'] for r in out))}")
    log(f"    质量分档: {dict(Counter(r['quality_tier'] for r in out))}")
    log(f"    剔除构成: {dict((k, v) for k, v in stats.items())}")
    log(f"    宽表 -> {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
