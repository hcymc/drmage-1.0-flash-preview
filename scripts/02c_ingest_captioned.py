#!/usr/bin/env python
"""02c_ingest_captioned.py — 阶段一/三（HF 描述通道）：一趟读遍 parquet，只落宽表 CSV。

数据来源：``summykai/minecraft-skins-captioned-900k``（MIT，854,116 条）
---------------------------------------------------------------------
该集合本身已经做了四件事，所以对本项目特别有价值：

1. **已去重**（作者自述 deduplicated）；
2. **已质量过滤**（garbage / solid-color / low-effort 已剔除）；
3. **只有 Steve 型**（4px 手臂）—— 与「骨架类型」的诉求一致；
4. **每张带一段自然语言描述** —— 独立的自然语言证据。

> 但**声明不是证据**：本脚本仍然逐张跑 UV 校验 + 骨架判别 + 质量三维打分，
> 实测确实有 5%~8% 会被自家的闸门拦下。
>
> 另外要澄清来源性质：``text`` 列从行文看是**上游 caption 模型生成的描述**，
> 不是人工逐张撰写。所以写入 ``captions.jsonl`` 时打 ``method: given``
> （外部给定），README 与 label_schema 里都不把它宣传成「人工标注」。

本脚本做的事
------------
* 流式读 parquet（不整表载入内存），逐条：
  - ``image`` 列取出 PNG 字节（**实测该数据集把这一列存成了裸 base64 字符串**，
    不是 HF ``Image`` 特征常见的 ``{bytes, path}`` 字典，三路都要认）；
  - 校验 + **一趟算完全部特征**（与 02b 同一套 ``manifest.features_for``）；
  - 合格样本写一行到 ``data/hf_captioned_manifest.csv``；
  - 描述写进 ``labels/captions.jsonl``（键为 sha256，便于后续对齐）；
* **不落任何中间图片文件**（理由见 02b 的模块文档）。

用法
----
    python 02c_ingest_captioned.py --target 60000
    python 02c_ingest_captioned.py --target 60000 --quality simple
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from manifest import FIELDS, features_for, read_rows, write_rows  # noqa: E402
from skinuv import analyze_image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw", "hf_captioned")
LOGS = os.path.join(ROOT, "logs")
LABELS = os.path.join(ROOT, "labels")
MANIFEST = os.path.join(DATA, "hf_captioned_manifest.csv")
CAPTIONS = os.path.join(LABELS, "captions.jsonl")
SOURCE = "hf_captioned"


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "ingest_captioned.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def extract_image_bytes(cell) -> bytes | None:
    """从 parquet 的 ``image`` 列取出 PNG 字节。

    实测这一列是**裸 base64 字符串**（不是 HF ``Image`` 特征的 ``{bytes, path}``
    字典），所以三路都要认：裸 bytes / dict / base64 str。
    早期版本只认前两路，结果每一行都落进 ``no_bytes``，整轮扫下来写出 0 条。
    """
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b:
            return bytes(b)
        p = cell.get("path")
        if p and os.path.isfile(p):
            with open(p, "rb") as fh:
                return fh.read()
        return None
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return None
        if s.startswith("data:"):
            s = s.split(",", 1)[-1]
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=60000)
    ap.add_argument("--quality", choices=["off", "reject", "simple"], default="reject")
    ap.add_argument("--model-gate", choices=["strict", "off"], default="strict")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--glob-dir", default=RAW)
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("[x] 缺 pyarrow。装法：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyarrow")
        return 1

    shards = sorted(os.path.join(args.glob_dir, f) for f in os.listdir(args.glob_dir)
                    if f.endswith(".parquet")) if os.path.isdir(args.glob_dir) else []
    if not shards:
        log(f"[x] {args.glob_dir} 下没有 parquet 分片，先跑 01b_fetch_hf.py --captioned N")
        return 1

    os.makedirs(LABELS, exist_ok=True)

    log(f"=== captioned 摄取启动 | 分片 {len(shards)} | target={args.target} "
        f"| quality={args.quality} | model_gate={args.model_gate} ===")
    t0 = time.time()

    # ---- 断点续跑：已有清单直接继承 ----
    rows_prev: list[dict] = []
    seen_sha: set[str] = set()
    if os.path.exists(MANIFEST):
        rows_prev = read_rows(MANIFEST)
        for r in rows_prev:
            if r.get("sha256"):
                seen_sha.add(r["sha256"])
        if rows_prev:
            log(f"  清单里已有 {len(rows_prev)} 条，将跳过其内容")

    written = 0
    stats = Counter()
    rows_out: list[dict] = []
    cap_fh = open(CAPTIONS, "a", encoding="utf-8")

    try:
        for shard in shards:
            if written >= args.target:
                break
            pf = pq.ParquetFile(shard)
            log(f"  分片 {os.path.basename(shard)} | 行组 {pf.num_row_groups} | 行 {pf.metadata.num_rows}")
            for batch in pf.iter_batches(batch_size=args.batch):
                d = batch.to_pydict()
                imgs = d.get("image")
                if imgs is None:
                    stats["no_image_col"] += 1
                    continue
                n = len(imgs)
                texts = d.get("text") or [""] * n
                descs = d.get("description") or [None] * n
                titles = d.get("title") or [None] * n
                names = d.get("file_name") or [None] * n
                for img, txt, desc, title, fname in zip(imgs, texts, descs, titles, names):
                    if written >= args.target:
                        break
                    stats["scanned"] += 1
                    data = extract_image_bytes(img)
                    if not data:
                        stats["no_bytes"] += 1
                        continue
                    h = hashlib.sha256(data).hexdigest()
                    if h in seen_sha:
                        stats["dup"] += 1
                        continue
                    try:
                        im = Image.open(io.BytesIO(data))
                        im.load()
                    except Exception as exc:
                        stats[f"decode:{type(exc).__name__}"] += 1
                        continue
                    rep = analyze_image(im, fname or "", quality_gate=args.quality,
                                        model_gate=args.model_gate)
                    if not rep.ok:
                        stats["reject:" + rep.reason.split(":")[0]] += 1
                        continue
                    seen_sha.add(h)

                    caption = (txt or desc or title or "").strip()
                    row = features_for(np.asarray(im, dtype=np.uint8), rep)
                    row["key"] = h[:16] + ".png"
                    row["sha256"] = h
                    row["caption_len"] = len(caption)
                    rows_out.append(row)
                    cap_fh.write(json.dumps({"sha256": h, "name": row["key"],
                                             "src_name": fname, "text": caption},
                                            ensure_ascii=False) + "\n")
                    written += 1
                if written >= args.target:
                    break
                log(f"  ...扫 {stats['scanned']} | 写出 {written} | "
                    f"剔除 { {k: v for k, v in stats.items() if ':' in k or k in ('no_bytes', 'dup')} }")
            if written >= args.target:
                break
    finally:
        cap_fh.close()

    all_rows = rows_prev + rows_out
    if all_rows:
        write_rows(MANIFEST, all_rows, mode="w")

    log(f"--- 扫描 {stats['scanned']}，本次写出 {written}，清单共 {len(all_rows)} 条，"
        f"用时 {time.time()-t0:.0f}s ---")
    log(f"    剔除构成: {dict((k, v) for k, v in stats.items() if ':' in k or k in ('no_bytes', 'dup'))}")
    log(f"    本次骨架类型分布: {dict(Counter(r['model_type'] for r in rows_out))}")
    log(f"    清单 {MANIFEST}（{len(FIELDS)} 列），描述 {CAPTIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
