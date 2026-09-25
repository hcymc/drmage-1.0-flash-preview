#!/usr/bin/env python
"""04_dataset.py — 阶段二之三：构建训练数据集（**全流程第二趟、也是最后一趟读像素**）。

数据流
------
``data/clean_manifest.csv``（02_clean 产出的宽表）里的每一行都带
``(source, key)`` 定位信息。本脚本按 **(来源 × split)** 分批，从**原始容器**
把像素取回来：

* ``hf_dedup``     → ``minecraft_skins_64x64.zip`` 里按成员名取
* ``hf_captioned`` → parquet 分片里按 sha256 取（列是裸 base64）
* ``mineskin``     → 磁盘上的松散 PNG

取回后直接写进**预分配的 memmap .npy**，不经过任何中间图片文件，
也不需要在内存里堆一份完整副本（样本数从清单里就能提前知道）。

产出
----
* ``data/processed/{train,val,test}.npy``
  形状 ``(N, 4, 64, 64)``，dtype ``uint8``，通道顺序 **RGBA**（CHW）。
* ``data/processed/{split}_cond.npy``
  形状 ``(N, 36)``，float32 条件向量（见 ``lib/labelset.py``），
  含**骨架类型 one-hot**（classic/slim/unknown）。
* ``data/processed/{split}_sids.json`` 与 npy 行序一一对应的样本 ID
* ``data/processed/meta.json`` 形状、切分统计、骨架类型分布、归一化参数

依赖顺序：**先跑 02_clean.py，再跑 10_label.py base，最后跑本脚本**
（条件向量来自 ``labels/annotations.jsonl``；缺了会明确告警，不会静默降级）。
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import sourceio  # noqa: E402
from labelset import COND_DIM, MODEL_ORDER, build_cond_matrix  # noqa: E402
from manifest import read_rows  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "processed")
CLEAN_MANIFEST = os.path.join(DATA, "clean_manifest.csv")
ANNOTATIONS = os.path.join(ROOT, "labels", "annotations.jsonl")

SHAPE = (4, 64, 64)


def log(msg: str) -> None:
    print(msg, flush=True)


def collect(rows: list[dict]) -> dict[str, np.ndarray]:
    """按来源把 ``rows`` 需要的像素一次性取回，返回 ``key -> (4,64,64)``。"""
    got: dict[str, np.ndarray] = {}
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    for src, items in by_source.items():
        keys = {r["key"] for r in items if r.get("key")}
        log(f"    来源 {src}: 需要 {len(keys)} 张")
        if src == "mineskin":
            for r in items:
                p = r.get("file_path") or ""
                try:
                    got[r["key"]] = sourceio.to_chw(sourceio.read_file(ROOT, p))
                except Exception:
                    continue
        else:
            raw = sourceio.fetch(src, keys, log=log)
            for k, b in raw.items():
                try:
                    got[k] = sourceio.to_chw(b)
                except Exception:
                    continue
            del raw
    return got


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isfile(CLEAN_MANIFEST):
        log(f"[x] 找不到 {CLEAN_MANIFEST}，先跑 02_clean.py")
        return 1
    if not os.path.isfile(ANNOTATIONS):
        log(f"[!] 找不到 {ANNOTATIONS} —— 条件向量会全是零向量。先跑 10_label.py base")

    rows = read_rows(CLEAN_MANIFEST)
    modern = [r for r in rows if not int(r.get("is_legacy") or 0)]
    log(f"现代皮肤 {len(modern)} 张，切分 {dict(Counter(r['split'] for r in modern))}")

    meta = {"shape": [4, 64, 64], "dtype": "uint8", "channel_order": "RGBA",
            "cond_dim": COND_DIM, "model_order": MODEL_ORDER,
            "splits": {}, "normalization": {}, "model_type_dist": {},
            "source": "原始容器按 (source,key) 流式取回，无中间图片文件"}

    for split in ("train", "val", "test"):
        sel = [r for r in modern if (r.get("split") or "train") == split]
        if not sel:
            log(f"  {split}: 空，跳过")
            continue

        log(f"  {split}: {len(sel)} 张，开始取像素")
        got = collect(sel)

        n = len(sel)
        path = os.path.join(OUT, f"{split}.npy")
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                       shape=(n, *SHAPE))
        sids: list[str] = []
        kept_rows: list[dict] = []
        i = 0
        for r in sel:
            a = got.get(r.get("key") or "")
            if a is None:
                continue
            mm[i] = a
            sids.append(r["sid"])
            kept_rows.append(r)
            i += 1
        mm.flush()
        del mm
        if i < n:                      # 取回时丢了几张 → 截断到真实长度
            arr = np.load(path, mmap_mode="r")[:i]
            np.save(path, np.ascontiguousarray(arr))
            del arr

        cond = build_cond_matrix(ANNOTATIONS, sids).astype(np.float32)
        np.save(os.path.join(OUT, f"{split}_cond.npy"), cond)
        with open(os.path.join(OUT, f"{split}_sids.json"), "w", encoding="utf-8") as fh:
            json.dump(sids, fh)

        x = np.load(path, mmap_mode="r")
        # 形状是 (N,4,64,64)：rgb 是 (N,3,64,64)，vis 是 (N,64,64)，
        # 所以按通道取统计量要写 rgb[:, c]（早期写成 rgb[c]，把「样本轴」当成了「通道轴」）
        vis = np.asarray(x[:, 3] > 0)
        rgb = np.asarray(x[:, :3]).astype(np.float32) / 255.0
        n_vis = int(vis.sum())
        if n_vis:
            mean = [float(rgb[:, c][vis].mean()) for c in range(3)]
            std = [float(rgb[:, c][vis].std()) for c in range(3)]
        else:
            mean, std = [0.5] * 3, [0.5] * 3
        mtd = Counter(r.get("model_type") for r in kept_rows)

        meta["splits"][split] = {"n": i, "file": f"{split}.npy",
                                 "cond_file": f"{split}_cond.npy",
                                 "shape": [i, *SHAPE]}
        meta["normalization"][split] = {"rgb_mean": [round(v, 4) for v in mean],
                                        "rgb_std": [round(v, 4) for v in std],
                                        "alpha_mean": round(float(np.asarray(x[:, 3]).mean()) / 255.0, 4)}
        meta["model_type_dist"][split] = dict(mtd)
        log(f"  {split}: img ({i},4,64,64) cond {cond.shape} | "
            f"{os.path.getsize(path)/1e6:.1f} MB | {dict(mtd)}")

    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    log(json.dumps({k: v for k, v in meta.items() if k != "normalization"},
                   indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
