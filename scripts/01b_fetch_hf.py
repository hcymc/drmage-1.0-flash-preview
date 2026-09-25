#!/usr/bin/env python
"""01b_fetch_hf.py — 阶段一（HF 通道）：拉取公开的大规模「已去重」Minecraft 皮肤数据集。

为什么走这条路
--------------
自建爬虫（``01_collect.py``，MineSkin 通道）能拿到最新、带来源溯源的样本，但
API 历史窗口有限。HuggingFace 上有现成的大规模集合，其中最贴合本项目需求的是：

* ``MihaiPopa-1/minecraft-skins-1.1m-deduped-64x64``      （Apache-2.0）
    Nyuuzyou 的 20M 原始集合，用 BLAKE3 逐像素去重 + 只保留合法 64×64，
    最终 **1,107,411** 张唯一皮肤，打包成 2.6GB ZIP。
    → 正是「已去重的超大 MC 皮肤数据集」。
* ``MihaiPopa-1/minecraft-skins-1.1m-deduped-64x64-2.0``
    对上一份的规则标签（color / bright / detailed / colorful / cape / human …），
    ``tags.jsonl`` 76.5MB，按 ``skin_XXXXXXXX.png`` 文件名对齐。
    作者自述已过滤 "troll skins"（单色涂块）。
* ``summykai/minecraft-skins-captioned-900k``              （MIT）
    854,116 张：**已去重 + 已质量过滤 + 仅 Steve 模型 + 每张带一段自然语言描述**。
    描述来自 caption 模型，是阶段三「语义 / 风格标注」最省力也最可靠的来源。

产出
----
* ``data/raw/hf_dedup/minecraft_skins_64x64.zip``   1.1M 去重皮肤包
* ``data/raw/hf_dedup/tags.jsonl``                  规则标签
* ``data/raw/hf_captioned/train-XXXXX-of-00007.parquet``  带描述的样本
* ``logs/fetch_hf.log``

用法
----
    python 01b_fetch_hf.py --list           # 只看远端大小
    python 01b_fetch_hf.py --all
    python 01b_fetch_hf.py --dedup          # 只拉去重包 + 标签
    python 01b_fetch_hf.py --captioned 2    # 拉 2 个 captioned 分片
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from hfdl import ParallelDownloader, remote_size  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
LOGS = os.path.join(ROOT, "logs")

REPO_DEDUP = "MihaiPopa-1/minecraft-skins-1.1m-deduped-64x64"
REPO_TAGS = "MihaiPopa-1/minecraft-skins-1.1m-deduped-64x64-2.0"
REPO_CAP = "summykai/minecraft-skins-captioned-900k"

DEDUP_FILES = [("minecraft_skins_64x64.zip", REPO_DEDUP)]
TAG_FILES = [("tags.jsonl", REPO_TAGS)]
CAP_SHARDS = [f"data/train-{i:05d}-of-00007.parquet" for i in range(7)]


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "fetch_hf.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--captioned", type=int, default=0, metavar="N")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mirror", action="store_true", default=True)
    ap.add_argument("--official", dest="mirror", action="store_false")
    args = ap.parse_args()

    if args.all:
        args.dedup = True
        if args.captioned == 0:
            args.captioned = 2
    if not (args.dedup or args.captioned or args.list):
        ap.print_help()
        return 1

    dl = ParallelDownloader(workers=args.workers, log=log)
    url = lambda repo, fn: dl.hf_url(repo, fn, mirror=args.mirror)

    jobs: list[tuple[str, str]] = []
    if args.dedup:
        for fn, repo in DEDUP_FILES:
            jobs.append((url(repo, fn), os.path.join(RAW, "hf_dedup", fn)))
        for fn, repo in TAG_FILES:
            jobs.append((url(repo, fn), os.path.join(RAW, "hf_dedup", fn)))
    for i in range(args.captioned):
        fn = CAP_SHARDS[i]
        jobs.append((url(REPO_CAP, fn), os.path.join(RAW, "hf_captioned", os.path.basename(fn))))

    log(f"=== HF 拉取启动 | mirror={args.mirror} workers={args.workers} jobs={len(jobs)} ===")
    for u, d in jobs:
        sz = remote_size(u)
        log(f"  {sz/1e6 if sz > 0 else -1:>9.1f}MB  {os.path.relpath(d, ROOT)}")
    if args.list:
        return 0

    ok = 0
    for u, d in jobs:
        if dl.download(u, d):
            ok += 1
    log(f"=== 完成 {ok}/{len(jobs)} ===")
    return 0 if ok == len(jobs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
