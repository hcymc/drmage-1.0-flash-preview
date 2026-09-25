"""sourceio.py — 从**原始容器**里按 key 取回样本字节（不落中间文件）。

设计口径
--------
清洗、标注只吃宽表 CSV；**只有构建训练集时才第二次碰像素**。
那次遍历不需要把图片解出来存成中间文件，而是直接按 ``(source, key)``
从原始容器里取：

======================  ====================================================
source                  取回方式
======================  ====================================================
``hf_dedup``            ``minecraft_skins_64x64.zip`` 里按成员名（``key``）
``hf_captioned``        parquet 分片里按 ``sha256`` 对（列里是裸 base64）
``mineskin``            磁盘上的松散 PNG（``file_path``）
======================  ====================================================

zip 与 parquet 都走**一趟顺序扫描 + 目标集合过滤**：
实测顺序读 zip 10,959/s、读 parquet 也不慢，而随机 seek 要贵得多。
所以接口是「给我一批想要的 key，我扫一遍把它们捞出来」，而不是逐条 ``get()``。
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import zipfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")

ZIP_DEDUP = os.path.join(DATA, "raw", "hf_dedup", "minecraft_skins_64x64.zip")
DIR_CAPTIONED = os.path.join(DATA, "raw", "hf_captioned")
DIR_MINEFILES = os.path.join(DATA, "raw", "mineskin")


def _decode(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data))
    im.load()
    return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def fetch_zip(zip_path: str, wanted: set[str], log=print) -> dict[str, bytes]:
    """顺序扫一遍 zip，把 ``wanted`` 里的成员（按 basename 匹配）取出来。"""
    out: dict[str, bytes] = {}
    if not wanted:
        return out
    with zipfile.ZipFile(zip_path) as zf:
        total = 0
        for info in zf.infolist():
            total += 1
            if not info.filename.lower().endswith(".png"):
                continue
            base = os.path.basename(info.filename)
            if base not in wanted:
                continue
            try:
                out[base] = zf.read(info)
            except Exception:
                continue
            if len(out) == len(wanted):
                break
        log(f"    zip 扫过 {total} 个成员，命中 {len(out)}/{len(wanted)}")
    return out


def fetch_parquet(dir_path: str, wanted: set[str], batch: int = 2048,
                  log=print) -> dict[str, bytes]:
    """顺序扫 parquet 分片；``image`` 列是裸 base64，按解码后 sha256 前 16 位匹配 key。"""
    import pyarrow.parquet as pq

    out: dict[str, bytes] = {}
    if not wanted:
        return out
    shards = sorted(os.path.join(dir_path, f) for f in os.listdir(dir_path)
                    if f.endswith(".parquet"))
    scanned = 0
    for shard in shards:
        if len(out) == len(wanted):
            break
        pf = pq.ParquetFile(shard)
        for b in pf.iter_batches(batch_size=batch):
            d = b.to_pydict()
            imgs = d.get("image")
            if imgs is None:
                continue
            for cell in imgs:
                scanned += 1
                if isinstance(cell, str):
                    s = cell.strip()
                    if s.startswith("data:"):
                        s = s.split(",", 1)[-1]
                    try:
                        data = base64.b64decode(s, validate=False)
                    except Exception:
                        continue
                elif isinstance(cell, (bytes, bytearray)):
                    data = bytes(cell)
                elif isinstance(cell, dict):
                    data = cell.get("bytes")
                    if not data:
                        continue
                    data = bytes(data)
                else:
                    continue
                key = hashlib.sha256(data).hexdigest()[:16] + ".png"
                if key in wanted and key not in out:
                    out[key] = data
            if len(out) == len(wanted):
                break
    log(f"    parquet 扫过 {scanned} 行，命中 {len(out)}/{len(wanted)}")
    return out


def fetch(source: str, wanted: set[str], log=print) -> dict[str, bytes]:
    """按来源分发。``wanted`` 是该来源的 key 集合。"""
    if source == "hf_dedup":
        return fetch_zip(ZIP_DEDUP, wanted, log=log)
    if source == "hf_captioned":
        return fetch_parquet(DIR_CAPTIONED, wanted, log=log)
    raise ValueError(f"来源 {source} 不支持按 key 批量取回（松散文件请直接读路径）")


def read_file(root: str, rel_path: str) -> bytes:
    with open(os.path.join(root, rel_path.replace("/", os.sep)), "rb") as fh:
        return fh.read()


def to_chw(data: bytes) -> np.ndarray:
    return np.transpose(_decode(data), (2, 0, 1))
