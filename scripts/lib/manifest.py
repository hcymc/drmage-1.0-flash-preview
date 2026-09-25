"""manifest.py — 全流程共享的「宽表清单」：一趟算完特征，下游只读 CSV。

为什么要宽表
------------
本机实测（6000 张真实样本，分段计时）::

    从 zip 顺序读         10,959 /s
    UV 校验+骨架判别+质量   687 /s
    创建小文件             13 ~ 50 /s     ← 真正的瓶颈

结论：**瓶颈在「产出中间文件」，不在解析**。所以正确做法不是「先物化成
15 万张小图 / 一个大 bin，再让清洗、标注、构建数据集各读一遍」，而是
**把能算的特征全挤进摄取那一趟**（反正图已经被解码在内存里了），
只落一份宽表 CSV。

于是整条链路的像素遍历次数降到 **两趟**：

1. 摄取（02b / 02c）：读一次原始容器 → 写宽表 CSV（零中间文件）
2. 构建数据集（04_dataset）：按 CSV 里的 (source, key) 再读一次 → 写训练用 npy

清洗（02_clean）与标注（10_label）**完全不碰像素**，只做 CSV 运算。

字段来源
--------
* 来自 ``skinuv.SkinReport``：UV 结构、骨架类型、质量三维
* 来自 ``labeling``：调色板、HSV、面部、对称度、复杂度档
* ``dhash``：16 位十六进制，供 02_clean 做近似去重（无需再读像素）

CSV 是扁平表，所以复杂结构（调色板 / HSV / 面部）序列化成 JSON 字符串存放，
读取时用 :func:`json_field` 还原。
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter

import numpy as np

from labeling import (classify_tone, complexity_class, edge_density,
                      eye_region_features, hsv_stats, palette, symmetry_score)
from skinuv import dhash_hex

#: 宽表列（摄取 02b/02c 产出的原始清单，与最终 clean_manifest 共用）
FIELDS = [
    # 身份 / 定位
    "source", "key", "sha256", "file_path", "width", "height", "mode", "is_legacy",
    # UV 结构
    "alpha_min", "alpha_max", "transparent_ratio", "overlay_ratio",
    "face_opaque_ratio", "torso_opaque_ratio",
    # 骨架类型（Steve / Alex）
    "model_type", "model_conf", "arm_ratio", "limb_l_ratio", "limb_r_ratio",
    # 质量三维 + 分档
    "quality_tier", "quality_reason",
    "n_colors", "dom_ratio", "edge_density", "color_entropy",
    # 标注用特征（10_label 直接消费，不再读像素）
    "dhash", "symmetry", "palette_json", "hsv_json", "face_json",
    "tone", "sat_class", "cpx_class", "is_near_solid", "pal_entropy",
    # 外部标签
    "hf_tags", "caption_len",
]

#: 02_clean 在此之上补的三列
CLEAN_EXTRA = ["sid", "bucket", "split"]


def json_field(s: str):
    """读 CSV 里的 JSON 字符串列；空值返回 None。"""
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _entropy(shares) -> float:
    xs = [float(x) for x in shares if x > 0]
    if not xs:
        return 0.0
    return float(-sum(x * np.log2(x) for x in xs))


def features_for(arr: np.ndarray, rep) -> dict:
    """从已解码的 (64,64,4) uint8 与 ``SkinReport`` 生成一行宽表数据。

    调用方（02b/02c）已经为了 UV 校验把图解码在内存里了，这里只是顺手多算
    几个便宜统计量——避免下游为了这些数字再遍历一遍全部像素。
    """
    px = arr[..., :3][arr[..., 3] > 0]
    hsv = hsv_stats(px)
    pal = palette(arr, k=8)
    face = eye_region_features(arr)
    sym = symmetry_score(arr)
    ed = edge_density(arr)
    n_uniq = int(np.unique(px.reshape(-1, 3), axis=0).shape[0]) if px.size else 0
    tone = classify_tone(hsv) if hsv else {}

    return {
        "source": "", "key": "", "sha256": "", "file_path": "",
        "width": rep.width, "height": rep.height, "mode": rep.mode,
        "is_legacy": int(rep.is_legacy),
        "alpha_min": rep.alpha_min, "alpha_max": rep.alpha_max,
        "transparent_ratio": rep.transparent_ratio,
        "overlay_ratio": rep.overlay_ratio,
        "face_opaque_ratio": rep.face_opaque_ratio,
        "torso_opaque_ratio": rep.torso_opaque_ratio,
        "model_type": rep.model_type, "model_conf": rep.model_conf,
        "arm_ratio": rep.arm_ratio,
        "limb_l_ratio": rep.limb_l_ratio, "limb_r_ratio": rep.limb_r_ratio,
        "quality_tier": rep.quality_tier, "quality_reason": rep.quality_reason,
        "n_colors": rep.unique_colors, "dom_ratio": rep.dom_ratio,
        "edge_density": rep.edge_density, "color_entropy": rep.color_entropy,
        "dhash": dhash_hex(arr),
        "symmetry": sym,
        "palette_json": json.dumps(pal, ensure_ascii=False),
        "hsv_json": json.dumps(hsv, ensure_ascii=False),
        "face_json": json.dumps(face, ensure_ascii=False),
        "tone": tone.get("tone", ""),
        "sat_class": tone.get("saturation_class", ""),
        "cpx_class": complexity_class(n_uniq, ed),
        "is_near_solid": int(n_uniq <= 3),
        "pal_entropy": round(_entropy([c["share"] for c in pal]), 3),
        "hf_tags": "", "caption_len": 0,
    }


def write_rows(path: str, rows: list[dict], extra: list[str] | None = None,
               mode: str = "a") -> None:
    fields = FIELDS + list(extra or [])
    new = not os.path.exists(path) or mode == "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def read_rows(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_clean(path: str) -> tuple[list[dict], list[str]]:
    """读 ``clean_manifest.csv``，返回 (行, 列名)。"""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        return list(r), list(r.fieldnames or [])


# --------------------------------------------------------------------------
# 宽表行 -> 标注记录（10_label base 直接消费，不再读像素）
# --------------------------------------------------------------------------

def annotation_from_row(row: dict) -> dict:
    hsv = json_field(row.get("hsv_json")) or {}
    pal = json_field(row.get("palette_json")) or []
    face = json_field(row.get("face_json")) or {}
    tone = (row.get("tone") or "", row.get("sat_class") or "")

    def f(k, d=0.0):
        try:
            return float(row.get(k) or d)
        except (TypeError, ValueError):
            return d

    def i(k, d=0):
        try:
            return int(float(row.get(k) or d))
        except (TypeError, ValueError):
            return d

    return {
        "sid": row.get("sid", ""),
        "path": row.get("file_path", ""),
        "source": row.get("source", ""),
        "sha256": row.get("sha256", ""),
        "key": row.get("key", ""),
        "width": i("width"), "height": i("height"),
        "is_legacy": bool(i("is_legacy")),
        "split": row.get("split", ""),
        "uv_valid": True,
        "uv_reason": "ok",
        "uv": {
            "alpha_min": i("alpha_min", 255), "alpha_max": i("alpha_max"),
            "transparent_ratio": f("transparent_ratio"),
            "overlay_ratio": f("overlay_ratio"),
            "face_opaque_ratio": f("face_opaque_ratio"),
            "torso_opaque_ratio": f("torso_opaque_ratio"),
        },
        "model": {
            "type": row.get("model_type", ""),
            "confidence": f("model_conf"),
            "method": "rule",
            "evidence": {"arm_ratio": f("arm_ratio"),
                         "limb_l_ratio": f("limb_l_ratio"),
                         "limb_r_ratio": f("limb_r_ratio")},
        },
        "palette": pal,
        "color": hsv,
        "tone": {"tone": tone[0], "saturation_class": tone[1]},
        "transparency": {
            "transparent_ratio": f("transparent_ratio"),
            "overlay_used": f("overlay_ratio") > 0.02,
            "overlay_fill_ratio": f("overlay_ratio"),
        },
        "complexity": {
            "unique_colors": i("n_colors"),
            "palette_k8_entropy": f("pal_entropy"),
            "edge_density": f("edge_density"),
            "symmetry": f("symmetry"),
            "class": row.get("cpx_class", ""),
            "is_near_solid": bool(i("is_near_solid")),
        },
        "face": face,
        "quality": {
            "tier": row.get("quality_tier", ""),
            "tier_reason": row.get("quality_reason", ""),
            "dom_ratio": f("dom_ratio"),
            "edge_density": f("edge_density"),
            "color_entropy": f("color_entropy"),
        },
        "hf_tags": (row.get("hf_tags") or "").split("|") if row.get("hf_tags") else [],
    }


def model_type_counter(rows) -> dict:
    return dict(Counter((r.get("model_type") or "?") for r in rows))
