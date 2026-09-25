"""labeling.py — 标注用的特征提取。

设计原则
--------
* **可复核**：每个字段都能被一个具体函数重算出来，不依赖不可解释的黑盒。
* **分层次**：规则可推导的字段（全量）与启发式/模型字段（抽样）用不同前缀区分，
  并在 ``method`` 里声明来源（``rule`` / ``cluster`` / ``model`` / ``manual``）。
* **不装懂**：启发式风格标签一律带 ``confidence``，且允许为 ``null``。

调色板提取用 PIL 的自适应中位切分量化（``convert('P', ADAPTIVE)``），
而不是逐图跑 KMeans——1.2 万张图上后者代价过高，中位切分对像素画足够且快两个数量级。
"""

from __future__ import annotations

import colorsys
import math
from collections import Counter

import numpy as np
from PIL import Image

#: 肤色 / 中性色判定的 HSV 阈值
NEUTRAL_SAT = 0.16
DARK_V = 0.35
BRIGHT_V = 0.72
HIGH_SAT = 0.62


def load_rgba(path: str) -> np.ndarray:
    a = np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)
    if a.shape != (64, 64, 4) and a.shape != (64, 32, 4):
        raise ValueError(f"unexpected shape {a.shape}")
    return a


def visible_pixels(arr: np.ndarray) -> np.ndarray:
    """返回形状 (M,3) 的可见像素 RGB。"""
    m = arr[..., 3] > 0
    return arr[..., :3][m]


# --------------------------------------------------------------------------
# 规则可推导特征
# --------------------------------------------------------------------------

def palette(arr: np.ndarray, k: int = 8) -> list[dict]:
    """主色调色板：中位切分量化后按占比排序，返回 top-k。"""
    im = Image.fromarray(arr, "RGBA")
    # 只在可见像素上做量化：把透明像素替换成一个不会被选中的哨兵色前先抠出可见区
    m = arr[..., 3] > 0
    if not m.any():
        return []
    ys, xs = np.nonzero(m)
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    q = Image.fromarray(crop, "RGBA").convert("RGBA")
    # 用 alpha 去背到黑底，避免透明像素干扰量化结果
    bg = Image.new("RGBA", q.size, (0, 0, 0, 255))
    flat = Image.alpha_composite(bg, q).convert("RGB")
    pal_img = flat.convert("P", palette=Image.ADAPTIVE, colors=k)
    counts = Counter(pal_img.getdata())
    palette_rgb = pal_img.getpalette()
    total = sum(counts.values()) or 1
    out = []
    for idx, cnt in counts.most_common(k):
        r, g, b = palette_rgb[idx * 3: idx * 3 + 3]
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}",
                    "rgb": [r, g, b],
                    "share": round(cnt / total, 4)})
    return out


def hsv_stats(px: np.ndarray) -> dict:
    if px.size == 0:
        return {}
    rgb = px.astype(np.float32) / 255.0
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    # 向量化 hue 计算
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    d = np.maximum(mx - mn, 1e-6)
    h = np.zeros_like(v)
    m1 = (mx == r)
    m2 = (mx == g) & ~m1
    m3 = ~m1 & ~m2
    h[m1] = ((g[m1] - b[m1]) / d[m1]) % 6
    h[m2] = ((b[m2] - r[m2]) / d[m2]) + 2
    h[m3] = ((r[m3] - g[m3]) / d[m3]) + 4
    h = h * 60.0
    hist, _ = np.histogram(h, bins=12, range=(0, 360), weights=None)
    return {
        "hue_mean_deg": round(float(h.mean()), 1),
        "sat_mean": round(float(s.mean()), 3),
        "val_mean": round(float(v.mean()), 3),
        "hue_hist12": [int(x) for x in hist],
        "neutral_ratio": round(float((s < NEUTRAL_SAT).mean()), 3),
        "dark_ratio": round(float((v < DARK_V).mean()), 3),
        "bright_ratio": round(float((v > BRIGHT_V).mean()), 3),
        "high_sat_ratio": round(float((s > HIGH_SAT).mean()), 3),
    }


def edge_density(arr: np.ndarray) -> float:
    """可见区域内的相邻像素差异比例（简化 Sobel）。"""
    m = arr[..., 3] > 0
    if m.sum() < 4:
        return 0.0
    g = arr[..., :3].astype(np.int16).sum(axis=2)
    dx = np.abs(np.diff(g, axis=1)) > 24
    dy = np.abs(np.diff(g, axis=0)) > 24
    mm_x = m[:, 1:] & m[:, :-1]
    mm_y = m[1:, :] & m[:-1, :]
    denom = mm_x.sum() + mm_y.sum()
    if denom == 0:
        return 0.0
    return round(float((dx & mm_x).sum() + (dy & mm_y).sum()) / float(denom), 4)


def symmetry_score(arr: np.ndarray) -> float:
    """左右对称度（在可见区域内比较左右镜像后的颜色一致性）。"""
    m = arr[..., 3] > 0
    flip = arr[:, ::-1]
    both = m & (flip[..., 3] > 0)
    if both.sum() < 16:
        return 0.0
    diff = np.abs(arr[..., :3].astype(np.int16) - flip[..., :3].astype(np.int16)).sum(axis=2)
    return round(float((diff[both] < 30).mean()), 4)


def eye_region_features(arr: np.ndarray) -> dict:
    """面部正面 8×8 里的「眼睛带」分析：像素画角色脸的典型判据。"""
    face = arr[8:16, 8:16]
    a = face[..., 3] > 0
    if a.sum() == 0:
        return {"face_dark_ratio": 0.0, "face_has_contrast_pair": False}
    v = face[..., :3].astype(np.float32).mean(axis=2) / 255.0
    dark = (v < 0.3) & a
    bright = (v > 0.7) & a
    return {
        "face_dark_ratio": round(float(dark.sum() / max(a.sum(), 1)), 3),
        "face_has_contrast_pair": bool(dark.sum() >= 2 and bright.sum() >= 2),
    }


def classify_tone(hsv: dict) -> dict:
    """明暗 / 饱和风格分类（阈值规则，可复核）。"""
    if not hsv:
        return {}
    v, s, neu = hsv["val_mean"], hsv["sat_mean"], hsv["neutral_ratio"]
    if neu > 0.75:
        tone = "grayscale"
    elif v < 0.35:
        tone = "dark"
    elif v > 0.68:
        tone = "bright"
    else:
        tone = "mid"
    if s > 0.55:
        sat = "high_saturation"
    elif s < 0.2:
        sat = "low_saturation"
    else:
        sat = "mid_saturation"
    return {"tone": tone, "saturation_class": sat}


def complexity_class(n_colors: int, edge: float) -> str:
    if n_colors <= 3:
        return "near_solid"
    if n_colors <= 8:
        return "simple"
    if n_colors <= 24:
        return "medium"
    if n_colors <= 64:
        return "detailed"
    return "very_detailed"


def feature_vector(arr: np.ndarray) -> np.ndarray:
    """给聚类用的低维特征：颜色直方图 + 结构统计 + 降采样外观。"""
    px = visible_pixels(arr)
    parts = []
    # 1) 12-bin hue 直方图（归一化）
    if px.size:
        hsv = hsv_stats(px)
        hist = np.array(hsv["hue_hist12"], dtype=np.float32)
        parts.append(hist / max(hist.sum(), 1))
        parts.append(np.array([hsv["sat_mean"], hsv["val_mean"],
                               hsv["neutral_ratio"], hsv["dark_ratio"],
                               hsv["bright_ratio"], hsv["high_sat_ratio"]], dtype=np.float32))
    else:
        parts.append(np.zeros(12, np.float32))
        parts.append(np.zeros(6, np.float32))
    # 2) 结构统计
    m = arr[..., 3] > 0
    apx = px if px.size else np.zeros((1, 3), np.uint8)
    parts.append(np.array([
        float(m.mean()),
        edge_density(arr),
        symmetry_score(arr),
        min(np.unique(apx.reshape(-1, 3), axis=0).shape[0], 256) / 256.0,
    ], dtype=np.float32))
    # 3) 8×8 RGBA 降采样外观（最能体现「看起来像什么」）
    small = np.array(Image.fromarray(arr, "RGBA").resize((8, 8), Image.BILINEAR),
                     dtype=np.float32) / 255.0
    parts.append(small.reshape(-1))
    return np.concatenate(parts).astype(np.float32)
