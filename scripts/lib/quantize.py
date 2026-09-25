#!/usr/bin/env python
"""调色板量化：**全项目唯一一份实现**。

为什么必须只有一份
------------------
这个项目已经反复吃过「同名不同实现」的亏（``flat_frac``、各种「覆盖率」、
``quantize_to_palette`` 漏掉的尺度检查）。而量化历史上在两条链路上各写了一份：

* ``scripts/30_generate.py``（导出）—— 只有全局板
* ``webui/infer/engine.py``（推理平台）—— 有 ``global`` / ``adaptive`` 两种

口径一旦分叉，「平台上看着好、导出却不是一回事」就会再次发生。
现在统一到这里，两边都只是薄封装。

实测口径（``scripts/_probe/_l1_quant_ab.py`` → ``logs/_l1_ab/report.txt``）
------------------------------------------------------------------------
同一批**真实条件**的模型输出（32 张）与 val 真实皮肤（256 张），指标同口径：

    方案         唯一色(均/中位)  精确相等率   饱和     结论
    真实(val)      134 / 66        0.475      0.353    —— 靶子
    原样          1454 / 1490      0.081      0.362    ±几色阶抖动盖住了结构
    全局 k48         26            0.659      0.298    过冲 + 偏灰(−16%)
    全局 k96         43            0.598      0.314    过冲 + 偏灰(−11%)
    逐图 k16         16            0.636      0.350    过度平涂（细节被抹掉）
    逐图 k48         48            0.523      0.355    ✓
    逐图 k64         64            0.498      0.355    ✓ 最贴真实

结论：**逐图自适应（K≈64 最贴、48 略硬）**。全局板会过冲且把图拉灰 ——
所以全局板只保留给 A/B 对照，不再是默认。

⚠️ 量化只能「封顶色数、把抖动做成硬色块」，**修不了边界位置**
（那是模型的问题，见 ``docs/first_layer_detail_fix.md``）。
"""

from __future__ import annotations

import os

import numpy as np

MODES = ("off", "per", "global")


def nearest(px: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """把 ``(N,3)`` 的像素吸附到 ``(K,3)`` 调色板的最近色。分块算，防爆内存。"""
    out = np.empty_like(px, dtype=np.float32)
    step = 1 << 16
    for i in range(0, len(px), step):
        seg = px[i:i + step]
        d = ((seg[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        out[i:i + step] = pal[d.argmin(axis=1)]
    return out


def kmeans_pp(px: np.ndarray, k: int, seed: int = 0, iters: int = 24) -> np.ndarray:
    """极小 k-means++（numpy 自研，不为这点事引 sklearn）。

    用 k-means++ 初始化而不是纯随机取点：单张图只有 ~2000 个可见像素，
    随机初始化会让同一张图在不同 seed 下差出几色，评估时不可复现。
    """
    x = np.asarray(px, dtype=np.float32)
    k = int(min(k, len(x)))
    if k <= 0 or len(x) == 0:
        return x[:1]
    if len(x) == k:
        return x
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    d2 = ((x - centers[0]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        # 【修复】float32 下 d2.sum() 可能是极小正数，除出来的概率向量
        # 求和 ≠1（差 1e-7），numpy 的 choice 会直接抛
        # "probabilities do not sum to 1"。近纯色图（全部像素几乎同色）
        # 必然触发——推理侧只量化生成图所以从没暴露，给真实皮肤做
        # 预处理时第 1 张纯色图就炸。改 float64 + 显式重归一 + 零和兜底。
        p = np.asarray(d2, dtype=np.float64)
        s = float(p.sum())
        if not np.isfinite(s) or s <= 0.0:
            centers.append(x[int(rng.integers(len(x)))])
        else:
            p /= s
            p = np.clip(p, 0.0, None)
            p /= p.sum()
            centers.append(x[int(rng.choice(len(x), p=p))])
        d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(axis=1))
    c = np.stack(centers).astype(np.float32)
    for _ in range(iters):
        lab = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        new = c.copy()
        for j in range(len(c)):
            m = lab == j
            if m.any():
                new[j] = x[m].mean(axis=0)
        if np.allclose(new, c, atol=0.25):
            c = new
            break
        c = new
    return c


def load_global_palette(k: int, models_dir: str) -> np.ndarray:
    """读 ``palette_k{K}.npy``，**统一到 0-255 尺度**。

    尺度检查不能省：仓库里的板有的存 [0,1]、有的存 [0,255]。
    缺这一行的后果是把整张图量化成一片白（历史上真的发生过，
    而且 ``30_generate`` 与 ``engine`` 两边曾经一个有一个没有）。
    """
    p = os.path.join(models_dir, f"palette_k{k}.npy")
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"找不到全局调色板 {p}（先跑 scripts/23_build_palette.py --ks {k}）")
    pal = np.load(p).astype(np.float32)
    if pal.max() > 1.5:
        pal = pal / 255.0
    return pal * 255.0


def quantize_u8(rgb: np.ndarray, vis: np.ndarray, mode: str = "per", k: int = 64,
                palette: np.ndarray | None = None, models_dir: str | None = None,
                seed: int = 0) -> np.ndarray:
    """把 ``rgb`` (64,64,3) **uint8** 的可见像素吸附到调色板，返回同形状 uint8。

    * ``mode="off"`` 或 ``k<=0``：原样返回。
    * ``mode="per"``：**逐图自适应** k-means（推荐，见模块 docstring）。
    * ``mode="global"``：用 ``palette`` 或 ``models_dir/palette_k{k}.npy``。

    只改 ``vis`` 为真的像素 —— 透明像素的 RGB 必须原样保留，
    否则「先量化再乘 alpha」的调用方会拿到被污染的颜色。
    """
    if mode in ("off", "", None) or k <= 0:
        return rgb
    vis = np.asarray(vis, dtype=bool)
    if not vis.any():
        return rgb
    out = rgb.copy()
    px = rgb[vis].astype(np.float32)
    if mode == "global":
        if palette is None:
            if models_dir is None:
                raise ValueError("mode=global 需要 palette 或 models_dir")
            palette = load_global_palette(k, models_dir)
        pal = np.asarray(palette, dtype=np.float32)
        if pal.max() <= 1.5:
            pal = pal * 255.0
    else:
        pal = kmeans_pp(px, k, seed=seed)
    out[vis] = np.clip(np.rint(nearest(px, pal)), 0, 255).astype(np.uint8)
    return out
