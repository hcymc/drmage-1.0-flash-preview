# -*- coding: utf-8 -*-
"""25_build_mask_bank.py — 从真实数据构建 **alpha mask 库**（按部位位分组）。

为什么需要它
------------
`skinatlas.sample_alpha_templates()` 用「逐面三态频率 + 逐像素伯努利」合成模板。
它把每个面的**覆盖率**复现得很准（overlay 面 0.179 vs 0.177），但**空间结构为零**：

    实测（scripts/_probe/_struct_audit.py，val 13200 张）
    --------------------------------------------------------------
    指标                    真实      模板       i.i.d. 理论
    面内邻接一致率          0.9568    0.8958     0.9182      <- 模板 ≈ 纯 i.i.d.
    hat 盒连通块数          5.3       31.8       -
    hat 盒连通块中位像素    11.5      1.5        -           <- 模板是孤立单点
    到最近真实 mask 的距离  1.0×      3.09×      -           <- 离流形 3 倍

也就是说：训练时模型看到的是**真实连通的帽子**，推理时看到的是**椒盐点**。
这是 conditioning 输入上的 train/test 分布错配 —— 属于「改了没用」的典型病根，
而且它还会污染导出结果（`30_generate.py:274` 把模板直接当导出图的 alpha，
于是第二层是「穿孔」的而不是帽子）。

修法：**不再合成 mask，直接从真实数据里检索。**
p(mask) 用经验分布代替参数化近似，训练/推理两侧**恒等分布**，结构 100% 真实。

产出
----
``models/mask_bank.npz``
  * ``masks``  (N, 512) uint8 —— N 张真实 alpha，逐行 packbits 压缩
  * ``bits``   (N, 4)  int8   —— 每张的 (ov_hat, ov_body, ov_arm, ov_leg)
  * ``src``    (N,)    int32  —— 来源在 train 里的行号（可追溯）

用法
----
    python scripts/25_build_mask_bank.py --limit 40000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))

DATA = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
VIS = 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=40000,
                    help="最多收多少张（真实 mask 的多样性远超需要，4 万足够）")
    ap.add_argument("--out", default=os.path.join(MODELS, "mask_bank.npz"))
    a = ap.parse_args()

    t0 = time.time()
    x = np.load(os.path.join(DATA, f"{a.split}.npy"), mmap_mode="r")
    n = min(a.limit, x.shape[0])
    print(f"{a.split}.npy {x.shape} -> 取前 {n} 张", flush=True)

    # 只读 alpha 通道：memmap 按页读，channel-3 切片实际只物化 1/4 数据
    al = np.asarray(x[:n, 3])
    vis = (al >= VIS).astype(np.uint8).reshape(n, 4096)
    print(f"alpha 读完 {time.time()-t0:.1f}s，可见率 {vis.mean():.4f}", flush=True)

    ov_path = os.path.join(DATA, f"{a.split}_ov4.npy")
    if os.path.isfile(ov_path):
        bits = np.load(ov_path)[:n].astype(np.int8)
    else:
        bits = np.zeros((n, 4), dtype=np.int8)
        print(f"[!] 缺 {os.path.basename(ov_path)}，部位位全 0（请先跑 24_build_ov_bits.py）")

    packed = np.packbits(vis, axis=1)          # (n, 512)
    np.savez_compressed(a.out, masks=packed, bits=bits,
                        src=np.arange(n, dtype=np.int32))
    sz = os.path.getsize(a.out) / 1024 ** 2
    print(f"[ok] {a.out}  {n} 张  {sz:.1f}MB  用时 {time.time()-t0:.1f}s")

    # ---- 自检：库里的 mask 必须带真实结构（否则这个库白建了）----
    key = bits.sum(axis=1)
    hist = {int(k): int((key == k).sum()) for k in range(5)}
    print(f"部位位组合分布 {hist}")

    def n_components(m: np.ndarray) -> int:
        if not m.any():
            return 0
        lab = np.where(m, np.arange(1, m.size + 1).reshape(m.shape), 0).astype(np.int32)
        for _ in range(80):
            p = lab.copy()
            p[1:, :] = np.maximum(p[1:, :], lab[:-1, :])
            p[:-1, :] = np.maximum(p[:-1, :], lab[1:, :])
            p[:, 1:] = np.maximum(p[:, 1:], lab[:, :-1])
            p[:, :-1] = np.maximum(p[:, :-1], lab[:, 1:])
            p = np.where(m, p, 0)
            if np.array_equal(p, lab):
                break
            lab = p
        return int(len(np.unique(lab[lab > 0])))

    hats = [n_components(vis[i].reshape(64, 64)[0:8, 8:16])
            for i in np.random.default_rng(0).choice(n, 200, replace=False)]
    print(f"抽样 200 张的 hat.top 连通块数 中位 {np.median(hats):.1f}"
          f"（真实 hat 盒基准 5.3；合成模板是 31.8）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
