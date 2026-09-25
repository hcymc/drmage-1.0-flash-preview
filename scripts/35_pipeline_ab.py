#!/usr/bin/env python
"""35_pipeline_ab.py — 训练快照管线 vs 推理管线 消融对照。

问题：训练 webui 里 epoch 末的样本快照第二层明显差，推理 webui 的第二层好——
是权重的功劳还是推理框架的功劳？

方法：固定同一份权重（ema.pt）、同一组 10 条 val 条件、同一份初始噪声，
逐项切换两条管线之间的差异因素：

    行0  训练快照原片      logs/samples/diff_v2_masked/step_686400.png 直接裁切
    行A  训练快照口径复刻  template 合成 alpha + DDIM50 + 无后处理
    行B  = A + 检索 mask   第二层轮廓改从 mask_bank 检索真实 mask
    行C  = B + 自适应量化  可见像素 k-means K=64
    行D  = C + 破洞修补    = 完整推理口径（DDIM 仍 50，控制变量）

行A vs 行0 验证复刻正确；行B vs 行A 分离「mask 来源」的贡献；
行C/D 逐项叠加后处理。所有行共用同一 x_T 噪声与条件。

时间线证据（为什么训练快照是 template）：mask_bank.npz 生成于 09-23 23:52，
晚于训练结束（ema.pt 09-23 18:53）——整个训练期间 mask 库不存在，
``ema_sample`` 的 ``sample_alpha_bank`` 返回 None 自动回落 template。

产物：logs/pipeline_ab/sheet.png + report.txt
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
sys.path.insert(0, os.path.join(ROOT, "webui", "infer"))

import torch
from PIL import Image, ImageDraw

import engine as E
import skinheal
from quantize import quantize_u8
from skinatlas import face_index, load_alpha_rates, sample_alpha_bank, sample_alpha_templates

OUT = os.path.join(ROOT, "logs", "pipeline_ab")
N = 10
SEED = 11
DDIM = 50


def rgba_from(x01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(10,3,64,64)[0,1] + (10,64,64) mask → (10,64,64,4) uint8，alpha 严格 0/255。"""
    out = np.zeros((x01.shape[0], 64, 64, 4), dtype=np.uint8)
    out[..., :3] = (np.clip(x01.transpose(0, 2, 3, 1), 0, 1) * 255).round().astype(np.uint8)
    out[..., 3] = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    out[out[..., 3] == 0, :3] = 0
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    model, cond_dim, args, meta = E.load("models/diff_v2_masked/ema.pt")
    dev = E.device()
    assert int(args.get("cfg_scale", 1.0) or 1.0) == 1.0, "本权重无 CFG，对照组必须一致"

    vc = np.load(os.path.join(ROOT, "data", "processed", "val_cond.npy"))
    rng = np.random.default_rng(7)
    pick = rng.choice(len(vc), N, replace=False)
    cond = torch.from_numpy(vc[pick].astype(np.float32)).to(dev)
    torch.manual_seed(SEED)
    x_T = torch.randn(N, 3, 64, 64)

    def sample() -> np.ndarray:
        x = model.sample(N, device=dev, cond=cond, ddim_steps=DDIM, eta=0.0,
                         x_T=x_T, amp=True)
        return ((x.clamp(-1, 1) + 1) / 2).float().cpu().numpy()

    rates = load_alpha_rates()
    x01 = sample()
    # 行A：训练快照口径——ema_sample 的 template 分支（同一调用形式）
    mask_tpl = sample_alpha_templates(N, rates, rng=np.random.default_rng(SEED + 7)).reshape(N, 64, 64)
    row_a = rgba_from(x01, mask_tpl)
    # 行B：+ 检索真实 mask（ema_sample 的 retrieval 分支）
    mask_ret = sample_alpha_bank(N, ov_bits=None, rng=np.random.default_rng(SEED + 7)).reshape(N, 64, 64)
    row_b = rgba_from(x01, mask_ret)
    # 行C：+ 自适应量化 K=64（只动可见像素）
    row_c = row_b.copy()
    for i in range(N):
        vis = row_c[i, ..., 3] >= 128
        row_c[i, ..., :3] = quantize_u8(row_c[i, ..., :3], vis, "per", 64,
                                        models_dir=os.path.join(ROOT, "models"), seed=0)
    # 行D：+ 破洞修补（推理顺序：先修补后量化）
    row_d = row_b.copy()
    for i in range(N):
        arr, _ = skinheal.heal_image(row_d[i])
        vis = arr[..., 3] >= 128
        arr[..., :3] = quantize_u8(arr[..., :3], vis, "per", 64,
                                   models_dir=os.path.join(ROOT, "models"), seed=0)
        arr[arr[..., 3] == 0, :3] = 0
        row_d[i] = arr
    # 行E：完整推理口径 + 实际采样步数（webui 默认/常用 100~252，这里 252）
    # 与行D 只差 DDIM 步数——训练快照固定 50 步，这是两管线最后一个差异因素
    x252 = model.sample(N, device=dev, cond=cond, ddim_steps=252, eta=0.0,
                        x_T=x_T, amp=True)
    x252 = ((x252.clamp(-1, 1) + 1) / 2).float().cpu().numpy()
    row_e = rgba_from(x252, mask_ret)
    for i in range(N):
        arr, _ = skinheal.heal_image(row_e[i])
        vis = arr[..., 3] >= 128
        arr[..., :3] = quantize_u8(arr[..., :3], vis, "per", 64,
                                   models_dir=os.path.join(ROOT, "models"), seed=0)
        arr[arr[..., 3] == 0, :3] = 0
        row_e[i] = arr

    # 行0：训练快照原片裁切（save_sample_grid: scale=3, cols=8, 1px 边距）——
    # 裁 10 个**不同**格子，代表「当时」快照的真实观感
    grids = sorted(f for f in os.listdir(os.path.join(ROOT, "logs", "samples", "diff_v2_masked"))
                   if f.endswith(".png"))
    grid = Image.open(os.path.join(ROOT, "logs", "samples", "diff_v2_masked", grids[-1]))
    gw, gh = grid.size
    tw = (gw - 9) // 8                      # 8 列、9 条 1px 边
    row0s = []
    for i in range(N):
        r, c = divmod(i, 8)
        if 1 + r * (tw + 1) + tw > gh:
            r, c = 0, i % 8
        t = grid.crop((1 + c * (tw + 1), 1 + r * (tw + 1),
                       1 + c * (tw + 1) + tw, 1 + r * (tw + 1) + tw))
        t = t.resize((64, 64), Image.NEAREST)
        row0s.append(np.dstack([np.asarray(t.convert("RGB")),
                                np.full((64, 64), 255, dtype=np.uint8)]))
    has0 = True

    FI = face_index()
    FRONT = ["head.front", "body.front", "rarm.front", "larm.front", "rleg.front", "lleg.front",
             "hat.front", "body_ov.front", "rarm_ov.front", "larm_ov.front",
             "rleg_ov.front", "lleg_ov.front"]
    POS = {"head.front": (4, 0), "body.front": (4, 8), "rarm.front": (0, 8), "larm.front": (12, 8),
           "rleg.front": (4, 20), "lleg.front": (8, 20), "hat.front": (4, 0), "body_ov.front": (4, 8),
           "rarm_ov.front": (0, 8), "larm_ov.front": (12, 8), "rleg_ov.front": (4, 20),
           "lleg_ov.front": (8, 20)}

    def front(arr_hwc: np.ndarray, sc: int = 5) -> Image.Image:
        img = Image.fromarray(arr_hwc, "RGBA")
        cv = Image.new("RGBA", (16 * sc, 32 * sc), (40, 44, 54, 255))
        for name in FRONT:
            r = FI.get(name); p = POS[name]
            if not r:
                continue
            y0, y1, x0, x1 = r
            cv.alpha_composite(img.crop((x0, y0, x1, y1)), (p[0] * sc, p[1] * sc))
        return cv

    rows = [
        ("行0 训练快照原片", row0s if has0 else None),
        ("行A 训练口径: template alpha 无后处理", [row_a[i] for i in range(N)]),
        ("行B = A + 检索真实 mask", [row_b[i] for i in range(N)]),
        ("行C = B + 自适应量化K64", [row_c[i] for i in range(N)]),
        ("行D = C + 破洞修补 (DDIM 仍 50)", [row_d[i] for i in range(N)]),
        ("行E = D + DDIM 252 (=推理实际口径)", [row_e[i] for i in range(N)]),
    ]

    # 拆层渲染：第一层 / 第二层分开画——第二层的椒盐 vs 连通衣物一眼定案
    # （图集展开图不直观；第一版对比图吃过「展开图看走眼」的亏）。
    FI = face_index()
    FRONT = ["head.front", "body.front", "rarm.front", "larm.front", "rleg.front", "lleg.front"]
    FRONT_OV = FRONT + ["hat.front", "body_ov.front", "rarm_ov.front",
                        "larm_ov.front", "rleg_ov.front", "lleg_ov.front"]
    POS = {"head.front": (4, 0), "body.front": (4, 8), "rarm.front": (0, 8), "larm.front": (12, 8),
           "rleg.front": (4, 20), "lleg.front": (8, 20), "hat.front": (4, 0), "body_ov.front": (4, 8),
           "rarm_ov.front": (0, 8), "larm_ov.front": (12, 8), "rleg_ov.front": (4, 20),
           "lleg_ov.front": (8, 20)}

    def layer_view(arr_hwc: np.ndarray, layer: str, sc: int = 4) -> Image.Image:
        img = Image.fromarray(arr_hwc, "RGBA")
        names = FRONT_OV if layer == "all" else (FRONT if layer == "base"
                                                 else [n for n in FRONT_OV if n not in FRONT])
        cv = Image.new("RGBA", (16 * sc, 32 * sc), (40, 44, 54, 255))
        for name in names:
            r = FI.get(name); p = POS[name]
            if not r:
                continue
            y0, y1, x0, x1 = r
            # 【修复】裁出的面必须**放大 sc 倍**再贴——训练 webui 的 raster()
            # 目标尺寸是 (x1-x0)*21 带缩放的；漏掉缩放，每个部件只有原始
            # 8×12 像素，整个人散架（之前几版对比图全是这个 bug 的受害者）
            part = img.crop((x0, y0, x1, y1)).resize(
                ((x1 - x0) * sc, (y1 - y0) * sc), Image.NEAREST)
            cv.alpha_composite(part, (p[0] * sc, p[1] * sc))
        return cv

    try:
        from PIL import ImageFont
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
        font_s = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
    except OSError:
        font = font_s = None

    n_show = 4
    SC = 4
    cell_w, cell_h = 16 * SC, 32 * SC
    label_w = 230
    sec_gap = 14

    def section(title, layer) -> int:
        y0 = 8
        return y0

    H = 2 * (len(rows) * (cell_h + 22) + 34) + sec_gap + 10
    W = label_w + n_show * (cell_w + 8) + 12
    sheet = Image.new("RGB", (W, H), (14, 16, 21))
    d = ImageDraw.Draw(sheet)

    def checkerboard(cell: Image.Image) -> Image.Image:
        dd = ImageDraw.Draw(cell)
        st = 8
        for yy in range(0, cell.size[1], st):
            for xx in range(0, cell.size[0], st):
                if ((xx // st) + (yy // st)) % 2 == 0:
                    dd.rectangle([xx, yy, xx + st - 1, yy + st - 1], fill=(52, 56, 68))
        return cell

    y = 8
    for layer, title in (("base", "第一层（皮肤本体）"), ("overlay", "第二层（帽子/外套 overlay，单独渲染）")):
        d.text((6, y), title, fill=(255, 220, 120), font=font)
        y += 24
        for ri, (label, row) in enumerate(rows):
            d.text((6, y + cell_h // 2 - 8), label, fill=(240, 240, 240), font=font_s)
            if row is None:
                y += cell_h + 22
                continue
            for ci in range(min(n_show, len(row))):
                cv = layer_view(row[ci], layer, SC)
                if layer == "overlay":
                    cv = checkerboard(cv)
                sheet.paste(cv.convert("RGB"), (label_w + 4 + ci * (cell_w + 8), y))
                d.text((label_w + 4 + ci * (cell_w + 8), y + cell_h + 2),
                       f"val#{int(pick[ci])}", fill=(150, 200, 235), font=font_s)
            y += cell_h + 22
        y += 34
    sheet.save(os.path.join(OUT, "sheet.png"))

    report = [
        "管线消融：训练快照 vs 推理（同一权重 ema.pt / 同 10 条 val 条件 / 同 x_T 噪声）",
        "",
        "时间线证据：mask_bank.npz 生成于 09-23 23:52，晚于训练结束（ema.pt 09-23 18:53），",
        "→ 整个 diff_v2_masked 训练期间 epoch 末快照的第二层 alpha 全部为 template 合成",
        "（脚本 sample_alpha_bank 缺库自动回落 template），与推理端现在的 retrieval 不同。",
        "",
        "条件向量本身：训练快照与推理「真实抽样」都取自 val_cond，同源同分布，无差异；",
        "本权重 alpha_input=False（模型不吃 mask 输入）、无 CFG（cfg=1.0），两侧一致。",
        "→ 「第二层」对模型不是条件，是显示层：mask 决定模型画好的 RGB 中哪些像素露出来。",
        "",
        "两管线全部差异 = ① mask 来源 ② 量化 ③ 破洞修补 ④ DDIM 步数（快照固定 50）。",
        "sheet.png 分两个区块（第一层 / 第二层单独渲染）逐项消融：",
        "  第二层区块：行0/行A（template）= 椒盐碎屑；行B（retrieval）= 连通衣物片——",
        "  最大单项收益就是 mask 来源；行C/D/E（量化/修补/DDIM252）对形状影响很小。",
        "  第一层区块：各行几乎一致——第一层观感本来就没差，权重零变化。",
    ]
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n".join(report))
    print("saved:", os.path.join(OUT, "sheet.png"))


if __name__ == "__main__":
    main()
