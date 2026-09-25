#!/usr/bin/env python
"""30_generate.py — 阶段五之一：用训练好的模型批量生成皮肤。

用法
----
    python 30_generate.py --model dcgan     --n 256
    python 30_generate.py --model diffusion --n 256
    python 30_generate.py --model diffusion --n 64 \
        --cond-spec '{"tone":"dark","hue_deg":0,"complexity_class":"detailed"}'

产出
----
* ``data/generated/<run>/skin_XXXX.png``   64×64 RGBA PNG（alpha 已阈值化）
* ``data/generated/<run>/manifest.csv``    生成参数与逐样本统计
* ``data/generated/<run>/contact_sheet.png``
* ``logs/generate_<run>.json``
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from labelset import (COND_DIM, describe, describe_ov_bits,  # noqa: E402
                      ov_bits_from_spec, ov_bits_to_box_paint, vector_from_spec)
from models import DDPM, DCGANDiscriminator, DCGANGenerator, SmallUNet, count_params  # noqa: E402
from quantize import quantize_u8  # noqa: E402
from skinatlas import (load_alpha_rates, sample_alpha_bank,  # noqa: E402
                       sample_alpha_templates)
from skinuv import LIMB_L_MIN, analyze, quality_metrics, skin_contact_sheet  # noqa: E402
from trainutil import load_ckpt, seed_everything  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "data", "generated")
LOGS = os.path.join(ROOT, "logs")
MODELS = os.path.join(ROOT, "models")


def build_dcgan(ckpt_path: str, device):
    """按 checkpoint 里存的 ``args`` **原样重建**生成器结构。

    必须把所有影响结构的开关都读回来（``palette_k`` / ``film`` / ``palette_hue_bins``）：
    漏掉任何一个，``load_state_dict`` 就会报一堆 missing/unexpected keys，
    或者更糟 —— 结构对得上但语义不同（例如 palette 头被换成普通 tanh 头，
    权重键名不一致时反而会静默不加载）。``init_palette`` 不用传，
    调色板本身就在 state_dict 里，会被覆盖掉。
    """
    ck = load_ckpt(ckpt_path, map_location="cpu")
    a = ck.get("args", {})
    cond_dim = COND_DIM if a.get("cond") else 0
    G = DCGANGenerator(
        z_dim=a.get("z_dim", 128), base=a.get("g_base", 256), cond_dim=cond_dim,
        palette_k=a.get("palette_k", 0),
        film=bool(a.get("film")),
        palette_hue_bins=a.get("palette_hue_bins", 0),
        alpha_mode=a.get("alpha_mode", "head"),
    )
    G.load_state_dict(ck["G"] if "G" in ck else ck)
    return G.to(device).eval(), cond_dim, ck


def build_diffusion(ckpt_path: str, device, prefer_ema: bool = True):
    ck = load_ckpt(ckpt_path, map_location="cpu")
    a = ck.get("args", {})
    cond_dim = COND_DIM if a.get("cond") else 0
    # 输入通道数与噪声调度**必须从 checkpoint 的 args 回读**：
    # 早期这里写死 in_ch=4 且不给 schedule，一旦模型是「3 通道 RGB + UV 模板 alpha」
    # 或换了调度，要么 load_state_dict 报错，要么采样过程与训练不一致（静默出坏图）。
    #
    # ``alpha_input`` 同理：模型把 UV 模板 alpha 当作第 4 个**输入**通道
    # （扩散张量仍是 3 通道 RGB），所以 ``in_ch = channels + 1``、``out_ch = channels``。
    # 漏掉这个开关，加载时 in_ch 少 1、out_ch 多 1，两边都对不上。
    channels = int(a.get("channels", 3))
    alpha_input = bool(a.get("alpha_input"))
    unet = SmallUNet(in_ch=channels + (1 if alpha_input else 0),
                     base=a.get("base", 48), t_dim=a.get("t_dim", 256),
                     cond_dim=cond_dim, out_ch=channels)
    # EMA 权重选择。``latest.pt`` 里同时存了 ``model``（原始权重）和 ``ema``
    # （EMA shadow）；``ema.pt`` 的 ``model`` 键本身就是 EMA shadow。
    # 修复说明：旧版这里有个**死参数**——签名里的 ``prefer_ema`` 从来没被读过，
    # 拿 latest.pt 出图时永远加载原始权重（EMA 才是训练交付口径），
    # 且调用方毫无办法察觉。现在 ``prefer_ema=True``（默认）时优先取 ``ck["ema"]``。
    state = None
    if prefer_ema and isinstance(ck.get("ema"), dict) \
            and any(torch.is_tensor(v) for v in ck["ema"].values()):
        state = ck["ema"]
    if state is None:
        state = ck.get("model", ck)
    # ema.pt 只存了 shadow 权重
    if "model" not in ck and isinstance(ck, dict) and all(torch.is_tensor(v) for v in ck.values()):
        state = ck
    # 训练时存的是整个 DDPM：既有 ``unet.*`` 前缀，又混着 DDPM 自己的 buffer
    # （``betas`` / ``alphas_cumprod`` / ``sqrt_acp`` / ``sqrt_1macp``）。
    # 这里只建了 UNet，所以要把 ``unet.`` 前缀剥掉、并把 buffer 丢掉。
    # 早期的判断是「所有键都以 unet. 开头才剥」——因为 buffer 没有前缀，
    # 这个条件永远为假，于是前缀一直没被剥掉，加载必然失败。
    if any(isinstance(k, str) and k.startswith("unet.") for k in state):
        state = {k[len("unet."):]: v for k, v in state.items()
                 if isinstance(k, str) and k.startswith("unet.")}
    missing, unexpected = unet.load_state_dict(dict(state), strict=False)
    # 旧 checkpoint（训练时还没有 FiLM）会缺 ``.film.`` 的键 —— 那些层是**零初始化**的，
    # 缺省值恰好等价于「无 FiLM」，所以可以放行。除此之外的任何缺失/多余
    # 都说明结构真的对不上（比如 channels / alpha_input 读错），必须直接报错，
    # 不能静默用随机权重出图。
    bad = [k for k in missing if ".film." not in k]
    if bad or unexpected:
        raise RuntimeError(
            f"权重与结构不匹配（{os.path.basename(ckpt_path)}）："
            f"missing={bad[:6]} unexpected={list(unexpected)[:6]}")
    model = DDPM(unet, timesteps=a.get("timesteps", 1000),
                 schedule=a.get("schedule", "cosine")).to(device)
    return model.eval(), cond_dim, a


def quantize_to_palette(x: torch.Tensor, k: int,
                        mask: torch.Tensor | None = None,
                        mode: str = "per") -> torch.Tensor:
    """把 (B,3,64,64) 的像素吸附到调色板。**薄封装**，实现在
    ``scripts/lib/quantize.py``（与推理平台共用同一份，口径不许分叉）。

    ``mode``：
      * ``per``（默认）—— 逐图自适应 k-means。实测最贴真实（见 lib 的 docstring 表）。
      * ``global`` —— ``models/palette_kK.npy`` 全局板。会过冲且把图拉灰
        （饱和 0.298 vs 真实 0.353），只留给 A/B 对照。

    ``mask`` 给定时**只量化可见像素**，不可见像素的 RGB 原样返回 ——
    否则「先量化再乘 alpha」的调用方会拿到被污染的颜色。
    """
    if k <= 0 or mode in ("off", "", None):
        return x
    b, _, h, w = x.shape
    arr = (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).contiguous().numpy()
    if mask is not None:
        vis = (mask.permute(0, 2, 3, 1).contiguous().numpy()[..., 0] > 0.5)
    else:
        vis = np.ones((b, h, w), dtype=bool)
    out = np.stack([quantize_u8(arr[i], vis[i], mode, k, models_dir=MODELS, seed=0)
                    for i in range(b)])
    t = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()
    return t.to(device=x.device, dtype=x.dtype).div(255.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dcgan", "diffusion"], default="diffusion")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--alpha-threshold", type=float, default=0.5)
    ap.add_argument("--quantize", type=int, default=0,
                    help="生成后把可见像素吸附到 K 色调色板。"
                         "0=关。**量化的正确位置是后处理而不是模型结构**——"
                         "模型连续输出（保留 dithering/渐变的表达力），量化最后兜底，"
                         "这是 Minecraft 皮肤生成管线的业界标准做法")
    ap.add_argument("--quantize-mode", choices=["per", "global"], default="per",
                    help="per=逐图自适应 k-means（默认，实测最贴真实：K=64 时色数 64 对真实 66、"
                         "精确相等率 0.497 对 0.475、饱和 0.356 对 0.353）；"
                         "global=用 models/palette_kK.npy 全局板（实测过冲且偏灰，只留给 A/B）")
    ap.add_argument("--ddim-steps", type=int, default=100)
    ap.add_argument("--cfg-scale", type=float, default=4.0,
                    help="classifier-free guidance 强度（1.0=关）。"
                         "只在训练时开了 --cond-dropout 的权重上有意义")
    ap.add_argument("--cond-spec", type=str, default=None,
                    help="JSON 字符串或 JSON 文件路径，用于条件生成")
    ap.add_argument("--model-type", choices=["none", "classic", "slim"], default="none",
                    help="快捷指定骨架类型：classic=Steve(4px 手臂)，slim=Alex(3px 手臂)")
    ap.add_argument("--alpha-source", choices=["retrieval", "template"], default="retrieval",
                    help="第二层 alpha 的来源。**retrieval（推荐）**：从 models/mask_bank.npz "
                         "检索真实 mask —— 训练侧喂的就是真实 alpha，推理侧必须同分布，"
                         "否则 conditioning 落在训练分布之外（实测离真实流形 3.09 倍、"
                         "hat 盒连通块中位 1.5px 对真实 11.5px，第二层会出椒盐/穿孔）。"
                         "template：旧的逐像素 i.i.d. 合成，只留给 A/B 对照")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = a.out or f"{a.model}_{datetime.now():%Y%m%d_%H%M%S}"
    outdir = os.path.join(GEN, run)
    os.makedirs(outdir, exist_ok=True)

    ckpt = a.ckpt or (
        os.path.join(MODELS, "diffusion", "ema.pt") if a.model == "diffusion"
        else os.path.join(MODELS, "dcgan", "latest.pt"))
    if not os.path.isfile(ckpt):
        print(f"找不到权重 {ckpt}，先训练模型")
        return
    print(f"载入 {ckpt}")

    spec = None
    if a.cond_spec:
        spec = (json.load(open(a.cond_spec, encoding="utf-8"))
                if os.path.isfile(a.cond_spec) else json.loads(a.cond_spec))
    if a.model_type != "none":
        spec = dict(spec or {})
        spec["model_type"] = a.model_type
        print(f"骨架类型指定为 {a.model_type}（写入条件向量第 33:36 段）")

    cond_vec = None
    alpha_rates = load_alpha_rates()   # 3 通道模型（alpha 模板化）时会用到
    if a.model == "dcgan":
        G, cond_dim, ck = build_dcgan(ckpt, device)
        # 注意：alpha_mode 存在 checkpoint 的 args 里，**不是** argparse 的 Namespace。
        # 早期这里写成 ``a.get("alpha_mode")``——Namespace 没有 .get，DCGAN 一生成就
        # 抛 AttributeError，整条生成链其实从没跑通过。
        ck_args = ck.get("args", {})
        print(f"DCGAN 参数量 {count_params(G)/1e6:.2f}M | cond_dim={cond_dim} | "
              f"alpha_mode={ck_args.get('alpha_mode', 'head')}")
        if cond_dim and spec:
            cond_vec = torch.tensor(vector_from_spec(spec), device=device).repeat(a.batch, 1)
    else:
        model, cond_dim, cfg = build_diffusion(ckpt, device)
        print(f"UNet 参数量 {count_params(model.unet)/1e6:.2f}M | cond_dim={cond_dim} | "
              f"T={model.T} | DDIM={a.ddim_steps}")

    # 条件向量统一在这里收口。**别在每个分支里各写一份**：
    # 早期条件模型在没有 --cond-spec 时 cond 保持 None，FiLM 生成器直接抛
    # "film=True 时 forward 必须提供 cond"，而无条件的路径却又是 None——两处口径不一致。
    #
    # 部位级 overlay 四位（模型 cond_dim > COND_DIM 时）：从 spec 取，
    # 追加到 36 维基础向量之后。**必须与下面的 alpha 模板同源**，见 box_paint。
    use_ov_bits = cond_dim > COND_DIM
    ov_bits = None
    if cond_dim:
        base_v = (vector_from_spec(spec) if spec
                  else np.zeros(COND_DIM, dtype=np.float32))
        if use_ov_bits:
            ov_bits = ov_bits_from_spec(spec or {})
            base_v = np.concatenate([base_v, ov_bits]).astype(np.float32)
            print(f"部位级 overlay 开关：{describe_ov_bits(ov_bits)}")
        if spec is None:
            print("注意：模型是条件模型但未给 --cond-spec，使用零条件向量"
                  "（等价于无条件边缘分布）")
        cond_vec = torch.tensor(base_v, device=device).repeat(a.batch, 1)

    rows, saved = [], []
    t0 = time.time()
    done = 0
    alpha_input = bool(cfg.get("alpha_input")) if a.model != "dcgan" else False
    with torch.no_grad():
        while done < a.n:
            b = min(a.batch, a.n - done)
            cond = cond_vec[:b] if cond_vec is not None else None
            a_t = None
            if a.model == "dcgan":
                z = torch.randn(b, G.z_dim, device=device)
                x = G(z, cond)
                x = x.clamp(0, 1)
            else:
                # alpha mask 必须在**采样前**抽好：alpha_input 的模型要把它当
                # 条件平面喂进去（它决定「哪里会露出来」）。
                #
                # ⚠️ 这里原先调的是 `sample_alpha_templates()`（逐像素 i.i.d. 合成）。
                # 训练侧喂的是**真实 alpha**，合成模板只复现了逐面覆盖率、空间结构为零
                # （实测 hat 盒连通块中位 1.5px vs 真实 11.5px，面内邻接一致率 0.8958 vs
                # 真实 0.9568）—— 推理侧 conditioning 落在训练分布之外，而且模板直接被
                # 当作导出图的 alpha，于是第二层是「穿孔」的而不是连通的衣物。
                # 改为从真实 mask 库检索（经验分布），训练/推理两侧恒等分布。
                # 开了部位位时按部位位**分桶**检索，保证「条件说画什么、mask 就露什么」。
                box_paint = (ov_bits_to_box_paint(np.tile(ov_bits, (b, 1)))
                             if ov_bits is not None else None)
                a_np = None
                if a.alpha_source == "retrieval":
                    a_np = sample_alpha_bank(b, ov_bits=np.tile(ov_bits, (b, 1))
                                             if ov_bits is not None else None,
                                             rng=np.random.default_rng(a.seed + done))
                    if a_np is None:
                        print("[!] models/mask_bank.npz 缺失，回落 template（分布会错配；"
                              "跑 scripts/25_build_mask_bank.py 生成）")
                if a_np is None:
                    a_np = sample_alpha_templates(b, alpha_rates,
                                                  rng=np.random.default_rng(a.seed + done),
                                                  box_paint=box_paint)
                a_t = torch.from_numpy(a_np).reshape(b, 1, 64, 64)
                extra = (a_t.to(device) * 2 - 1) if alpha_input else None
                x = model.sample(b, device=device, cond=cond, ddim_steps=a.ddim_steps,
                                 extra=extra,
                                 cfg_scale=(a.cfg_scale if cond is not None else 1.0))
                x = ((x.clamp(-1, 1) + 1) / 2)
            x = x.float().cpu()
            if x.shape[1] == 3:
                # alpha_mode="none" 的模型：alpha 由 UV 模板合成（两级伯努利，
                # 频率是真实数据实测的），模型完全不学 alpha
                if a_t is None:
                    a_np = None
                    if a.alpha_source == "retrieval":
                        a_np = sample_alpha_bank(
                            b, ov_bits=np.tile(ov_bits, (b, 1)) if ov_bits is not None else None,
                            rng=np.random.default_rng(a.seed + done))
                    if a_np is None:
                        a_np = sample_alpha_templates(
                            b, alpha_rates, rng=np.random.default_rng(a.seed + done),
                            box_paint=(ov_bits_to_box_paint(np.tile(ov_bits, (b, 1)))
                                       if ov_bits is not None else None))
                    a_t = torch.from_numpy(a_np).reshape(b, 1, 64, 64)
                a_t = a_t.float().cpu()
                if a.quantize > 0:
                    # 传 mask：只量化可见像素，透明区 RGB 保持 0
                    x = quantize_to_palette(x, a.quantize, mask=a_t,
                                            mode=a.quantize_mode)
                x = torch.cat([x * a_t, a_t], dim=1)
            alpha = (x[:, 3:4] > a.alpha_threshold).float()
            x = torch.cat([x[:, :3] * alpha, alpha], dim=1)
            # NCHW -> NHWC。早期写成 permute(0, 2, 3, 0)，同一个维度用了两次，
            # 直接抛 "permute(): duplicate dims are not allowed"。
            arr = (x * 255).round().byte().permute(0, 2, 3, 1).contiguous().numpy()
            for i in range(b):
                p = os.path.join(outdir, f"skin_{done + i:04d}.png")
                Image.fromarray(arr[i], "RGBA").save(p)
                # uv_valid 只该反映「格式/UV 是否合法」，所以这里两条闸门都关掉；
                # 骨架类型是否可用单独记在 model_gate_pass，避免两个概念混在一列里。
                rep = analyze(p, quality_gate="off", model_gate="off")
                q = quality_metrics(arr[i])
                rows.append({
                    "path": os.path.relpath(p, ROOT).replace("\\", "/"),
                    "uv_valid": int(rep.ok), "uv_reason": rep.reason,
                    "model_type": rep.model_type, "model_conf": rep.model_conf,
                    "arm_ratio": rep.arm_ratio, "limb_l_ratio": rep.limb_l_ratio,
                    "model_gate_pass": int(rep.model_type != "unknown"
                                           and rep.limb_l_ratio >= LIMB_L_MIN),
                    "quality_tier": q["tier"], "n_colors": q["n_colors"],
                    "transparent_ratio": rep.transparent_ratio,
                    "overlay_ratio": rep.overlay_ratio,
                    "face_opaque_ratio": rep.face_opaque_ratio,
                    "unique_colors": rep.unique_colors,
                    "alpha_min": rep.alpha_min, "alpha_max": rep.alpha_max,
                })
                saved.append(p)
            done += b
            print(f"  生成 {done}/{a.n}（{time.time()-t0:.1f}s）", flush=True)

    with open(os.path.join(outdir, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    sheet = skin_contact_sheet(saved, cols=16, scale=4)
    sheet.save(os.path.join(outdir, "contact_sheet.png"))

    n_ok = sum(r["uv_valid"] for r in rows)
    n_gate = sum(r["model_gate_pass"] for r in rows)
    mt_req = (spec or {}).get("model_type")
    mt_dist = _counter([r["model_type"] for r in rows])
    mt_hit = (sum(v for k, v in mt_dist.items() if k == mt_req) if mt_req else None)
    summary = {
        "run": run, "model": a.model, "ckpt": ckpt, "n": len(rows),
        "uv_valid": n_ok, "uv_valid_rate": round(n_ok / max(len(rows), 1), 4),
        "model_gate_pass": n_gate,
        "model_gate_pass_rate": round(n_gate / max(len(rows), 1), 4),
        "transparent_ratio_mean": round(float(np.mean([r["transparent_ratio"] for r in rows])), 4),
        "unique_colors_mean": round(float(np.mean([r["unique_colors"] for r in rows])), 1),
        # 骨架类型可控性：请求的类型与「从像素反判出来的类型」一致率
        "model_type_requested": mt_req,
        "model_type_detected_dist": mt_dist,
        "model_type_hit_rate": (round(mt_hit / max(len(rows), 1), 4) if mt_hit is not None else None),
        "quality_tier_dist": _counter([r["quality_tier"] for r in rows]),
        "cond_spec": spec,
        "cond_described": describe(vector_from_spec(spec)) if spec else None,
        "reject_reasons": _counter([r["uv_reason"] for r in rows if not r["uv_valid"]]),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(LOGS, f"generate_{run}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def _counter(xs: list[str]) -> dict:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


if __name__ == "__main__":
    main()
