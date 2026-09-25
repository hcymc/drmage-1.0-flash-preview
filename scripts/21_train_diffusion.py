#!/usr/bin/env python
"""21_train_diffusion.py — 阶段四之二：DDPM（像素空间扩散）。

为什么走扩散而不是继续修 DCGAN
-------------------------------
DCGAN 那条路反复撞同一堵墙：**判别器塌成常数函数**（``d_loss=ln2``、
``‖∇D‖≈0.06``），于是对抗信号、特征匹配、特征多样性这些「学习型」机制
全部失效，只剩下手工统计量在拉 G —— 结果就是「统计量全对、语义为零」。
实测证据见 ``logs/v4_alpha_feat/metrics.jsonl`` 与
``logs/semantic_diversity.json``（v4 语义多样性 0.35，真实 1.00）。

扩散的训练目标是**回归**（预测噪声，MSE），没有判别器 → 没有对抗失衡、
没有模式坍塌。三个公开的 MC 皮肤生成项目也都用扩散。

关键设计
--------
* **3 通道 RGB + UV 模板 alpha**（``--channels 3``，默认）：
  alpha 完全交给 ``skinatlas.sample_alpha_templates``（频率是真实数据实测的三态模型），
  模型不学 alpha。既省容量，也根除「剪影是噪点」的问题。
* **cosine schedule**：线性 schedule 在小图上把绝大多数时间步花在「几乎全噪」，
  低噪段（决定细节的步）占比太小。
* **EMA**：扩散对权重平均敏感，采样与交付都用 EMA 权重。
* **DDIM 采样**：100 步出图。

用法
----
    python 21_train_diffusion.py --epochs 60 --batch 64 --base 96 --tag diff_v1
    python 21_train_diffusion.py --resume models/diff_v1/latest.pt --tag diff_v1
    python 21_train_diffusion.py --channels 4 --alpha-binary-weight 0.05   # 让模型自己学 alpha

产出
----
* ``models/<tag>/latest.pt`` ``ema.pt`` ``epoch_XXXX.pt``
* ``logs/<tag>/metrics.jsonl`` / ``train.log`` / ``loss_curve.png``
* ``logs/samples/<tag>/step_XXXXXX.png``
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from labelset import (COND_DIM, OV_BITS, build_cond_matrix,  # noqa: E402
                      filter_by_model_type, load_model_types,
                      ov_bits_to_box_paint)
from losses import flat_fraction, unique_color_count  # noqa: E402
from models import DDPM, SmallUNet, count_params  # noqa: E402
from skinatlas import (load_alpha_rates, sample_alpha_bank,  # noqa: E402
                       sample_alpha_templates)
from trainutil import (Heartbeat, RunLogger, alpha_binarize, alpha_binary_loss,  # noqa: E402
                       jitter_mask, load_ckpt, rng_state, save_ckpt, save_sample_grid,
                       seed_everything, set_rng_state, write_curve)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
LOGS = os.path.join(ROOT, "logs")
ANNOT = os.path.join(ROOT, "labels", "annotations.jsonl")
CLEAN_MANIFEST = os.path.join(ROOT, "data", "clean_manifest.csv")


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _load_ov4(split: str, keep) -> np.ndarray:
    """读 ``{split}_ov4.npy`` 并按 ``keep`` 取子集（顺序与 ``*_sids`` 一致）。

    ``keep`` 是「从原始 npy 里取哪些行」的索引（见下面 ``tr_keep`` 的构造），
    所以必须用它去索引，不能直接按位置对齐 —— 那会在 ``--model-type``
    筛掉一部分样本之后把标签错位（这种错位不会报错，只会让条件与图像
    对不上，是最难查的一类）。
    """
    p = os.path.join(DATA, f"{split}_ov4.npy")
    if not os.path.isfile(p):
        raise SystemExit(
            f"找不到 {p}；先跑：python scripts/24_build_ov_bits.py --split {split}")
    arr = np.load(p)
    return arr if keep is None else arr[keep]


class EMA:
    """指数滑动平均。decay 随步数 warmup，避免早期被随机初始化拖住。"""

    def __init__(self, model: nn.Module, decay: float = 0.9995, warmup: int = 1000):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step)) if self.step < self.warmup else self.decay
        for k, v in model.state_dict().items():
            s = self.shadow.get(k)
            if s is None:
                # 从旧 checkpoint 继承的 shadow 里没有后加的层（如 FiLM）：
                # 直接用当前权重建档，而不是 KeyError。
                self.shadow[k] = v.detach().clone().float()
                continue
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v.detach())

    def apply_to(self, model: nn.Module) -> None:
        # 宽容加载：从旧 checkpoint 继承的 shadow 可能没有后加的层（如 FiLM），
        # 那些层用模型当前的值（零初始化）。用 strict=True 会直接崩在采样上。
        missing, unexpected = model.load_state_dict(
            {k: v for k, v in self.shadow.items()}, strict=False)
        if unexpected:
            raise RuntimeError(f"EMA shadow 含多余键：{list(unexpected)[:6]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--t-dim", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")
    ap.add_argument("--ddim-steps", type=int, default=100)
    ap.add_argument("--channels", type=int, choices=[3, 4], default=3,
                    help="3=RGB + UV 模板 alpha（推荐）；4=让模型自己学 alpha")
    ap.add_argument("--mask-loss", dest="mask_loss", action="store_true", default=True,
                    help="只在实际可见像素上算损失（默认开）。关掉会重现历史"
                         "「overlay 区域被学成暗色」的 bug：透明像素参与损失后，"
                         "模型把 overlay 区域的 RGB 学成黑色而不是不画")
    ap.add_argument("--no-mask-loss", dest="mask_loss", action="store_false")
    ap.add_argument("--alpha-binary-weight", type=float, default=0.05,
                    help="仅 --channels 4 时生效")
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--cond-dropout", type=float, default=0.15,
                    help="条件 dropout 概率 —— CFG 的训练前提（0=关）")
    ap.add_argument("--cfg-scale", type=float, default=4.0,
                    help="采样时的 CFG 强度（1.0=关）。只影响 EMA 采样与导出，"
                         "不影响训练；必须先跑过带 --cond-dropout 的训练")
    ap.add_argument("--alpha-input", dest="alpha_input", action="store_true", default=False,
                    help="把 UV 模板 alpha 作为第 4 个输入通道（UNet in_ch=4 / out_ch=3）。"
                         "让模型看到「这里推理时会不会露出来」，是治第二层的核心一步")
    ap.add_argument("--overlay-weight", type=float, default=4.0,
                    help="overlay 盒的损失权重（1.0=不加权）。overlay 有效可见像素只有"
                         "base 的 1/6.7，不加权它的梯度会被 base 淹没")
    ap.add_argument("--ov-bits", dest="ov_bits", action="store_true", default=False,
                    help="条件追加 4 个**部位级** overlay 开关（hat/body/arm/leg），"
                         "COND 36→40。让模型知道「哪个部位要画外层」——"
                         "原来的全局 overlay_used 一个 bit 表达不了"
                         "「头有帽子、手臂没袖子」这种占 53%% 的常见组合。"
                         "需要 data/processed/{train,val}_ov4.npy")
    ap.add_argument("--model-type", choices=["all", "classic", "slim"], default="classic")
    ap.add_argument("--tag", type=str, default="diffusion",
                    help="产物子目录名；多版本并行时必须区分，否则互相覆盖")
    ap.add_argument("--alpha-source", choices=["retrieval", "template"], default="retrieval",
                    help="可视化样本与推理时的 alpha 来源。retrieval=从真实 mask 库检索"
                         "（推荐，与训练 loss 喂的真实 alpha 同分布）；template=旧的 i.i.d. 合成")
    ap.add_argument("--mask-cross-sample", dest="mask_cross_sample", action="store_true",
                    default=False,
                    help="条件平面用**批内另一个样本**的 alpha，切断「mask 形状 ↔ 内容」的相关性。"
                         "不切断的话模型会走「认出是哪张皮肤」的捷径，推理时拿到无关 mask 就在"
                         "暴露区域输出满熵噪声（「纯噪点」那张图）。损失仍只算在本样本可见区上")
    ap.add_argument("--mask-jitter", type=float, default=0.0,
                    help="对条件平面的 mask 做随机形态扰动（膨胀/腐蚀概率 + 平移），"
                         "进一步切断精确对应关系，同时保留真实连通结构。0=关")
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None, help="冒烟测试用：跑够步数就停")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    models_dir = os.path.join(ROOT, "models", args.tag)
    logger = RunLogger(args.tag, LOGS)
    logger.info({"event": "config", "args": vars(args), "device": str(device)})
    # 心跳：监控面板据此判断「在跑 / 已停 / 卡死」，并拿到确切 PID 做启停控制
    hb = Heartbeat(os.path.join(logger.dir, "heartbeat.json"),
                   tag=args.tag, script="21_train_diffusion.py", epochs=args.epochs)
    # **启动即写一次心跳**：指标是每 100 步才记一条（约 11 秒），
    # 短实验（--max-steps 几十步）或刚开始的这十几秒里如果没心跳，
    # 监控面板会完全看不到这个实验在跑。
    hb.beat(step=0, epoch_idx=0, phase="init")

    tr_x = np.load(os.path.join(DATA, "train.npy"), mmap_mode="r")
    va_x = np.load(os.path.join(DATA, "val.npy"), mmap_mode="r")
    with open(os.path.join(DATA, "train_sids.json"), "r", encoding="utf-8") as fh:
        tr_sids_all = json.load(fh)
    with open(os.path.join(DATA, "val_sids.json"), "r", encoding="utf-8") as fh:
        va_sids_all = json.load(fh)

    # ---- 骨架类型筛选：实测 10 万样本里 slim 只有 1 张，骨架是档案属性 ----------
    # 用**索引映射**而不是把子集抠出来：train.npy 是 1.7GB memmap，
    # ``tr_x[mask]`` 会把它整块读进内存，而我们要的只是「从哪些行采样」。
    tr_keep = np.arange(len(tr_sids_all))
    va_keep = np.arange(len(va_sids_all))
    if args.model_type != "all":
        sid2type = load_model_types(CLEAN_MANIFEST)
        if not sid2type:
            logger.say(f"[!] 读不到 {CLEAN_MANIFEST}，--model-type 筛选失效，按全量训练")
        else:
            tr_keep = np.where(np.array(
                filter_by_model_type(tr_sids_all, args.model_type, sid2type), bool))[0]
            va_keep = np.where(np.array(
                filter_by_model_type(va_sids_all, args.model_type, sid2type), bool))[0]
    tr_sids = [tr_sids_all[i] for i in tr_keep]
    va_sids = [va_sids_all[i] for i in va_keep]
    logger.say(f"骨架 {args.model_type}：train {len(tr_keep)} / val {len(va_keep)}")

    cond_dim = COND_DIM if args.cond else 0
    tr_cond = build_cond_matrix(ANNOT, tr_sids) if args.cond else None
    va_cond = build_cond_matrix(ANNOT, va_sids) if args.cond else None
    va_ov = None
    # ---- 部位级 overlay 四位：追加到条件尾部（36 → 40）----
    # 标签由 scripts/24_build_ov_bits.py 从**真实 alpha** 算出；
    # 训练时的条件平面（extra）用的是同一份真实 alpha，两者天然一致。
    if args.cond and args.ov_bits:
        tr_ov = _load_ov4("train", tr_keep)
        va_ov = _load_ov4("val", va_keep)
        if tr_cond is not None:
            tr_cond = np.concatenate([tr_cond, tr_ov], axis=1).astype(np.float32)
        if va_cond is not None:
            va_cond = np.concatenate([va_cond, va_ov], axis=1).astype(np.float32)
        cond_dim = COND_DIM + OV_BITS
        logger.say(f"部位级 overlay 四位已启用 | cond_dim={cond_dim} | "
                   f"train 为真比例 {[round(float(v), 3) for v in tr_ov.mean(axis=0)]}")

    alpha_rates = load_alpha_rates() if args.channels == 3 else None
    if args.alpha_input and args.channels != 3:
        raise SystemExit("--alpha-input 只支持 --channels 3（RGB 目标 + 模板 alpha 作条件平面）")
    unet_in = args.channels + (1 if args.alpha_input else 0)
    unet = SmallUNet(in_ch=unet_in, base=args.base, t_dim=args.t_dim,
                     cond_dim=cond_dim, out_ch=args.channels)
    model = DDPM(unet, timesteps=args.timesteps, schedule=args.schedule).to(device)
    logger.say(f"UNet {count_params(unet)/1e6:.2f}M | in_ch={unet_in} out_ch={args.channels} | "
               f"T={args.timesteps} | schedule={args.schedule} | cond_dim={cond_dim} | "
               f"alpha_input={args.alpha_input} | cond_dropout={args.cond_dropout}")
    logger.say(f"train {tr_x.shape} | val {va_x.shape} | 数据 {tr_x.shape[0]} 张")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = make_scaler(use_amp)
    ema = EMA(model, decay=args.ema_decay)

    start_epoch, gstep = 0, 0
    if args.resume and os.path.isfile(args.resume):
        ck = load_ckpt(args.resume, map_location=device)
        # 宽容加载：旧 checkpoint 没有 FiLM（那些层零初始化 ⇒ 缺省等价于无 FiLM）。
        # 其余任何缺失都说明架构真的变了，直接报错而不是静默随机初始化。
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        bad = [k for k in missing if ".film." not in k]
        if bad or unexpected:
            raise RuntimeError(f"resume 结构不匹配：missing={bad[:6]} "
                               f"unexpected={list(unexpected)[:6]}")
        try:
            opt.load_state_dict(ck["opt"])
        except (ValueError, RuntimeError) as exc:
            # 跨架构续训（例如新增了 FiLM / alpha 输入通道）时优化器状态对不上，
            # 跳过是正常的 —— Adam 的动量会在几十步内自己重建。
            logger.say(f"[!] 优化器状态与当前结构不匹配，跳过：{exc}")
        start_epoch = ck.get("epoch", 0) + 1
        gstep = ck.get("gstep", 0)
        if ck.get("ema"):
            ema.shadow = ck["ema"]; ema.step = ck.get("ema_step", 0)
        if ck.get("rng"):
            set_rng_state(ck["rng"])
        logger.say(f"从 {args.resume} 续训：epoch {start_epoch} step {gstep}")

    n_vis = 32
    fixed_noise = torch.randn(n_vis, args.channels, 64, 64, device=device)
    fixed_cond = None
    pick = (np.linspace(0, len(va_cond) - 1, n_vis).astype(int)
            if (args.cond and va_cond is not None) else None)
    if pick is not None:
        fixed_cond = torch.tensor(va_cond[pick], device=device)
    # 3 通道模式下可视化样本的 alpha 也固定，保证跨 step 可比。
    # 开了 --ov-bits 时 mask 必须**按 fixed_cond 的那 4 位**取 —— 否则
    # 「条件说没外套、mask 却画了外套」，可视化本身就在骗自己。
    #
    # ⚠️ 这里的 alpha 来源必须与**推理侧**一致，否则可视化看到的样本网格
    # 与导出/平台的产物是两个东西（之前正是这个问题：训练脚本用合成模板，
    # 而训练 loss 喂的是真实 alpha）。
    fixed_a = None
    if args.channels == 3:
        if args.ov_bits and va_ov is not None and pick is not None:
            _bits = va_ov[pick]
            box_paint = ov_bits_to_box_paint(_bits)
        else:
            _bits, box_paint = None, None
        _a_np = None
        if args.alpha_source == "retrieval":
            _a_np = sample_alpha_bank(n_vis, ov_bits=_bits,
                                      rng=np.random.default_rng(args.seed + 7))
            if _a_np is None:
                logger.say("[!] models/mask_bank.npz 缺失，可视化回落 template")
        if _a_np is None:
            _a_np = sample_alpha_templates(n_vis, alpha_rates,
                                           rng=np.random.default_rng(args.seed + 7),
                                           box_paint=box_paint)
        fixed_a = torch.from_numpy(_a_np).reshape(n_vis, 1, 64, 64).to(device)

    n = len(tr_keep)
    steps_per_epoch = max(1, n // args.batch)
    logger.say(f"batch {args.batch} | {steps_per_epoch} step/epoch | "
               f"目标 {args.epochs} epoch | AMP={use_amp}")
    # 数据已载入，正式进入训练循环（监控面板可据此区分「在加载」与「在训练」）
    hb.beat(step=gstep, epoch_idx=start_epoch, phase="training",
            steps_per_epoch=steps_per_epoch)
    # 把训练脚本自己的监控目标也写进心跳，面板才能算 ETA
    hb._static["steps_per_epoch"] = steps_per_epoch

    # ---- 诊断读数：第二层(overlay) / 第一层(base) 的「近黑像素占比」 ----
    # 这是「overlay 区域被学成暗色」那个历史 bug 的**直接观测量**。
    # 不加这个读数，就只能靠人肉进游戏看；加了之后每次采样都会记进 metrics.jsonl。
    from skinatlas import OVERLAY_BOXES as _OV  # noqa: E402
    from skinatlas import face_index as _face_index  # noqa: E402
    _fid = _face_index()

    def _box_idx(pred) -> np.ndarray:
        return np.concatenate([
            (np.arange(y0, y1)[:, None] * 64 + np.arange(x0, x1)[None, :]).reshape(-1)
            for nm, (y0, y1, x0, x1) in _fid.items() if pred(nm.split(".")[0])])

    _OV_IDX = _box_idx(lambda b: b in _OV)
    _BS_IDX = _box_idx(lambda b: b not in _OV)

    # overlay 区域的损失权重图 (1,1,64,64)：可见的 overlay 像素乘 overlay_weight。
    # 依据：overlay 有效可见像素只有 base 的 1/6.7（覆盖率 0.149 vs 0.994），
    # masked_loss 之后它的梯度像素比 base 少一个量级 —— 不加权就永远学不出东西。
    _OV_W = None
    if args.overlay_weight != 1.0:
        _w = np.ones((64 * 64,), dtype=np.float32)
        _w[_OV_IDX] = float(args.overlay_weight)
        _OV_W = torch.from_numpy(_w.reshape(1, 1, 64, 64)).to(device)

    def _dark_rate(rgba_t, idx) -> float | None:
        """可见像素里「三通道最大值 < 38/255」的占比（跨样本平均）。"""
        arr = (rgba_t[:, :3].detach().float().cpu().numpy().transpose(0, 2, 3, 1) * 255)
        am = rgba_t[:, 3].detach().cpu().numpy() > 0.5
        frgb = arr.reshape(arr.shape[0], -1, 3)
        fvis = am.reshape(am.shape[0], -1)
        vals = []
        for i in range(frgb.shape[0]):
            m = fvis[i][idx]
            if m.sum() == 0:
                continue
            v = frgb[i][idx][m]
            vals.append(float((v.max(axis=1) < 38).mean()))
        return round(float(np.mean(vals)), 4) if vals else None

    sdir = os.path.join(LOGS, "samples", args.tag)

    def ema_sample(step: int) -> dict:
        backup = copy.deepcopy(model.state_dict())
        ema.apply_to(model)
        model.eval()
        sample_extra = (fixed_a * 2.0 - 1.0) if args.alpha_input else None
        with torch.no_grad():
            smp = model.sample(n_vis, device=device, cond=fixed_cond,
                               ddim_steps=args.ddim_steps, extra=sample_extra,
                               cfg_scale=(args.cfg_scale if args.cond else 1.0))
        model.train()
        model.load_state_dict(backup)
        raw = (smp.clamp(-1, 1) + 1) / 2
        if args.channels == 3:
            rgb, a = raw, fixed_a
            rgba = torch.cat([rgb * a, a], dim=1)
            uc = unique_color_count(rgb, a)
            ff = float(flat_fraction(rgb.float(), a.float(), target=None))
        else:
            rgba = alpha_binarize(raw)
            uc = unique_color_count(rgba[:, :3], rgba[:, 3:4])
            ff = float(flat_fraction(rgba[:, :3].float(), rgba[:, 3:4].float(), target=None))
        os.makedirs(sdir, exist_ok=True)
        save_sample_grid(rgba, os.path.join(sdir, f"step_{step:06d}.png"), cols=8)
        am = rgba[:, 3:4]
        return {"vis_alpha_mean": round(float(am.mean()), 4),
                "vis_gen_unique_colors": round(uc, 1),
                "vis_gen_flat_frac": round(ff, 4),
                "ov_dark_frac": _dark_rate(rgba, _OV_IDX),
                "bs_dark_frac": _dark_rate(rgba, _BS_IDX),
                "real_ov_dark_frac": 0.25, "real_bs_dark_frac": 0.05,
                "real_unique_colors": 142.2, "real_flat_frac": 0.5285}

    for epoch in range(start_epoch, args.epochs):
        perm = np.random.permutation(n)
        ep_t0 = time.time()
        acc, cnt = 0.0, 0
        vis_acc, vis_cnt = 0.0, 0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            if len(idx) < 4:
                continue
            rows = tr_keep[idx]
            # fancy indexing 按给定顺序返回行，所以 rows 无需排序，
            # 返回顺序天然与 cond[idx] 对齐。
            full = np.asarray(tr_x[rows])
            xb = full[:, :args.channels]
            x0 = torch.from_numpy(xb).to(device, non_blocking=True).float().div_(127.5).sub_(1.0)
            cond = (torch.from_numpy(tr_cond[idx]).to(device) if tr_cond is not None else None)
            # 可见性掩码：数据里 alpha 是 0/255，用 >=128 二值化 —— 与项目里其它
            # 所有「可见像素」判据保持同一个阈值（ALPHA_VIS_THRESHOLD 的等价物）。
            a_mask = None
            extra = None
            if args.mask_loss and args.channels == 3:
                a_bin = (full[:, 3:4] >= 128).astype(np.float32)
                a_mask = torch.from_numpy(a_bin).to(device, non_blocking=True)
                if args.alpha_input:
                    # 条件平面：训练侧用**真实** alpha，推理侧用**检索的真实 mask**。
                    # 两者同源（都是真实数据里长出来的连通图形），所以
                    # 「给定 mask 补内容」在推理时是个适定任务。
                    #
                    # ⚠️ 但不能直接用目标自己的 alpha：
                    # `extra = a_mask`（目标自己的 mask）会让 mask 与内容**强相关**，
                    # 模型可以走「从 mask 形状认出是哪张皮肤 → 直接倒出对应颜色」的
                    # 捷径。推理时拿到的是**无关**的真实 mask，落在这条捷径的
                    # 分布之外 → 在 mask 露出、训练时无监督的位置输出满熵噪声
                    # （这正是「纯噪点」那张图的生成机制）。
                    #
                    # `--mask-cross-sample` 用**批内另一个样本**的 alpha 作条件平面，
                    # 使 mask 与内容解耦；损失仍然只算在本样本自己的可见区上
                    # （`a_mask` 独立传递，不参与 extra），所以监督信号没有被削弱。
                    extra_mask = a_mask
                    if args.mask_cross_sample:
                        perm = torch.randperm(a_mask.size(0), device=a_mask.device)
                        extra_mask = a_mask[perm]
                    if args.mask_jitter > 0:
                        j = jitter_mask(extra_mask, args.mask_jitter,
                                        generator=torch.Generator(device=device).manual_seed(
                                            int(torch.randint(0, 2 ** 31 - 1, (1,)).item())))
                        extra_mask = j
                    extra = extra_mask * 2.0 - 1.0

            with torch.cuda.amp.autocast(enabled=use_amp):
                if args.channels == 4 and args.alpha_binary_weight > 0:
                    # 「模型自己学 alpha」模式：从预测噪声反解 x0_hat，对 alpha 通道
                    # 加二值化 hinge，压掉雾状半透明边缘。
                    pred, noise, t, xt = model.forward_step(x0, cond, extra, args.cond_dropout)
                    loss = torch.nn.functional.mse_loss(pred, noise)
                    a_hat = (model.predict_x0(xt, t, pred)[:, 3:4] + 1) / 2
                    loss = loss + args.alpha_binary_weight * alpha_binary_loss(a_hat)
                elif a_mask is not None:
                    loss = model.masked_loss(x0, a_mask, cond, extra, _OV_W,
                                             cond_dropout=args.cond_dropout)
                    vis_acc += float(a_mask.mean()); vis_cnt += 1
                else:
                    loss = model.loss(x0, cond, extra, cond_dropout=args.cond_dropout)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if (s + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
                gstep += 1

            acc += float(loss.detach()) * args.grad_accum; cnt += 1

            if gstep and gstep % 100 == 0:
                rec = {
                    "mse": round(acc / max(cnt, 1), 5),
                    "lr": opt.param_groups[0]["lr"],
                    "vram_peak_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                                     if device.type == "cuda" else 0),
                }
                if vis_cnt:
                    # 掩码生效时记一下可见率（应≈0.48）。同时提醒：这个 mse 的分母是
                    # **可见像素数**，与历史「全图像素均值」不可直接比较。
                    rec["visible_frac"] = round(vis_acc / vis_cnt, 4)
                    rec["mse_scope"] = "visible_only"
                else:
                    rec["mse_scope"] = "all_pixels"
                logger.metric(gstep, epoch=epoch, **rec)
                # epoch_idx 与 metrics.jsonl 的 epoch 同口径（0 起编号）
                hb.beat(step=gstep, epoch_idx=epoch, mse=rec["mse"],
                        vram_gb=rec["vram_peak_gb"], phase="train")

            if gstep and args.sample_every and gstep % args.sample_every == 0:
                logger.metric(gstep, **ema_sample(gstep))
                hb.beat(step=gstep, epoch_idx=epoch, phase="sampling")

            if args.max_steps and gstep >= args.max_steps:
                break

        dt = time.time() - ep_t0
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0
        logger.say(f"epoch {epoch+1}/{args.epochs} | {dt:.1f}s | mse={acc/max(cnt,1):.5f} | "
                   f"step={gstep} | vram_peak={peak:.2f}GB")
        hb.beat(step=gstep, epoch_idx=epoch, epochs_done=epoch + 1,
                mse=round(acc / max(cnt, 1), 5), epoch_seconds=round(dt, 1),
                vram_gb=round(peak, 2), phase="epoch_end")

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
              "gstep": gstep, "args": vars(args), "rng": rng_state(),
              "ema": ema.shadow, "ema_step": ema.step}
        os.makedirs(models_dir, exist_ok=True)
        save_ckpt(os.path.join(models_dir, "latest.pt"), **ck)
        save_ckpt(os.path.join(models_dir, "ema.pt"),
                  model=ema.shadow, gstep=gstep, epoch=epoch, args=vars(args))
        if (epoch + 1) % 20 == 0:
            save_ckpt(os.path.join(models_dir, f"epoch_{epoch+1:04d}.pt"), **ck)

        if args.max_steps and gstep >= args.max_steps:
            logger.say(f"到达 max_steps={args.max_steps}，停止（冒烟测试）")
            break

    logger.metric(gstep or 1, **ema_sample(gstep or 1), final=1)
    write_curve(os.path.join(logger.dir, "metrics.jsonl"),
                os.path.join(logger.dir, "loss_curve.png"), ["mse"])
    logger.say("训练结束")
    hb.finish(step=gstep, epoch_idx=epoch, epochs_done=epoch + 1, reason="completed")


if __name__ == "__main__":
    main()
