"""trainutil.py — 训练通用工具：随机种子、日志、断点续训、样本网格、显存观测。

设计要点
--------
* **样本网格带棋盘底**：皮肤有 alpha，直接拼图看不出透明区，棋盘底才能看出
  模型是否学会了「该透明的地方透明」。
* **断点续训**：checkpoint 里同时存模型、优化器、epoch、全局步数与 RNG 状态，
  ``--resume`` 后接着跑，loss 曲线不断档。
* **显存观测**：每个 epoch 记录 ``torch.cuda.max_memory_allocated``，
  这样「有没有爆显存」是有日志证据的，而不是靠感觉。
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_report() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {"cuda": True, "device": torch.cuda.get_device_name(0),
            "vram_total_gb": round(total / 1024 ** 3, 2),
            "vram_free_gb": round(free / 1024 ** 3, 2)}


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------

class RunLogger:
    """JSONL 逐步日志 + 每次调用 append 的文本日志。"""

    def __init__(self, name: str, log_dir: str):
        self.dir = os.path.join(log_dir, name)
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl = os.path.join(self.dir, "metrics.jsonl")
        self.text = os.path.join(self.dir, "train.log")
        self.t0 = time.time()
        self.info({
            "event": "start",
            "time": datetime.now().isoformat(timespec="seconds"),
            "gpu": gpu_report(),
        })

    def _write(self, obj: dict) -> None:
        with open(self.jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def info(self, obj: dict) -> None:
        self._write({"elapsed_s": round(time.time() - self.t0, 1), **obj})

    def say(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.text, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def metric(self, step: int, **kw) -> None:
        self.info({"step": step, **kw})


def write_curve(metrics_path: str, out_png: str, keys: list[str]) -> bool:
    """把 metrics.jsonl 画成折线图。用 PIL 手绘，避免 matplotlib 依赖。"""
    if not os.path.isfile(metrics_path):
        return False
    series: dict[str, list[tuple[int, float]]] = {k: [] for k in keys}
    with open(metrics_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "step" not in o:
                continue
            for k in keys:
                if k in o and isinstance(o[k], (int, float)):
                    series[k].append((o["step"], float(o[k])))
    if not any(series.values()):
        return False

    W, H, pad = 900, 420, 48
    img = Image.new("RGB", (W, H), (24, 24, 30))
    dr = ImageDraw.Draw(img)
    colors = [(90, 170, 250), (250, 130, 110), (140, 220, 150), (240, 200, 90), (200, 140, 240)]
    all_pts = [p for v in series.values() for p in v]
    xmin = min(p[0] for p in all_pts)
    xmax = max(p[0] for p in all_pts)
    # 必须显式兜住 xmax == xmin：短跑（只写过一个 metric 点）或所有点同 step 时
    # 这里会 ZeroDivisionError，而且是在**训练结束之后**才崩，白跑一整轮。
    # 早先写的 ``max(...) or 1`` 只能防 xmax==0，防不住 xmax==xmin。
    if xmax - xmin < 1e-9:
        xmax = xmin + 1
    ymin = min(p[1] for p in all_pts); ymax = max(p[1] for p in all_pts)
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0
    dr.rectangle([pad, pad, W - pad, H - pad], outline=(70, 70, 82))
    for i, (k, pts) in enumerate(series.items()):
        if not pts:
            continue
        col = colors[i % len(colors)]
        xy = [((x - xmin) / (xmax - xmin) * (W - 2 * pad) + pad,
               H - pad - (y - ymin) / (ymax - ymin) * (H - 2 * pad)) for x, y in pts]
        if len(xy) == 1:
            # 单点：画一个小方块，否则 line() 什么都画不出来，曲线图会是空的
            x, y = xy[0]
            dr.rectangle([x - 3, y - 3, x + 3, y + 3], fill=col)
        else:
            dr.line(xy, fill=col, width=2)
        dr.text((pad + 8, pad + 8 + i * 16), f"{k}", fill=col)
    dr.text((pad, H - pad + 8), f"step {xmin}..{xmax}", fill=(180, 180, 190))
    dr.text((W - pad - 150, pad - 26), f"{ymin:.3f} .. {ymax:.3f}", fill=(180, 180, 190))
    img.save(out_png)
    return True


# --------------------------------------------------------------------------
# 样本网格
# --------------------------------------------------------------------------

def tensor_to_pil(t: torch.Tensor, scale: int = 3) -> Image.Image:
    """(4,H,W) 值域 [0,1] → RGBA PIL，棋盘底衬出透明区。"""
    a = (t.detach().float().cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(a, "RGBA").resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    bg = Image.new("RGBA", im.size, (46, 46, 54, 255))
    step = max(4, im.size[0] // 16)
    dr = ImageDraw.Draw(bg)
    for y in range(0, im.size[1], step):
        for x in range(0, im.size[0], step):
            if ((x // step) + (y // step)) % 2 == 0:
                dr.rectangle([x, y, x + step - 1, y + step - 1], fill=(70, 70, 80, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def save_sample_grid(samples: torch.Tensor, path: str, cols: int = 10, scale: int = 3) -> None:
    """samples: (N,4,64,64) in [0,1]。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tiles = [tensor_to_pil(samples[i], scale) for i in range(samples.size(0))]
    if not tiles:
        return
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1), rows * th + (rows + 1)), (18, 18, 22))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (1 + c * (tw + 1), 1 + r * (th + 1)))
    sheet.save(path)


def alpha_binarize(x: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """把 alpha 阈值化成 0/1。皮肤里半透明很少见，这一步能显著提升「可被解析」的观感。"""
    out = x.clone()
    a = (out[:, 3:4] > thresh).float()
    return torch.cat([out[:, :3], a], dim=1)


def alpha_binary_loss(alpha: torch.Tensor, margin: float = 0.18) -> torch.Tensor:
    """鼓励 alpha 靠近 0 或 1（hinge）：只惩罚落在 (margin, 1-margin) 区间的像素。"""
    d = torch.minimum(alpha.abs(), (1.0 - alpha).abs())
    return F.relu(margin - d).mean()


def jitter_mask(mask: torch.Tensor, p: float = 0.5,
                generator: "torch.Generator | None" = None) -> torch.Tensor:
    """对条件平面的 mask 做**形态扰动**：随机膨胀 / 腐蚀 / 平移。

    为什么需要
    ----------
    训练时若把「目标自己的 alpha」当条件平面（``extra``），mask 与内容**强相关**
    —— 模型可以走「从 mask 形状认出是哪张皮肤 → 倒出对应颜色」的捷径。
    推理时拿到的是**无关**的真实 mask，落在这条捷径的分布之外，于是模型在
    「mask 露出、但训练时无监督」的位置输出满熵噪声（用户看到的「纯噪点」）。

    ``--mask-cross-sample`` 从**样本维度**切断相关性（用别的样本的 mask）；
    本函数从**形状维度**再补一刀：让 mask 的形状不完全等于目标的 alpha 边界。
    两者都保留真实连通结构（不是逐像素随机翻），所以不引入 i.i.d. 盐噪声那种
    新的分布错配。

    ``p`` 是每一刀生效的概率。3×3 max/min pool 做膨胀/腐蚀；平移 ±1 像素。
    """
    if p <= 0:
        return mask
    g = generator
    def uni() -> float:
        t = torch.rand(1, generator=g)
        return float(t)
    out = mask
    # 膨胀（3×3 max）：只有 mask 区域内的像素可以长出去，保持 base 层几乎不变
    if uni() < p:
        out = F.max_pool2d(out, 3, 1, 1)
    if uni() < p:
        out = -F.max_pool2d(-out, 3, 1, 1)     # 腐蚀 = 对补集做膨胀再取补
    if uni() < p * 0.6:
        dy, dx = int(torch.randint(-1, 2, (1,)).item()), int(torch.randint(-1, 2, (1,)).item())
        if dy or dx:
            out = torch.roll(out, shifts=(dy, dx), dims=(2, 3))
    return out.clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# 断点
# --------------------------------------------------------------------------

def save_ckpt(path: str, **state) -> None:
    """原子写 checkpoint：先写 ``path.tmp`` 再 ``os.replace`` 覆盖。

    ⚠️ **Windows 上 ``os.replace`` 会间歇性抛 ``PermissionError``（WinError 5 拒绝访问）**
    —— 实测在 ``pc_soft`` 训练到 step 6600（epoch 2 末）时，``latest.pt`` 写成功、
    紧跟着的 ``ema.pt`` 替换失败，异常一路冒泡**直接终止了整个训练**
    （17 分钟进度，只靠 latest.pt 先存成功才没丢）。

    成因：目标文件那一刻被别的进程**独占**。最常见的是杀软 / Windows Search /
    索引服务的实时扫描刚盯上这个新写的大文件（本项目 ``latest.pt`` 有 218 MB）。
    这类锁**通常几十毫秒内释放**，所以正确做法是**带退避重试**，而不是失败即崩。

    重试用尽后**不删 tmp**：``path.tmp`` 内容此时是完好的最新权重，
    保留它就能手动改名为正式文件恢复（比在锁住的目标上再 copy 更可靠）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    delay, last = 0.1, None
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:                       # WinError 5：被瞬时独占
            last = exc
            if attempt < 9:
                time.sleep(delay)
                delay = min(delay * 1.6, 2.0)
    raise PermissionError(
        f"save_ckpt: 替换 {path} 连续 10 次失败（{last}）。"
        f"临时文件 {tmp} 内容完好，可直接改名为 {os.path.basename(path)} 恢复，"
        f"或稍后重试。") from last


def load_ckpt(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


class Heartbeat:
    """把「训练进程还活着、跑到哪了」写成一个原子更新的 JSON 文件。

    为什么需要它：训练监控面板要判断的是「在跑 / 已停 / 卡死」，
    光看日志文件的修改时间不可靠（一个 epoch 要 4 分钟，中间完全没有写盘）。
    心跳由**训练进程自己**写，所以里面的 ``pid`` 一定是训练进程，
    监控面板据此做启停控制，不需要去枚举系统进程、也不用猜。

    文件是「先写临时文件再 rename」的原子更新，读到半截 JSON 的概率为 0。
    """

    def __init__(self, path: str, **static_fields):
        self.path = path
        self.t0 = time.time()
        self._static = dict(static_fields)
        self._last: dict = {}          # 记住最后一次 beat 的字段，供 finish 继承
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _write(self, d: dict) -> None:
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass      # 心跳失败绝不能影响训练

    def beat(self, **fields) -> None:
        d = {"pid": os.getpid(), "ts": time.time(),
             "alive_s": round(time.time() - self.t0, 1), "finished": False}
        d.update(self._static)
        d.update(fields)
        self._last = d
        self._write(d)

    def finish(self, **fields) -> None:
        # 继承最后一次 beat 的字段（step / epoch / mse / vram…），
        # 否则训练正常结束后这些读数会全部消失，监控只能显示「最终 MSE —」。
        d = dict(self._last)
        d.update({"pid": os.getpid(), "ts": time.time(),
                  "alive_s": round(time.time() - self.t0, 1), "finished": True,
                  "phase": "finished"})
        d.update(fields)
        self._last = d
        self._write(d)


def rng_state() -> dict:
    return {"py": random.getstate(), "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    try:
        random.setstate(s["py"])
        np.random.set_state(s["np"])
        torch.set_rng_state(s["torch"])
        if s.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s["cuda"])
    except Exception:
        pass
