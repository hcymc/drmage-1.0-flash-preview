"""engine.py — drmage 推理内核（模型载入 / 条件构造 / DDIM 采样 / 后处理量化）。

设计约束
--------
1. **不重复实现采样链**：`build_diffusion` 与 `quantize_to_palette` 只有一份——
   `scripts/30_generate.py`。这里按路径 importlib 导入它（文件名以数字开头，
   不能直接 `import`）。本项目有过「同一件事写两份、只错一处」的教训
   （见 `skinatlas.limb_mirror_index` 的注释），推理侧不再犯。
2. **模型常驻但要能换**：权重按 ``(路径, mtime, 大小)`` 做失效判断，
   换 checkpoint 不需要重启服务。
3. **一次只跑一个采样**：全局锁 + 显存检查。训练在跑时**必须拒绝**——
   采样要 ~4.6GB，与本机 8GB 卡上的训练叠加必 OOM。
4. **不动训练侧任何文件**：本模块只读项目数据，只写 ``webui/infer/``。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEBUI = os.path.dirname(HERE)
ROOT = os.path.dirname(WEBUI)

for _p in (os.path.join(ROOT, "scripts", "lib"), os.path.join(ROOT, "scripts"), WEBUI):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")

#: 对外展示名。路径 / tag 一律不改（路径与 tag 仍为 diff_v2_masked，与训练日志、
#: 复现文档里的名字保持一致，改名只发生在展示层）。
DISPLAY_NAME = "drmage-1.0-flash-preview"
DISPLAY_NOTE = "drm + image；flash = 参数极小（11.9M）。发布性质：实验预览。"

#: 默认权重 = 当前成绩最好的那一版（masked 续训 + EMA）
DEFAULT_CKPT = os.path.join("models", "diff_v2_masked", "ema.pt")

#: 推理平台只面向这个实验（阶段一预训练权重 diff_v1 不进下拉框，
#: 它只作为 docs/REPRODUCE.md 两阶段链条的中间产物存在）。
SCAN_TAG = "diff_v2_masked"

#: 采样时至少要留的显存（GB）。实测单次 16 张 DDIM100 峰值约 1.2GB，
#: 但训练并存才是真风险，所以这里给的是「不让训练跑起来」的量级。
MIN_FREE_VRAM_GB = 3.0

#: 建议优先展示的权重（下拉框排序用）：ema 出图明显更干净，
#: latest 保留给「想在原始权重上对比」的场景。
PREFERRED = [
    "models/diff_v2_masked/ema.pt",
    "models/diff_v2_masked/latest.pt",
]

_LOCK = threading.RLock()
_CACHE: dict = {"key": None, "model": None, "meta": None, "args": None}
_LAST_ERROR: str | None = None


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _import_30():
    """按路径导入 ``scripts/30_generate.py``（数字开头，无法普通 import）。"""
    path = os.path.join(ROOT, "scripts", "30_generate.py")
    key = "_gen30_infer"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vram() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {"cuda": True, "free_gb": round(free / 1024 ** 3, 2),
            "total_gb": round(total / 1024 ** 3, 2),
            "device": torch.cuda.get_device_name(0)}


def _norm_rel(rel_or_abs: str) -> str:
    """把用户给的权重路径收敛成「相对项目根、用正斜杠」的形式。"""
    p = (rel_or_abs or DEFAULT_CKPT).replace("\\", "/")
    if os.path.isabs(p):
        p = os.path.relpath(p, ROOT).replace("\\", "/")
    return p


def list_ckpts() -> list[dict]:
    """扫出可用权重，按「推荐优先 + mtime 倒序」排。

    只扫 ``models/<SCAN_TAG>/``（本发布只面向 diff_v2_masked）。
    ``ema.pt`` 是出图该用的（latest.pt 存的是原始权重 + 优化器状态，
    EMA 明显更干净）；这里把两者都列出来但标出 kind，默认选 ema。
    """
    rows: list[dict] = []
    root = os.path.join(MODELS, SCAN_TAG)
    if not os.path.isdir(root):
        return rows
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".pt"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT).replace("\\", "/")
            tag = os.path.relpath(dirpath, MODELS).replace("\\", "/")
            try:
                st = os.stat(full)
            except OSError:
                continue
            kind = ("ema" if fn == "ema.pt" else
                    "latest" if fn == "latest.pt" else
                    "epoch" if fn.startswith("epoch_") else "snapshot")
            rows.append({"rel": rel, "tag": tag, "name": fn, "kind": kind,
                         "size_mb": round(st.st_size / 1024 ** 2, 1),
                         "mtime": st.st_mtime,
                         "mtime_cn": datetime.fromtimestamp(st.st_mtime).strftime("%m-%d %H:%M"),
                         "preferred": rel in PREFERRED,
                         "pref_rank": PREFERRED.index(rel) if rel in PREFERRED else 99})
    rows.sort(key=lambda r: (r["pref_rank"], -r["mtime"]))
    return rows


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------

def _ckey(path: str) -> tuple:
    st = os.stat(path)
    return (path, int(st.st_mtime), st.st_size)


def load(rel_or_abs: str = DEFAULT_CKPT, force: bool = False) -> tuple:
    """载入权重（带缓存）。返回 ``(model, cond_dim, args, meta)``。

    ``meta`` 里有参数量 / in_ch / schedule / 训练步数，直接给前端展示，
    这样「现在玩的是哪一版」永远是可查的，不用靠记忆。
    """
    global _LAST_ERROR
    rel = _norm_rel(rel_or_abs)
    path = os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"权重不存在：{rel}")

    key = _ckey(path)
    with _LOCK:
        if not force and _CACHE["key"] == key and _CACHE["model"] is not None:
            # ⚠️ `args` 必须一起缓存：第 161 行读它，而 _CACHE 初值是 None。
            # 写缓存时漏了 args → **缓存命中就返回 None** → generate() 里
            # `args.get("alpha_input")` 抛 AttributeError（只有第一次请求能过）。
            # 这是「同一个键写一份、读一份」的典型漏项，别再把 update 写窄。
            return _CACHE["model"], _CACHE["meta"]["cond_dim"], _CACHE["args"], _CACHE["meta"]

        gen = _import_30()
        dev = device()
        t0 = time.time()
        model, cond_dim, args = gen.build_diffusion(path, dev)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # 键名按 tag 而异：训练脚本写的是 ``gstep``（global step），
        # 老快照里才可能叫 step/global_step/steps，四个都试一遍。
        # ``ema.pt`` 只存 shadow 权重 + gstep/epoch/args，没有优化器状态。
        step = (ck.get("gstep") or ck.get("step") or ck.get("global_step")
                or ck.get("steps"))
        epoch = ck.get("epoch")
        from models import count_params
        meta = {
            "rel": rel, "abs": path,
            "cond_dim": int(cond_dim or 0),
            "in_ch": int(model.unet.in_ch),
            "params_m": round(count_params(model.unet) / 1e6, 3),
            "timesteps": int(model.T),
            "schedule": str(model.schedule),
            "base": int(args.get("base", 0) or 0),
            "channels": int(args.get("channels", 0) or 0),
            "mask_loss": bool(args.get("mask_loss")),
            "trained_model_type": args.get("model_type"),
            "step": int(step) if step is not None else None,
            "epoch": int(epoch) if epoch is not None else None,
            "size_mb": round(os.path.getsize(path) / 1024 ** 2, 1),
            "mtime_cn": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M"),
            "load_seconds": round(time.time() - t0, 2),
            "device": str(dev),
        }
        _CACHE.update({"key": key, "model": model, "meta": meta, "args": args})
        torch.cuda.empty_cache()
        return model, meta["cond_dim"], args, meta


def cached_meta() -> dict | None:
    with _LOCK:
        return _CACHE.get("meta")


def last_error() -> str | None:
    return _LAST_ERROR


# ---------------------------------------------------------------------------
# alpha：由 UV 模板给出，模型完全不学 alpha（channels=3）
# ---------------------------------------------------------------------------

ALPHA_MODES = {
    "retrieval": "从真实 mask 库检索（**推荐**）——与训练侧同分布，空间结构 100% 真实",
    "template": "按实测频率逐像素 i.i.d. 抽（**旧行为**）。分布错配，会出椒盐/穿孔第二层",
    "shared": "整批共用同一张 mask（换条件对比时更公平）",
    "full": "alpha 全满（把模型画的所有像素都显示出来，便于看 UV 全貌）",
    "none": "全部透明（只用来验证渲染管线）",
}

#: 旧默认 `template` 实际上**不是**训练时的做法 —— 训练侧喂的是**真实 alpha**。
#: `sample_alpha_templates` 只复现了逐面覆盖率，空间结构为零（实测面内邻接一致率
#: 0.8958 vs 真实 0.9568；hat 盒连通块中位 1.5px vs 真实 11.5px）。把它当默认
#: 等于让推理侧的 conditioning 落在训练分布之外。默认改为 `retrieval`。
DEFAULT_ALPHA_MODE = "retrieval"


def make_alpha(n: int, mode: str, seed: int,
               box_paint: dict | None = None,
               ov_bits: "np.ndarray | None" = None) -> np.ndarray:
    """抽 ``(n, 4096)`` 的 alpha mask。

    ``box_paint``：``{盒名: (n,) 的 0/1}``，来自条件里的**部位级 overlay 四位**。
    给定时 mask 会按它决定「哪个盒画」——这样「条件说要有袖子」和「mask 确实
    露出袖子」是同一件事。

    ``ov_bits``：``(n,4)`` 的 0/1，``labelset.OV_BIT_NAMES`` 顺序。``retrieval``
    模式下用它做**精确分桶检索**（只在部位位完全一致的真实 mask 里抽），
    等价于 ``box_paint`` 的约束但形状来自真实作者。

    ``retrieval`` 缺库时**自动回落到 template 并打印警告**，不静默换分布。
    """
    rng = np.random.default_rng(seed)
    if mode == "full":
        return np.ones((n, 64 * 64), dtype=np.float32)
    if mode == "none":
        return np.zeros((n, 64 * 64), dtype=np.float32)
    from skinatlas import load_alpha_rates, sample_alpha_bank, sample_alpha_templates
    if mode in ("retrieval", "shared"):
        out = sample_alpha_bank(1 if mode == "shared" else n,
                                ov_bits=ov_bits, rng=rng)
        if out is None:
            print("[!] models/mask_bank.npz 缺失 —— 回落 template（分布会错配；"
                  "跑 scripts/25_build_mask_bank.py 生成）")
        else:
            return np.repeat(out, n, axis=0) if mode == "shared" else out
    return sample_alpha_templates(n, load_alpha_rates(), rng=rng,
                                  box_paint=box_paint)


# ---------------------------------------------------------------------------
# 条件
# ---------------------------------------------------------------------------

def spec_vector(spec: dict, ov_bits: bool = False) -> np.ndarray:
    """条件向量。

    ``ov_bits=True`` 时在 36 维基础向量后追加 **4 个部位级 overlay 位**
    （``labelset.OV_BIT_NAMES``：hat/body/arm/leg），总长 40 —— 与开了
    ``--ov-bits`` 训练的模型匹配。默认 ``False`` 以兼容旧权重。
    """
    from labelset import ov_bits_from_spec, vector_from_spec
    v = vector_from_spec(spec or {})
    if ov_bits:
        v = np.concatenate([v, ov_bits_from_spec(spec or {})]).astype(np.float32)
    return v


def describe_vector(v) -> dict:
    from labelset import describe
    return describe(np.asarray(v, dtype=np.float32))


_VAL_COND = None


def val_cond():
    """``data/processed/val_cond.npy``（真实皮肤的条件向量，13200×36）。

    已核对：``model_type`` 段**全是 classic**（var=0），所以直接拿它抽样
    不会外推到 slim —— 这是「用真实条件」这条路成立的前提。
    """
    global _VAL_COND
    if _VAL_COND is None:
        _VAL_COND = np.load(os.path.join(DATA, "processed", "val_cond.npy"))
    return _VAL_COND


_VAL = None


def val_skins():
    """真实 val 皮肤（mmap，不占内存）。``(N,4,64,64) uint8``。"""
    global _VAL
    if _VAL is None:
        _VAL = np.load(os.path.join(DATA, "processed", "val.npy"), mmap_mode="r")
    return _VAL


def real_index(n: int, seed: int, same_tone: str | None = None) -> list[int]:
    """从 val 里抽 n 个真实样本的下标（可选按 tone 段筛）。"""
    vc = val_cond()
    idx = np.arange(len(vc))
    if same_tone:
        from labelset import TONE_ORDER
        if same_tone in TONE_ORDER:
            k = TONE_ORDER.index(same_tone)
            idx = idx[vc[:, 19 + k] > 0.5]
    if len(idx) == 0:
        return []
    rng = np.random.default_rng(seed)
    take = min(n, len(idx))
    return rng.choice(idx, take, replace=False).tolist()


def gather_cond(mode: str, n: int, seed: int, spec: dict | None = None,
                ref_index: int | None = None,
                tone: str | None = None) -> tuple[np.ndarray, list[int] | None, dict]:
    """构造 ``(n, 36)`` 条件矩阵。三种来源：

    ``spec``  —— 用界面上的滑块合成一份条件，整批复制（对比参数变化用）
    ``real``  —— 从 val 里抽真实条件（每张不同，看模型的「模拟真实分布」能力）
    ``ref:``  —— 完全复刻某个真实样本的整条条件向量（与真实图逐张对照用）

    ``tone`` 只对 ``real`` 生效：按 tone one-hot 段筛一遍再抽。

    .. warning::
       **返回的矩阵行数可能小于 ``n``**（``real`` + ``tone`` 筛完就那么多）。
       调用方必须用 ``cond.shape[0]`` 当实际批量，不能继续用请求里的 ``n``——
       否则 ``model.sample(n, cond=cc)`` 收到行数不足的 ``cond``，
       形状对不上会静默出错或直接崩。
    """
    cd = _cond_dim()
    from labelset import COND_DIM, OV_BITS, ov_bits_from_spec
    use_ov = cd >= COND_DIM + OV_BITS        # 模型开了 --ov-bits
    _VAL_OV = os.path.join(DATA, "processed", "val_ov4.npy")

    def _augment(base: np.ndarray, rows=None) -> np.ndarray:
        """把 4 个部位位拼到 36 维基础向量之后。

        ``rows`` 给定时取该真实样本自己的位（real / ref 模式：条件与真实图
        必须同源）；不给时取自 ``spec``（spec 模式：整批同一份设置）。
        """
        if not use_ov:
            return base.astype(np.float32)
        if rows is not None:
            if os.path.isfile(_VAL_OV):
                ov = np.load(_VAL_OV)[rows].astype(np.float32)
            else:
                ov = np.zeros((base.shape[0], OV_BITS), dtype=np.float32)
        else:
            ov = np.tile(ov_bits_from_spec(spec or {})[None, :],
                         (base.shape[0], 1)).astype(np.float32)
        return np.concatenate([base, ov], axis=1).astype(np.float32)

    if mode == "real":
        idx = real_index(n, seed, same_tone=tone)
        vc = val_cond()
        cond = _augment(vc[idx].astype(np.float32), rows=np.asarray(idx))
        return cond, idx, {
            "mode": "real", "cond_dim": cd, "tone": tone, "n_actual": len(idx),
            "real_index": idx}
    if mode == "ref" and ref_index is not None:
        vc = val_cond()
        v = vc[int(ref_index)].astype(np.float32)
        cond = _augment(v[None, :], rows=np.asarray([int(ref_index)]))
        return np.repeat(cond, n, axis=0), None, {
            "mode": "ref", "ref_index": int(ref_index),
            "described": describe_vector(v)}
    v = spec_vector(spec or {}, ov_bits=use_ov).astype(np.float32)
    return np.repeat(v[None, :], n, axis=0), None, {"mode": "spec",
                                                    "described": describe_vector(v),
                                                    "spec": spec or {}}


def _cond_dim() -> int:
    m = cached_meta()
    return int(m["cond_dim"]) if m else 36


# ---------------------------------------------------------------------------
# 后处理量化
# ---------------------------------------------------------------------------

def quantize(rgb: np.ndarray, vis: np.ndarray, mode: str, k: int,
             global_pal: str | None = None, seed: int = 0) -> np.ndarray:
    """把可见像素吸附到调色板。``rgb`` 为 ``(64,64,3)`` 的 [0,1] 浮点。

    **实现在 ``scripts/lib/quantize.py``（全项目唯一一份）**：推理平台和
    ``30_generate.py``（导出）必须同口径，否则就是「平台上看着好、导出却是
    另一回事」。哪种更好有实测，见那份 docstring 里的表：**逐图自适应 K≈64**
    （色数 64 对真实 66、精确相等率 0.497 对 0.475、饱和 0.356 对 0.353）；
    全局板会过冲且把图拉灰（饱和 0.298）。

    * ``adaptive`` —— 每张图自己 k-means 选 K 色（**推荐**）。
    * ``global`` —— ``models/palette_kK.npy`` 全局板（保留给 A/B）。
    """
    if mode in ("off", "", None) or k <= 0:
        return rgb
    from quantize import quantize_u8          # scripts/lib，唯一实现
    u8 = np.clip(np.rint(np.asarray(rgb, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    pal = None
    if mode == "global":
        p = global_pal or os.path.join(MODELS, f"palette_k{k}.npy")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到全局调色板 {os.path.relpath(p, ROOT)}")
        pal = np.load(p).astype(np.float32)
        if pal.max() <= 1.5:                  # 有的板存 [0,1]
            pal = pal * 255.0
    out = quantize_u8(u8, vis, "global" if mode == "global" else "per", k,
                      palette=pal, models_dir=MODELS, seed=seed)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

def generate(cond: np.ndarray, n: int, ddim_steps: int = 100, eta: float = 0.0,
             seed: int = 1234, alpha_mode: str = DEFAULT_ALPHA_MODE,
             quant: dict | None = None, ckpt: str = DEFAULT_CKPT,
             heal: bool = True, amp: bool = False,
             alpha_np: "np.ndarray | None" = None) -> dict:
    """跑一次采样，返回 ``{images: [np.uint8 (64,64,4)], rgba_vis: [...], summary}``。

    ``heal``：破洞修补（skinheal，默认开）——把第一层孤立的透明像素用
    周围颜色补上。放在**量化之前**，让补进来的颜色也被吸进调色板，
    与整图观感一致。

    ``amp``：采样套 fp16 autocast（抽奖模式的提速开关，单次小批量可关）。

    ``alpha_np``：外部预抽好的 alpha mask ``(n, 4096)``。抽奖模式用它保证
    mask 分配只取决于 (seed, 候选序号)，与内部微批的切法无关。
    """
    global _LAST_ERROR
    with _LOCK:
        model, cond_dim, args, meta = load(ckpt)
        dev = device()
        info = vram()
        if info.get("cuda") and info["free_gb"] < MIN_FREE_VRAM_GB:
            raise RuntimeError(
                f"可用显存只有 {info['free_gb']}GB（低于安全线 {MIN_FREE_VRAM_GB}GB）。"
                "训练很可能正在跑——采样要 ~1.2GB，与训练叠加会 OOM，已拒绝。")

        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2 ** 31 - 1))
        n = int(max(1, min(64, n)))

        cc = torch.from_numpy(np.asarray(cond[:n], dtype=np.float32)).to(dev)
        if cond_dim and cc.shape[1] != cond_dim:
            raise ValueError(f"条件维度 {cc.shape[1]} != 模型要求的 {cond_dim}")

        # mask 必须**在采样前**抽好：alpha_input 的模型把它当条件平面，
        # 「哪里会露出来」是采样时的已知输入，而不是采样之后才决定的事。
        # 部位级 overlay 四位（cond_dim > COND_DIM 时）**直接从条件尾部取出**。
        # retrieval 模式下这四位用于**分桶检索**；template 模式下用于 box_paint。
        # 两条路都保证「条件说要有袖子」与「mask 确实露出袖子」是同一件事。
        from labelset import COND_DIM, OV_BITS, ov_bits_to_box_paint
        box_paint = bits = None
        if cond_dim >= COND_DIM + OV_BITS:
            bits = cc[:, COND_DIM:COND_DIM + OV_BITS].detach().cpu().numpy()
            box_paint = ov_bits_to_box_paint(bits)
        a_np = (np.asarray(alpha_np, dtype=np.float32)[:n]
                if alpha_np is not None
                else make_alpha(n, alpha_mode, int(seed), box_paint=box_paint, ov_bits=bits))
        a_t = torch.from_numpy(a_np).reshape(n, 1, 64, 64)
        alpha_input = bool(args.get("alpha_input"))
        extra = (a_t.to(dev) * 2 - 1) if alpha_input else None
        cfg_scale = float(args.get("cfg_scale", 1.0) or 1.0) if cond_dim else 1.0

        t0 = time.time()
        try:
            with torch.no_grad():
                x = model.sample(n, device=dev, cond=cc if cond_dim else None,
                                 ddim_steps=int(ddim_steps), eta=float(eta),
                                 extra=extra, cfg_scale=cfg_scale, amp=amp)
            x = ((x.clamp(-1, 1) + 1) / 2).float().cpu()
        except torch.cuda.OutOfMemoryError as exc:      # noqa: PERF203
            torch.cuda.empty_cache()
            _LAST_ERROR = f"CUDA OOM: {exc}"
            raise RuntimeError("显存不足（OOM）。把批量降到 8 或更小再试。") from exc
        samp_s = time.time() - t0

        q = quant or {}
        qmode = q.get("mode", "off")
        qk = int(q.get("k", 48) or 0)
        tq0 = time.time()
        rgb = x.permute(0, 2, 3, 1).numpy()             # (n,64,64,3) [0,1]
        vis = a_np.reshape(n, 64, 64) > 0.5
        # ---- 破洞修补（heal）：在量化**之前**做，补进来的颜色跟着进调色板 ----
        h_st = {"enabled": bool(heal), "holes": 0, "filled": 0, "skipped_big": 0}
        h_per = [0] * n
        if heal:
            bm = _base_mask()
            for i in range(n):
                arr8 = np.zeros((64, 64, 4), dtype=np.uint8)
                arr8[..., :3] = (np.clip(rgb[i], 0, 1) * 255).round().astype(np.uint8)
                arr8[..., 3] = np.where(vis[i], 255, 0).astype(np.uint8)
                import skinheal
                arr8, st = skinheal.heal_image(arr8, bm)
                h_st["holes"] += st["holes"]
                h_st["filled"] += st["filled"]
                h_st["skipped_big"] += st["skipped_big"]
                h_per[i] = st["filled"]
                rgb[i] = arr8[..., :3].astype(np.float32) / 255.0
                vis[i] = arr8[..., 3] >= 128
        if qmode not in ("off", "", None) and qk > 0:
            for i in range(n):
                rgb[i] = quantize(rgb[i], vis[i], qmode, qk,
                                  global_pal=q.get("palette"),
                                  seed=int(seed) + i)
        quant_s = time.time() - tq0

        images, per = [], []
        for i in range(n):
            a = vis[i]
            arr = np.zeros((64, 64, 4), dtype=np.uint8)
            arr[..., :3] = (np.clip(rgb[i], 0, 1) * 255).round().astype(np.uint8)
            arr[..., 3] = np.where(a, 255, 0).astype(np.uint8)
            arr[~a, :3] = 0                             # 透明区 RGB 归零（游戏内不显示但保持一致）
            images.append(arr)
            px = arr[..., :3][a] if a.any() else np.zeros((0, 3), np.uint8)
            per.append({
                "visible": round(float(a.mean()), 4),
                "unique_colors": int(len(np.unique(px, axis=0))) if len(px) else 0,
                "near_black": round(float((px.max(axis=1) < 25).mean()), 4) if len(px) else None,
                "mean_rgb": [int(v) for v in px.mean(axis=0).round(0)] if len(px) else None,
                "healed_px": (h_per[i] if heal else 0),
            })

        out = {
            "images": images, "per_sample": per,
            "summary": {
                "n": n, "ddim_steps": int(ddim_steps), "eta": float(eta),
                "seed": int(seed), "alpha_mode": alpha_mode,
                "quant": {"mode": qmode, "k": qk},
                "heal": h_st,
                "amp": bool(amp),
                "sample_seconds": round(samp_s, 2),
                "quant_seconds": round(quant_s, 2),
                "seconds": round(time.time() - t0, 2),
                "per_image_seconds": round((time.time() - t0) / max(n, 1), 3),
                "visible_mean": round(float(np.mean([p["visible"] for p in per])), 4),
                "unique_colors_mean": round(float(np.mean(
                    [p["unique_colors"] for p in per])), 1),
            },
        }
        _LAST_ERROR = None
        return out


_BASE_MASK = None


def _base_mask() -> "np.ndarray":
    """基础层修补区域缓存（skinheal.base_region_mask 的进程级单例）。"""
    global _BASE_MASK
    if _BASE_MASK is None:
        import skinheal
        _BASE_MASK = skinheal.base_region_mask()
    return _BASE_MASK


# ---------------------------------------------------------------------------
# 生成记录（可追溯：哪一版权重、什么条件、什么参数 → 什么结果）
# ---------------------------------------------------------------------------

HISTORY = os.path.join(HERE, "history.jsonl")


def append_history(rec: dict) -> None:
    try:
        with open(HISTORY, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def read_history(limit: int = 40) -> list[dict]:
    if not os.path.isfile(HISTORY):
        return []
    out: list[dict] = []
    with open(HISTORY, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out[-limit:][::-1]
