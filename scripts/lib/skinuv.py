"""skinuv.py — Minecraft 皮肤 UV 规范校验与结构分析。

Minecraft 皮肤是固定 UV 布局的纹理图。本模块把「能否被 Minecraft 正常解析」
写成可执行的判定逻辑，供采集过滤、清洗、生成验证三个阶段共用。

64×64（现代格式，1.8+）
------------------------
每个部位占 32×16 的方块，左半是**外层 / overlay**，右半是**本体 / base**。
历史上（1.8 之前）所有本体都在左半、右半留空；1.8 之后把「左臂 / 左腿 / 全部
overlay」放到了右半，于是图从 64×32 变成 64×64。

坐标以 (x0, y0, x1, y1) 半开区间表示。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from PIL import Image

#: 现代 64×64 布局：(部件名, x0, y0, x1, y1, 是否 overlay)
REGIONS_64: tuple[tuple[str, int, int, int, int, bool], ...] = (
    ("head",          0,  0, 32, 16, False),
    ("head_overlay", 32,  0, 64, 16, True),
    ("body",         16, 16, 40, 32, False),
    ("right_arm",    40, 16, 56, 32, False),
    ("right_leg",     0, 16, 16, 32, False),
    ("body_overlay", 16, 32, 40, 48, True),
    ("right_arm_overlay", 40, 32, 56, 48, True),
    ("right_leg_overlay",  0, 32, 16, 48, True),
    ("left_arm",     32, 48, 48, 64, False),
    ("left_leg",     16, 48, 32, 64, False),
    ("left_arm_overlay", 48, 48, 64, 64, True),
    ("left_leg_overlay",  0, 48, 16, 64, True),
)

#: 旧版 64×32 布局：只有本体，左臂左腿由右臂右腿镜像而来，无 overlay
REGIONS_32: tuple[tuple[str, int, int, int, int, bool], ...] = (
    ("head",       0,  0, 32, 16, False),
    ("body",      16, 16, 40, 32, False),
    ("right_arm", 40, 16, 56, 32, False),
    ("right_leg",  0, 16, 16, 32, False),
)

#: 面部正面 8×8 区域（64×64 与 64×32 相同）。皮肤是否「有意义」的主要判据。
FACE_BOX = (8, 8, 16, 16)
#: 身体正面 (20,20)-(28,32) —— 用于判断躯干是否被填充
TORSO_FRONT_BOX = (20, 20, 28, 32)

# ======================================================================
# 左右肢体本体区（用于识别「错位 / 只画半张」的破损皮肤）
# ======================================================================
# 现代 64×64 里：右腿 (0,16)、右臂 (40,16) 在上半部；左腿 (16,48)、左臂 (32,48) 在下半部。
# 一张正常的 64×64 皮肤左右肢体都会画；如果下半部整片空，说明它其实是把
# 64×32 旧版硬塞进上半部（或画布错位），游戏里左臂左腿会渲染成透明 —— 破损。
LIMB_R_BOXES: tuple[tuple[int, int, int, int], ...] = (
    (0, 16, 16, 32),    # 右腿
    (40, 16, 56, 32),   # 右臂
)
LIMB_L_BOXES: tuple[tuple[int, int, int, int], ...] = (
    (16, 48, 32, 64),   # 左腿
    (32, 48, 48, 64),   # 左臂
)
#: 现代 64×64 皮肤左肢体本体占用下限。
#: 实测 5 万样本里这一项是**双峰**的：要么 ≈0，要么 ≥0.78（p10 就是 0.781），
#: 中间几乎没有样本。阈值取 0.10 落在谷底，安全且不误杀。
LIMB_L_MIN = 0.10


# ======================================================================
# Steve(经典/classic，手臂 4px) vs Alex(纤细/slim，手臂 3px) 判别
# ======================================================================
# 原理：手臂在贴图上占一个 16×16 的盒子 UV 展开区。标准 net 布局：
#
#            [  d  ][  w  ][  d  ][  w  ]
#   上排(h=d) [  top   ][ bottom ]
#   下排(h=h) [ right][ front ][ left ][ back ]
#
#   经典 (w=4,d=4,h=12)：展开宽 = 2*(4+4) = 16，正好铺满 16 宽的盒子；
#        back 面落在 x∈[u+12,u+16)，y∈[v+4,v+16)。
#   纤细 (w=3,d=3,h=12)：展开宽 = 2*(3+3) = 12，只铺前 12 列、前 15 行；
#        back 面落在 x∈[u+9,u+12)，而 x∈[u+12,u+16) **永远空着**。
#
# 于是有两条互不依赖的判据（旧版只用了第二条的弱化版，实测误判很多）：
#
#   判据 A（物理上限，最硬）：纤细的手臂盒最多只能填 12*15 = 180/256 px，
#       占用率上限 0.7031。**超过这个值就绝不可能是纤细**。
#   判据 B（排他区）：x∈[u+12,u+16)、y∈[v+4,v+16) 与 y=v+15、x∈[u,u+12)
#       这块 L 形区域（60 px）经典会画、纤细永远画不到。里面有笔画 → 经典。
#
# 旧版为什么错：用「加权被使用率 used」落区间，`used=(0.5+0.5+0+0)/3=0.333`
# 刚好卡在 classic 阈值 0.35 之下掉进 unknown；而「背面没画但手臂画满」的经典
# 皮肤又会被判成 slim。实测 unknown 占 16%、slim 占 0.14%，两个数都不可信。
#
# 右臂盒 (40,16)  / 左臂盒 (32,48)      —— 本体
# 右臂 overlay 盒 (40,32) / 左臂 overlay 盒 (48,48)
ARM_BASE_BOXES: tuple[tuple[str, int, int], ...] = (
    ("rarm", 40, 16),
    ("larm", 32, 48),
)
ARM_OVERLAY_BOXES: tuple[tuple[str, int, int], ...] = (
    ("rarmov", 40, 32),
    ("larmov", 48, 48),
)
#: 手臂盒里各条带的相对坐标（dx0, dy0, dx1, dy1）
_ARM_EXCL_RIGHT = (12, 4, 16, 16)   # 经典 back face，48 px
_ARM_EXCL_BOTTOM = (0, 15, 12, 16)  # 经典侧面末行，12 px
_ARM_FOOT = (0, 0, 12, 15)          # 纤细真正使用的脚印区，180 px
#: 纤细手臂盒占用率的物理上限
SLIM_ARM_BOX_MAX = 12 * 15 / 256    # = 0.703125
#: 判据 B 的阈值
CLASSIC_EXCL_MIN = 0.06
SLIM_FOOT_MIN = 0.35

#: 兼容旧字段名（其它脚本/清单里出现过）
MODEL_PROBE_STRIPS: tuple[tuple[str, int, int, int, int], ...] = (
    ("rarm_back",   52, 20, 56, 32),
    ("larm_back",   44, 52, 48, 64),
    ("rarmov_back", 52, 36, 56, 48),
    ("larmov_back", 60, 52, 64, 64),
)
#: 手臂"一定有内容"的参考区（用于判断这张皮肤到底画没画手臂）
ARM_BODY_BOXES: tuple[tuple[int, int, int, int], ...] = (
    (40, 16, 56, 32),   # 右臂本体
    (32, 48, 48, 64),   # 左臂本体
)


# ======================================================================
# 质量打分：过滤「低质量 / 颜色特别简陋」的皮肤
# ======================================================================
# 三个正交维度：
#   n_colors      —— 可见像素里的唯一 RGB 数（颜色是否"简陋"）
#   dom_ratio     —— 最高频颜色占比（是否大面积纯色块）
#   edge_density  —— 可见区域内相邻像素颜色变化率（是否有细节）
QUALITY_MIN_COLORS = 8          # 少于此直接判死（近似 1~2 色的"涂色块"）
QUALITY_DOM_RATIO_MAX = 0.72    # 单一颜色占比超此值 = 大面积纯色板
QUALITY_MIN_EDGE = 0.015        # 可见区边缘密度下限（近乎无细节）
QUALITY_SIMPLE_COLORS = 14      # 「简陋」上限：颜色数低于此……
QUALITY_SIMPLE_EDGE = 0.06      # ……且边缘密度也低于此 → 判为简陋


def _arm_box_stats(alpha: np.ndarray, u: int, v: int) -> dict:
    """返回一个 16×16 手臂盒的 box / excl / foot 三项占用率。"""
    sub = alpha[v:v + 16, u:u + 16] > 0
    x0, y0, x1, y1 = _ARM_EXCL_RIGHT
    excl_right = int(np.count_nonzero(sub[y0:y1, x0:x1]))
    a0, b0, a1, b1 = _ARM_EXCL_BOTTOM
    excl_bottom = int(np.count_nonzero(sub[b0:b1, a0:a1]))
    excl_n = (y1 - y0) * (x1 - x0) + (b1 - b0) * (a1 - a0)
    f0, g0, f1, g1 = _ARM_FOOT
    foot = int(np.count_nonzero(sub[g0:g1, f0:f1]))
    return {
        "box": float(np.count_nonzero(sub) / 256),
        "excl": (excl_right + excl_bottom) / excl_n,
        "foot": foot / ((g1 - g0) * (f1 - f0)),
    }


def detect_model_type(arr: np.ndarray) -> tuple[str, float, dict]:
    """判断经典 / 纤细模型。

    返回 ``(type, confidence, evidence)``，type ∈ {classic, slim, unknown}。
    confidence ∈ [0,1]；evidence 记录每条判据的实测值，便于人工复核与回归。

    判定顺序（先硬后软）::

        两臂盒占用都 < 0.02          -> unknown（手臂压根没画，无从判断）
        占用率 > SLIM_ARM_BOX_MAX    -> classic（超出纤细的物理上限，铁证）
        排他区(L 形 60px)有笔画      -> classic（纤细永远画不到那块）
        排他区为 0 且脚印区够满      -> slim
        其余                          -> unknown（证据不足）
    """
    a = arr[..., 3]
    base = {tag: _arm_box_stats(a, u, v) for tag, u, v in ARM_BASE_BOXES}
    ov = {tag: _arm_box_stats(a, u, v) for tag, u, v in ARM_OVERLAY_BOXES}

    arm_ratio = float(np.mean([s["box"] for s in base.values()]))
    box_max = max(s["box"] for s in base.values())
    box_min = min(s["box"] for s in base.values())
    excl_max = max(s["excl"] for s in base.values())
    excl_ov_max = max(s["excl"] for s in ov.values())
    foot_min = min(s["foot"] for s in base.values())

    # 兼容旧 evidence 键名（四条判别带占用率）
    occ: dict[str, float] = {}
    for name, x0, y0, x1, y1 in MODEL_PROBE_STRIPS:
        sub = a[y0:y1, x0:x1]
        occ[name] = round(float(np.count_nonzero(sub > 0) / sub.size), 4)

    ev = {
        "arm_ratio": round(arm_ratio, 4),
        "box_max": round(box_max, 4), "box_min": round(box_min, 4),
        "excl_max": round(excl_max, 4), "excl_ov_max": round(excl_ov_max, 4),
        "foot_min": round(foot_min, 4), **occ,
    }

    if box_max < 0.02:
        return "unknown", 0.0, ev

    # 判据 A：超出纤细物理上限 -> 铁定经典
    if box_max > SLIM_ARM_BOX_MAX + 1e-6:
        over = (box_max - SLIM_ARM_BOX_MAX) / (1.0 - SLIM_ARM_BOX_MAX)
        return "classic", round(min(1.0, 0.75 + 0.25 * over), 3), ev

    # 判据 B：排他区有笔画 -> 经典
    if excl_max > CLASSIC_EXCL_MIN or excl_ov_max > CLASSIC_EXCL_MIN:
        conf = min(1.0, 0.6 + 2.0 * max(excl_max, excl_ov_max))
        return "classic", round(conf, 3), ev

    # 判据 B 反向：排他区干净 + 脚印区饱满 -> 纤细
    if excl_max <= 1e-9 and excl_ov_max <= 1e-9 and foot_min >= SLIM_FOOT_MIN:
        return "slim", round(min(1.0, 0.6 + foot_min / 2.5), 3), ev

    return "unknown", round(0.4 + foot_min / 2.5, 3), ev



# ======================================================================
# 感知哈希（供清洗阶段做近似去重，不需要再读一次像素）
# ======================================================================

def dhash_bits64(arr: np.ndarray) -> int:
    """8×8 dHash 打成 64 位整数。

    先把 alpha 合成到中灰底上再转灰度，避免「透明区」在不同皮肤间被当成相同；
    然后对 9×8 的灰度图做水平差分，得到 8×8=64 个比特。

    为什么打成 int：15 万样本的近似去重内层是 Python 循环，
    ``(a ^ b).bit_count()`` 是纯 C，比 numpy bool 数组比较快一个数量级。
    """
    rgba = Image.fromarray(np.asarray(arr, dtype=np.uint8), "RGBA")
    bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
    flat = Image.alpha_composite(bg, rgba).convert("L")
    g = np.array(flat.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def dhash_hex(arr: np.ndarray) -> str:
    """64 位 dHash 的 16 位十六进制表示（写进宽表 CSV 用）。"""
    return f"{dhash_bits64(arr):016x}"


def quality_metrics(arr: np.ndarray) -> dict:
    """质量三维度 + 分档。``tier``: good / ok / simple / reject。"""
    a = arr[..., 3]
    rgb = arr[..., :3]
    vis = a > 0
    n_vis = int(np.count_nonzero(vis))
    if n_vis == 0:
        return {"n_colors": 0, "dom_ratio": 1.0, "edge_density": 0.0,
                "color_entropy": 0.0, "tier": "reject", "q_reason": "empty"}

    flat_rgb = rgb[vis]
    # 唯一色
    codes = (flat_rgb[:, 0].astype(np.uint32) << 16 | flat_rgb[:, 1].astype(np.uint32) << 8
             | flat_rgb[:, 2].astype(np.uint32))
    uniq, counts = np.unique(codes, return_counts=True)
    n_colors = int(uniq.size)
    dom_ratio = float(counts.max() / n_vis)
    p = counts / n_vis
    color_entropy = float(-(p * np.log2(p)).sum())

    # 边缘密度：只看「两个像素都可见」的相邻对，比较颜色差
    g = rgb.astype(np.int16)
    dx = np.abs(g[:, 1:, :] - g[:, :-1, :]).sum(axis=2)
    dy = np.abs(g[1:, :, :] - g[:-1, :, :]).sum(axis=2)
    vx = vis[:, 1:] & vis[:, :-1]
    vy = vis[1:, :] & vis[:-1, :]
    ex = int(np.count_nonzero((dx > 0) & vx))
    ey = int(np.count_nonzero((dy > 0) & vy))
    denom = int(np.count_nonzero(vx)) + int(np.count_nonzero(vy))
    edge_density = float((ex + ey) / denom) if denom else 0.0

    # ---- 分档 ----
    if n_colors < QUALITY_MIN_COLORS:
        tier, why = "reject", f"few_colors:{n_colors}"
    elif dom_ratio > QUALITY_DOM_RATIO_MAX and n_colors < 24:
        tier, why = "reject", f"flat_block:{dom_ratio:.2f}"
    elif edge_density < QUALITY_MIN_EDGE:
        tier, why = "reject", f"no_detail:{edge_density:.4f}"
    elif n_colors < QUALITY_SIMPLE_COLORS and edge_density < QUALITY_SIMPLE_EDGE:
        tier, why = "simple", f"simple:{n_colors}/{edge_density:.3f}"
    elif n_colors >= 24 and edge_density >= 0.15:
        tier, why = "good", "fine"
    else:
        tier, why = "ok", "normal"

    return {"n_colors": n_colors, "dom_ratio": round(dom_ratio, 4),
            "edge_density": round(edge_density, 4),
            "color_entropy": round(color_entropy, 3),
            "tier": tier, "q_reason": why}


@dataclass
class SkinReport:
    path: str
    ok: bool
    mode: str = ""
    width: int = 0
    height: int = 0
    is_legacy: bool = False
    reason: str = ""
    # 结构统计
    alpha_min: int = 255
    alpha_max: int = 0
    transparent_ratio: float = 0.0   # 全图 alpha==0 占比
    overlay_ratio: float = 0.0       # overlay 区域里被使用的像素占比
    face_opaque_ratio: float = 0.0   # 面部正面不透明像素占比
    torso_opaque_ratio: float = 0.0
    unique_colors: int = 0           # 只看 alpha>0 的像素
    # 左右肢体本体占用（识别错位/半张皮肤）
    limb_l_ratio: float = 0.0
    limb_r_ratio: float = 0.0
    # 模型类型（Steve / Alex）
    model_type: str = ""             # classic | slim | unknown | legacy_na
    model_conf: float = 0.0
    arm_ratio: float = 0.0           # 手臂区域被画的像素占比
    # 质量
    quality_tier: str = ""           # good | ok | simple | reject
    quality_reason: str = ""
    dom_ratio: float = 0.0
    edge_density: float = 0.0
    color_entropy: float = 0.0

    def to_row(self) -> dict:
        return asdict(self)


def _regions_for(h: int):
    return REGIONS_64 if h == 64 else REGIONS_32


def analyze(path: str, quality_gate: str = "reject", model_gate: str = "strict") -> SkinReport:
    """校验并统计单张皮肤。不抛异常——所有问题都落在 ``ok`` / ``reason`` 上。

    ``quality_gate``: ``"reject"`` 只丢最低档；``"simple"`` 连「简陋」一档也丢；
    ``"off"`` 不做质量判定。
    ``model_gate``: ``"strict"`` 会丢掉骨架类型判不出来 / 左肢体缺画的破损皮肤；
    ``"off"`` 只统计不淘汰（做分布诊断时用）。
    """
    rep = SkinReport(path=path, ok=False)
    try:
        im = Image.open(path)
        im.load()
    except Exception as exc:  # 损坏 / 非图片
        rep.reason = f"unreadable:{type(exc).__name__}"
        return rep
    return analyze_image(im, path, quality_gate, model_gate)


def analyze_bytes(data: bytes, path: str = "", quality_gate: str = "reject",
                  model_gate: str = "strict") -> SkinReport:
    """直接从内存里的 PNG 字节校验（HF zip / parquet 流式处理用，不落盘）。"""
    import io
    rep = SkinReport(path=path, ok=False)
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        rep.reason = f"unreadable:{type(exc).__name__}"
        return rep
    return analyze_image(im, path, quality_gate, model_gate)


def analyze_image(im: Image.Image, path: str = "", quality_gate: str = "reject",
                  model_gate: str = "strict") -> SkinReport:
    rep = SkinReport(path=path, ok=False)
    rep.mode = im.mode
    rep.width, rep.height = im.size

    # ---- 硬性格式条件 ------------------------------------------------
    if im.mode != "RGBA":
        rep.reason = f"not_rgba:{im.mode}"
        return rep
    if (rep.width, rep.height) not in {(64, 64), (64, 32)}:
        rep.reason = f"bad_size:{rep.width}x{rep.height}"
        return rep

    arr = np.asarray(im, dtype=np.uint8)
    alpha = arr[..., 3]
    rep.is_legacy = rep.height == 32

    rep.alpha_min = int(alpha.min())
    rep.alpha_max = int(alpha.max())

    total = alpha.size
    n_transparent = int(np.count_nonzero(alpha == 0))
    rep.transparent_ratio = round(n_transparent / total, 4)

    # 全透明 = 损坏 / 空皮肤
    if rep.alpha_max == 0:
        rep.reason = "fully_transparent"
        return rep

    # 只看有效像素
    visible = alpha > 0
    n_visible = int(np.count_nonzero(visible))
    if n_visible < 16:
        rep.reason = f"too_few_pixels:{n_visible}"
        return rep

    rgb = arr[..., :3][visible]
    rep.unique_colors = int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0])
    if rep.unique_colors <= 1:
        rep.reason = "single_color"   # 纯色 = 无信息
        return rep

    # ---- 模型类型（Steve / Alex）------------------------------------
    if rep.height == 64:
        mt, mc, ev = detect_model_type(arr)
        rep.model_type, rep.model_conf = mt, mc
        rep.arm_ratio = float(ev.get("arm_ratio", 0.0))
        # 左右肢体本体占用（识别「旧版塞上半部 / 画布错位」的破损皮肤）
        lp = rp = ln = rn = 0
        for x0, y0, x1, y1 in LIMB_L_BOXES:
            sub = alpha[y0:y1, x0:x1]
            lp += int(np.count_nonzero(sub > 0))
            ln += sub.size
        for x0, y0, x1, y1 in LIMB_R_BOXES:
            sub = alpha[y0:y1, x0:x1]
            rp += int(np.count_nonzero(sub > 0))
            rn += sub.size
        rep.limb_l_ratio = round(lp / ln, 4) if ln else 0.0
        rep.limb_r_ratio = round(rp / rn, 4) if rn else 0.0
    else:
        rep.model_type, rep.model_conf = "legacy_na", 0.0
        rep.limb_l_ratio = rep.limb_r_ratio = 0.0

    # ---- 质量（颜色简陋 / 无细节）-----------------------------------
    q = quality_metrics(arr)
    rep.quality_tier = q["tier"]
    rep.quality_reason = q["q_reason"]
    rep.dom_ratio = q["dom_ratio"]
    rep.edge_density = q["edge_density"]
    rep.color_entropy = q["color_entropy"]

    # ---- 结构条件 ----------------------------------------------------
    fx0, fy0, fx1, fy1 = FACE_BOX
    face_a = alpha[fy0:fy1, fx0:fx1]
    rep.face_opaque_ratio = round(float(np.count_nonzero(face_a > 0) / face_a.size), 4)

    tx0, ty0, tx1, ty1 = TORSO_FRONT_BOX
    torso_a = alpha[ty0:ty1, tx0:tx1]
    rep.torso_opaque_ratio = round(float(np.count_nonzero(torso_a > 0) / torso_a.size), 4)

    regions = _regions_for(rep.height)
    overlay_used, overlay_total = 0, 0
    for _name, x0, y0, x1, y1, is_overlay in regions:
        if not is_overlay:
            continue
        sub = alpha[y0:y1, x0:x1]
        overlay_used += int(np.count_nonzero(sub > 0))
        overlay_total += sub.size
    rep.overlay_ratio = round(overlay_used / overlay_total, 4) if overlay_total else 0.0

    # 面部正面完全空 => Minecraft 会渲染成一团透明，判定不合格
    if rep.face_opaque_ratio < 0.25:
        rep.reason = f"face_empty:{rep.face_opaque_ratio}"
        return rep

    # 可见像素集中在极小区域 => 近似空白
    if n_visible / total < 0.02:
        rep.reason = f"nearly_blank:{round(n_visible/total, 4)}"
        return rep

    # ---- 骨架类型闸门 -------------------------------------------------
    # 实测 5 万样本：7.1% 的「皮肤」两根手臂盒完全没画，肉眼复核后确认是
    # 品牌 logo / 平铺 2D 插画 / 近空白模板 —— 它们的内容根本不按 UV 布局摆放，
    # 在游戏里渲染出来是一团乱，属于垃圾数据，必须剔除而不是硬塞进训练集。
    if model_gate == "strict" and rep.height == 64:
        if rep.model_type == "unknown":
            rep.reason = f"modeltype_unknown:arm={rep.arm_ratio}"
            return rep
        if rep.limb_l_ratio < LIMB_L_MIN:
            rep.reason = f"left_limb_empty:{rep.limb_l_ratio}"
            return rep

    # ---- 质量闸门 ----------------------------------------------------
    if quality_gate == "simple" and rep.quality_tier in ("simple", "reject"):
        rep.reason = f"quality_{rep.quality_tier}:{rep.quality_reason}"
        return rep
    if quality_gate == "reject" and rep.quality_tier == "reject":
        rep.reason = f"quality_reject:{rep.quality_reason}"
        return rep

    rep.ok = True
    rep.reason = "ok"
    return rep


def region_view(im: Image.Image):
    """返回 ``name -> (子图, 是否 overlay)``，给可视化 / 分部位统计用。"""
    out = {}
    for name, x0, y0, x1, y1, is_overlay in _regions_for(im.size[1]):
        out[name] = (im.crop((x0, y0, x1, y1)), is_overlay)
    return out


def visible_mask(path: str) -> np.ndarray:
    """返回布尔掩码：True = 该像素可见（alpha>0）。"""
    arr = np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)
    return arr[..., 3] > 0


# ======================================================================
# 相似度指纹（供清洗阶段做近似去重，不必再读像素）
# ======================================================================
DHASH_N = 16          # 16×16 → 256 bit → 64 个十六进制字符


def dhash_hex(arr: np.ndarray, n: int = DHASH_N) -> str:
    """16×16 dHash → 定长十六进制字符串。

    透明像素先按 alpha 合成到中灰底（皮肤大片区域是透明的，直接算灰度会让
    所有皮肤都长得一样），再转灰度、求水平相邻差分二值化。

    把它存进清单，清洗阶段的近似去重就**不需要再打开任何图片**——
    两个指纹的 Hamming 距离用整数异或 + bit_count 就能算。
    """
    a = np.asarray(arr, dtype=np.uint8)
    alpha = a[..., 3:4].astype(np.uint16)
    rgb = (a[..., :3].astype(np.uint16) * alpha + 128 * (255 - alpha)) // 255
    g = np.asarray(
        Image.fromarray(rgb.astype(np.uint8), "RGB")
             .convert("L").resize((n + 1, n), Image.BILINEAR),
        dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:0{(n * n) // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    """两个 dHash 十六进制串的 Hamming 距离。"""
    if not a or not b:
        return 1 << 30
    return (int(a, 16) ^ int(b, 16)).bit_count()



def skin_contact_sheet(paths: Iterable[str], cols: int = 12, scale: int = 4,
                       bg: tuple[int, int, int] = (40, 40, 46)) -> Image.Image:
    """把若干皮肤拼成一张总览图（棋盘底，便于观察 alpha）。"""
    paths = list(paths)
    if not paths:
        return Image.new("RGB", (1, 1), bg)
    rows = (len(paths) + cols - 1) // cols
    cw, ch = 64 * scale, 64 * scale
    sheet = Image.new("RGB", (cols * cw, rows * ch), bg)
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGBA").resize((cw, ch), Image.NEAREST)
        except Exception:
            continue
        # 棋盘底衬出透明区
        tile = Image.new("RGB", (cw, ch), bg)
        step = max(4, cw // 16)
        for y in range(0, ch, step):
            for x in range(0, cw, step):
                if ((x // step) + (y // step)) % 2 == 0:
                    tile.paste((70, 70, 78), (x, y, min(x + step, cw), min(y + step, ch)))
        tile.paste(im, (0, 0), im)
        sheet.paste(tile, ((i % cols) * cw, (i // cols) * ch))
    return sheet
