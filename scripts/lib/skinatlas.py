"""skinatlas.py — 把 64×64 皮肤图集拆成「部件 × 6 个立方体面」，并预计算
结构损失需要的像素邻接 / 镜像配对索引。

为什么需要它
------------
Minecraft 皮肤不是一张普通图片，而是**固定 UV 布局的图集**：
每个部件（头/身/左右臂/左右腿）的贴图都占一个盒子展开区，盒子再按
上/下/左/右/前/后 6 个面摊开。这意味着两件普通图像任务里没有的先验：

1. **面内应该近似分片常数**（像素画就是大色块 + 硬边）；
   但**面与面之间的边界是可以突变的**——所以不能用全图 TV 去平滑。
2. **左右肢体的面互为镜像**：右臂「前面」和左臂「前面」讲的是同一件事。

盒子展开（标准 net，宽 = 2(w+d)，高 = d+h）::

        [  d  ][  w  ][  d  ][  w  ]
  上排  [  top   ][ bottom ]           高度 d
  下排  [right][ front ][ left ][ back] 高度 h

用法::

    from skinatlas import FACE_MASKS, in_face_edges, limb_pairs
    masks = FACE_MASKS                      # [(name, (y0,y1,x0,x1)), ...]
    e = in_face_edges()                     # LongTensor (2, E) 展平后的像素下标
    pairs = limb_pairs()                    # [((name_l, flip), (name_r, flip)), ...]
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

SIZE = 64

#: 部件盒子：(名字, u, v, w, d, h, 是否 overlay)
#: 尺寸来自 Minecraft 标准皮肤：头 8³、身体 8×4×12、四肢 4×4×12
PART_BOXES: tuple[tuple[str, int, int, int, int, int, bool], ...] = (
    ("head",        0,  0, 8, 8, 8, False),
    ("hat",        32,  0, 8, 8, 8, True),
    ("body",       16, 16, 8, 4, 12, False),
    ("rarm",       40, 16, 4, 4, 12, False),
    ("rleg",        0, 16, 4, 4, 12, False),
    ("body_ov",    16, 32, 8, 4, 12, True),
    ("rarm_ov",    40, 32, 4, 4, 12, True),
    ("rleg_ov",     0, 32, 4, 4, 12, True),
    ("larm",       32, 48, 4, 4, 12, False),
    ("lleg",       16, 48, 4, 4, 12, False),
    ("larm_ov",    48, 48, 4, 4, 12, True),
    ("lleg_ov",     0, 48, 4, 4, 12, True),
)

#: 一个盒子里 6 个面的相对位置：(面名, dx, dy, fw, fh)
#: dx/dy 相对盒子左上角；fw/fh 是该面的宽高
_FACES = [
    ("top",     None, 0,       None, None),      # 特殊：dx=d, dy=0, fw=w, fh=d
    ("bottom",  None, 0,       None, None),      # 特殊：dx=d+w, dy=0
    ("right",   0,    None,    None, None),      # dx=0,      dy=d, fw=d, fh=h
    ("front",   None, None,    None, None),      # dx=d,      dy=d, fw=w, fh=h
    ("left",    None, None,    None, None),      # dx=d+w,    dy=d, fw=d, fh=h
    ("back",    None, None,    None, None),      # dx=d+w+d,  dy=d, fw=w, fh=h
]


def face_rects() -> list[tuple[str, int, int, int, int]]:
    """展开所有部件的 6 个面，返回 ``[(face_name, y0, y1, x0, x1), ...]``。

    ``face_name`` 形如 ``rarm.front`` / ``lleg.top`` / ``head_ov`` 里的 ``hat.back``。
    """
    out = []
    for name, u, v, w, d, h, _ov in PART_BOXES:
        rects = [
            ("top",    u + d,         v,         w, d),
            ("bottom", u + d + w,     v,         w, d),
            ("right",  u,             v + d,     d, h),
            ("front",  u + d,         v + d,     w, h),
            ("left",   u + d + w,     v + d,     d, h),
            ("back",   u + d + w + d, v + d,     w, h),
        ]
        for fname, x, y, fw, fh in rects:
            if fw <= 0 or fh <= 0:
                continue
            out.append((f"{name}.{fname}", y, y + fh, x, x + fw))
    return out


#: ``[(face_name, y0, y1, x0, x1), ...]``，一次性算好
FACE_MASKS: list[tuple[str, int, int, int, int]] = face_rects()

#: 立体化（front/back/left/right 才是「看得到的那一圈」）—— 用于统计与镜像配对
_LOOP_FACES = ("front", "back", "left", "right")


def face_index() -> dict[str, tuple[int, int, int, int]]:
    return {n: (y0, y1, x0, x1) for n, y0, y1, x0, x1 in FACE_MASKS}


def in_face_edges() -> np.ndarray:
    """面内相邻像素对。

    返回 ``(2, E)`` 的 int32 数组，每列是一对**同一个面内部**、且水平或垂直相邻的
    像素在展平后的下标（``idx = y*64 + x``）。

    只取面内：面与面之间的接缝在真实皮肤里本来就可能突变，
    放进去会逼模型把合理的硬边抹平。
    """
    pairs = []
    for _name, y0, y1, x0, x1 in FACE_MASKS:
        idx = (np.arange(y0, y1)[:, None] * SIZE + np.arange(x0, x1)[None, :])
        if x1 - x0 > 1:
            a = idx[:, :-1].ravel()
            b = idx[:, 1:].ravel()
            pairs.append(np.stack([a, b]))
        if y1 - y0 > 1:
            a = idx[:-1, :].ravel()
            b = idx[1:, :].ravel()
            pairs.append(np.stack([a, b]))
    return np.concatenate(pairs, axis=1).astype(np.int32)


@lru_cache(maxsize=1)
def pixel_face_id() -> np.ndarray:
    """``(4096,)`` int16：每个像素所属的面 id（``-1`` = 不属于任何面，即 padding 区）。

    这是「结构先验」类改造的底座：有了它才能区分「同一面内的相邻像素」
    与「跨 UV 缝的相邻像素」——而标准 3×3 卷积把这两类一视同仁。
    """
    m = np.full(SIZE * SIZE, -1, dtype=np.int16)
    for i, (_nm, y0, y1, x0, x1) in enumerate(FACE_MASKS):
        m[_flat_idx(y0, y1, x0, x1)] = i
    return m


def cross_face_edges() -> np.ndarray:
    """UV 上相邻、但**属于不同面**的像素对，返回 ``(2, E)`` int32。

    这批对子是「卷积跨 UV 缝」的现场。它们在 3D 上多数**并不相邻**
    （只有正好落在盒体棱上的那些才相邻），却被标准卷积当成邻域。
    把它们的色差分布与「面内相邻对」的色差分布做比值，就能定量回答
    「模型有没有跨缝串色 / 边界暗边」——真实数据先量一遍作为目标值。
    """
    fid = pixel_face_id().reshape(SIZE, SIZE)
    yy_all = np.arange(SIZE)[:, None] * SIZE
    xx_all = np.arange(SIZE)[None, :]
    a, b = [], []
    for dy, dx in ((0, 1), (1, 0)):
        p = fid[:SIZE - dy, :SIZE - dx]
        q = fid[dy:, dx:]
        ok = (p >= 0) & (q >= 0) & (p != q)
        yy = yy_all[:SIZE - dy, :SIZE - dx]
        xx = xx_all[:, :SIZE - dx]
        a.append((yy + xx)[ok])
        b.append((yy + dy + xx + dx)[ok])
    return np.stack([np.concatenate(a), np.concatenate(b)]).astype(np.int32)


def limb_pairs() -> list[tuple[str, str]]:
    """需要**互为镜像**的面对。

    语义：把角色左右镜像之后，右臂「前面」应该对应左臂「前面」（水平翻一次）。
    所以配对是 ``(右肢面, 左肢面)``，比较时右侧取水平翻转。

    只配 front / back / top / bottom —— 侧面的镜像关系是「右肢的右边」对应
    「左肢的左边」，跨了不同的面名，容易写错，先不纳入。
    """
    out = []
    for r, l in (("rarm", "larm"), ("rleg", "lleg")):
        for f in ("front", "back", "top", "bottom"):
            out.append((f"{r}.{f}", f"{l}.{f}"))
    return out


@lru_cache(maxsize=1)
def limb_mirror_index() -> tuple[tuple[str, str, np.ndarray, np.ndarray], ...]:
    """``[(部位, 配对名, 右肢面像素下标, 左肢面水平翻转后下标), ...]``。

    **这是唯一的权威实现，损失和各探针脚本都必须用它。**
    以前 losses 和各探针各写一份镜像下标，结果踩了一个很难看的坑：
    ``FACE_MASKS`` 里的矩形是 ``(y0, y1, x0, x1)``，其中 **x1 是开区间上界**
    （不含），水平镜像应当是 ``x0 + (x1-1-x)``；但如果调用方是用
    ``xx.max()`` 反推端点，拿到的就是**闭区间最大值** ``x1-1``，
    此时正确的式子变成 ``x0 + x1 - x``。

    两套写法各自都对，混用才会错 —— 而且只错 1 个像素，
    表现为「对称度莫名其妙差 20%」这种极难归因的现象（实测 arm 差 12%）。
    现在统一从 ``face_index()`` 取开区间端点，只有一处公式。
    """
    fi = face_index()
    out = []
    for rn, ln in limb_pairs():
        yr0, yr1, xr0, xr1 = fi[rn]
        yl0, yl1, xl0, xl1 = fi[ln]

        def _idx(y0, y1, x0, x1):
            yy = np.arange(y0, y1)[:, None]
            xx = np.arange(x0, x1)[None, :]
            return (yy * SIZE + xx).reshape(-1)

        ir = _idx(yr0, yr1, xr0, xr1)
        il = _idx(yl0, yl1, xl0, xl1)
        # x1 是开区间上界，所以水平镜像 = x0 + (x1 - 1 - x)
        mirrored = (il // SIZE) * SIZE + (xl0 + xl1 - 1 - (il % SIZE))
        out.append(("arm" if rn.startswith("rarm") else "leg", f"{rn}~{ln}",
                    ir, mirrored))
    return tuple(out)


#: 用于统计「真实数据长什么样」的分区（面级），把 loss 标定成**分布匹配**而不是盲目最小化
def stats_regions() -> list[tuple[str, int, int, int, int]]:
    return FACE_MASKS


# ======================================================================
# 分区（region）—— 为什么必须分开量
# ======================================================================
# 把 72 个面一起平均会掩盖真实差异：真实皮肤的腿经常接近纯色块，脸却有
# 眼睛/嘴这种硬细节，躯干 base 层常有大片纯色而 overlay 层常整片透明。
# 一个全局平均数会被这些区域互相抵消，于是「生成值略高」看起来无害，
# 实际上可能是「脸糊了、腿却是对的」。
#
# 分区方案（互斥、全覆盖，不重复计数）：
#
#   face       —— ``head.front`` 单独切出来（唯一的五官区）
#   *.base     —— 基础层：游戏里必然渲染的那层
#   *.overlay  —— 外层：帽子/夹克，作者经常整片不画
#
# 用它做损失标定后，每一项都对齐「同区域」的真实值。

#: 部件盒子 → 身体部位（互斥、全覆盖）
PART_GROUPS: dict[str, str] = {
    "head": "head", "hat": "head",
    "body": "body", "body_ov": "body",
    "rarm": "arm", "rarm_ov": "arm", "larm": "arm", "larm_ov": "arm",
    "rleg": "leg", "rleg_ov": "leg", "lleg": "leg", "lleg_ov": "leg",
}

#: 部件盒子 → 图层（base=本体，overlay=帽子/夹克这类可整片透明的外层）
LAYER_GROUPS: dict[str, str] = {
    "head": "base", "body": "base", "rarm": "base", "rleg": "base",
    "larm": "base", "lleg": "base",
    "hat": "overlay", "body_ov": "overlay", "rarm_ov": "overlay",
    "rleg_ov": "overlay", "larm_ov": "overlay", "lleg_ov": "overlay",
}

#: 支持的分区方案
SCHEMES: tuple[str, ...] = ("face", "part", "layer", "part_layer", "detail")

#: 推荐默认：把脸单独切出来，其余按「部件.图层」
DEFAULT_SCHEME = "detail"


def box_of(face_name: str) -> str:
    """``'lleg_ov.back'`` → ``'lleg_ov'``"""
    return face_name.split(".", 1)[0]


def region_label(scheme: str, face_name: str) -> str:
    """把一个面名映射到它所属的区域标签。"""
    box = box_of(face_name)
    if box not in PART_GROUPS:
        raise KeyError(f"未知部件 {box!r}（面 {face_name!r}）")
    if scheme == "face":
        return face_name
    if scheme == "part":
        return PART_GROUPS[box]
    if scheme == "layer":
        return LAYER_GROUPS[box]
    if scheme == "part_layer":
        return f"{PART_GROUPS[box]}.{LAYER_GROUPS[box]}"
    if scheme == "detail":
        # 脸是唯一有五官硬细节的区域，单列；其余按 部件.图层
        if face_name == "head.front":
            return "face"
        return f"{PART_GROUPS[box]}.{LAYER_GROUPS[box]}"
    raise KeyError(f"未知分区方案 {scheme!r}，可选 {SCHEMES}")


@lru_cache(maxsize=32)
def region_faces(scheme: str = DEFAULT_SCHEME) -> dict[str, list[tuple[str, int, int, int, int]]]:
    """``{区域名: [(面名, y0, y1, x0, x1), ...]}``，按区域聚好，键有序。"""
    out: dict[str, list] = {}
    for rect in FACE_MASKS:
        out.setdefault(region_label(scheme, rect[0]), []).append(rect)
    return {k: out[k] for k in sorted(out)}


@lru_cache(maxsize=32)
def region_edges(scheme: str = DEFAULT_SCHEME) -> dict[str, np.ndarray]:
    """``{区域名: (2, E_r) 面内邻接像素对}``。

    E_r 只含**该区域内部**的相邻对（跨区域、跨面接缝一律不算）。
    把区域切细之后每个区域的 E_r 会变小，所以损失里**不要直接用求和**，
    要么按区域取平均，要么按 E_r 加权。
    """
    out: dict[str, np.ndarray] = {}
    for name, rects in region_faces(scheme).items():
        pairs = []
        for _fn, y0, y1, x0, x1 in rects:
            idx = (np.arange(y0, y1)[:, None] * SIZE + np.arange(x0, x1)[None, :])
            if x1 - x0 > 1:
                pairs.append(np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]))
            if y1 - y0 > 1:
                pairs.append(np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()]))
        out[name] = np.concatenate(pairs, axis=1).astype(np.int32)
    return out


def region_sizes(scheme: str = DEFAULT_SCHEME) -> dict[str, int]:
    """``{区域名: 像素数}``，用于检查分区确实互斥且覆盖 64×64。"""
    return {k: sum((y1 - y0) * (x1 - x0) for _n, y0, y1, x0, x1 in v)
            for k, v in region_faces(scheme).items()}


# ======================================================================
# alpha 模板 —— alpha 在这个域里几乎是确定性函数，不该让模型学
# ======================================================================
# 实测（3000 张真实皮肤，scripts/_probe/alpha_rates.py）：
#   * base 层 62 个面 98~100% 全满（例外：手臂 bottom/back ≈72.5% 全满，
#     但「有内容」仍有 99.5%）→ 模板置 1，误差可忽略
#   * overlay 层是两级结构：盒级「画不画」(hat 0.90 / body_ov 0.58 / …)
#     × 盒内面级「画哪些」(外套正面 0.91、底面 0.22)
#   * padding 区（832px）18.5% 有内容，但游戏内不可见 → 强制 0
#
# 为什么必须两级而不是每面独立抽：独立抽会让「overlay 区域至少一面有内容」
# 的概率冲到 ~0.96，而真实只有 ~0.55~0.90 —— 真实作者「画了外套就整件画」，
# 面与面强相关。两级模型同时复现盒级与面级覆盖率。

_OVERLAY_BOXES: tuple[str, ...] = ("hat", "body_ov", "rarm_ov", "rleg_ov",
                                   "larm_ov", "lleg_ov")
OVERLAY_BOXES = _OVERLAY_BOXES

#: 兜底频率（models/alpha_rates.json 不存在时用；来自 3000 张实测）
DEFAULT_ALPHA_RATES: dict = {
    "q_box": {"hat": 0.9017, "body_ov": 0.5793, "rarm_ov": 0.5833,
              "rleg_ov": 0.5623, "larm_ov": 0.5773, "lleg_ov": 0.5650},
    "faces": {
        "hat.top":      {"full": 0.14, "partial": 0.59, "fill": 0.34},
        "hat.bottom":   {"full": 0.06, "partial": 0.48, "fill": 0.33},
        "hat.front":    {"full": 0.04, "partial": 0.88, "fill": 0.44},
        "hat.left":     {"full": 0.06, "partial": 0.85, "fill": 0.37},
        "hat.right":    {"full": 0.06, "partial": 0.85, "fill": 0.37},
        "hat.back":     {"full": 0.13, "partial": 0.69, "fill": 0.43},
        "body_ov.top":    {"full": 0.06, "partial": 0.31, "fill": 0.36},
        "body_ov.bottom": {"full": 0.08, "partial": 0.15, "fill": 0.27},
        "body_ov.front":  {"full": 0.06, "partial": 0.85, "fill": 0.28},
        "body_ov.left":   {"full": 0.11, "partial": 0.36, "fill": 0.23},
        "body_ov.right":  {"full": 0.10, "partial": 0.36, "fill": 0.23},
        "body_ov.back":   {"full": 0.12, "partial": 0.66, "fill": 0.31},
        "rarm_ov.top":    {"full": 0.13, "partial": 0.21, "fill": 0.42},
        "rarm_ov.bottom": {"full": 0.06, "partial": 0.15, "fill": 0.45},
        "rarm_ov.front":  {"full": 0.04, "partial": 0.92, "fill": 0.28},
        "rarm_ov.left":   {"full": 0.04, "partial": 0.67, "fill": 0.28},
        "rarm_ov.right":  {"full": 0.04, "partial": 0.91, "fill": 0.32},
        "rarm_ov.back":   {"full": 0.04, "partial": 0.92, "fill": 0.25},
        "larm_ov.top":    {"full": 0.14, "partial": 0.20, "fill": 0.43},
        "larm_ov.bottom": {"full": 0.06, "partial": 0.15, "fill": 0.43},
        "larm_ov.front":  {"full": 0.04, "partial": 0.92, "fill": 0.30},
        "larm_ov.left":   {"full": 0.04, "partial": 0.91, "fill": 0.33},
        "larm_ov.right":  {"full": 0.04, "partial": 0.61, "fill": 0.29},
        "larm_ov.back":   {"full": 0.04, "partial": 0.91, "fill": 0.24},
        "rleg_ov.top":    {"full": 0.09, "partial": 0.13, "fill": 0.35},
        "rleg_ov.bottom": {"full": 0.30, "partial": 0.26, "fill": 0.53},
        "rleg_ov.front":  {"full": 0.07, "partial": 0.90, "fill": 0.30},
        "rleg_ov.left":   {"full": 0.06, "partial": 0.59, "fill": 0.23},
        "rleg_ov.right":  {"full": 0.07, "partial": 0.85, "fill": 0.31},
        "rleg_ov.back":   {"full": 0.07, "partial": 0.85, "fill": 0.30},
        "lleg_ov.top":    {"full": 0.09, "partial": 0.13, "fill": 0.35},
        "lleg_ov.bottom": {"full": 0.30, "partial": 0.26, "fill": 0.53},
        "lleg_ov.front":  {"full": 0.07, "partial": 0.89, "fill": 0.30},
        "lleg_ov.left":   {"full": 0.07, "partial": 0.84, "fill": 0.31},
        "lleg_ov.right":  {"full": 0.06, "partial": 0.59, "fill": 0.23},
        "lleg_ov.back":   {"full": 0.07, "partial": 0.84, "fill": 0.29},
    },
}


def load_alpha_rates(path: str | None = None) -> dict:
    """读实测的 alpha 三态频率；缺文件时回落到内置常数。"""
    p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "models", "alpha_rates.json")
    p = os.path.normpath(p)
    if os.path.isfile(p):
        import json
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("q_box") and d.get("faces"):
            return {"q_box": d["q_box"], "faces": d["faces"]}
    return DEFAULT_ALPHA_RATES


def sample_alpha_templates(batch: int, rates: dict | None = None,
                           rng: np.random.Generator | None = None,
                           box_paint: dict | None = None) -> np.ndarray:
    """按实测三态频率抽 ``(batch, 4096)`` 的展平 alpha 掩码（0/1 float）。

    每个面三种状态（频率都是真实数据实测的，见 :func:`load_alpha_rates`）：

    * ``full``    —— 整面全 1
    * ``partial`` —— 整面按 ``fill`` 的逐像素伯努利填充（**真实皮肤的面经常
      只画一部分**，全有全无模型会把逐像素可见率高估 4 倍：body_ov.bottom
      实测 0.067，全有全无模型给 0.29）
    * 空 —— 全 0

    盒级先抽「画不画」（hat 0.90 / body_ov 0.58 / …），不画则整盒为 0；
    base 层没有盒级开关。padding 区（不属于任何面）恒 0。

    ``box_paint``：``{盒名: (batch,) 的 0/1}``。**给定时该盒的画不画由它决定，
    不再随机抽** —— 这是「条件说画什么，模板就必须露出什么」的接口。
    不做这一步，``cond`` 里说「要袖子」而模板没露 arm.overlay，
    模型拿到的就是一个自相矛盾的监督信号（这是第二层学不会的直接原因之一）。
    构造它用 ``labelset.ov_bits_to_box_paint``。
    """
    rates = rates or DEFAULT_ALPHA_RATES
    rng = rng or np.random.default_rng()
    q_box = rates["q_box"]
    faces = rates["faces"]

    alpha = np.zeros((batch, SIZE * SIZE), dtype=np.float32)
    fmap = face_index()

    def _idx(y0, y1, x0, x1):
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        return (yy * SIZE + xx).reshape(-1)

    for nm, (y0, y1, x0, x1) in fmap.items():
        box = nm.split(".", 1)[0]
        st = faces.get(nm)
        if st is None:                       # base 层缺参数时按全满处理
            if box not in OVERLAY_BOXES:
                alpha[:, _idx(y0, y1, x0, x1)] = 1.0
            continue
        idx = _idx(y0, y1, x0, x1)
        u = rng.random(batch)
        p_full, p_part = st["full"], st["partial"]
        full_m = u < p_full
        part_m = (u >= p_full) & (u < p_full + p_part)
        fill = (rng.random((batch, idx.size)) < st["fill"]).astype(np.float32)
        face_a = np.where(full_m[:, None], 1.0,
                          np.where(part_m[:, None], fill, 0.0))
        if box in OVERLAY_BOXES:
            forced = (box_paint or {}).get(box)
            if forced is not None:
                paint_box = np.asarray(forced, dtype=np.float32).reshape(-1, 1)
            else:
                paint_box = (rng.random(batch) < q_box.get(box, 0.5))[:, None]
            face_a = face_a * paint_box
        alpha[:, idx] = face_a
    return alpha


def alpha_face_index() -> dict[str, np.ndarray]:
    """``{面名: 像素下标}``，供模板构建与覆盖率校验用。"""
    return {nm: _flat_idx(y0, y1, x0, x1) for nm, y0, y1, x0, x1 in FACE_MASKS}


def _flat_idx(y0, y1, x0, x1) -> np.ndarray:
    yy = np.arange(y0, y1)[:, None]
    xx = np.arange(x0, x1)[None, :]
    return (yy * SIZE + xx).reshape(-1)


# ======================================================================
# 检索式 mask 采样 —— 用经验分布取代「逐像素 i.i.d.」的参数化近似
# ======================================================================
# 为什么合成模板不能用了
# ----------------------
# `sample_alpha_templates` 把每个面的**覆盖率**复现得很准，但空间结构为零。
# 实测（scripts/_probe/_struct_audit.py，val 13200 张）:
#
#     面内邻接一致率       真实 0.9568 | 合成 0.8958 | 纯 i.i.d. 理论 0.9182
#     hat 盒连通块中位像素  真实 11.5   | 合成 1.5
#     到最近真实 mask 距离  真实 1.0×   | 合成 3.09×
#
# 后果有两层：
#   1) **训练/推理 conditioning 错配** —— 训练喂真实连通的帽子，推理喂椒盐点。
#      模型在推理时见到的是训练里从没出现过的输入分布。
#   2) **导出的第二层是「穿孔」的** —— `30_generate.py` 把模板直接当导出图的
#      alpha，于是帽子/外套变成随机点阵，而不是连通的衣物。
#
# 修法不是调参，是换「分布的表达方式」：mask 直接**从真实数据检索**（经验分布），
# 训练与推理两侧恒等分布，空间结构 100% 真实。库由
# `scripts/25_build_mask_bank.py` 构建（默认 4 万张，2.8MB）。

_MASK_BANK: dict | None = None


def load_mask_bank(path: str | None = None) -> dict | None:
    """读 ``models/mask_bank.npz``，解包成 ``{"masks": (N,4096) uint8, "bits": (N,4)}``。

    缺文件时返回 ``None``（调用方应回落到 :func:`sample_alpha_templates`
    并在日志里说明），**不抛异常** —— 训练/推理链不该因为一个可选资产缺失就断。
    """
    global _MASK_BANK
    if _MASK_BANK is not None:
        return _MASK_BANK or None
    p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "models", "mask_bank.npz")
    p = os.path.normpath(p)
    if not os.path.isfile(p):
        _MASK_BANK = {}
        return None
    z = np.load(p)
    masks = np.unpackbits(z["masks"], axis=1)[:, :SIZE * SIZE].astype(np.uint8)
    _MASK_BANK = {"masks": masks, "bits": np.asarray(z["bits"]).astype(np.int8)}
    return _MASK_BANK


def sample_alpha_bank(batch: int, ov_bits: np.ndarray | None = None,
                      rng: np.random.Generator | None = None,
                      bank_path: str | None = None) -> np.ndarray | None:
    """从真实 mask 库里**检索** ``batch`` 张 ``(batch, 4096)`` 的 0/1 mask。

    ``ov_bits``：``(batch, 4)`` 的 0/1（``labelset.OV_BIT_NAMES`` 顺序）。
    给定时只在**部位位完全一致**的真实 mask 里抽 —— 这样「条件说要有袖子」
    与「模板真的露出袖子」仍然对齐（保留了 `box_paint` 那条约束），
    但露出的形状是**真实作者画的袖子**，不是伯努利点阵。

    返回 ``None`` 表示库不可用，调用方自行回落。**返回 0/1 float32。**
    """
    bank = load_mask_bank(bank_path)
    if not bank:
        return None
    rng = rng or np.random.default_rng()
    masks, bits = bank["masks"], bank["bits"]
    n = masks.shape[0]
    if ov_bits is None:
        idx = rng.integers(0, n, batch)
    else:
        want = np.asarray(ov_bits).astype(np.int8).reshape(batch, -1)[:, :4]
        keys = bits[:, 0].astype(np.int32) * 8 + bits[:, 1].astype(np.int32) * 4 \
            + bits[:, 2].astype(np.int32) * 2 + bits[:, 3].astype(np.int32)
        buckets: dict[int, np.ndarray] = {}
        for k in np.unique(keys):
            buckets[int(k)] = np.nonzero(keys == k)[0]
        out = np.empty((batch, SIZE * SIZE), dtype=np.float32)
        for i in range(batch):
            k = int(want[i, 0]) * 8 + int(want[i, 1]) * 4 + int(want[i, 2]) * 2 + int(want[i, 3])
            pool = buckets.get(k)
            if pool is None or pool.size == 0:      # 该组合在库里没有 -> 退到全库
                pool = np.arange(n)
            out[i] = masks[pool[rng.integers(0, pool.size)]]
        return out
    return masks[idx].astype(np.float32)
