"""losses.py — 针对「小分辨率 + 高结构」的 Minecraft 皮肤自研损失项。

设计原则：**分布匹配，不是盲目最小化**
------------------------------------
一条朴素的平滑损失（比如全图 TV 直接压到 0）会把像素画里本来就该有的硬边也抹掉，
最后得到一张模糊的色块。所以这里每一项都先**测出真实数据的目标值**，
再把生成值往那个数上拉（``|stat_gen − stat_real|`` 或 ``relu(stat_gen − stat_real)``）。

真实目标值由 ``_probe/atlas_stats.py`` 在 13 万张清洗后的皮肤上实测，
写进 ``logs/atlas_stats.json``，训练时直接读。

四项损失
--------
======================  ==========================================================
名称                     作用
======================  ==========================================================
``face_flatness``        面内分片常数：一个小立方体面内部应该是大色块
``edge_bimodal``         **边界锐化**：相邻像素差要双峰（要么 0、要么 ≥τ），
                         不允许「到处都是小差异」——这正是 v1 输出 2500 色噪声的病根
``limb_mirror``          左右肢体的对应面互为镜像
``palette_concentration``调色板集中：软颜色直方图的熵要对齐真实值
``r1_penalty``           判别器 R1 正则，替代「压低 D 学习率」这种土办法
======================  ==========================================================

配套的**结构性**手段（比损失更硬）见 ``models.py::PaletteHead``：
把 RGB 输出改成「K 个调色板色的软加权和」，图像最多只能有 K 种颜色，
从架构上让「2500 色噪声」不可能出现。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

#: 面内相邻像素对 / 镜像面配对，由 skinatlas 提供
_ATLAS_CACHE: dict = {}

#: alpha 可见判据（在 [0,1] 尺度上）。
#:
#: 实测真实皮肤 99.85% 的像素 alpha 严格等于 0 或 255，只有 **0.068%**
#: 落在 ``1..127`` 这个「``>0`` 与 ``≥128`` 会给出不同答案」的区间。
#: 影响很小，但**标定脚本和损失必须用同一个阈值**，否则目标值对应的
#: 是一个不同的可见像素集合（上一轮的 hard/soft 计数错位就是这么来的）。
#: numpy 侧等价写法：``alpha_uint8 >= 128``。
ALPHA_VIS_THRESHOLD = 0.5


def _atlas(device, dtype=torch.long):
    """惰性构造并缓存索引张量，避免每次 forward 重算。

    缓存内容：

    * ``edges``   —— 全局面内邻接对 ``(2, 5504)``
    * ``regions`` —— ``{分区方案: {区域名: (2, E_r)}}``，见 ``skinatlas.region_edges``
    * ``limb``    —— 左右肢体镜像配对 ``[(右肢面下标, 左肢面水平翻转后下标), ...]``
    """
    key = str(device)
    if key not in _ATLAS_CACHE:
        from skinatlas import (DEFAULT_SCHEME, SCHEMES, in_face_edges,
                               limb_mirror_index, region_edges)

        def _t(a):
            return torch.as_tensor(a, device=device).long()

        e = _t(in_face_edges())
        regions = {s: {n: _t(v) for n, v in region_edges(s).items()} for s in SCHEMES}

        # 镜像配对：**必须**用 skinatlas.limb_mirror_index()，不要在这里重写一遍
        # 水平翻转公式。矩形端点是开区间，自己反推端点极易差 1 个像素，
        # 表现为「对称度差 20%」而完全看不出原因（已经栽过一次）。
        pairs = [(_t(ir), _t(im), part, name)
                 for part, name, ir, im in limb_mirror_index()]
        _ATLAS_CACHE[key] = {"edges": e, "regions": regions, "limb": pairs,
                             "scheme": DEFAULT_SCHEME}
    return _ATLAS_CACHE[key]


def _flat(x: torch.Tensor) -> torch.Tensor:
    """(B,C,64,64) -> (B,C,4096)"""
    return x.reshape(x.shape[0], x.shape[1], -1)


def _pair_dist(rgb: torch.Tensor, alpha: torch.Tensor, idx: torch.Tensor):
    """返回面内相邻像素对的色差 ``d`` (B,E) 与可见掩码 ``m`` (B,E)。

    **掩码不能省**：透明像素的 RGB 是 0，把它算进去会让「不透明↔透明」
    这种接缝对产生巨大的假色差。numpy 参考实现（``struct_metrics.py``）只在
    **两端都可见**的像素对上统计，torch 版必须完全一致，否则两边对不上——
    实测漏掉掩码时 limb_mirror 会从 0.126 虚高到 0.233。
    """
    a = _flat(rgb)                       # (B,3,4096)
    al = _flat(alpha)[:, 0] > ALPHA_VIS_THRESHOLD   # (B,1,4096) -> (B,4096)
    d = (a[:, :, idx[0]] - a[:, :, idx[1]]).abs().sum(dim=1)      # (B,E)
    m = (al[:, idx[0]] & al[:, idx[1]])                            # (B,E)
    return d, m


def _masked_mean(x: torch.Tensor, m: torch.Tensor):
    """按样本求带掩码的均值。

    返回 ``(per_sample (B,), valid (B,))``。**调用方必须只在 valid 上取平均**：
    掩码全空的样本（该样本在这个面上一个可见像素对都没有）numpy 参考实现是
    **直接跳过**的，如果把它的 0 也计入平均，会把整体系统性拉低。
    """
    n = m.sum(dim=1)
    return (x * m).sum(dim=1) / n.clamp(min=1), (n > 0)


def _batch_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """在「有效样本」上取标量均值；一个都没有时返回 0。"""
    per, valid = _masked_mean(x, m)
    if not bool(valid.any()):
        return torch.zeros((), device=x.device)
    return per[valid].mean()


# ======================================================================
# 0) 近同色占比损失（主项）—— 唯一被实测证明有区分度的平坦度指标
# ======================================================================

def flat_fraction(rgb: torch.Tensor, alpha: torch.Tensor, tau: float = 0.25,
                  eps: float = 0.05, delta: float = 0.02,
                  target: float | None = 0.562) -> torch.Tensor:
    """面内相邻像素中「几乎完全相同」的占比，对齐真实值。

    ``target=None`` 时返回**原始值** ``F``（不取绝对差），供训练时打印进度用。

    数学形式（用 sigmoid 把硬计数软化成可导）::

        s_pq = σ( (ε − d_pq) / δ )        # d_pq = ||I_p − I_q||₁
        F    = (1/|E|) Σ_{(p,q)∈E} s_pq
        L    = |F − F_real|

    实测（``logs/atlas_stats.json``）：

        ==================  =========  =========
        集合                 F(ε=0.05)  邻接色差中位数
        ==================  =========  =========
        真实 13 万张皮肤       **0.562**   0.0196
        dcgan_v1 生成          0.089     0.1725
        ==================  =========  =========

    也就是真实皮肤**一半以上的相邻像素是完全相同的**（像素画的大色块），
    而生成结果只有 8.9%，典型色差是真实的 8.8 倍 —— 「满屏微差噪声」。

    **为什么它比 ``face_flatness``（平均色差）更好当主项**：
    平均色差是个和，优化它会把大色差一起往下拉（过平滑）；
    而「近同色占比」只关心有多少对是平的，不惩罚那些本来就该大的色差，
    所以能一边把噪声压成平块、一边保住硬边。实测它也是唯一把
    真实/生成区分开 6 倍以上的指标。
    """
    d, m = _pair_dist(rgb, alpha, _atlas(rgb.device)["edges"])
    s = torch.sigmoid((eps - d) / delta)
    val = _batch_mean(s, m)
    if target is None:
        return val
    return (val - target).abs()


def unique_color_count(rgb: torch.Tensor, alpha: torch.Tensor) -> float:
    """一批图的**平均唯一色数**（不可导，只用于监控）。

    这是「输出是不是满熵噪声」最直接的判据：真实皮肤 142（中位 73），
    dcgan_v1 是 2551（4096 像素的上限附近）。PaletteHead 生效后应当 ≤ K。
    """
    a = rgb.detach().float().cpu().numpy()
    al = alpha.detach().float().cpu().numpy()
    if a.ndim == 4:                                  # (B,3,H,W)
        a = a.transpose(0, 2, 3, 1)
    if al.ndim == 4:
        al = al[:, 0]
    out = []
    for i in range(a.shape[0]):
        v = (a[i][al[i] > ALPHA_VIS_THRESHOLD] * 255).round().astype(np.uint8)
        out.append(int(np.unique(v.reshape(-1, 3), axis=0).shape[0]) if v.size else 0)
    return float(np.mean(out)) if out else 0.0


# ======================================================================
# 1) UV 分区平坦损失（面内分片常数）
# ======================================================================

def face_flatness(rgb: torch.Tensor, alpha: torch.Tensor,
                  target: float | None = None) -> torch.Tensor:
    """面内相邻像素的平均 L1 色差。

    ``rgb`` (B,3,64,64) ∈[0,1]，``alpha`` (B,1,64,64)。

    数学形式::

        L_TV = (1/|E|) Σ_{(p,q)∈E} ||I_p − I_q||₁ ,   E = 同一面内部的相邻对

    ``E`` **只取面内**：面与面之间的接缝在真实皮肤里本来就可能突变。

    为什么用 ``|L_TV − target|`` 而不是 ``L_TV``：
    直接最小化会连面内合理的色阶过渡一起压平，最终整张脸变成一块色。
    实测真实皮肤的 ``L_TV = 0.2213``（**不是 0**），所以要对齐到那个数。
    它作为辅助项（权重低于 ``flat_fraction``），负责兜住「整体碎的粒度」。
    """
    d, m = _pair_dist(rgb, alpha, _atlas(rgb.device)["edges"])
    if target is None:
        return _batch_mean(d, m)
    return (_batch_mean(d, m) - target).abs()


# ======================================================================
# 2) 边界锐化损失（梯度双峰）
# ======================================================================

def edge_bimodal(rgb: torch.Tensor, alpha: torch.Tensor, tau: float = 0.25,
                 target: float | None = None) -> torch.Tensor:
    """逼相邻像素色差取双峰分布：要么≈0（同色块），要么≥τ（硬边）。

    令 ``d = ||I_p − I_q||₁``，``u = min(d, τ)``，则::

        g(u) = u·(τ − u) / τ²

        g(0) = g(τ) = 0 ，在 u = τ/2 取最大 0.25

    所以**最小化** ``g`` 会让 ``d`` 从 τ/2 向两端跑：
    比 τ/2 小的被推向 0，比 τ/2 大的被推向 τ。它是「双峰」而不是「趋零」——
    这正是它和普通 TV 平滑的本质区别。

    病因对得上：v1 生成的皮肤唯一色数 2551（真实仅 149），
    也就是**每一个相邻像素对都有一个小而不为零的差异**，恰好落在 τ/2 附近。

    ``tau`` 取真实皮肤「硬边」的典型色差量级（0-255 尺度约 60，归一化后 0.25），
    由 ``_probe/atlas_stats.py`` 实测标定。
    """
    d, m = _pair_dist(rgb, alpha, _atlas(rgb.device)["edges"])
    u = d.clamp(max=tau)
    g = u * (tau - u) / (tau * tau)
    val = _batch_mean(g, m)
    if target is None:
        return val
    return (val - target).abs()


# ======================================================================
# 3) 左右肢体镜像损失
# ======================================================================

def limb_mirror(rgb: torch.Tensor, alpha: torch.Tensor,
                target: float | None = None) -> torch.Tensor:
    """右肢的面与左肢的对应面（水平翻转后）应该几乎一样。

    数学形式::

        L_sym = (1/|P|) Σ_{(f,g)∈P}  mean_{p∈vis} || I_p − I_{flip(p)} ||₁

    ``P`` = {右臂/左臂, 右腿/左腿} × {front, back, top, bottom}。

    ⚠️ 不能强行把它压到 0：实测真实皮肤 ``L_sym = 0.1261``（中位 0.0584），
    说明作者本来就经常把左右肢画得不一样（护腕、单边纹身）。
    强行对称等于引入错误先验 → 所以用 ``|L_sym − L_sym_real|`` 对齐，且权重小。
    """
    al = _flat(alpha)[:, 0] > ALPHA_VIS_THRESHOLD   # (B,4096)
    a = _flat(rgb)
    total, n = 0.0, 0
    for ir, im, _part, _pair in _atlas(rgb.device)["limb"]:
        m = al[:, ir] & al[:, im]
        d = (a[:, :, ir] - a[:, :, im]).abs().sum(dim=1)
        total = total + _batch_mean(d, m)
        n += 1
    if n == 0:
        return torch.zeros((), device=rgb.device)
    val = total / n
    if target is None:
        return val
    return (val - target).abs()


def limb_mirror_per_pair(rgb: torch.Tensor, alpha: torch.Tensor) -> dict[str, torch.Tensor]:
    """逐「面配对」返回镜像色差，键形如 ``'rarm.front~larm.front'``。

    有它才能定位「是哪一对面不对称」——只看 ``arm`` 一个汇总数，
    分不清是前胸还是背面对不上。
    """
    al = _flat(alpha)[:, 0] > ALPHA_VIS_THRESHOLD
    a = _flat(rgb)
    out: dict[str, torch.Tensor] = {}
    for ir, im, _part, pair in _atlas(rgb.device)["limb"]:
        m = al[:, ir] & al[:, im]
        d = (a[:, :, ir] - a[:, :, im]).abs().sum(dim=1)
        out[pair] = _batch_mean(d, m)
    return out


def limb_mirror_by_part(rgb: torch.Tensor, alpha: torch.Tensor) -> dict[str, torch.Tensor]:
    """按部位分开量左右镜像色差：``{'arm': …, 'leg': …}``。

    分开量的理由：手臂和腿的对称程度在真实皮肤里**并不相同**——
    护腕/单边袖套让手臂比腿更常被故意画得不对称。混在一起平均会把
    两种不同的先验压成一个人为的中间值，标定出来的目标值对两边都不准。
    """
    pairs = limb_mirror_per_pair(rgb, alpha)
    acc: dict[str, list] = {}
    for _ir, _im, part, pair in _atlas(rgb.device)["limb"]:
        acc.setdefault(part, []).append(pairs[pair])
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def palette_usage_guard(probs: torch.Tensor, min_entropy: float = 3.47) -> torch.Tensor:
    """调色板「使用熵」下限护栏：``relu(H_min − H(usage))``，不可导以外全可导。

    ``probs`` 是 :class:`models.PaletteHead` 内部的软分配 ``(B,K,H,W)``；
    ``usage = probs.mean(dim=(0,2,3))`` 是 K 个调色板色的平均使用率，
    ``H(usage) = −Σ_k usage_k log usage_k``。

    **为什么必须有这一项**：硬量化（STE + argmax）配上「奖励平坦」的损失，
    会收敛到「只用极少数几个颜色」的退化解 —— 因为一张纯色图天然满足
    「相邻像素完全相同」。实测（3 epoch，K=96）：``flat_frac`` 直接跳到
    **0.53**（真实 0.5285，几乎完美），但 ``unique_colors`` 掉到 **3.9~9.0**
    （真实中位 73）。**平坦度对了，但那是靠「把所有地方涂成同一色」换来的。**

    ⚠️ 这一项**不是**从数据标定出来的分布匹配项，而是一条**下限护栏**：
    它只在 ``H < H_min`` 时生效，不去规定使用分布应该长什么样。
    ``H_min = ln(32) ≈ 3.47`` 的意思是「至少用出 ~32 种有效颜色」，
    这是设计选择（真实皮肤唯一色数中位 73，取 32 是留了余量的下限），
    不是实测值 —— 写清楚免得以后被误当成标定结果。
    """
    usage = probs.mean(dim=(0, 2, 3))
    ent = -(usage * torch.log(usage + 1e-8)).sum()
    return F.relu(min_entropy - ent)


# ======================================================================
# 6) 区域化（按部件 × 图层）—— 把「全局平均」换成「逐区域对齐」
# ======================================================================
# 为什么必须分区：把 72 个面（5504 个邻接对）一起平均会**互相抵消**。
# 真实皮肤的腿常常接近纯色块、脸却有眼睛/嘴这种硬细节；躯干 base 层多为
# 大色块而 overlay 层经常整片透明。全局数偏高时，可能是「腿是正确的细碎、
# 脸糊了」，也可能是「脸对了、腿碎了」——一个标量分不出来，也就无从优化。
#
# 分区方案见 ``skinatlas.region_label``（默认 ``detail``）:
#   face / head.base / head.overlay / body.base / body.overlay /
#   arm.base / arm.overlay / leg.base / leg.overlay
# 九个区域**互斥且全覆盖**，所以逐区域对齐不会重复计数。

def _m_flat(d, tau, eps, delta):
    return d


def _m_flat_frac(d, tau, eps, delta):
    return torch.sigmoid((eps - d) / delta)


def _m_edge(d, tau, eps, delta):
    u = d.clamp(max=tau)
    return u * (tau - u) / (tau * tau)


#: 可区域化的三个配对统计量（与全局版一一对应，保证同一个估计量）
PAIR_METRICS = {
    "flat": _m_flat,
    "flat_frac": _m_flat_frac,
    "edge": _m_edge,
}

#: 区域权重 ω_r —— **不按像素数加权**，否则 64px 的脸只占 2% 会被淹没。
#: 脸给 1.6 是因为它语义上最关键（64×64 里只有 8×8 是五官）。
REGION_WEIGHTS = {
    "face": 1.6,
    "head.base": 1.0, "head.overlay": 0.5,
    "body.base": 1.0, "body.overlay": 0.5,
    "arm.base": 0.8, "arm.overlay": 0.4,
    "leg.base": 0.8, "leg.overlay": 0.4,
}

#: 指标权重 λ_m —— 与全局版一致（``flat_frac`` 是唯一有 6 倍区分度的主项）
METRIC_WEIGHTS = {"flat_frac": 2.0, "flat": 0.5, "edge": 0.6}


def region_pair_stats(rgb: torch.Tensor, alpha: torch.Tensor, scheme: str = "detail",
                      tau: float = 0.4275, eps: float = 0.05, delta: float = 0.02,
                      taus: dict[str, float] | None = None,
                      only: set[str] | None = None) -> dict[str, dict[str, torch.Tensor]]:
    """逐区域算出三个配对统计量，**返回的每一项都可导**。

    返回 ``{区域名: {'flat': …, 'flat_frac': …, 'edge': …}}``。

    估计量与全局版**完全相同**（``_batch_mean``：样本内先按有效对取均值，
    再跨样本平均）—— 这条必须守住，否则标定出来的目标值和损失优化的不是同一个量。

    ``taus`` 可为每个区域单独指定硬边量级 τ（**只有 ``edge`` 用到它**）。
    必须允许分区给 τ：实测脸的 ``τ_local = 1.251`` 是全局 ``0.4275`` 的 3 倍，
    而 ``edge`` 项里 ``u = min(d, τ)`` —— 拿全局 τ 去量脸，脸部的色差几乎全部
    被削到 τ 上饱和，``g(u)`` 恒为 0，这一项在脸上等于失效。
    不传 ``taus`` 时全区域共用 ``tau``。
    """
    groups = _atlas(rgb.device)["regions"].get(scheme) or {}
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, idx in groups.items():
        if only and name not in only:
            continue
        d, m = _pair_dist(rgb, alpha, idx)
        tr = (taus or {}).get(name, tau)
        out[name] = {k: _batch_mean(f(d, tr, eps, delta), m)
                     for k, f in PAIR_METRICS.items()}
    return out


def region_struct_loss(rgb: torch.Tensor, alpha: torch.Tensor,
                       region_targets: dict[str, dict[str, float]],
                       scheme: str = "detail",
                       region_weights: dict[str, float] | None = None,
                       metric_weights: dict[str, float] | None = None,
                       tau: float = 0.4275, eps: float = 0.05, delta: float = 0.02,
                       ) -> tuple[torch.Tensor, dict[str, float]]:
    """区域化结构损失。

    数学形式::

        L_region = Σ_r ω_r Σ_m λ_m · |S_m(I; r) − T_m(r)|
                   ────────────────────────────────────────
                          Σ_r ω_r Σ_m λ_m   （只累加有标定的项）

    * ``S_m(I; r)`` —— 区域 ``r`` 内部的第 ``m`` 个统计量（``PAIR_METRICS``）
    * ``T_m(r)``    —— 该区域真实数据上的实测值（``logs/atlas_stats.json``）
    * ``ω_r``       —— ``REGION_WEIGHTS``
    * ``λ_m``       —— ``METRIC_WEIGHTS``

    **没有标定的区域会被跳过，不会用全局值兜底**：拿全局值当某区域的
    目标是引入错误先验（脸的目标和腿的目标本来就不一样），宁可该项不生效。

    返回 ``(损失, 逐项数值)``；逐项数值按 ``"{区域}.{指标}"`` 命名，直接进
    metrics.jsonl 就能看到「是脸在退化还是腿在退化」。
    """
    rw = {**REGION_WEIGHTS, **(region_weights or {})}
    mw = {**METRIC_WEIGHTS, **(metric_weights or {})}
    # 每个区域用自己的 τ（标定时实测的局部硬边量级）；缺省回落全局 τ
    taus = {r: (t.get("tau_used") or tau)
            for r, t in (region_targets or {}).items() if isinstance(t, dict)}
    stats = region_pair_stats(rgb, alpha, scheme, tau, eps, delta, taus=taus,
                              only={r for r, w in rw.items() if w > 0})
    num: torch.Tensor | None = None
    den = 0.0
    detail: dict[str, float] = {}
    for r, ms in stats.items():
        tm = (region_targets or {}).get(r)
        if not tm:
            continue
        wr = rw.get(r, 0.0)
        if wr <= 0:
            continue
        for mname, val in ms.items():
            wm = mw.get(mname, 0.0)
            t = tm.get(mname)
            if wm <= 0 or t is None:
                continue
            l = (val - t).abs()
            term = wr * wm * l
            num = term if num is None else num + term
            den += wr * wm
            detail[f"{r}.{mname}"] = float(l.detach())
    if num is None or den == 0:
        return torch.zeros((), device=rgb.device), detail
    return num / den, detail


# ======================================================================
# 4) 调色板集中损失
# ======================================================================

def soft_color_hist(rgb: torch.Tensor, alpha: torch.Tensor,
                    palette: torch.Tensor, sigma: float = 0.08) -> torch.Tensor:
    """把可见像素软分配到 ``palette``（K,3）上，返回 (K,) 的占比直方图。

    ``sigma`` 控制软分配宽度；越小越接近硬量化。
    """
    a = _flat(rgb)                                          # (B,3,N)
    al = _flat(alpha) > ALPHA_VIS_THRESHOLD                  # (B,N)
    x = a.permute(0, 2, 1).reshape(-1, 3)                   # (B*N,3)
    v = al.reshape(-1)                                      # (B*N,)
    x = x[v]
    if x.numel() == 0:
        return torch.zeros(palette.shape[0], device=rgb.device)
    dist2 = ((x[:, None, :] - palette[None, :, :]) ** 2).sum(dim=2)   # (M,K)
    w = torch.softmax(-dist2 / (2 * sigma * sigma), dim=1)
    return w.mean(dim=0)


def palette_concentration(rgb: torch.Tensor, alpha: torch.Tensor,
                          palette: torch.Tensor, sigma: float = 0.08,
                          h_target: float | None = None) -> torch.Tensor:
    """软颜色直方图的熵要与真实值对齐。

    ``H(h) = −Σ_k h_k log h_k``。

    ⚠️ **实测结论：这一项在当前形式下没有区分度，不要当主项用。**
    同一套指标（K=96, σ=0.08）量出来：

        真实 13 万张    H = 4.2025
        dcgan_v1 生成   H = 4.0306      ← 噪声的熵反而**更低**

    原因：σ=0.08 的软分配对「远离所有真实调色板中心」的噪声色会摊到多个
    相邻中心上，反而把直方图抹平；而真实皮肤虽然只用 142 种色，
    但这 142 种在 96 个中心上的使用分布本身就比较均匀。
    两者熵接近纯属巧合，方向甚至是反的。

    → 想要「颜色不杂乱」，**必须靠结构手段**（``models.PaletteHead``：
    图像最多只能有 K 种颜色），而不是靠熵损失去劝。
    这个函数保留下来只作为监控量，权重设为 0 或极小。
    """
    h = soft_color_hist(rgb, alpha, palette, sigma)
    ent = -(h * torch.log(h + 1e-8)).sum()
    if h_target is None:
        return ent
    return (ent - h_target).abs()


# ======================================================================
# 5) 判别器 R1 正则
# ======================================================================

def r1_penalty(d_out: torch.Tensor, x_real: torch.Tensor,
               gamma: float = 10.0, lazy_interval: int = 1) -> torch.Tensor:
    """R1 梯度惩罚：``γ/2 · E[ ||∇_x D(x_real)||² ]``。

    为什么需要它：v1 训练里 D 单方面压制 G（d_loss 0.45→0.21、
    g_loss 2.4→6.5），当时的处理是把 D 学习率压到 1/4 —— 那是土办法，
    等于在削弱判别器的同时牺牲了它的引导能力。

    ⚠️ **``lazy_interval`` 不是可有可无的参数，漏了会把惩罚强度放大 N 倍**。
    惰性正则只在每 N 步执行一次，为了让**平均**惩罚强度与全量执行时一致，
    必须把 γ 除以 N（StyleGAN2 里的 ``lazy_gamma = gamma / r1_interval``）。
    实测漏掉这一步时：``γ=10``、每 16 步执行 → 等效强度 16 倍，
    R1 项在第 50 步就达到 1.33（BCE 才 0.73），把 D 一路压成常数函数
    （``‖∇D‖`` 从 0.52 掉到 **0.023**）。

    为什么 D 一旦被压平就再也好不了：``d_loss = ln2`` 且 ``R1 = 0`` 是
    **稳定的驻点** —— 常数 D 让 BCE 在 ``logit=0`` 处的梯度恰好为 0
    （``dL/dlogit = 0.5·σ(0) − 0.5·(1−σ(0)) = 0``），R1 也为 0。
    掉进这个「判别器完全无判别力但损失看起来正常」的状态后，
    它**不会自己爬出来**，只能靠随机噪声，而 Adam 不会。
    所以监控 ``d_grad_rms = √(2·r1/γ)`` 比看 ``d_loss`` 有用得多。

    ⚠️ **R1 与谱归一化不能共存**：R1 需要二阶求导（``create_graph=True``），
    而谱归一化的权重靠幂迭代算出来，二阶图会穿过那个反馈回路并放大到数值崩坏
    （实测 ``r1`` 从 0.002 雪崩到 **8056**）。用 R1 时 D 里不要加 ``spectral_norm``。
    """
    grad = torch.autograd.grad(outputs=d_out.sum(), inputs=x_real,
                               create_graph=True, retain_graph=True)[0]
    pen = gamma * 0.5 * grad.reshape(grad.shape[0], -1).pow(2).sum(dim=1).mean()
    if lazy_interval > 1:
        pen = pen / float(lazy_interval)
    return pen


# ======================================================================
# 组合
# ======================================================================

#: 建议起始权重，全部由 ``logs/atlas_stats.json`` 的实测值标定。
#: 升温策略（``ramp``）比权重本身更重要：结构项一开始给满会直接塌成纯色块。
DEFAULT_WEIGHTS = {
    "w_alpha": 0.15,        # alpha 二值化 hinge（沿用，已证明有效）
    "w_region": 2.0,        # **主项**：区域化结构损失（逐部件对齐，见 region_struct_loss）
    "w_flatfrac": 0.5,      # 全局兜底：防止「各区域都对但整体粒度跑偏」
    "w_flat": 0.15,         # 全局兜底
    "w_edge": 0.2,          # 全局兜底
    "w_sym": 0.2,           # 肢体镜像对齐 0.1261（真实本就不对称，权重必须小）
    "w_pal": 0.0,           # 调色板熵 —— **实测无区分度，默认关掉**
    "ramp_start": 3,        # 前 3 个 epoch 不加结构损失，先让对抗损失定大形
    "ramp_full": 15,        # 第 15 个 epoch 时结构损失到满权重
    "r1_gamma": 10.0,       # R1 系数（替代「压低 D 学习率」的土办法）
    "r1_every": 16,         # R1 惰性执行间隔
}


def total_generator_loss(g_adv, x_gen, alpha, palette, targets: dict,
                         w: dict | None = None,
                         scale: float = 1.0,
                         scheme: str = "detail") -> tuple[torch.Tensor, dict]:
    """把结构损失挂到对抗损失上。

    ``scale`` 是 ``ramp()`` 的输出（0→1），用来让结构约束逐步生效。
    ``targets`` 里带 ``regions`` 字段时走**区域化主项**，同时保留三项全局兜底项
    （权重已相应调小）：区域损失管「每个部件各自像不像」，
    全局兜底管「整体粒度有没有跑偏」，两者互补而不是二选一。

    返回 ``(总损失, 逐项数值)``，逐项数值直接进 metrics.jsonl 便于诊断；
    区域项按 ``"{区域}.{指标}"`` 命名，能直接看出是哪个部位在退化。
    """
    w = {**DEFAULT_WEIGHTS, **(w or {})}
    parts: dict[str, object] = {}
    struct = 0.0

    reg_t = targets.get("regions") or {}
    if w.get("w_region", 0) > 0 and reg_t:
        reg_loss, reg_detail = region_struct_loss(
            x_gen, alpha, reg_t, scheme=scheme,
            region_weights=w.get("region_weights"),
            metric_weights=w.get("metric_weights"),
            tau=targets.get("tau", 0.4275))
        parts["region"] = reg_loss
        struct = struct + w["w_region"] * reg_loss
        for k, v in reg_detail.items():
            parts[f"r.{k}"] = v

    if w.get("w_flatfrac", 0) > 0:
        parts["flatfrac"] = flat_fraction(x_gen, alpha, targets.get("tau", 0.4275),
                                          target=targets.get("flat_frac_target", 0.562))
        struct = struct + w["w_flatfrac"] * parts["flatfrac"]
    if w.get("w_flat", 0) > 0 and targets.get("flat") is not None:
        parts["flat"] = face_flatness(x_gen, alpha, targets.get("flat"))
        struct = struct + w["w_flat"] * parts["flat"]
    if w.get("w_edge", 0) > 0 and targets.get("edge") is not None:
        parts["edge"] = edge_bimodal(x_gen, alpha, targets.get("tau", 0.4275),
                                     targets.get("edge"))
        struct = struct + w["w_edge"] * parts["edge"]
    if w.get("w_sym", 0) > 0 and targets.get("sym") is not None:
        parts["sym"] = limb_mirror(x_gen, alpha, targets.get("sym"))
        struct = struct + w["w_sym"] * parts["sym"]
    if w.get("w_pal", 0) > 0:
        parts["pal"] = palette_concentration(
            x_gen, alpha, palette, targets.get("sigma", 0.08),
            targets.get("pal_entropy"))
        struct = struct + w["w_pal"] * parts["pal"]

    total = g_adv + scale * struct
    # 逐项数值里既有 tensor（各项损失）也有 float（区域损失返回的逐项诊断值），
    # 统一转成 float 再返回，调用方直接 json 落盘即可。
    out: dict[str, float] = {}
    for k, v in parts.items():
        out[k] = float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
    return total, out


def ramp(epoch: int, start: int, full: int) -> float:
    """结构损失的线性升温：epoch<start 时为 0，到 full 时到 1。

    动机：结构项一开始就给满，模型还没学会「大致长什么样」就被逼着分片常数，
    会直接塌成纯色块。先让对抗损失把大形定下来，再逐步加结构约束。
    """
    if epoch <= start:
        return 0.0
    if epoch >= full:
        return 1.0
    return (epoch - start) / max(1, full - start)
