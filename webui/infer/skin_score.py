"""skin_score.py — 抽奖模式评分器。

回答一个问题：一次「抽奖」抽出的 N 张候选皮肤，哪几张最值得交付出去？

设计立场（先说清楚边界）
------------------------
* **不评「像不像某张具体的皮肤」**——同条件下真实分布是多峰的，
  没有「标准答案」可比；
* **评「这张图作为一张 Minecraft 皮肤的成色」**——像素画的结构质感
  （同色块、镜像、脸部）+ 颜色分布是否落在真实范围 + 是否听条件的话。
  三类信号对应本模型三种已知失败模式：
  高频噪声（结构信号≈0）、涂色块（颜色与结构信号双低）、
  不听指挥（条件贴合度低）。

五个组件（各归一到 [0,1]，加权和为总分）
----------------------------------------
| 组件 | 权重 | 信号 | 真实参照（出处见 docs/LOTTERY.md） |
|---|---|---|---|
| color_realism 颜色真实度 | 0.25 | 唯一色数落在真实中位（≈66）的 log 带内 | 真实中位 66 |
| block_structure 色块结构 | 0.25 | ≥8px 同色连通块覆盖率，单调奖赏 | 真实 0.139（丰富子集） |
| limb_symmetry 肢体镜像 | 0.20 | 左右肢 UV 镜像色差（复用 limb_mirror_index） | 真实不对称度 0.075~0.176 |
| face_coherence 脸部连贯 | 0.15 | head.front 面内相邻精确相等率，单调奖赏 | 真实 0.398 |
| cond_adherence 条件贴合 | 0.15 | 生成图特征 vs 它自己的条件向量 | —（自校准） |

退化保护：唯一色 <8（涂色块）总分 ×0.3；出现非有限像素直接 0 分。

口径纪律：所有「真实参照」数字都来自项目已有实测（`docs/MODEL_CARD.md`
与训练期探针），且是**随机真实皮肤**口径——本评分器只做候选间的
相对排序，不把「追上真实绝对值」当目标（本模型当前达不到）。
"""

from __future__ import annotations

import os
import sys

import numpy as np

_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

WEIGHTS = {
    "color_realism": 0.25,
    "block_structure": 0.25,
    "limb_symmetry": 0.20,
    "face_coherence": 0.15,
    "cond_adherence": 0.15,
}

# ---------------------------------------------------------------------------
# 学习型评分器（主评分）：真实 vs 生成 判别器
# ---------------------------------------------------------------------------
# 手搓五组件有方向性偏置：平涂皮肤天然拿满结构分，精细设计反而被
# 「色块不够大 / 脸部相邻相等率低」扣分（实测优质生成图 0.695 输给纯色西装
# 0.947）。「整体像不像真的皮肤」是手搓规则枚举不了的高维直觉，所以让
# scripts/33_train_lottery_scorer.py 训练的小判别器来给主分：
#   score = P(real)，两侧样本过完全相同的后处理管线（量化档位随机化）。
# 五组件保留为**参考诊断**显示在弹窗里，不再参与排序。

_SCORER: dict = {"mtime": None, "model": None, "device": None}


def _scorer_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "models", "lottery_scorer", "latest.pt")


def load_scorer(force: bool = False) -> bool:
    """懒加载判别器权重（按 mtime 缓存）。可用返回 True。"""
    import torch
    p = _scorer_path()
    if not os.path.isfile(p):
        return False
    mtime = os.path.getmtime(p)
    if not force and _SCORER["mtime"] == mtime and _SCORER["model"] is not None:
        return True
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train33", os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "scripts",
            "33_train_lottery_scorer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build_model()
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    _SCORER.update({"mtime": mtime, "model": model, "device": dev,
                    "auc": ck.get("val_auc")})
    return True


def learned_score(arr: np.ndarray) -> float | None:
    """P(real) ∈ [0,1]；判别器不可用时返回 None（调用方回落规则分）。"""
    import torch
    if not load_scorer():
        return None
    model, dev = _SCORER["model"], _SCORER["device"]
    x = torch.from_numpy(
        np.ascontiguousarray(arr.transpose(2, 0, 1))[None].astype(np.float32) / 255.0).to(dev)
    with torch.no_grad():
        p = float(torch.sigmoid(model(x)).item())
    return p

#: 颜色真实度的参照：真实皮肤可见像素唯一色中位 ≈66（ ingest 实测），
#: 基准可见像素数 ≈2000（真实可见率中位 0.48 × 4096）。
UC_TARGET = 66.0
UC_BASE_PX = 2000.0
#: log 带宽：唯一色差 4 倍 → 该项衰减到 e^-1。
UC_TAU = float(np.log(4.0))

#: 色块结构的饱和点：覆盖率达到 0.10 即给满分（真实丰富子集 0.139）。
BLOCK_CAP = 0.10

#: 肢体镜像：不对称度 ≤0.15 满分、≥0.45 零分（真实 0.075~0.176，v1 生成 0.33~0.48）。
SYM_BEST, SYM_WORST = 0.15, 0.45

#: 脸部连贯的饱和点：面内精确相等率达到 0.25 给满分（真实脸区 0.398）。
FACE_CAP = 0.25

#: 条件贴合的衰减尺度。自检（``python skin_score.py``）会打印真实 val 对的
#: 实测距离分布——τ 取「真实中位距离 → 该项 0.80 分」的标定值。
ADH_TAU = 0.36

#: 退化保护：可见像素唯一色低于该值视为涂色块，总分乘罚系数。
DEGEN_UC = 8
DEGEN_PENALTY = 0.3


# ---------------------------------------------------------------------------
# 基础量
# ---------------------------------------------------------------------------

def _unique_colors(rgb: np.ndarray, vis: np.ndarray) -> int:
    px = rgb[vis]
    if px.size == 0:
        return 0
    return int(np.unique(px.reshape(-1, 3), axis=0).shape[0])


class _DSU:
    """小规模并查集（4096 节点，够快，不引依赖）。"""

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _block_cover(rgb: np.ndarray, vis: np.ndarray, min_px: int = 8) -> float:
    """≥``min_px`` 的同色（三通道精确相等）4-连通块覆盖可见像素的比例。"""
    h, w = vis.shape
    idx = np.full((h, w), -1, dtype=np.int64)
    flat = vis.reshape(-1)
    idx.reshape(-1)[flat] = np.arange(int(flat.sum()))
    n = int(flat.sum())
    if n == 0:
        return 0.0
    dsu = _DSU(n)
    r = rgb.reshape(-1, 3)
    ii = idx.reshape(-1)

    def _union_pairs(a_mask, b_mask, a_idx, b_idx):
        pairs = (a_mask & b_mask).reshape(-1)     # a_idx/b_idx 是展平的，掩码同序展平
        if not pairs.any():
            return
        ai, bi = a_idx[pairs], b_idx[pairs]
        eq = (r[ai] == r[bi]).all(axis=1)
        for a, b in zip(ai[eq], bi[eq]):
            dsu.union(int(a), int(b))

    # 水平 / 垂直相邻且同为可见、颜色精确相等 → 并
    hm = vis[:, :-1] & vis[:, 1:]
    _union_pairs(hm, hm, idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1))
    vm = vis[:-1, :] & vis[1:, :]
    _union_pairs(vm, vm, idx[:-1, :].reshape(-1), idx[1:, :].reshape(-1))

    from collections import Counter
    roots = Counter(dsu.find(i) for i in range(n) if ii[i] >= 0)
    big = sum(c for c in roots.values() if c >= min_px)
    return big / n


def _face_adj_eq(rgb: np.ndarray, vis: np.ndarray) -> float:
    """head.front（8×8 脸区）内相邻像素**精确相等**的比例（面内口径）。"""
    fi = _face_rect()
    if fi is None:
        return 0.0
    y0, y1, x0, x1 = fi
    sub_v = vis[y0:y1, x0:x1]
    sub_r = rgb[y0:y1, x0:x1]
    pairs = 0
    eq = 0
    for sl_a, sl_b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        both = sub_v[sl_a] & sub_v[sl_b]
        pairs += int(both.sum())
        if both.any():
            eq += int(((sub_r[sl_a] == sub_r[sl_b]).all(axis=2) & both).sum())
    return eq / pairs if pairs else 0.0


_FACE_RECT = None


def _face_rect():
    global _FACE_RECT
    if _FACE_RECT is None:
        from skinatlas import face_index
        _FACE_RECT = face_index().get("head.front")
    return _FACE_RECT


def _limb_asym(rgb: np.ndarray, vis: np.ndarray) -> float:
    """左右肢 UV 镜像不对称度：|ΔRGB| 均值 / 255（按臂/腿的可见对数加权）。"""
    from skinatlas import limb_mirror_index
    rf = rgb.reshape(-1, 3).astype(np.float32)
    vf = vis.reshape(-1)
    num = 0.0
    den = 0
    for _part, _pair, ir, im in limb_mirror_index():
        both = vf[ir] & vf[im]
        c = int(both.sum())
        if c == 0:
            continue
        num += float(np.abs(rf[ir][both] - rf[im][both]).mean()) * c
        den += c
    return (num / den / 255.0) if den else 1.0


def _adh_dist(rgb: np.ndarray, vis: np.ndarray, cond: np.ndarray) -> float:
    """生成图特征 vs 它自己的条件向量的距离（与训练特征同一实现：labeling.hsv_stats）。"""
    from labeling import hsv_stats
    px = rgb[vis]
    if px.size == 0:
        return 1.0
    hs = hsv_stats(px)
    hist = np.asarray(hs["hue_hist12"], dtype=np.float32)
    hist = hist / max(hist.sum(), 1.0)
    c_hist = np.asarray(cond[:12], dtype=np.float32)
    c_hist = c_hist / max(c_hist.sum(), 1e-6)
    l1 = float(np.abs(hist - c_hist).sum())
    h_img = float(hs["hue_mean_deg"])
    h_cond = float(cond[12]) * 360.0
    dh = abs(h_img - h_cond)
    dh = min(dh, 360.0 - dh)
    ds = abs(float(hs["sat_mean"]) - float(cond[13]))
    dv = abs(float(hs["val_mean"]) - float(cond[14]))
    return 0.5 * l1 + dh / 360.0 + 0.5 * ds + 0.5 * dv


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def score_image(arr: np.ndarray, cond: np.ndarray | None = None) -> dict:
    """给一张 ``(64,64,4)`` uint8 RGBA 皮肤计算**原始特征**（不含最终总分）。

    最终总分由 :func:`combine_pool` 在**整个候选池内排名归一**后给出——
    v1 的教训是绝对阈值（色块覆盖 cap、唯一色目标 66）会系统性卡死某类风格；
    v2 的教训是单一判别器学到的是「生成分布的典型性」，与我们的审美判断正交。
    v3：透明特征 + 池内排名 + 评审口味权重 + 判别器只做垃圾守门。

    返回 ``{"total", "parts", "stats"}``；此函数内 ``total`` 先置为规则分
    （无池上下文时的回落值），``combine_pool`` 会覆写。
    """
    arr = np.asarray(arr)
    vis = arr[..., 3] >= 128
    rgb = arr[..., :3]
    n_vis = int(vis.sum())

    if n_vis == 0 or not np.isfinite(rgb.astype(np.float64)).all():
        return {"total": 0.0, "parts": dict.fromkeys(WEIGHTS, 0.0),
                "stats": {"visible_px": n_vis, "degenerate": True}}

    uc = _unique_colors(rgb, vis)
    cover = _block_cover(rgb, vis)
    face_eq = _face_adj_eq(rgb, vis)
    asym = _limb_asym(rgb, vis)

    # ---- v3 原始特征（池内排名归一在 combine_pool 做）----
    fpx = rgb[vis].astype(np.float32)
    mx, mn = fpx.max(1), fpx.min(1)
    v_ch = mx / 255.0
    s_ch = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    sat_mean = float(s_ch.mean())
    # 设计密度：与可见像素主导色（中位 RGB）距离 > 60 的占比——
    # 「这张皮肤除了底色还画了多少东西」，直接对齐我们的评审口味（偏爱有设计内容的）
    dom = np.median(fpx, axis=0)
    design_density = float((np.sqrt(((fpx - dom) ** 2).sum(1)) > 60).mean())
    # 色相熵：hue 直方图的归一化熵（0=单色，1=均匀铺满 12 桶）
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = fpx[:, 0], fpx[:, 1], fpx[:, 2]
    h = np.where(mx == r, ((g - b) / d) % 6,
                 np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    hh, _ = np.histogram(h[mx > mn], bins=12, range=(0, 360))
    p = hh / max(hh.sum(), 1); p = p[p > 0]
    hue_entropy = float(-(p * np.log(p)).sum() / np.log(12)) if len(p) else 0.0
    # 边缘密度（细节量）
    g16 = arr[..., :3].astype(np.int16).sum(2)
    em = vis[:, 1:] & vis[:, :-1]
    edge = float((np.abs(np.diff(g16, axis=1)) > 24)[em].mean()) if em.any() else 0.0
    # 可读脸部：head.front 里同时存在暗(<0.3)与亮(>0.7)像素（=有眼睛）
    face_pair = 0.0
    fr = _face_rect()
    if fr is not None:
        y0, y1, x0, x1 = fr
        fm = vis[y0:y1, x0:x1]
        fv = rgb[y0:y1, x0:x1].astype(np.float32).mean(2) / 255.0
        if fm.any():
            face_pair = float((((fv < 0.3) & fm).sum() >= 2
                               and ((fv > 0.7) & fm).sum() >= 2))

    s_uc = float(np.exp(-abs(np.log(max(uc, 1) /
        (UC_TARGET * np.sqrt(max(n_vis, 1) / UC_BASE_PX)))) / UC_TAU))
    s_sym = float(np.clip((SYM_WORST - asym) / (SYM_WORST - SYM_BEST), 0.0, 1.0))

    parts = {
        "color_realism": s_uc,
        "block_structure": float(min(cover / BLOCK_CAP, 1.0)),
        "limb_symmetry": s_sym,
        "face_coherence": float(min(face_eq / FACE_CAP, 1.0)),
    }
    stats = {
        "visible_px": n_vis, "unique_colors": uc, "block_cover": round(cover, 4),
        "face_adj_eq": round(face_eq, 4), "limb_asym": round(asym, 4),
        "design_density": round(design_density, 4), "hue_entropy": round(hue_entropy, 4),
        "sat_mean": round(sat_mean, 4), "edge": round(edge, 4), "face_pair": face_pair,
    }

    w = dict(WEIGHTS)
    if cond is not None and np.size(cond) >= 15:
        d = _adh_dist(rgb, vis, np.asarray(cond, dtype=np.float64))
        parts["cond_adherence"] = float(np.exp(-d / ADH_TAU))
        stats["adh_dist"] = round(float(d), 4)
    else:
        w = {k: v for k, v in w.items() if k != "cond_adherence"}
        parts["cond_adherence"] = None
    z = sum(w.values())
    rule_total = sum(w[k] * parts[k] for k in w) / z
    parts["rule_total"] = round(float(rule_total), 4)

    p_real = learned_score(arr)
    parts["learned"] = (round(p_real, 4) if p_real is not None else None)

    # 无池上下文（单图调用）时的回落总分 = 规则分；池模式由 combine_pool 覆写
    total = rule_total
    if uc < DEGEN_UC:
        total *= DEGEN_PENALTY
        stats["degenerate"] = True
    return {"total": round(float(total), 4), "parts": parts, "stats": stats}


# ---------------------------------------------------------------------------
# v3：池内排名归一融合
# ---------------------------------------------------------------------------

#: 排名融合权重（由我们的评审反馈标定：丰富的设计 > 中庸的整齐）。
#: 判别器 P(real) **不参与排序、只在弹窗展示**——实测它学到的信号与我们
#: 的审美判断正交（鲜艳精细的设计反而低分，两轮实测均被否决），任何正权重
#: 都会把「我们喜欢的那类」重新按下去。防垃圾改由连贯度 + 覆盖率因子承担。
W_RICHNESS = 0.45
W_COHERENCE = 0.55
#: 覆盖率因子：可见率 <30% 线性压分（近空/碎屑候选），0.30 以上不罚。
COVERAGE_LO = 0.30


def _ranks(vals: list[float]) -> np.ndarray:
    """并列取平均名的 [0,1] 排名（0=池内最小，1=池内最大）。"""
    x = np.asarray(vals, dtype=np.float64)
    n = len(x)
    order = np.argsort(x, kind="stable")
    rk = np.empty(n, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2.0 / max(n - 1, 1)
        i = j + 1
    return rk


def combine_pool(score_dicts: list[dict]) -> list[dict]:
    """在整个候选池内做排名归一并写回每项的 ``total``。

    * 丰富度 = 排名均值(设计密度, 色相熵, 饱和, 边缘细节)——我们明确偏好
      细节丰富、色彩鲜明的候选，正向计分（v1 把它们全判负了）；
    * 连贯度 = 排名均值(肢体镜像, 可读脸部, 覆盖率)——噪声/碎屑在这里
      天然垫底，不需要再请一个黑盒守门员；
    * 覆盖率因子硬性压近空候选；涂色块/非有限像素维持退化否决。
    单张池（n=1）时排名恒为 0.5，等于退化为常数底分。
    """
    n = len(score_dicts)
    if n == 0:
        return score_dicts

    def col(name):
        return [float(d["stats"].get(name) or 0.0) for d in score_dicts]

    richness = (_ranks(col("design_density")) + _ranks(col("hue_entropy"))
                + _ranks(col("sat_mean")) + _ranks(col("edge"))) / 4.0
    coherence = (_ranks([float(d["parts"]["limb_symmetry"]) for d in score_dicts])
                 + _ranks(col("face_pair")) + _ranks(col("visible_px"))) / 3.0
    coverage = np.clip(np.array(col("visible_px")) / COVERAGE_LO, 0.0, 1.0)

    for i, d in enumerate(score_dicts):
        t = W_RICHNESS * float(richness[i]) + W_COHERENCE * float(coherence[i])
        t *= float(coverage[i])
        if d["stats"].get("degenerate"):
            t *= DEGEN_PENALTY
        d["total"] = round(float(t), 4)
        d["parts"]["richness_rank"] = round(float(richness[i]), 4)
        d["parts"]["coherence_rank"] = round(float(coherence[i]), 4)
        d["parts"]["coverage_factor"] = round(float(coverage[i]), 4)
        d["stats"]["scorer"] = "v3_pool"
    return score_dicts


# ---------------------------------------------------------------------------
# 自检 / 标定：真实 val 分布 + 退化样本对照
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    val = np.load(os.path.join(ROOT, "data", "processed", "val.npy"), mmap_mode="r")
    vc = np.load(os.path.join(ROOT, "data", "processed", "val_cond.npy"))
    rng = np.random.default_rng(7)
    idx = rng.choice(len(val), 256, replace=False)
    rows = []
    for i in idx:
        # 数据集是 CHW（(4,64,64)），评分器吃 HWC——与 engine.real_items 同一转法
        rows.append(score_image(np.asarray(val[i]).transpose(1, 2, 0), vc[i]))
    tot = np.array([r["total"] for r in rows])
    print("== 真实 val 256 张（应整体偏高、分布宽）")
    print(f"total  median={np.median(tot):.3f}  p10={np.quantile(tot, .1):.3f}  "
          f"p90={np.quantile(tot, .9):.3f}")
    for k in WEIGHTS:
        v = [r["parts"][k] for r in rows if r["parts"].get(k) is not None]
        print(f"  {k:<16} median={np.median(v):.3f}")
    d = [r["stats"].get("adh_dist") for r in rows if r["stats"].get("adh_dist") is not None]
    d = np.array(d)
    print(f"adh_dist median={np.median(d):.3f}  →  令真实中位=0.80 分的 τ* = "
          f"{np.median(d) / np.log(1 / 0.80):.3f}（当前 ADH_TAU={ADH_TAU}）")

    print("== 退化样本（应显著低于真实下限）")
    a0 = np.asarray(val[int(idx[0])]).transpose(1, 2, 0).copy()
    noise = a0.copy()
    noise[..., :3][noise[..., 3] >= 128] = rng.integers(
        0, 256, size=(int((noise[..., 3] >= 128).sum()), 3))
    flat = a0.copy()
    m0 = flat[..., 3] >= 128
    flat[..., :3][m0] = (120, 90, 70)
    sn = score_image(noise, vc[int(idx[0])])
    sf = score_image(flat, vc[int(idx[0])])
    print(f"噪声图 total={sn['total']:.3f} parts={ {k: round(v,2) for k,v in sn['parts'].items() if v is not None} }")
    print(f"纯色图 total={sf['total']:.3f} parts={ {k: round(v,2) for k,v in sf['parts'].items() if v is not None} }")
    assert sn["total"] < 0.35 and sf["total"] < 0.35, "退化样本必须显著低分"
    assert np.median(tot) > 0.45, "真实皮肤中位分应明显高于退化样本"
    print("SKIN_SCORE_SELFTEST_OK")
