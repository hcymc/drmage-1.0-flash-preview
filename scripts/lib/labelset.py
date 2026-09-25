"""labelset.py — 把 labels/annotations.jsonl 转成条件生成用的向量。

条件向量由**规则可推导**的标签拼成（不用启发式语义标签，避免把噪声喂进条件）：

| 片段 | 维度 | 来源 |
|---|---|---|
| 色相直方图 | 12 | `color.hue_hist12` 归一化 |
| 色相均值 | 1 | `color.hue_mean_deg` / 360 |
| 饱和 / 明度 | 2 | `color.sat_mean` `val_mean` |
| 中性 / 暗 / 亮 / 高饱和占比 | 4 | `color.*_ratio` |
| tone one-hot | 4 | `tone.tone` ∈ {grayscale,dark,bright,mid} |
| 饱和档 one-hot | 3 | `tone.saturation_class` |
| 复杂度档 one-hot | 5 | `complexity.class` |
| 用过 overlay | 1 | `transparency.overlay_used` |
| 透明像素占比 | 1 | `transparency.transparent_ratio` |
| **模型类型 one-hot** | 3 | `model.type` ∈ {classic, slim, unknown} |
| **合计** | **36** | |

> **模型类型（Steve / Alex）**：Minecraft 皮肤有两套骨架 —— 经典（classic，Steve 型，
> 手臂 4 像素宽）与纤细（slim，Alex 型，手臂 3 像素宽）。两者的贴图 UV 布局不同：
> 纤细模型手臂贴图只用到 16 宽盒子里的前 12 列，最右 4 列**永远空着**。
> 这个差异必须显式标注出来，否则模型会把两种骨架混在一起学，生成的手臂宽度是乱的。
> 这里把它作为条件向量的一个 one-hot 段，训练与推理都能指定 / 区分。

同时提供 ``describe(vector)``，把条件向量反解回「人话」，用于验证条件是否真的生效。
"""

from __future__ import annotations

import json
import os

import numpy as np

TONE_ORDER = ["grayscale", "dark", "bright", "mid"]
SAT_ORDER = ["high_saturation", "mid_saturation", "low_saturation"]
CPX_ORDER = ["near_solid", "simple", "medium", "detailed", "very_detailed"]
MODEL_ORDER = ["classic", "slim", "unknown"]

COND_DIM = 36

#: **部位级 overlay 开关**的位顺序（在 36 维基础向量之后追加）。
#:
#: 为什么要这一组位：原来的条件里关于第二层只有 ``transparency.overlay_used``
#: （**全局 1 个 bit**）和 ``transparent_ratio``（1 个 float）。模型因此**分不清**
#: 「要帽子」和「要外套」—— 而这两件事在真实皮肤里是完全独立的
#: （head.overlay 有内容的比例 0.896，body/arm/leg.overlay 只有 0.527~0.581）。
#: 全局 1 个 bit 表达不了「头有帽子、手臂没袖子」这种最常见的组合。
#:
#: 有了这 4 位，**条件、alpha 模板、损失加权三者才能对齐**：
#: 说「要袖子」→ 模板在 arm.overlay 露出 → 加权让模型认真学这块。
#: 只做后者不做前者，模型仍然不知道「这个部位该不该画」。
#:
#: 这 4 位**不并入** :data:`COND_DIM`（保持 36），是为了让旧 checkpoint 仍然能按
#: 36 维重建；开启时训练侧把维度变成 ``36 + 4 = 40``，由 ``--ov-bits`` 控制。
OV_BIT_NAMES = ("ov_hat", "ov_body", "ov_arm", "ov_leg")
OV_BITS = len(OV_BIT_NAMES)

#: 位名 → 对应的 overlay 盒前缀（见 ``skinatlas.OVERLAY_BOXES``）
OV_BIT_BOXES: dict[str, tuple[str, ...]] = {
    "ov_hat": ("hat",),
    "ov_body": ("body_ov",),
    "ov_arm": ("rarm_ov", "larm_ov"),
    "ov_leg": ("rleg_ov", "lleg_ov"),
}


def _onehot(value, order) -> list[float]:
    v = [0.0] * len(order)
    if value in order:
        v[order.index(value)] = 1.0
    return v


def vector_from_record(rec: dict) -> np.ndarray:
    c = rec.get("color") or {}
    t = rec.get("tone") or {}
    comp = rec.get("complexity") or {}
    tr = rec.get("transparency") or {}
    md = rec.get("model") or {}

    hist = np.array(c.get("hue_hist12") or [0] * 12, dtype=np.float32)
    hist = hist / max(hist.sum(), 1.0)

    parts = [
        hist,
        [float(c.get("hue_mean_deg", 0.0)) / 360.0],
        [float(c.get("sat_mean", 0.0)), float(c.get("val_mean", 0.0))],
        [float(c.get("neutral_ratio", 0.0)), float(c.get("dark_ratio", 0.0)),
         float(c.get("bright_ratio", 0.0)), float(c.get("high_sat_ratio", 0.0))],
        _onehot(t.get("tone"), TONE_ORDER),
        _onehot(t.get("saturation_class"), SAT_ORDER),
        _onehot(comp.get("class"), CPX_ORDER),
        [1.0 if tr.get("overlay_used") else 0.0, float(tr.get("transparent_ratio", 0.0))],
        _onehot(md.get("type") or "unknown", MODEL_ORDER),
    ]
    v = np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])
    if v.shape[0] != COND_DIM:
        raise ValueError(f"cond dim {v.shape[0]} != {COND_DIM}")
    return v


def load_model_types(clean_manifest: str) -> dict[str, str]:
    """读 ``data/clean_manifest.csv``，返回 ``sid -> model_type``。

    用于「只拿 classic(Steve) 或只拿 slim(Alex) 训练」这种筛选。
    """
    import csv
    out: dict[str, str] = {}
    if not os.path.isfile(clean_manifest):
        return out
    with open(clean_manifest, "r", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r.get("sid", "")] = r.get("model_type", "") or "unknown"
    return out


def filter_by_model_type(sids: list[str], model_type: str,
                         sid2type: dict[str, str]) -> np.ndarray:
    """返回布尔掩码：True = 保留。``model_type='all'`` 时全 True。"""
    if model_type in ("all", "", None):
        return np.ones(len(sids), dtype=bool)
    return np.array([sid2type.get(s, "unknown") == model_type for s in sids], dtype=bool)


def build_cond_matrix(annotations_path: str, sids: list[str]) -> np.ndarray:
    """按 ``sids`` 顺序返回 ``(len(sids), COND_DIM)``；缺标注的样本回退为零向量。"""
    recs: dict[str, dict] = {}
    if os.path.isfile(annotations_path):
        with open(annotations_path, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                recs[r.get("sid", "")] = r
    out = np.zeros((len(sids), COND_DIM), dtype=np.float32)
    n_missing = 0
    for i, sid in enumerate(sids):
        rec = recs.get(sid)
        if not rec or "color" not in rec:
            n_missing += 1
            continue
        out[i] = vector_from_record(rec)
    if n_missing:
        print(f"[labelset] {n_missing}/{len(sids)} 个样本缺少规则标注，条件回退为零向量")
    return out


#: 人话条件规格的默认值（对应一个「中间调、中等饱和、中等复杂度」的皮肤）
SPEC_DEFAULTS = {
    "hue_deg": 15.0,            # 主导色相，0-360
    "hue_spread": 0.18,         # 色相直方图的集中度（越大越集中）
    "sat_mean": 0.45,
    "val_mean": 0.55,
    "neutral_ratio": 0.1,
    "dark_ratio": 0.25,
    "bright_ratio": 0.25,
    "high_sat_ratio": 0.25,
    "tone": "mid",
    "saturation_class": "mid_saturation",
    "complexity_class": "medium",
    "overlay_used": True,
    "transparent_ratio": 0.35,
    "model_type": "classic",     # classic(Steve) | slim(Alex) | unknown
    # ---- 部位级 overlay 开关 ----
    # 注意它们**不属于** 36 维向量（见 OV_BIT_NAMES）：给 UI 与条件生成共用，
    # 由 ``ov_bits_from_spec`` 单独取出，训练/推理时追加到向量尾部。
    "ov_hat": True,              # 头部外层（帽子 / 头发）
    "ov_body": True,             # 躯干外层（外套 / 上衣）
    "ov_arm": True,              # 手臂外层（袖子）
    "ov_leg": True,              # 腿部外层（裤腿）
}


def vector_from_spec(spec: dict) -> np.ndarray:
    """从「人话」规格构造条件向量，用于条件生成与可控性验证。

    ``spec`` 里没给的字段走 :data:`SPEC_DEFAULTS`。示例::

        vector_from_spec({"tone": "dark", "hue_deg": 0, "complexity_class": "detailed"})
    """
    s = {**SPEC_DEFAULTS, **{k: v for k, v in spec.items() if v is not None}}

    # 由 hue 生成一个围绕该色相的半高斯直方图
    hue = float(s["hue_deg"]) % 360.0
    centers = np.arange(12) * 30 + 15
    d = np.abs(centers - hue)
    d = np.minimum(d, 360 - d)
    sigma = max(12.0, 360.0 * float(s["hue_spread"]))
    hist = np.exp(-(d ** 2) / (2 * sigma ** 2))
    hist = hist / max(hist.sum(), 1e-6)

    parts = [
        hist,
        [hue / 360.0],
        [float(s["sat_mean"]), float(s["val_mean"])],
        [float(s["neutral_ratio"]), float(s["dark_ratio"]),
         float(s["bright_ratio"]), float(s["high_sat_ratio"])],
        _onehot(s["tone"], TONE_ORDER),
        _onehot(s["saturation_class"], SAT_ORDER),
        _onehot(s["complexity_class"], CPX_ORDER),
        [1.0 if s["overlay_used"] else 0.0, float(s["transparent_ratio"])],
        _onehot(s["model_type"], MODEL_ORDER),
    ]
    v = np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])
    if v.shape[0] != COND_DIM:
        raise ValueError(f"cond dim {v.shape[0]} != {COND_DIM}")
    return v


def describe(v: np.ndarray) -> dict:
    """条件向量 → 人话，用于验证「条件是否真的可控」。

    向量切分（必须与 :func:`vector_from_record` 严格一致）::

        0:12   色相直方图 12
        12     色相均值
        13,14  饱和度 / 明度
        15..18 中性 / 暗 / 亮 / 高饱和占比
        19..22 tone one-hot (4)
        23..25 饱和档 one-hot (3)
        26..30 复杂度档 one-hot (5)
        31,32  用过 overlay / 透明像素占比
        33..35 模型类型 one-hot (3)  classic / slim / unknown
    """
    v = np.asarray(v).reshape(-1)
    hist = v[0:12]
    tone = TONE_ORDER[int(np.argmax(v[19:23]))] if v[19:23].sum() > 0 else "?"
    sat = SAT_ORDER[int(np.argmax(v[23:26]))] if v[23:26].sum() > 0 else "?"
    cpx = CPX_ORDER[int(np.argmax(v[26:31]))] if v[26:31].sum() > 0 else "?"
    mdl = MODEL_ORDER[int(np.argmax(v[33:36]))] if v[33:36].sum() > 0 else "unknown"
    return {
        "dominant_hue_deg": int(np.argmax(hist) * 30 + 15),
        "hue_mean_deg": round(float(v[12]) * 360, 1),
        "sat_mean": round(float(v[13]), 3),
        "val_mean": round(float(v[14]), 3),
        "tone": tone,
        "saturation_class": sat,
        "complexity_class": cpx,
        "overlay_used": bool(v[31] > 0.5),
        "transparent_ratio": round(float(v[32]), 4),
        "model_type": mdl,
    }


# ---------------------------------------------------------------------------
# 部位级 overlay 开关（见 OV_BIT_NAMES）
# ---------------------------------------------------------------------------

def overlay_bits_from_alpha(alpha_flat, thresh: float = 0.02) -> np.ndarray:
    """从 ``(4096,)`` / ``(64,64)`` 的可见性掩码算 4 个部位级 overlay 位。

    判据：该部位的**任一** overlay 面里可见像素占比 > ``thresh``。

    用「任一」而不是「全部」：真实作者画外套经常只画正面
    （``body_ov.front`` 的 ``partial`` 频率 0.85，而 ``body_ov.top`` 只有 0.31），
    要求「所有面都有」会把大量真实外套判成「没有外套」——目标位会系统性偏低，
    而条件与标签不一致正是「改了没用」的经典来源。
    """
    from skinatlas import face_index

    fid = face_index()
    a = np.asarray(alpha_flat).reshape(-1) > 0.5
    out = np.zeros(OV_BITS, dtype=np.float32)
    for gi, name in enumerate(OV_BIT_NAMES):
        boxes = OV_BIT_BOXES[name]
        for nm, (y0, y1, x0, x1) in fid.items():
            if nm.split(".", 1)[0] not in boxes:
                continue
            yy = np.arange(y0, y1)[:, None]
            xx = np.arange(x0, x1)[None, :]
            sub = a[(yy * 64 + xx).reshape(-1)]
            if sub.size and float(sub.mean()) > thresh:
                out[gi] = 1.0
                break
    return out


def ov_bits_from_spec(spec: dict) -> np.ndarray:
    """从人话 ``spec`` 取 4 个部位 overlay 位（缺省走 :data:`SPEC_DEFAULTS`）。"""
    s = {**SPEC_DEFAULTS, **(spec or {})}
    return np.array([1.0 if s.get(n) else 0.0 for n in OV_BIT_NAMES],
                    dtype=np.float32)


def describe_ov_bits(v) -> dict:
    """4 位 → 人话（给 UI 回显用）。"""
    v = np.asarray(v).reshape(-1)
    return {n: bool(v[i] > 0.5) for i, n in enumerate(OV_BIT_NAMES) if i < v.size}


def ov_bits_to_box_paint(bits) -> dict[str, np.ndarray]:
    """``(N,4)`` 的位 → ``{盒名: (N,) 的 0/1}``，喂给
    ``skinatlas.sample_alpha_templates(box_paint=...)``。

    一个位对应一到两个盒（``ov_arm`` → ``rarm_ov`` + ``larm_ov``），
    展开后左右袖一起开关 —— 与真实数据一致（作者画袖子通常两边都画，
    但 ``limb_mirror`` 实测左右仍允许不同，所以**不是**强制对称，
    只是模板层面对齐）。
    """
    b = np.asarray(bits, dtype=np.float32).reshape(-1, OV_BITS)
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(OV_BIT_NAMES):
        for box in OV_BIT_BOXES[name]:
            out[box] = b[:, i]
    return out
