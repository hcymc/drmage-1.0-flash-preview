#!/usr/bin/env python
"""10_label.py — 阶段三：数据标注。

分层推进，对应三种可信度不同的来源：

  Layer 1  base    —— 基础元数据 + UV 有效性。**全量**，事实型，无歧义。
  Layer 2  rules   —— 规则可推导：调色板 / 明暗 / 饱和 / 透明度使用度 / 复杂度 /
                      对称性 / 面部特征。**全量**，每个字段都有明确函数可复核。
  Layer 3  cluster —— 无监督聚类分组，得到「风格族」。**全量**，但族名需要人工命名
                      （脚本给出每族代表图，命名写入 labels/cluster_names.json）。
  Layer 4  semantic—— 启发式主题猜测。**部分覆盖**，带 confidence，明确标注为
                      heuristic，需人工抽检校正。

所有结果写进 ``labels/annotations.jsonl``（一行一个样本）。
脚本**可增量重跑**：已存在的 sid 默认跳过，加 ``--force`` 才重算。

用法
----
    python 10_label.py base      # Layer 1+2，全量
    python 10_label.py cluster   # Layer 3，全量
    python 10_label.py semantic  # Layer 4
    python 10_label.py schema    # 生成 labels/label_schema.md
    python 10_label.py report    # 覆盖率与分布统计
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from labeling import feature_vector  # noqa: E402  （只有聚类那一层还需要像素）
from manifest import annotation_from_row, read_clean, read_rows  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
LABELS = os.path.join(ROOT, "labels")
CLEAN_MANIFEST = os.path.join(DATA, "clean_manifest.csv")
PROCESSED = os.path.join(DATA, "processed")
ANNOTATIONS = os.path.join(LABELS, "annotations.jsonl")
CAPTIONS = os.path.join(LABELS, "captions.jsonl")
CLUSTER_NAMES = os.path.join(LABELS, "cluster_names.json")
CLUSTER_REPS = os.path.join(LABELS, "cluster_reps")

LABEL_VERSION = 1


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "label.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------
# JSONL 读写（流式，容错）
# --------------------------------------------------------------------------

def load_annotations() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not os.path.isfile(ANNOTATIONS):
        return out
    with open(ANNOTATIONS, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if rec.get("sid"):
                out[rec["sid"]] = rec
    return out


def save_annotations(recs: dict[str, dict]) -> None:
    os.makedirs(LABELS, exist_ok=True)
    tmp = ANNOTATIONS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in recs.values():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, ANNOTATIONS)


def load_clean_rows() -> list[dict]:
    with open(CLEAN_MANIFEST, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# Layer 1 + 2
# --------------------------------------------------------------------------

def layer_base(force: bool = False, limit: int | None = None) -> None:
    """Layer 1+2：把宽表行组装成标注记录。

    **这一步不读像素。** 所有需要的数字（UV 结构、骨架类型、质量三维、
    调色板 / HSV / 面部 / 对称度 / 复杂度档）都是上游摄取那一趟
    在内存里顺手算完写进 ``data/clean_manifest.csv`` 的
    （见 ``lib/manifest.py`` 的说明）。这里只是把它们摆成标注的层级结构。
    """
    rows, _fields = read_clean(CLEAN_MANIFEST)
    recs = load_annotations()
    todo = [r for r in rows if force or r["sid"] not in recs]
    if limit:
        todo = todo[:limit]
    log(f"Layer1+2：待标注 {len(todo)} / 总 {len(rows)}（已有 {len(recs)}）")
    t0 = time.time()

    for i, r in enumerate(todo, 1):
        rec = recs.get(r["sid"], {})
        rec.update(annotation_from_row(r))
        rec["quality"]["is_noise"] = bool(
            int(r.get("n_colors") or 0) <= 2 or float(r.get("transparent_ratio") or 0) > 0.9)
        rec["quality"]["is_near_blank"] = bool(float(r.get("face_opaque_ratio") or 0) < 0.5)
        rec["quality"]["dup_group"] = r.get("bucket", "")
        rec["quality"]["review_flag"] = None
        rec["label_meta"] = {
            "base_method": "rule@ingest",     # 特征在摄取那趟算好，这里只做组装
            "version": LABEL_VERSION,
            "annotated_at": datetime.now().isoformat(timespec="seconds"),
        }
        recs[r["sid"]] = rec
        if i % 20000 == 0:
            log(f"  {i}/{len(todo)}（{time.time()-t0:.0f}s）")
            save_annotations(recs)   # 中途落盘，防丢

    save_annotations(recs)
    log(f"Layer1+2 完成：{len(todo)} 条，耗时 {time.time()-t0:.1f}s，累计 {len(recs)}")


def _entropy(shares: list[float]) -> float:
    s = np.array([x for x in shares if x > 0], dtype=np.float64)
    if s.size == 0:
        return 0.0
    s = s / s.sum()
    return float(-(s * np.log2(s)).sum())


# --------------------------------------------------------------------------
# Layer 3：聚类
# --------------------------------------------------------------------------

def layer_cluster(k: int = 24, force: bool = False) -> None:
    from sklearn.cluster import KMeans

    rows, _ = read_clean(CLEAN_MANIFEST)
    recs = load_annotations()
    missing = [r for r in rows if r["sid"] not in recs]
    if missing:
        log(f"警告：{len(missing)} 条样本尚未跑 base 层，聚类前先跑 base 更稳妥")

    # 特征直接从 data/processed/*.npy 上算——那里已经是清洗后的全部像素，
    # 不必再按 (source,key) 回原始容器取一趟（理由见 lib/manifest.py）。
    src: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        p = os.path.join(PROCESSED, f"{split}.npy")
        sp = os.path.join(PROCESSED, f"{split}_sids.json")
        if not (os.path.isfile(p) and os.path.isfile(sp)):
            log(f"  (缺 {split}.npy / _sids.json，先跑 04_dataset.py)")
            continue
        x = np.load(p, mmap_mode="r")
        for j, sid in enumerate(json.load(open(sp, encoding="utf-8"))):
            src[sid] = x[j]
    if not src:
        log("[x] 没有可用的 processed npy，先跑 04_dataset.py")
        return

    order = [r for r in rows if r["sid"] in src]
    log(f"提取特征（{len(order)} 张，来自 processed npy）…")
    feats, sids = [], []
    for i, r in enumerate(order, 1):
        arr = np.asarray(src[r["sid"]]).transpose(1, 2, 0)   # CHW -> HWC
        feats.append(feature_vector(arr))
        sids.append(r["sid"])
        if i % 20000 == 0:
            log(f"  特征 {i}/{len(order)}")
    X = np.stack(feats)
    log(f"特征矩阵 {X.shape}，KMeans k={k} …")

    km = KMeans(n_clusters=k, n_init=6, random_state=42)
    lab = km.fit_predict(X)
    log(f"聚类完成，分布：{dict(Counter(lab.tolist()))}")

    # 每族取离质心最近的一张作为代表图
    os.makedirs(CLUSTER_REPS, exist_ok=True)
    names = {}
    if os.path.isfile(CLUSTER_NAMES):
        with open(CLUSTER_NAMES, "r", encoding="utf-8") as fh:
            names = json.load(fh)
    sizes = Counter(lab.tolist())
    for c in range(k):
        idx = np.nonzero(lab == c)[0]
        if idx.size == 0:
            continue
        d = ((X[idx] - km.cluster_centers_[c]) ** 2).sum(axis=1)
        best = idx[int(np.argmin(d))]
        dst = os.path.join(CLUSTER_REPS, f"c{c:02d}_{sids[best]}.png")
        try:
            arr = np.asarray(src[sids[best]]).transpose(1, 2, 0)
            Image.fromarray(arr, "RGBA").resize((256, 256), Image.NEAREST).save(dst)
        except Exception:
            pass
        names.setdefault(str(c), {"name": f"cluster_{c:02d}", "size": sizes[c],
                                  "representative": sids[best],
                                  "named_by": "auto"})
        names[str(c)]["size"] = sizes[c]
        names[str(c)]["representative"] = sids[best]

    with open(CLUSTER_NAMES, "w", encoding="utf-8") as fh:
        json.dump(names, fh, indent=1, ensure_ascii=False)

    for sid, c in zip(sids, lab.tolist()):
        rec = recs.get(sid)
        if rec is None:
            continue
        rec["style_group"] = {
            "cluster_id": int(c),
            "cluster_name": names.get(str(c), {}).get("name", f"cluster_{c:02d}"),
            "method": "cluster",
            "algo": f"KMeans(k={k}, seed=42)",
            "cluster_size": int(sizes[c]),
        }
    save_annotations(recs)
    log(f"Layer3 完成：{k} 个族，代表图在 {CLUSTER_REPS}")

    # 生成一张总览图便于人工命名
    reps = sorted(f for f in os.listdir(CLUSTER_REPS) if f.endswith(".png"))
    if reps:
        cols = min(6, len(reps))
        rows_n = (len(reps) + cols - 1) // cols
        cell = 256
        from PIL import ImageDraw
        sheet = Image.new("RGB", (cols * cell, rows_n * (cell + 22)), (28, 28, 34))
        dr = ImageDraw.Draw(sheet)
        for i, fn in enumerate(reps):
            r_, c_ = divmod(i, cols)
            im = Image.open(os.path.join(CLUSTER_REPS, fn)).convert("RGB")
            sheet.paste(im, (c_ * cell, r_ * (cell + 22)))
            dr.text((c_ * cell + 4, r_ * (cell + 22) + cell + 4),
                    f"{fn[:-4]}  n={names.get(fn.split('_')[0][1:], {}).get('size','?')}",
                    fill=(225, 225, 230))
        sheet.save(os.path.join(LOGS, "cluster_overview.png"))
        log(f"总览图 -> {os.path.join(LOGS, 'cluster_overview.png')}")


def layer_semantic(force: bool = False) -> None:
    """启发式主题标签。明确标注为 heuristic，带 confidence，需人工抽检。"""
    recs = load_annotations()
    n_done = 0
    for sid, rec in recs.items():
        if not force and rec.get("theme"):
            continue
        c = rec.get("color") or {}
        comp = rec.get("complexity") or {}
        face = rec.get("face") or {}
        uv = rec.get("uv") or {}
        tone = (rec.get("tone") or {}).get("tone", "mid")

        theme, conf, why = "unknown", 0.0, []
        uniq = comp.get("unique_colors", 0)
        if comp.get("is_near_solid") or uniq <= 3:
            theme, conf, why = "solid_minimal", 0.85, ["unique_colors<=3"]
        elif face.get("face_has_contrast_pair") and face.get("face_dark_ratio", 0) > 0.03:
            theme, conf, why = "humanoid", 0.5, ["face contrast pair present"]
        elif c.get("neutral_ratio", 0) > 0.75 and uniq <= 16:
            theme, conf, why = "monochrome_graphic", 0.55, ["neutral_ratio>0.75"]
        elif uv.get("overlay_ratio", 0) > 0.35:
            theme, conf, why = "layered_costume", 0.45, ["overlay heavily used"]
        elif comp.get("class") in ("detailed", "very_detailed"):
            theme, conf, why = "complex_character", 0.4, ["high color count"]
        else:
            theme, conf, why = "simple_character", 0.35, ["default bucket"]

        style = "monochrome" if c.get("neutral_ratio", 0) > 0.75 else (
            "neon" if c.get("high_sat_ratio", 0) > 0.5 else (
                "dark" if tone == "dark" else ("pastel" if tone == "bright" else "balanced")))

        rec["theme"] = {"label": theme, "method": "heuristic",
                        "confidence": conf, "evidence": why}
        rec["style"] = {"label": style, "method": "heuristic", "confidence": 0.45}
        n_done += 1
    save_annotations(recs)
    log(f"Layer4 完成：{n_done} 条启发式标签")


# --------------------------------------------------------------------------
# Layer 2.7：并入选集自带的人工描述
# --------------------------------------------------------------------------

def layer_captions(force: bool = False) -> None:
    """把 ``labels/captions.jsonl``（HF 描述集自带的文本）并入标注。

    这个字段的定位要说清楚：**它是外部给定的，不是本项目推断的**。
    上游 ``summykai/minecraft-skins-captioned-900k`` 的 ``text`` 列从行文看
    是 caption 模型（视觉语言模型）生成的描述，不是人工逐张撰写——
    所以 ``method`` 记成 ``given``（给定）而不是 ``human``，
    README 与 label_schema 里也不得把它宣传成「人工标注」。

    它的价值仍然真实：这是一段**独立的自然语言证据**，本项目只做 sha256 对齐，
    不改写、不扩写、不补全。覆盖不到的样本保持没有该字段，
    绝不用启发式规则编一段填进去——那样等于把噪声伪装成事实。

    对齐键用 sha256 而不是文件名：sha256 是内容哈希，跨来源也能对上。
    """
    if not os.path.isfile(CAPTIONS):
        log(f"[!] 找不到 {CAPTIONS}，先跑 02c_ingest_captioned.py")
        return
    by_sha: dict[str, str] = {}
    bad = 0
    with open(CAPTIONS, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                bad += 1
                continue
            h = r.get("sha256") or ""
            t = (r.get("text") or "").strip()
            if h and t:
                by_sha[h] = t
    log(f"载入描述 {len(by_sha)} 条（解析失败 {bad}）")

    recs = load_annotations()
    if not recs:
        log("[!] 标注为空，先跑 10_label.py base")
        return
    hit, miss, chars = 0, 0, 0
    for _sid, rec in recs.items():
        if not force and rec.get("caption"):
            continue
        t = by_sha.get(rec.get("sha256") or "")
        if not t:
            miss += 1
            continue
        rec["caption"] = {
            "text": t,
            "n_chars": len(t),
            "source": "hf_captioned",
            "method": "given",     # 给定（上游 caption 模型产出），非本项目生成、非人工
        }
        hit += 1
        chars += len(t)
    save_annotations(recs)
    log(f"Layer2.7 完成：命中 {hit} 条，未覆盖 {miss} 条，"
        f"平均长度 {chars/max(hit,1):.0f} 字符")


# --------------------------------------------------------------------------
# schema / report
# --------------------------------------------------------------------------

def cmd_schema() -> None:
    md = """# 标注字段说明（label_schema.md）

样本 ID 形如 `M000123`（64×64 现代版）或 `L000012`（64×32 旧版），
与 `data/clean_manifest.csv` 的 `sid` 一一对应。

每条标注是一个 JSON 对象，按**来源可信度**分成四层。字段里的 `method` /
`*_method` 表明该字段是规则算出来的、聚类得到的、模型猜的还是人工确认的。

---

## 第 1 层 · 基础元数据（全量覆盖 · 事实型）

| 字段 | 类型 | 含义 |
|---|---|---|
| `sid` | string | 样本唯一 ID |
| `path` | string | 相对项目根的 PNG 路径 |
| `source` | string | 来源站点/通道（`mineskin` / `account` …） |
| `sha256` | string | 文件字节级哈希，精确去重依据 |
| `key` | string | 来源侧的内容键（MineSkin 纹理哈希 / UUID） |
| `width` `height` | int | 像素尺寸，只允许 64×64 或 64×32 |
| `is_legacy` | bool | 是否为 64×32 旧版格式 |
| `split` | string | `train` / `val` / `test` |
| `uv_valid` | bool | 是否通过 UV 规范校验 |
| `uv_reason` | string | 校验结论，`ok` 或具体失败原因 |

## 第 2 层 · 规则可推导标签（全量覆盖 · 可复核）

| 字段 | 含义 |
|---|---|
| `uv.transparent_ratio` | 全图 alpha==0 像素占比 |
| `uv.overlay_ratio` | overlay 区域被使用的像素占比 |
| `uv.face_opaque_ratio` | 面部正面 8×8 的不透明像素占比 |
| `uv.torso_opaque_ratio` | 躯干正面区域的不透明像素占比 |
| `palette` | top-8 主色，每项 `{hex, rgb, share}`。中位切分量化得到 |
| `color.hue_hist12` | 12 段色相直方图（可见像素） |
| `color.sat_mean` `val_mean` | 平均饱和度 / 明度 |
| `color.neutral_ratio` | 近中性色像素占比（低饱和） |
| `color.dark_ratio` `bright_ratio` | 暗 / 亮像素占比 |
| `tone.tone` | `grayscale` / `dark` / `bright` / `mid` |
| `tone.saturation_class` | `high_saturation` / `mid_saturation` / `low_saturation` |
| `transparency.overlay_used` | 是否明显使用了 overlay 层（阈值 0.02） |
| `complexity.unique_colors` | 可见像素的去重颜色数 |
| `complexity.palette_k8_entropy` | top-8 调色板占比的信息熵 |
| `complexity.edge_density` | 相邻像素差异比例（边缘密度） |
| `complexity.symmetry` | 左右镜像一致像素占比 |
| `complexity.class` | `near_solid` / `simple` / `medium` / `detailed` / `very_detailed` |
| `face.face_dark_ratio` | 面部暗像素占比 |
| `face.face_has_contrast_pair` | 面部是否同时存在暗像素与亮像素 |

## 第 2.5 层 · 骨架类型 Steve / Alex（全量覆盖 · 规则判定）

Minecraft 皮肤有两套骨架，贴图 UV 布局不同，**必须分开**，否则生成的手臂宽度会乱：

| 取值 | 含义 | 手臂宽度 |
|---|---|---|
| `classic` | 经典，即 Steve 默认皮肤那一套 | 4 像素 |
| `slim` | 纤细，即 Alex 默认皮肤那一套 | 3 像素 |
| `unknown` | 手臂区域几乎没画，无法判定 | — |
| `legacy_na` | 64×32 旧版皮肤，不适用 | — |

| 字段 | 含义 |
|---|---|
| `model.type` | 上述四值之一 |
| `model.confidence` | 0-1，规则判定的把握程度 |
| `model.method` | 固定为 `rule` |
| `model.evidence` | 两条判据的实测值（`box_max` / `excl_max` / `foot_min` + 四条判别带占用率） |
| `model.arm_ratio` | 手臂两个盒子区域里被绘制的像素占比 |
| `model.limb_l_ratio` / `model.limb_r_ratio` | 左 / 右肢体本体（腿+臂）被绘制的像素占比 |

> 判据（两条，互相独立）：
>
> **① 物理上限。** 纤细模型 w=d=3、h=12，手臂盒最多只能填 `12×15 = 180/256 = 0.7031`。
> 占用率**超过这个值就绝不可能是纤细**——这是最硬的一条。
>
> **② 排他区。** 手臂在贴图上占一个 16 宽的盒子 UV 展开区。经典展开宽 `2*(4+4)=16`
> 正好铺满；纤细展开宽 `2*(3+3)=12` 只铺前 12 列 15 行。于是
> 「最右 4 列（`x∈[u+12,u+16)`，`y∈[v+4,v+16)`）+ 末行左 12 列」这块 L 形共 60 px，
> 经典会画、纤细永远画不到。里面出现笔画 → 经典。
>
> > 早期版本只用②的弱化版做加权投票，实测在 4 万条上吐出 16% unknown + 0.14% slim，
> > 两个数都不可信：短袖型经典皮肤（手臂背面只画一半）会被推进 unknown，
> > 而「手臂画满但背面整片没画」的经典又会被误判成 slim。加上判据①之后，
> > 5 万条实测 classic 92.65% / unknown 7.08% / slim 0.00%。
>
> **重要前提**：骨架类型本质上是**角色档案属性**（profile 里的 `model: slim`），
> 不是纹理自带的信息。一张画满 16 宽手臂的纹理既能给经典模型用，也能给纤细模型用。
> 所以本项目能保证的是「这张纹理**与经典骨架的 UV 占用一致**」，
> 不能反推出作者当初选了哪个模型。实测该数据集里瘦骨架纹理占比 ≈0。

## 第 3 层 · 聚类风格族（全量 · 半自动）

| 字段 | 含义 |
|---|---|
| `style_group.cluster_id` | KMeans 族编号 |
| `style_group.cluster_name` | 可读族名，**由人工在 `cluster_names.json` 中确认** |
| `style_group.method` | 固定为 `cluster` |
| `style_group.algo` | 算法与参数，如 `KMeans(k=24, seed=42)` |
| `style_group.cluster_size` | 该族样本数 |

> 说明：族名默认是 `cluster_NN` 占位。代表图见 `labels/cluster_reps/` 与
> `logs/cluster_overview.png`。人工命名后重跑本脚本即可回填 `cluster_name`。

## 第 4 层 · 启发式主题 / 风格（部分覆盖 · 需人工抽检）

| 字段 | 含义 |
|---|---|
| `theme.label` | 主题猜测：`solid_minimal` / `humanoid` / `monochrome_graphic` / `layered_costume` / `complex_character` / `simple_character` / `unknown` |
| `theme.method` | 固定为 `heuristic` |
| `theme.confidence` | 0-1，**这是启发式置信度，不是概率校准值** |
| `theme.evidence` | 触发该标签的具体证据（字段名与阈值） |
| `style.label` | 风格猜测：`monochrome` / `neon` / `dark` / `pastel` / `balanced` |

## 质量与可用性

| 字段 | 含义 |
|---|---|
| `quality.tier` | `good` / `ok` / `simple` / `reject`。`reject` 的样本**不进数据集** |
| `quality.tier_reason` | 触发该档位的具体规则与数值 |
| `quality.dom_ratio` | 最高频颜色占可见像素的比例（是否大面积纯色块） |
| `quality.edge_density` | 可见区内相邻像素颜色变化率（是否有细节） |
| `quality.color_entropy` | 可见像素颜色的信息熵 |
| `quality.is_noise` | 是否明显噪声/近似空白 |
| `quality.is_near_blank` | 面部填充率过低 |
| `quality.dup_group` | LSH 分桶键，同键为近似重复候选 |
| `quality.review_flag` | 人工复核备注（默认 `null`） |

> **关于「质量低下 / 颜色特别简陋」的淘汰口径**（`scripts/lib/skinuv.py`）：
>
> | 规则 | 阈值 | 判定 |
> |---|---|---|
> | 可见像素唯一色数 | `< 8` | `reject`（涂色块） |
> | 单色占比 且 颜色数 | `> 0.72` 且 `< 24` | `reject`（大面积纯色板） |
> | 边缘密度 | `< 0.015` | `reject`（没有任何细节） |
> | 颜色数 且 边缘密度 | `< 14` 且 `< 0.06` | `simple`（简陋，默认剔除，可用 `--quality simple` 收紧） |
> | 颜色数 且 边缘密度 | `>= 24` 且 `>= 0.15` | `good` |
>
> 阈值是在正式数据的分布上校准出来的（见 `logs/ingest_hf.log` 的分档统计），
> 不是拍脑袋定的。

---

## 诚实说明

- 第 1、2 层是**确定性规则**，可被 `scripts/lib/labeling.py` 与 `scripts/lib/skinuv.py`
  里的函数完全重算，不存在模型猜测。
- 第 3 层是**无监督聚类**，族语义由人来命名；脚本不会替人下断言。
- 第 4 层是**启发式**（阈值 + 结构判据），不是训练过的分类器。
  它给的是「大概像什么」，用于分层抽样与数据筛选，**不能当作 ground truth**。
  需要更可靠的语义标签时，应在这一层之上再做人工抽检校正。
"""
    os.makedirs(LABELS, exist_ok=True)
    with open(os.path.join(LABELS, "label_schema.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    log("label_schema.md 已生成")


def cmd_report() -> None:
    recs = load_annotations()
    total = len(recs)
    if not total:
        log("还没有标注结果")
        return
    cov = {
        "total": total,
        "has_base": sum(1 for r in recs.values() if "uv" in r),
        "has_rules": sum(1 for r in recs.values() if "palette" in r and r.get("palette")),
        "has_cluster": sum(1 for r in recs.values() if "style_group" in r),
        "has_semantic": sum(1 for r in recs.values() if "theme" in r),
        "has_caption": sum(1 for r in recs.values() if "caption" in r),
    }
    dist = {
        "model_type": dict(Counter((r.get("model") or {}).get("type") for r in recs.values())),
        "quality_tier": dict(Counter((r.get("quality") or {}).get("tier") for r in recs.values())),
        "tone": dict(Counter((r.get("tone") or {}).get("tone") for r in recs.values())),
        "saturation": dict(Counter((r.get("tone") or {}).get("saturation_class") for r in recs.values())),
        "complexity_class": dict(Counter((r.get("complexity") or {}).get("class") for r in recs.values())),
        "overlay_used": dict(Counter((r.get("transparency") or {}).get("overlay_used") for r in recs.values())),
        "theme": dict(Counter((r.get("theme") or {}).get("label") for r in recs.values())),
        "style": dict(Counter((r.get("style") or {}).get("label") for r in recs.values())),
        "cluster": dict(Counter((r.get("style_group") or {}).get("cluster_id") for r in recs.values())),
    }
    rep = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "coverage": cov, "coverage_pct": {k: round(v / total, 4) for k, v in cov.items() if k != "total"},
           "distributions": dist}
    with open(os.path.join(LOGS, "label_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)

    md = ["# 阶段三 · 标注覆盖率与分布", "", f"样本总数：**{total}**", "",
          "## 覆盖率", "", "| 层 | 已标注 | 覆盖率 |", "|---|---|---|"]
    for k in ("has_base", "has_rules", "has_cluster", "has_semantic", "has_caption"):
        md.append(f"| {k} | {cov[k]} | {cov[k]/total:.1%} |")
    md += ["", "> `has_caption` 是唯一来自**外部给定**的语义层"
           "（HF 描述集自带的 caption 文本，由上游 caption 模型产出），"
           "其余三层都是本项目用规则/聚类/启发式算出来的。"]
    for name, d in dist.items():
        md += ["", f"## {name} 分布", "", "| 取值 | 数量 |", "|---|---|"]
        for k, v in sorted(d.items(), key=lambda x: -(x[1] or 0)):
            md.append(f"| {k} | {v} |")
    with open(os.path.join(LOGS, "label_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    log(json.dumps(cov, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("base"); p1.add_argument("--force", action="store_true"); p1.add_argument("--limit", type=int)
    p2 = sub.add_parser("cluster"); p2.add_argument("--k", type=int, default=24); p2.add_argument("--force", action="store_true")
    p3 = sub.add_parser("semantic"); p3.add_argument("--force", action="store_true")
    p4 = sub.add_parser("captions"); p4.add_argument("--force", action="store_true")
    sub.add_parser("schema")
    sub.add_parser("report")
    a = ap.parse_args()
    os.makedirs(LABELS, exist_ok=True)
    if a.cmd == "base":
        layer_base(a.force, a.limit)
    elif a.cmd == "cluster":
        layer_cluster(a.k, a.force)
    elif a.cmd == "semantic":
        layer_semantic(a.force)
    elif a.cmd == "captions":
        layer_captions(a.force)
    elif a.cmd == "schema":
        cmd_schema()
    elif a.cmd == "report":
        cmd_report()


if __name__ == "__main__":
    main()
