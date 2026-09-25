"""server.py — drmage 临时推理平台（纯标准库，零第三方服务端依赖）。

和 `webui/server.py`（训练监控）**完全独立**：不同文件、不同端口（8850）、
不同进程。可以同时开，互不影响；也不要互相 kill。

接口
----
    GET  /                      实验台页面
    GET  /static/*              静态资源
    GET  /api/meta              模型/权重列表/条件目录/3D 几何/选项（一次拿全）
    POST /api/generate          起一次生成（异步任务）→ {job}
    GET  /api/job?id=           查任务状态与结果
    GET  /api/img/<key>         取 64×64 RGBA PNG
    GET  /api/real?n=&seed=     从真实 val 抽 n 张（对照用）
    GET  /api/export?keys=&layer=  把选中的若干张打成 zip
                                    layer=base → 第二层整片置透明；缺省=全部两层
    GET  /api/history           最近的生成记录
    POST /api/reload            重新载入权重（换了 ckpt / 文件被覆盖时用）

安全边界
--------
* 只监听 127.0.0.1
* 只读项目数据；唯一写入是 `webui/infer/` 下的 history.jsonl、
  exports/（导出 zip 留档）与内存缓存
* 生成前检查：训练在跑 → 拒绝（8GB 卡上采样 1.2GB + 训练 4.6GB 必 OOM）；
  可用显存低于安全线 → 拒绝
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEBUI = os.path.dirname(HERE)
ROOT = os.path.dirname(WEBUI)
for _p in (HERE, WEBUI):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine as E  # noqa: E402
import geom  # noqa: E402

STATIC = os.path.join(HERE, "static")
SERVER_LOG = os.path.join(HERE, "server.log")

DEFAULT_PORT = 8850
IMG_CACHE_MAX = 800
KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

_LOCK = threading.RLock()
_IMGS: "dict[str, bytes]" = {}
_IMGS_ORDER: list[str] = []
_JOBS: dict[str, dict] = {}
_JOBS_ORDER: list[str] = []
_GEN_BUSY = False


def log(msg: str, level: str = "info") -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level.upper():<7}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(SERVER_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 图片缓存
# ---------------------------------------------------------------------------

def _put_img(rgba) -> str:
    from PIL import Image
    key = uuid.uuid4().hex[:16]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    with _LOCK:
        _IMGS[key] = buf.getvalue()
        _IMGS_ORDER.append(key)
        while len(_IMGS_ORDER) > IMG_CACHE_MAX:
            old = _IMGS_ORDER.pop(0)
            _IMGS.pop(old, None)
    return key


def _img(key: str) -> bytes | None:
    with _LOCK:
        return _IMGS.get(key)


# ---------------------------------------------------------------------------
# 训练占用检查
# ---------------------------------------------------------------------------

def training_state() -> dict:
    """读训练监控那边的心跳（只读，不干扰）。"""
    try:
        import collect as C
        st = C.run_state(None)
        return {"status": st.get("status"), "status_cn": st.get("status_cn"),
                "tag": st.get("tag"), "step": st.get("step"),
                "epoch": st.get("epoch"), "pid": st.get("pid")}
    except Exception as exc:                      # noqa: BLE001
        return {"status": "unknown", "error": f"{type(exc).__name__}: {exc}"}


def guard(force: bool = False) -> str | None:
    """返回拒绝原因；None = 可以生成。"""
    if _GEN_BUSY:
        return "已有一个生成任务在跑，等它结束（或刷新页面看结果）。"
    tr = training_state()
    if tr.get("status") == "running" and not force:
        return (f"训练正在跑（{tr.get('tag')} · epoch {tr.get('epoch')}），"
                "已拒绝生成：采样要 ~1.2GB 显存，与训练叠加会 OOM。"
                "先停训练，或用 force 强制（风险自负）。")
    info = E.vram()
    if info.get("cuda") and info["free_gb"] < E.MIN_FREE_VRAM_GB and not force:
        return (f"可用显存只有 {info['free_gb']}GB（安全线 {E.MIN_FREE_VRAM_GB}GB），"
                "已拒绝。关掉占显存的程序，或用 force 强制。")
    return None


# ---------------------------------------------------------------------------
# 条件目录（给前端生成控件，避免前端硬编码 36 维向量结构）
# ---------------------------------------------------------------------------

def cond_catalog() -> dict:
    from labelset import CPX_ORDER, MODEL_ORDER, SAT_ORDER, SPEC_DEFAULTS, TONE_ORDER
    return {
        "dim": 36,
        "defaults": SPEC_DEFAULTS,
        "tone": TONE_ORDER,
        "saturation_class": SAT_ORDER,
        "complexity_class": CPX_ORDER,
        "model_type": MODEL_ORDER,
        "trained_model_type": "classic",
        "model_type_note": "训练集 100% 是 classic（Steve，4px 手臂），"
                           "slim 属外推、本版不支持——UI 里已锁死。",
        "dead_knobs": [31, 32],
        "dead_knobs_note": "第 31/32 维（overlay_used / transparent_ratio）在当前架构下是"
                           "死旋钮：alpha 已改由 UV 模板生成，模型不产出 alpha。",
        "constant_dims": [26, 33, 34, 35],
    }


def quant_opts() -> list[dict]:
    out = [{"value": "off", "label": "不量化（模型原样输出）",
            "note": "保留模型的连续输出，细节最多但色数会到 1000+"}]
    pal_dir = E.MODELS
    for fn in sorted(os.listdir(pal_dir)):
        m = re.match(r"^palette_k(\d+)\.npy$", fn)
        if m:
            out.append({"value": "global", "k": int(m.group(1)),
                        "palette": os.path.join(pal_dir, fn),
                        "label": f"全局调色板 K={m.group(1)}",
                        "note": "真实可见像素的全局 k-means。实测**过冲且偏灰**——"
                                "饱和 0.298 vs 真实 0.353，单张常落到 26 色（真实中位 66）。"
                                "留给 A/B 对照，不建议日常用"})
    out.append({"value": "adaptive", "k": 64, "label": "自适应 K=64（推荐）",
                "note": "每张图按自己的配色做 k-means，保留本图色调。"
                        "实测最贴真实：色数 64 对真实 66、精确相邻相等率 0.497 对 0.475、"
                        "饱和 0.356 对 0.353"})
    out.append({"value": "adaptive", "k": 48, "label": "自适应 K=48（略硬一点）",
                "note": "精确相邻相等率 0.522（真实 0.475）、色数 48——比 K=64 更「平涂」"})
    out.append({"value": "adaptive", "k": 32, "label": "自适应 K=32（偏硬）",
                "note": "色数 32，已明显低于真实中位 66，细节开始被抹掉"})
    out.append({"value": "adaptive", "k": 16, "label": "自适应 K=16（过度平涂）",
                "note": "实测过冲：色数 16 对真实 66（**只剩四分之一**），"
                        "精确相等率 0.633（比真实高 33%）。整片糊成色板，细节会被抹掉"})
    return out


# ---------------------------------------------------------------------------
# 生成任务
# ---------------------------------------------------------------------------

def start_job(req: dict) -> dict:
    global _GEN_BUSY
    # 抽奖参数在入队前就同步校验：非法参数立刻报错回给前端，
    # 而不是让任务跑起来再在轮询里发现 error（否则看起来像按钮坏了）。
    if ((req.get("cond") or {}).get("mode") == "lottery"):
        _lottery_params(req)
    err = guard(bool(req.get("force")))
    if err:
        return {"error": err, "code": 409}
    with _LOCK:
        if _GEN_BUSY:
            return {"error": "已有任务在跑", "code": 409}
        _GEN_BUSY = True
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "state": "queued", "t0": time.time(), "req": req,
           "items": [], "summary": None, "error": None,
           "cond_info": None, "created": datetime.now().strftime("%m-%d %H:%M:%S")}
    with _LOCK:
        _JOBS[job_id] = job
        _JOBS_ORDER.append(job_id)
        while len(_JOBS_ORDER) > 60:
            old = _JOBS_ORDER.pop(0)
            if _JOBS.get(old, {}).get("state") in ("done", "error"):
                _JOBS.pop(old, None)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job": job_id}


def _hist_record(job_id, req, meta, n, seed, cond_mode, cinfo, quant, out, seconds):
    """把一次生成记进 history.jsonl。

    异步任务（``_run_job``）和探针用的同步接口（``_sync_generate``）**必须**共用
    这一份，否则「哪些生成没被记录」会随代码走样而漂移——历史上就是同步口漏记。
    """
    return {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "job": job_id,
        "ckpt": meta.get("rel"), "step": meta.get("step"),
        "n": n, "seed": seed, "ddim": req.get("ddim"), "eta": req.get("eta"),
        "alpha": req.get("alpha"), "quant": quant, "cond_mode": cond_mode,
        "cond": cinfo, "summary": out["summary"], "seconds": seconds,
    }


def _items_of(out: dict, ref_idx, cinfo: dict) -> list[dict]:
    """把一次采样的结果摊成 items。

    异步口（``_run_job``）和探针用的同步口（``_sync_generate``）**必须**共用这一份。
    两边各写一遍的话，同步口就会少字段——历史上就漏过：它一度只有 ``key/idx/per``，
    没有 ``cond`` / ``real_index``，害得探针拿不到「这一张用的是哪条真实条件」，
    只能去猜。同一个坑在 ``_hist_record`` 上已经踩过一次，不再踩第二次。
    """
    items = []
    for i, arr in enumerate(out["images"]):
        desc = None
        if cinfo.get("mode") == "real" and ref_idx:
            desc = E.describe_vector(E.val_cond()[ref_idx[i]])
        elif cinfo.get("described"):
            desc = cinfo["described"]
        items.append({"key": _put_img(arr), "idx": i,
                      "per": out["per_sample"][i], "cond": desc,
                      "real_index": (ref_idx[i] if ref_idx else None)})
    return items


def _run_job(job: dict) -> None:
    global _GEN_BUSY
    req = job["req"]
    try:
        # 抽奖模式走独立的编排（多批 + 打分 + 挑 top-K），与普通单批分叉
        if ((req.get("cond") or {}).get("mode") == "lottery"):
            _run_lottery(job)
            return
        job["state"] = "loading"
        ckpt = req.get("ckpt") or E.DEFAULT_CKPT
        model, cond_dim, args, meta = E.load(ckpt)
        job["model"] = {k: meta[k] for k in
                        ("rel", "params_m", "in_ch", "schedule", "step",
                         "trained_model_type", "mtime_cn")}

        job["state"] = "sampling"
        n = int(req.get("n", 8))
        seed = int(req.get("seed", 1234))
        cond_mode = (req.get("cond") or {}).get("mode", "spec")
        spec = (req.get("cond") or {}).get("spec")
        ref = (req.get("cond") or {}).get("ref_index")
        cond, ref_idx, cinfo = E.gather_cond(cond_mode, n=n, seed=seed,
                                             spec=spec, ref_index=ref,
                                             tone=(req.get("cond") or {}).get("tone"))
        # real 模式筛了 tone 之后行数可能少于请求的 n（真实里那个色调本来就少）。
        # 必须以**实际条数**为准：继续用请求的 n 会让 model.sample 收到行数不足的
        # cond，形状对不上（这个坑在 gather_cond 的 docstring 里也标了）。
        n = int(cond.shape[0])
        if n == 0:
            raise RuntimeError(f"这个色调（tone={cinfo.get('tone')}）在真实样本里一条都没有，"
                               "换个色调或换 seed 再试")

        quant = dict(req.get("quant") or {})
        heal = bool(req.get("heal", True))
        # 选项是「选项表里的一项」，所以按 label 反查参数比较可靠
        out = E.generate(cond, n=n, ddim_steps=int(req.get("ddim", 100)),
                         eta=float(req.get("eta", 0.0)), seed=seed,
                         alpha_mode=req.get("alpha", "template"),
                         quant=quant, ckpt=ckpt, heal=heal)

        job["items"] = _items_of(out, ref_idx, cinfo)
        job["summary"] = out["summary"]
        job["cond_info"] = {k: v for k, v in cinfo.items()}
        job["state"] = "done"
        job["seconds"] = round(time.time() - job["t0"], 2)
        log(f"生成完成 job={job['id']} n={n} ckpt={meta['rel']} "
            f"ddim={req.get('ddim')} alpha={req.get('alpha')} "
            f"quant={quant.get('mode')}/{quant.get('k')} → {job['seconds']}s")
        E.append_history(_hist_record(job["id"], req, meta, n, seed, cond_mode,
                                      cinfo, quant, out, job["seconds"]))
    except Exception as exc:                      # noqa: BLE001
        job["state"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        log(f"生成失败 job={job['id']}：{job['error']}", "error")
        log(traceback.format_exc(limit=5), "error")
    finally:
        global_busy_reset()


def global_busy_reset() -> None:
    global _GEN_BUSY
    with _LOCK:
        _GEN_BUSY = False


# ---------------------------------------------------------------------------
# 抽奖模式（lottery）：抽 N 条真实条件 → 逐条生成 → 评分 → 输出 top-K
# ---------------------------------------------------------------------------

LOT_POOL_RANGE = (16, 128)
LOT_BATCH_RANGE = (16, 128)

#: 耗时估算常数（秒/张）：DDIM 每步耗时 + 单张固定开销（量化/修补/拷贝）。
#: 初值来自 2080S + fp16 autocast 的实测，之后每次抽奖结束用真实读数修正
#: （指数滑动，见 ``_update_estimate``）。
_EST = {"sec_per_img_per_step": 0.0035, "overhead_per_img": 0.035}


def _update_estimate(per_img_s: float, ddim: int) -> None:
    if per_img_s <= 0 or ddim <= 0:
        return
    a = 0.3
    spps = max(0.0, (per_img_s - _EST["overhead_per_img"]) / ddim)
    _EST["sec_per_img_per_step"] = (1 - a) * _EST["sec_per_img_per_step"] + a * spps


def estimate_seconds(n_pool: int, ddim: int, loaded: bool) -> dict:
    per_img = ddim * _EST["sec_per_img_per_step"] + _EST["overhead_per_img"]
    sec = n_pool * per_img + (0.0 if loaded else 5.0)
    return {"seconds": round(sec, 1),
            "seconds_low": round(sec * 0.7, 1),
            "seconds_high": round(sec * 1.6, 1),
            "per_image": round(per_img, 3),
            "calibrated_from_jobs": _EST.get("jobs", 0)}


def _lottery_params(req: dict) -> tuple[int, int, int]:
    """校验抽奖三参数：抽取数 / 批量在 16~128，最终输出 ≥1 且**严格小于**两者。"""
    c = req.get("cond") or {}
    try:
        n_pool = int(c.get("n_pool"))
        batch = int(c.get("batch"))
        final_k = int(c.get("final_k", 1))
    except (TypeError, ValueError):
        raise ValueError("抽奖参数缺失：n_pool / batch / final_k 必须是整数")
    if not (LOT_POOL_RANGE[0] <= n_pool <= LOT_POOL_RANGE[1]):
        raise ValueError(f"真实条件抽取数必须在 {LOT_POOL_RANGE[0]}~{LOT_POOL_RANGE[1]}，当前 {n_pool}")
    if not (LOT_BATCH_RANGE[0] <= batch <= LOT_BATCH_RANGE[1]):
        raise ValueError(f"生成批量必须在 {LOT_BATCH_RANGE[0]}~{LOT_BATCH_RANGE[1]}，当前 {batch}")
    if final_k < 1:
        raise ValueError(f"最终输出最少 1 张，当前 {final_k}")
    if final_k >= min(n_pool, batch):
        raise ValueError(f"最终输出（{final_k}）必须小于抽取数（{n_pool}）和批量（{batch}）")
    return n_pool, batch, final_k


def _run_lottery(job: dict) -> None:
    """抽奖编排。与普通单批生成（``_run_job`` 主体）刻意分开：
    它有候选序号、评分、top-K 三件自己的事，混在一起只会把两边都搅浑。

    结果只依赖 (seed, 候选序号)：初始噪声与 alpha mask 都按候选序号
    **一次性预生成**，再切成微批喂 GPU——所以「批量」只影响显存占用与
    进度上报的粒度，绝不影响抽出来的结果（测试有专门验证）。
    """
    global _GEN_BUSY
    req = job["req"]
    try:
        job["state"] = "loading"
        ckpt = req.get("ckpt") or E.DEFAULT_CKPT
        _m, _cd, args, meta = E.load(ckpt)
        job["model"] = {k: meta[k] for k in
                        ("rel", "params_m", "in_ch", "schedule", "step",
                         "trained_model_type", "mtime_cn")}

        job["state"] = "sampling"
        seed = int(req.get("seed", 1234))
        ddim = int(req.get("ddim", 50))
        eta = float(req.get("eta", 0.0))
        n_pool, batch, final_k = _lottery_params(req)
        tone = (req.get("cond") or {}).get("tone")
        quant = dict(req.get("quant") or {})
        heal = bool(req.get("heal", True))
        alpha_mode = req.get("alpha", "retrieval")

        # 候选 = val 真实条件（可按色调筛）。筛完不足时按实际数收紧 final_k。
        cond, _ref, cinfo = E.gather_cond("real", n=n_pool, seed=seed, tone=tone)
        n_act = int(cond.shape[0])
        if n_act == 0:
            raise RuntimeError(f"这个色调（tone={tone}）在真实样本里一条都没有，换色调或换 seed")
        if final_k >= min(n_act, batch):
            final_k = max(1, min(n_act, batch) - 1)
        job["lottery_plan"] = {"n_pool": n_pool, "n_actual": n_act,
                               "batch": batch, "final_k": final_k}

        # 预生成全部候选的初始噪声与 mask（按候选序号，与微批切法无关）
        import torch
        ch = int(args.get("channels", 3) or 3)
        torch.manual_seed(seed)
        x_T_all = torch.randn(n_act, ch, 64, 64)
        a_all = E.make_alpha(n_act, alpha_mode, seed)

        # 内部微批：单次 GPU 批量上限 64（``generate`` 内部还有 n≤64 的钳制），
        # 并按空闲显存再收紧——「批量」参数是逻辑批（进度/上限），显存超了
        # 自动切成更小的微批，结果不变（x_T/alpha 都按序号预生成）。
        micro = max(4, min(batch, 64))
        info = E.vram()
        if info.get("cuda"):
            # 实测口径：16 张 DDIM100 峰值 ≈1.2GB → 每张每步 ≈0.00075GB（fp16 后更低，
            # 这里按 fp32 保守取）。可用显存打七折换算成单批张数上限。
            per_img_gb = 0.00075 * max(ddim, 1)
            vram_cap = max(4, int(info["free_gb"] * 0.7 / max(per_img_gb, 1e-6)))
            micro = min(micro, vram_cap)

        import skin_score
        scores: list = [None] * n_act
        per_all: list = [None] * n_act
        items_img: list = [None] * n_act
        done, t0 = 0, time.time()
        for s0 in range(0, n_act, micro):
            s1 = min(s0 + micro, n_act)
            out = E.generate(cond[s0:s1], n=s1 - s0, ddim_steps=ddim, eta=eta,
                             seed=seed, alpha_mode=alpha_mode, quant=quant,
                             ckpt=ckpt, heal=heal, amp=True,
                             alpha_np=a_all[s0:s1])
            for j in range(s0, s1):
                arr = out["images"][j - s0]
                scores[j] = skin_score.score_image(arr, cond[j])
                items_img[j] = arr
                per_all[j] = dict(out["per_sample"][j - s0])
            done = s1
            job["progress"] = round(done / n_act, 4)
            job["state"] = "sampling"
        # v3：池内排名归一融合（丰富度+连贯度，判别器只做垃圾守门）——
        # 排名必须在**整个池**上算，单张打分没有「相对中庸/相对丰富」的参照
        skin_score.combine_pool(scores)
        for j in range(n_act):
            per_all[j]["score"] = scores[j]["total"]
        total_s = time.time() - t0
        _update_estimate(total_s / max(n_act, 1), ddim)
        _EST["jobs"] = _EST.get("jobs", 0) + 1

        order = sorted(range(n_act), key=lambda i: (-scores[i]["total"], i))
        top = order[:final_k]

        vc = E.val_cond()
        items = []
        for rank, i in enumerate(top, 1):
            items.append({
                "key": _put_img(items_img[i]), "idx": i,
                "per": per_all[i], "real_index": int(i), "origin": "lottery_gen",
                "cond": E.describe_vector(vc[i]),
                "score": scores[i]["total"], "score_parts": scores[i]["parts"],
                "score_stats": scores[i]["stats"], "final_rank": rank,
            })
        for i in order[final_k:]:
            items.append({
                "key": _put_img(items_img[i]), "idx": i,
                "per": per_all[i], "real_index": int(i), "origin": "lottery_gen",
                "cond": E.describe_vector(vc[i]),
                "score": scores[i]["total"], "score_parts": scores[i]["parts"],
                "score_stats": scores[i]["stats"], "final_rank": None,
            })
        totals = [scores[i]["total"] for i in range(n_act)]
        job["items"] = items
        job["lottery"] = {
            "n_pool": n_pool, "n_actual": n_act, "batch": batch,
            "micro_batch": int(micro), "final_k": final_k,
            "score_mean": round(float(np.mean(totals)), 4),
            "score_median": round(float(np.median(totals)), 4),
            "score_top": round(float(max(totals)), 4),
            "score_min": round(float(min(totals)), 4),
            "seconds": round(total_s, 2),
            "per_image_seconds": round(total_s / max(n_act, 1), 3),
            "heal": heal, "amp": True, "tone": tone,
        }
        job["summary"] = {
            "n": n_act, "ddim_steps": ddim, "eta": eta, "seed": seed,
            "alpha_mode": alpha_mode, "quant": quant,
            "seconds": round(total_s, 2),
            "per_image_seconds": round(total_s / max(n_act, 1), 3),
            "unique_colors_mean": round(float(np.mean(
                [p["unique_colors"] for p in per_all])), 1),
        }
        job["cond_info"] = {"mode": "lottery", "tone": tone, "n_pool": n_pool,
                            "n_actual": n_act, "batch": batch, "final_k": final_k}
        job["state"] = "done"
        job["seconds"] = round(time.time() - job["t0"], 2)
        log(f"抽奖完成 job={job['id']} 候选={n_act} micro={micro} "
            f"final_k={final_k} top={totals[top[0]] if top else '—'} "
            f"均值={job['lottery']['score_mean']} → {job['seconds']}s")
        E.append_history({
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "job": job["id"],
            "ckpt": meta.get("rel"), "step": meta.get("step"),
            "mode": "lottery", "n": n_act, "seed": seed, "ddim": ddim,
            "final_k": final_k, "score_top": job["lottery"]["score_top"],
            "score_mean": job["lottery"]["score_mean"],
            "summary": job["summary"], "seconds": job["seconds"],
        })
    except Exception as exc:                      # noqa: BLE001
        job["state"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        log(f"抽奖失败 job={job['id']}：{job['error']}", "error")
        log(traceback.format_exc(limit=5), "error")
    finally:
        global_busy_reset()


def real_items(n: int, seed: int, tone: str | None) -> dict:
    idx = E.real_index(n, seed, tone)
    vc = E.val_cond()
    skins = E.val_skins()
    items = []
    for i in idx:
        arr = skins[i].transpose(1, 2, 0)
        key = _put_img(arr)
        items.append({"key": key, "real_index": int(i),
                      "cond": E.describe_vector(vc[i])})
    return {"items": items, "n": len(items)}


_OV_MASK = None


def overlay_mask():
    """``(64,64)`` bool，True = 该像素落在第二层（overlay）面上。

    面名走 ``geom.overlay_faces()``——**和前端 ``S.overlayFaces`` 同一份来源**。
    所以「预览时算第二层、被隐藏的那些像素」与「导出时被清掉的那些像素」
    永远是同一批；各写一份就会出现「看得见却导不出」或者反过来的错位。
    """
    global _OV_MASK
    if _OV_MASK is None:
        import numpy as np
        from skinatlas import face_index
        fi = face_index()
        m = np.zeros((64, 64), dtype=bool)
        for name in geom.overlay_faces():
            r = fi.get(name)
            if r:
                y0, y1, x0, x1 = r
                m[y0:y1, x0:x1] = True
        _OV_MASK = m
    return _OV_MASK


def _png_base_only(b: bytes) -> bytes:
    """把一张皮肤 PNG 的第二层整片置为透明，返回新的 PNG 字节。

    只清第二层覆盖到的那片矩形，**第一层一个像素都不动**——两层在 UV 图上
    本来就是不相交的矩形（``skinatlas.face_index`` 保证），所以直接按掩码清零
    是精确的，不需要「只清该被第二层遮住的那些像素」这种逻辑
    （那反而是错的：模型画的第一层像素本来就该留着）。
    透明像素连 RGB 一起归零，与 ``engine.py::generate`` 出图时的约定一致。
    """
    import numpy as np
    from PIL import Image
    a = np.array(Image.open(io.BytesIO(b)).convert("RGBA"))
    a[overlay_mask()] = 0
    buf = io.BytesIO()
    Image.fromarray(a, "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_README_ALL = (
    "drmage-1.0-flash-preview 生成的 Minecraft 皮肤（64x64 RGBA）\n"
    "包含全部两层：第一层（基础）+ 第二层（帽子/外套等 overlay）。\n"
    "alpha 严格 0/255；直接放进 .minecraft/... 或皮肤站即可。\n"
    "注意：本版模型只训了 classic（Steve）骨架，手臂 4px。\n"
)
_README_BASE = (
    "drmage-1.0-flash-preview 生成的 Minecraft 皮肤（64x64 RGBA）\n"
    "**仅第一层**：第二层（帽子/外套等 overlay）的像素已整片置为透明。\n"
    "alpha 严格 0/255。想连第二层一起要，把导出选项换回「含第二层」。\n"
    "注意：本版模型只训了 classic（Steve）骨架，手臂 4px。\n"
)


def export_zip(keys: list[str], layer: str = "all") -> tuple[bytes, str | None]:
    """打包选中的皮肤。同时在 ``webui/infer/exports/`` 留档一份——
    内嵌浏览器/部分浏览器会拦截下载，磁盘副本保证文件永远拿得到。
    返回 ``(zip 字节, 留档绝对路径或 None)``，路径同时写进响应头。"""
    base_only = (layer == "base")
    buf = io.BytesIO()
    saved = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, k in enumerate(keys):
            b = _img(k)
            if not b:
                continue
            if base_only:
                b = _png_base_only(b)
            z.writestr(f"drmage_{i:02d}_{k}.png", b)
            saved += 1
        z.writestr("README.txt", _README_BASE if base_only else _README_ALL)
    body = buf.getvalue()
    path = None
    if saved:
        try:
            out_dir = os.path.join(HERE, "exports")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{datetime.now():%Y%m%d_%H%M%S}_"
                                + ("base" if base_only else "all")
                                + f"_{saved}.zip")
            with open(path, "wb") as fh:
                fh.write(body)
        except OSError:
            path = None
    return body, path


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "drmage-infer/1.0"

    def _send(self, code, body: bytes, ctype: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str)
                   .encode("utf-8"), "application/json; charset=utf-8")

    def _file(self, path, ctype=None):
        if not os.path.isfile(path):
            self._json({"error": "not found", "path": path}, 404)
            return
        ct = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ct.startswith("text/") or ct.endswith(("javascript", "json", "svg")):
            ct += "; charset=utf-8"
        with open(path, "rb") as fh:
            self._send(200, fh.read(), ct)

    def log_message(self, fmt, *args):
        pass

    # ---- GET ----
    def do_GET(self):                              # noqa: N802
        try:
            self._route_get()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:                   # noqa: BLE001
            log(f"GET {self.path} → {type(exc).__name__}: {exc}", "error")
            log(traceback.format_exc(limit=4), "error")
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            except Exception:
                pass

    def _route_get(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        q = parse_qs(u.query)

        def one(k, d=None):
            return (q.get(k) or [d])[0]

        if path in ("/", "/index.html"):
            self._file(os.path.join(STATIC, "index.html")); return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon"); return
        if path.startswith("/static/"):
            rel = path[len("/static/"):].replace("/", os.sep)
            self._file(os.path.join(STATIC, rel)); return

        if path == "/api/meta":
            meta = E.cached_meta()
            self._json({
                "display_name": E.DISPLAY_NAME,
                "display_note": E.DISPLAY_NOTE,
                "root": ROOT,
                "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ckpts": E.list_ckpts(),
                "default_ckpt": E.DEFAULT_CKPT,
                "loaded": meta,
                "gpu": E.vram(),
                "min_free_vram_gb": E.MIN_FREE_VRAM_GB,
                "training": training_state(),
                "cond": cond_catalog(),
                "quant_opts": quant_opts(),
                "alpha_modes": E.ALPHA_MODES,
                "geometry": geom.model_geometry(),
                "face_index": {n: list(v) for n, v in
                               __import__("skinatlas").face_index().items()},
                "overlay_faces": geom.overlay_faces(),
                "real_pool": int(len(E.val_cond())),
            })
            return
        if path == "/api/job":
            jid = one("id", "")
            with _LOCK:
                j = _JOBS.get(jid)
                j = json.loads(json.dumps(j, default=str)) if j else None
            if not j:
                self._json({"error": "no such job"}, 404); return
            j.pop("req", None)
            self._json(j); return
        if path == "/api/img":
            k = one("key", "")
            if not KEY_RE.match(k):
                self._json({"error": "bad key"}, 400); return
            b = _img(k)
            if b is None:
                self._json({"error": "expired key"}, 404); return
            self._send(200, b, "image/png"); return
        if path == "/api/real":
            n = max(1, min(64, int(one("n", 12))))
            seed = int(one("seed", 0))
            tone = one("tone") or None
            self._json(real_items(n, seed, tone)); return
        if path == "/api/export":
            keys = [k for k in (one("keys", "") or "").split(",") if KEY_RE.match(k)][:200]
            if not keys:
                self._json({"error": "no keys"}, 400); return
            # layer=base → 第二层整片置透明；其余值（含缺省）都是完整的
            layer = "base" if one("layer", "all") == "base" else "all"
            body, saved_path = export_zip(keys, layer)
            name = "drmage_skins_base.zip" if layer == "base" else "drmage_skins.zip"
            extra = {"Content-Disposition": f'attachment; filename="{name}"'}
            if saved_path:
                extra["X-Saved-Path"] = saved_path
            self._send(200, body, "application/zip", extra)
            return
        if path == "/api/history":
            self._json({"items": E.read_history(int(one("limit", 30)))}); return
        if path == "/api/lottery_estimate":
            n_pool = int(one("n_pool", 64))
            ddim = int(one("ddim", 50))
            self._json(estimate_seconds(
                n_pool, ddim, loaded=bool(E.cached_meta())))
            return
        self._json({"error": "not found", "path": path}, 404)

    # ---- POST ----
    def do_POST(self):                             # noqa: N802
        try:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            ln = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(ln) if ln else b"{}"
            try:
                req = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                self._json({"error": f"bad json: {exc}"}, 400); return

            if u.path == "/api/generate":
                r = start_job(req)
                self._json(r, r.pop("code", 200) if "error" in r else 200)
                return
            if u.path == "/api/reload":
                E.load(req.get("ckpt") or E.DEFAULT_CKPT, force=True)
                self._json({"ok": True, "loaded": E.cached_meta()}); return
            if u.path == "/api/generate_sync":
                # 同步版：给探针用（避免探针还要轮询）。批量必须小。
                self._json(_sync_generate(req)); return
            self._json({"error": "not found", "path": u.path}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:                   # noqa: BLE001
            log(f"POST {self.path} → {type(exc).__name__}: {exc}", "error")
            log(traceback.format_exc(limit=4), "error")
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            except Exception:
                pass


def _sync_generate(req: dict) -> dict:
    err = guard(bool(req.get("force")))
    if err:
        return {"error": err}
    global _GEN_BUSY
    with _LOCK:
        _GEN_BUSY = True
    try:
        t0 = time.time()
        seed = int(req.get("seed", 1234))
        n = int(req.get("n", 4))
        cond_mode = (req.get("cond") or {}).get("mode", "spec")
        ckpt = req.get("ckpt") or E.DEFAULT_CKPT
        # **先载入权重再构造条件**：gather_cond 要靠模型的 cond_dim 判断是否把
        # 4 个部位级 overlay 位拼进去（40 维 vs 36 维）。顺序反过来会在第一次
        # 请求时按 36 维构造，然后喂给要 40 维的模型（形状对不上）。
        meta = E.load(ckpt)[3]
        cond, ref_idx, cinfo = E.gather_cond(
            cond_mode, n=n, seed=seed,
            spec=(req.get("cond") or {}).get("spec"),
            ref_index=(req.get("cond") or {}).get("ref_index"),
            tone=(req.get("cond") or {}).get("tone"))
        # 同 _run_job：以实际条数为准（real + tone 筛选后可能变少）
        n = int(cond.shape[0])
        if n == 0:
            raise RuntimeError(f"这个色调（tone={cinfo.get('tone')}）在真实样本里一条都没有")
        quant = dict(req.get("quant") or {})
        out = E.generate(cond, n=n, ddim_steps=int(req.get("ddim", 20)),
                         eta=float(req.get("eta", 0.0)), seed=seed,
                         alpha_mode=req.get("alpha", "template"),
                         quant=quant, ckpt=ckpt)
        items = _items_of(out, ref_idx, cinfo)
        seconds = round(time.time() - t0, 2)
        E.append_history(_hist_record(f"sync-{int(time.time() * 1000)}", req, meta,
                                      n, seed, cond_mode, cinfo, quant, out, seconds))
        return {"items": items, "summary": out["summary"], "cond_info": cinfo}
    finally:
        with _LOCK:
            _GEN_BUSY = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--preload", action="store_true",
                    help="启动时就把权重载进来（首屏就能出图，代价是启动慢几秒）")
    a = ap.parse_args()

    log(f"drmage 推理平台启动 → http://{a.host}:{a.port}  (root={ROOT})")
    log(f"默认权重 {E.DEFAULT_CKPT} | 训练状态 {training_state().get('status')}")
    if a.preload:
        try:
            _m, _cd, _ar, meta = E.load(E.DEFAULT_CKPT)
            log(f"预载完成 {meta['rel']} {meta['params_m']}M "
                f"in_ch={meta['in_ch']} {meta['load_seconds']}s")
        except Exception as exc:                   # noqa: BLE001
            log(f"预载失败（不影响启动，点生成时会重试）：{exc}", "warn")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("收到 Ctrl-C，退出")


if __name__ == "__main__":
    main()
