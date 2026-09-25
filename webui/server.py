"""server.py — 训练监控平台的服务端（纯标准库，零第三方依赖）。

为什么不用 FastAPI/uvicorn
-------------------------
服务端只用 ``http.server`` + ``ThreadingHTTPServer``（发布目标环境不装
Web 框架，保持零第三方依赖），
前端也不用任何 CDN——整个平台是**自包含**的，双击就能跑。

接口一览
--------
    GET  /                      前端页面
    GET  /static/*              静态资源
    GET  /api/bootstrap         一次性：项目信息、图集面坐标、可用 tag、阈值
    GET  /api/overview          训练概要 + 系统概要 + 事件计数
    GET  /api/series?tag=       训练曲线（多序列、可降采样）
    GET  /api/system            系统运行状态 + 内存中的历史环
    GET  /api/artifacts?tag=    样本快照 / 断点 / 皮肤产出 / 生成记录
    GET  /api/reports           项目已有诊断报告（图集标定、颜色审计…）
    GET  /api/logs?tag=&name=   中文结构化日志
    GET  /api/events?since=     监控事件 / 系统消息
    GET  /api/server_log        本服务的运行日志
    GET  /api/stream            SSE：每 2 秒推送一次快照
    POST /api/train/start       启动训练（需 confirm=true）
    POST /api/train/stop        停止训练（需 confirm=true）

安全边界
--------
* 默认只监听 ``127.0.0.1``（要局域网访问得显式 ``--host 0.0.0.0``）
* ``/media/`` 只在白名单目录内取文件，且做路径穿越检查
* 启动训练的参数全部过白名单/范围校验，tag 只允许 ``[A-Za-z0-9_-]``
* 只读接口不修改项目任何文件
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect as C  # noqa: E402

STATIC = os.path.join(C.WEBUI, "static")
SERVER_LOG = os.path.join(C.WEBUI, "server.log")
LAUNCH_LOG = os.path.join(C.WEBUI, "launches.jsonl")

DEFAULT_PORT = 8848
TICK_SECONDS = 2.0
MAX_POINTS = 1200

#: ``/media/`` 允许暴露的目录（相对项目根）
MEDIA_ROOTS = {
    "logs/samples": os.path.join(C.LOGS, "samples"),
    "exports": C.EXPORTS,
    "data/generated": os.path.join(C.DATA, "generated"),
    "logs": C.LOGS,
}

STATE_LOCK = threading.Lock()
STATE: dict = {"overview": {}, "system": {}, "artifacts": {}, "events": {},
               "started_at": time.time(), "ticks": 0, "errors": 0, "last_error": None}
_SSE_CLIENTS: list = []
_SSE_LOCK = threading.Lock()
_LAUNCHED: dict = {}          # tag -> Popen（本服务启动的训练进程）


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
# 后台采集线程
# ---------------------------------------------------------------------------

def _artifacts(tag: str | None) -> dict:
    return C.artifacts_overview(tag)


def build_overview(tag: str | None, system: dict | None = None,
                   art: dict | None = None) -> dict:
    """组装一份概览。``tag`` 可显式指定，用于查看历史实验（不只是当前活跃的）。"""
    system = system if system is not None else C.system_snapshot()
    art = art if art is not None else _artifacts(tag)
    # run_state 读的是逐实验的 metrics/heartbeat，任何一个实验的文件有幺蛾子
    # 都不能把整个概览拖死 —— 早先 metrics.jsonl 为空时 max() 直接抛，
    # 结果是**整个 overview 停止更新**，前端看起来就是「哪一块都没加载」。
    try:
        train = C.run_state(tag)
    except Exception as exc:                      # noqa: BLE001
        log(f"run_state({tag}) 失败：{type(exc).__name__}: {exc}", "error")
        train = {"tag": tag, "status": "unknown", "status_cn": "状态读取失败",
                 "error": f"{type(exc).__name__}: {exc}"}
    return {
        "now": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "project": {"name": C.PROJECT_NAME, "desc": C.PROJECT_DESC, "root": C.ROOT},
        "train": train,
        "system": {k: system.get(k) for k in
                   ("cpu_percent", "memory", "disk_project", "gpus")},
        "artifacts": {"grids": art["grids"], "skins": art["skins"],
                      "latest_grid": art["latest_grid"],
                      "checkpoints": len(art["checkpoints"])},
        "runs": C.list_runs()[:16],
    }


def monitor_loop(engine: C.EventEngine) -> None:
    log("采集线程已启动，采样间隔 %.1f 秒" % TICK_SECONDS)
    last_save = 0.0
    while True:
        t0 = time.time()
        try:
            tag = C.pick_active_run()
            system = C.system_snapshot()
            C.push_history(system)
            art = _artifacts(tag)
            overview = build_overview(tag, system, art)
            engine.tick(overview, system, art)
            with STATE_LOCK:
                STATE["overview"] = overview
                STATE["system"] = system
                STATE["artifacts"] = art
                STATE["events"] = engine.snapshot()
                STATE["ticks"] += 1
                STATE["tag"] = tag
        except Exception as exc:                     # 采集失败不能让线程死掉
            with STATE_LOCK:
                STATE["errors"] += 1
                STATE["last_error"] = f"{type(exc).__name__}: {exc}"
            log(f"采集异常：{type(exc).__name__}: {exc}", "error")
            log(traceback.format_exc(limit=3), "error")

        if time.time() - last_save > 60:
            C.save_history()
            last_save = time.time()

        # 推送 SSE
        payload = None
        with STATE_LOCK:
            payload = {"overview": STATE["overview"],
                       "system_summary": STATE["overview"].get("system"),
                       "artifacts_summary": STATE["overview"].get("artifacts"),
                       "events": STATE["events"],
                       "server": {"uptime_s": round(time.time() - STATE["started_at"], 1),
                                  "ticks": STATE["ticks"], "errors": STATE["errors"]}}
        _broadcast(payload)
        time.sleep(max(0.5, TICK_SECONDS - (time.time() - t0)))


def _broadcast(payload: dict) -> None:
    data = ("data: " + json.dumps(payload, ensure_ascii=False, default=str)
            + "\n\n").encode("utf-8")
    with _SSE_LOCK:
        dead = []
        for q in _SSE_CLIENTS:
            try:
                q.append(data)
            except Exception:
                dead.append(q)
        for d in dead:
            _SSE_CLIENTS.remove(d)


# ---------------------------------------------------------------------------
# 曲线
# ---------------------------------------------------------------------------

def build_series(tag: str, max_points: int = MAX_POINTS) -> dict:
    recs = [m for m in C.read_metrics(tag) if not m.get("event")]
    if not recs:
        return {"tag": tag, "charts": [], "n_records": 0}

    def grab(key):
        out = []
        for m in recs:
            v = m.get(key)
            if v is None or not isinstance(v, (int, float)):
                continue
            out.append([m.get("step", len(out)), round(float(v), 6)])
        return out

    def downsample(pts, k):
        if len(pts) <= k:
            return pts
        step = len(pts) / k
        return [pts[int(i * step)] for i in range(k)] + [pts[-1]]

    def smooth(pts, win=20):
        if len(pts) <= 2:
            return pts
        out = []
        acc, q = 0.0, []
        for x, y in pts:
            q.append(y)
            acc += y
            if len(q) > win:
                acc -= q.pop(0)
            out.append([x, round(acc / len(q), 6)])
        return out

    def first_of(*keys):
        """按优先级取第一个有数据的指标列（跨架构指标名兜底）。"""
        for k in keys:
            pts = grab(k)
            if pts:
                return k, pts
        return None, []

    # 【兼容性】损失字段名随架构变：旧架构 ``mse``，PIDiff ``ce``（另有 ce1/ce2 两级）。
    # 原来写死 ``grab("mse")``：PIDiff 实验打开后「训练损失」整张空白。
    loss_key, loss_pts = first_of("mse", "ce")
    loss_name = {"mse": "MSE", "ce": "交叉熵 CE"}.get(loss_key, "MSE")
    charts = []

    if loss_key:
        loss_series = [
            {"key": loss_key, "label": f"{loss_name}（每 100 步）", "color": "cyan",
             "points": downsample(loss_pts, max_points), "width": 1, "alpha": 0.45},
            {"key": f"{loss_key}_smooth", "label": f"{loss_name}（20 点滑动平均）",
             "color": "violet",
             "points": downsample(smooth(loss_pts, 20), max_points), "width": 2.4},
        ]
        # PIDiff 把两级码本的交叉熵也画上：ce1 = 粗色码本、ce2 = 残差码本。
        # 两条分开看才能判断是「粗色学不会」还是「残差学不会」。
        for sub, lab, col in (("ce1", "粗色码本 CE（ce1）", "emerald"),
                              ("ce2", "残差码本 CE（ce2）", "amber")):
            pts = grab(sub)
            if pts:
                loss_series.append({"key": sub, "label": lab, "color": col,
                                    "points": downsample(smooth(pts, 20), max_points),
                                    "width": 1.6, "dash": True})
        charts.append({
            "id": "loss", "title": f"训练损失 {loss_name}", "x_label": "训练步数",
            "y_label": loss_name, "height": 260,
            "series": loss_series, "refs": [],
        })

    lr = grab("lr")
    if lr:
        charts.append({
            "id": "lr", "title": "学习率", "x_label": "训练步数", "y_label": "LR",
            "height": 180,
            "series": [{"key": "lr", "label": "学习率", "color": "amber",
                        "points": downsample(lr, max_points), "width": 2}],
            "refs": [], "scale": 1e6, "unit": "×10⁻⁶",
        })

    vram = grab("vram_peak_gb")
    if vram:
        charts.append({
            "id": "vram", "title": "峰值显存（GiB）", "x_label": "训练步数",
            "y_label": "GiB", "height": 160, "y_min": 0,
            "series": [{"key": "vram", "label": "峰值显存", "color": "rose",
                        "points": downsample(vram, max_points), "width": 2}],
            "refs": [{"y": 8.0, "label": "显卡总显存 8 GiB", "color": "muted"}],
        })

    # 逐 epoch 汇总（一轮一个点）：损失与每轮耗时
    ep_last: dict[int, dict] = {}
    for m in recs:
        if m.get("epoch") is not None:
            ep_last[m["epoch"]] = m
    # 每轮耗时从 elapsed_s 差分得到
    seq = sorted(ep_last.items())
    ep_secs = []
    for i in range(1, len(seq)):
        d = seq[i][1].get("elapsed_s", 0) - seq[i - 1][1].get("elapsed_s", 0)
        if 0 < d < 3600:
            ep_secs.append([seq[i][0] + 1, round(d, 1)])
    ep_mse = [[e + 1, round(m[loss_key], 6)] for e, m in seq
              if loss_key and m.get(loss_key) is not None]
    if ep_mse:
        charts.append({
            "id": "epoch", "title": "逐轮损失与耗时（按轮次编号，1 起）",
            "x_label": "轮次（epoch）", "y_label": "MSE", "height": 240,
            "series": [
                {"key": "mse_epoch", "label": "每轮末 MSE", "color": "emerald",
                 "points": ep_mse, "width": 2.2, "axis": "left"},
                {"key": "sec_epoch", "label": "每轮耗时（秒）", "color": "amber",
                 "points": ep_secs, "width": 1.6, "axis": "right", "dash": True},
            ],
            "refs": [], "dual_axis": True,
        })

    # 训练内探针（定期采样出的验收值）
    # 【兼容性】探针指标名同样随架构变：
    #   旧架构 → vis_gen_unique_colors / vis_gen_flat_frac / real_unique_colors
    #   PIDiff  → pl_uc（末段硬量化路径的唯一色）/ pl_adj_eq / real_uc / real_adj_eq
    # 写死旧名 → 探针图整张空白。
    probe_key, probe_uc = first_of("vis_gen_unique_colors", "pl_uc", "p_uc")
    probe_ff_key, probe_ff = first_of("vis_gen_flat_frac", "pl_adj_eq", "p_adj_eq")
    is_pidiff_probe = probe_key is not None and probe_key != "vis_gen_unique_colors"
    if probe_uc or probe_ff:
        real_uc = None
        real_ff = None
        for m in recs:
            for _k in ("real_unique_colors", "real_uc"):
                if m.get(_k) is not None:
                    real_uc = m[_k]
                    break
            for _k in ("real_flat_frac", "real_adj_eq"):
                if m.get(_k) is not None:
                    real_ff = m[_k]
                    break
            if real_uc is not None and real_ff is not None:
                break
        charts.append({
            "id": "probe", "title": "生成质量探针（训练中定期采样）",
            "x_label": "训练步数", "y_label": "唯一色数", "height": 230,
            "series": [
                {"key": "uc", "label": "单张唯一色数", "color": "cyan",
                 "points": probe_uc, "width": 2, "axis": "left"},
                {"key": "ff",
                 "label": ("相邻精确相等 adj_eq" if is_pidiff_probe
                           else "近同色占比 flat_frac"),
                 "color": "emerald", "points": probe_ff, "width": 2, "axis": "right"},
            ],
            "refs": ([{"y": real_uc, "label": f"真实唯一色数 {real_uc:.0f}",
                       "color": "muted", "axis": "left"}] if real_uc else []),
            "dual_axis": True,
        })

    # PIDiff 专属：码本诊断。命中率反映「模型有没有学会选码字」，
    # 利用率反映「码本有没有被塌缩到少数几个码字上」——
    # 后者是判断「塑料感 / 只会配少数颜色」的直接读数。
    for _id, _title, _keys, _labels, _cols, _ymax in (
            ("cb_acc", "码本命中率（argmax 猜对训练像素的比例）",
             ("acc1", "acc2"), ("粗色码本 acc1", "残差码本 acc2"),
             ("emerald", "amber"), None),
            ("cb_used", "码本利用率（被用到的码字比例）",
             ("used1", "used2"), ("粗色码本 used1", "残差码本 used2"),
             ("emerald", "amber"), 1.0)):
        _pts = [(k, grab(k)) for k in _keys]
        if not any(p for _, p in _pts):
            continue
        _chart = {"id": _id, "title": _title, "x_label": "训练步数",
                  "y_label": "比例", "height": 200, "y_min": 0,
                  "series": [{"key": k, "label": lb, "color": c,
                              "points": downsample(p, max_points), "width": 2}
                             for (k, p), lb, c in zip(_pts, _labels, _cols) if p],
                  "refs": []}
        if _ymax is not None:
            _chart["y_max"] = _ymax
        charts.append(_chart)

    return {"tag": tag, "n_records": len(recs), "charts": charts,
            "last": recs[-1] if recs else None}


# ---------------------------------------------------------------------------
# 训练控制
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")

#: 解释器 CUDA 检测缓存（``exe -> (ok, 说明)``）
_CUDA_CHECK: dict = {}


def _cuda_ok(exe=None):
    """确认某个解释器能用 CUDA。

    **为什么要查**：机器上常有多个 Python 环境，有的只装了 CPU 版 torch
    （甚至没装 torch）。用错环境起 webui，
    训练会**静默跑在 CPU 上**（慢一个数量级），日志里看不出任何异常 ——
    这正是「点了没反应 / 一晚上才跑两轮」的头号原因。
    """
    exe = exe or sys.executable
    if exe in _CUDA_CHECK:
        return _CUDA_CHECK[exe]
    try:
        r = subprocess.run(
            [exe, "-c", "import torch;print('1' if torch.cuda.is_available() else '0')"],
            capture_output=True, text=True, timeout=90)
        ok = r.stdout.strip().endswith("1")
        why = "" if ok else (r.stdout + r.stderr).strip()[-200:]
    except Exception as exc:                       # noqa: BLE001
        ok, why = False, f"{type(exc).__name__}: {exc}"
    _CUDA_CHECK[exe] = (ok, why)
    return ok, why


def _find_cuda_python():
    """在常见 conda 位置找一个装了 CUDA 版 torch 的解释器（只为给出可复制的建议）。"""
    import glob
    cands = [sys.executable]
    for pat in (os.path.expanduser("~/anaconda3/envs/*/python.exe"),
                os.path.expanduser("~/miniconda3/envs/*/python.exe"),
                os.path.expanduser("~/.conda/envs/*/python.exe")):
        cands += sorted(glob.glob(pat))
    for exe in cands:
        if _cuda_ok(exe)[0]:
            return exe
    return None

#: **训练预设** —— 发布权重 ``diff_v2_masked`` 的两阶段官方配方（见 docs/REPRODUCE.md）。
#:
#: 阶段一 ``pretrain``：无掩码损失的基础 DDPM（历史配置）。它单独训出来的 alpha
#: 通道有「overlay 灌黑」缺陷，**只作为阶段二的初始化存在**。
#: 阶段二 ``masked``：从阶段一断点续训 + 掩码损失，即发布权重本身。
#: 旧仓库里还有更多实验性预设（alpha 条件平面 / 部位位 / CFG 等），
#: 它们不属于本发布，已裁掉——本发布只复现 diff_v2_masked。
TRAIN_PRESETS: dict = {
    "masked": {
        "label": "阶段二 · diff_v2_masked 官方配方（推荐）",
        "desc": "掩码损失（只在可见像素上算）+ 与发布权重完全一致的其余配置。"
                "标准用法：tag 填 diff_v2_masked，断点填 models/diff_v1/latest.pt 续训"
                "（两阶段链条见 docs/REPRODUCE.md）",
        "args": [],
        "tag_hint": "diff_v2_masked",
        "needs": [],
    },
    "pretrain": {
        "label": "阶段一 · 预训练（diff_v1 配方）",
        "desc": "无掩码损失的基础 DDPM。alpha 由模型自己回归，overlay 区域会被"
                "学成暗色（历史已确认的缺陷），所以它只用来给阶段二提供初始化",
        "args": ["--no-mask-loss"],
        "tag_hint": "diff_v1",
        "needs": [],
    },
}


def _clamp(v, lo, hi, default):
    try:
        v = type(default)(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def start_training(params: dict) -> dict:
    tag = str(params.get("tag") or "").strip()
    if not _TAG_RE.match(tag):
        return {"ok": False, "error": "tag 只能包含字母、数字、下划线、短横线（1~40 字符）"}

    # 训练用的是「启动本服务的那个解释器」的 sys.executable。它没有 CUDA 时
    # 训练会**静默跑在 CPU 上**（慢一个数量级），所以直接拦下并给出正确命令，
    # 而不是让用户等一晚上才发现只跑了两轮。
    cuda_ok, why = _cuda_ok()
    if not cuda_ok:
        alt = _find_cuda_python()
        hint = (f"\n本机可用的解释器：{alt}" if alt
                else "\n没在常见 conda 位置找到可用的 CUDA 解释器。")
        return {"ok": False, "error":
                f"当前 WebUI 用的解释器没有可用的 CUDA：\n  {sys.executable}\n"
                f"  原因：{why or 'torch.cuda.is_available() 为 False'}\n"
                f"这样启动训练会跑在 CPU 上（慢一个数量级）。{hint}\n"
                "请用它重新启动本服务，例如：\n"
                f'  "{alt or sys.executable}" webui\\server.py --preload'}

    cur = C.pick_active_run()
    st = C.run_state(cur)
    if st.get("status") == "running":
        return {"ok": False,
                "error": f"已有训练在跑（{cur}，PID {st.get('pid')}），请先停止"}

    epochs = _clamp(params.get("epochs", 200), 1, 5000, 200)
    batch = _clamp(params.get("batch", 48), 1, 512, 48)
    base = _clamp(params.get("base", 64), 8, 512, 64)
    schedule = str(params.get("schedule") or "cosine")
    if schedule not in ("cosine", "linear"):
        schedule = "cosine"
    channels = _clamp(params.get("channels", 3), 3, 4, 3)
    model_type = str(params.get("model_type") or "classic")
    if model_type not in ("all", "classic", "slim"):
        model_type = "classic"
    lr = _clamp(params.get("lr", 2e-4), 1e-6, 1e-1, 2e-4)

    resume = params.get("resume") or ""
    if resume:
        p = os.path.abspath(os.path.join(C.ROOT, resume))
        # 只允许续训项目内 models/ 下的断点，防止任意路径
        if not p.startswith(os.path.abspath(os.path.join(C.ROOT, "models"))) \
                or not os.path.isfile(p):
            return {"ok": False, "error": f"续训断点不存在或不在 models/ 下：{resume}"}

    # ---- 预设：一组固化的「正确开关组合」（见 TRAIN_PRESETS）----
    preset = str(params.get("preset") or "").strip()
    preset_args: list[str] = []
    if preset:
        pr = TRAIN_PRESETS.get(preset)
        if pr is None:
            return {"ok": False, "error": f"未知预设：{preset}"}
        for rel in pr.get("needs", []):
            if not os.path.isfile(os.path.join(C.ROOT, rel)):
                hint = ("python scripts/24_build_ov_bits.py" if "ov4" in rel
                        else "python scripts/25_build_mask_bank.py")
                return {"ok": False,
                        "error": f"预设「{pr['label']}」需要 {rel}，但它不存在。"
                                 f"先在项目根跑：{hint}"}
        if "--alpha-input" in pr["args"] and channels != 3:
            return {"ok": False,
                    "error": f"预设「{pr['label']}」要求 channels=3"
                             "（RGB 目标 + 模板 alpha 作条件平面），当前是 "
                             f"{channels}"}
        preset_args = list(pr["args"])
        log(f"训练预设 {preset}（{pr['label']}）：{' '.join(preset_args) or '(无额外开关)'}")

    cmd = [sys.executable, os.path.join("scripts", "21_train_diffusion.py"),
           "--tag", tag, "--epochs", str(epochs), "--batch", str(batch),
           "--base", str(base), "--schedule", schedule,
           "--channels", str(channels), "--lr", str(lr),
           "--model-type", model_type, "--cond", "--mask-loss",
           "--sample-every", str(_clamp(params.get("sample_every", 2200), 50, 100000, 2200)),
           "--ddim-steps", str(_clamp(params.get("ddim_steps", 50), 5, 500, 50)),
           "--ema-decay", str(_clamp(params.get("ema_decay", 0.9995), 0.9, 0.99999, 0.9995))]
    cmd += preset_args
    if resume:
        cmd += ["--resume", os.path.relpath(p, C.ROOT)]
    # 两个预设与发布权重**结构完全一致**（in_ch=3、cond 36 维、无 FiLM），
    # 所以 ``--resume`` 是合法且被期待的：阶段二（masked）就是从阶段一断点续训来的。
    # 旧仓库里「新预设禁续训」的守卫针对的是改结构的实验预设，本发布已裁掉。
    _ = preset  # 预设只影响额外参数，不再有结构分支

    max_steps = params.get("max_steps")
    if max_steps not in (None, "", 0, "0"):
        cmd += ["--max-steps", str(_clamp(max_steps, 1, 200000, 100))]

    out_dir = os.path.join(C.LOGS, tag)
    os.makedirs(out_dir, exist_ok=True)
    console = open(os.path.join(out_dir, "console.log"), "a", encoding="utf-8")
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONIOENCODING"] = "utf-8"

    flags = 0
    if hasattr(subprocess, "DETACHED_PROCESS"):
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    # 【关键】如果本服务自己是被某个「作业对象（Job Object）」拉起来的
    # （例如从 IDE / 自动化会话里启动），子进程默认会**继承这个 job**，
    # 宿主一退出就整棵树被杀 —— 实测训练只活了 120 秒就被连坐干掉。
    # CREATE_BREAKAWAY_FROM_JOB 让子进程脱离父 job，真正做到独立存活。
    if hasattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB"):
        flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
    try:
        proc = subprocess.Popen(cmd, cwd=C.ROOT, env=env,
                                stdout=console, stderr=subprocess.STDOUT,
                                creationflags=flags)
    except OSError as exc:
        return {"ok": False, "error": f"启动失败：{exc}"}
    except Exception as exc:                       # 脱离 job 失败时不能放弃启动
        log(f"带 breakaway 启动失败（{exc}），退回普通后台启动", "warn")
        try:
            proc = subprocess.Popen(cmd, cwd=C.ROOT, env=env, stdout=console,
                                    stderr=subprocess.STDOUT,
                                    creationflags=subprocess.DETACHED_PROCESS
                                    | subprocess.CREATE_NEW_PROCESS_GROUP)
        except OSError as exc2:
            return {"ok": False, "error": f"启动失败：{exc2}"}
    _LAUNCHED[tag] = proc
    # **立刻补一个心跳文件**：训练进程要几十秒后才写完第一次心跳（数据载入 +
    # 第一个 100 步），在那之前监控面板会把它当成「不存在」。这里用 Popen 的
    # pid 先占位，训练进程随后会用同一个 pid 覆盖它。
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "heartbeat.json"), "w", encoding="utf-8") as fh:
            json.dump({"pid": proc.pid, "ts": time.time(), "alive_s": 0.0,
                       "finished": False, "tag": tag, "epochs": epochs,
                       "step": 0, "epoch_idx": 0, "phase": "launching",
                       "launched_by": "webui"}, fh, ensure_ascii=False)
    except OSError as exc:
        log(f"写初始心跳失败：{exc}", "warn")
    try:
        with open(LAUNCH_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(),
                                 "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "tag": tag, "pid": proc.pid, "cmd": cmd,
                                 "resume": resume or None, "epochs": epochs,
                                 "batch": batch, "base": base}, ensure_ascii=False) + "\n")
    except OSError:
        pass
    log(f"已启动训练：tag={tag} PID={proc.pid} epochs={epochs} batch={batch} "
        f"base={base} resume={resume or '从头'}")
    return {"ok": True, "pid": proc.pid, "tag": tag, "cmd": " ".join(cmd)}


#: 正在进行的「按当前断点出图」任务：run_name -> Popen（防止重复点按钮叠加任务）
_GEN_JOBS: dict = {}


def generate_skins(params: dict) -> dict:
    """用某个实验的最新断点出一批原始皮肤（``data/generated/<tag>_stepNNNNNN/``）。

    为什么放在面板里：v2 这种刚续训的实验在 ``exports/`` 里**什么都没有**，
    用户想看"v2 现在长什么样"只能靠现出 —— 这正是「最新皮肤」的来源。
    """
    tag = str(params.get("tag") or "").strip()
    if not _TAG_RE.match(tag):
        return {"ok": False, "error": "tag 只能包含字母、数字、下划线、短横线"}
    ckpt = os.path.join(C.MODELS, tag, "latest.pt")
    if not os.path.isfile(ckpt):
        return {"ok": False, "error": f"没有可用断点：models/{tag}/latest.pt 还不存在"}

    # 生成约吃 4.6GB 显存（batch 16），训练占着 4.6GB 时叠上去必 OOM —— 直接拒绝并说清楚
    st = C.run_state(C.pick_active_run())
    if st.get("status") == "running":
        return {"ok": False,
                "error": f"训练正在跑（{st.get('tag')}，显存已用 {st.get('vram_gb')}GB）。"
                         f"先停止训练再出图，否则会 OOM"}

    n = _clamp(params.get("n", 16), 1, 256, 16)
    ddim = _clamp(params.get("ddim_steps", 50), 5, 500, 50)
    step = C.latest_ckpt_step(tag)
    run_name = f"{tag}_step{step:06d}"
    outdir = os.path.join(C.DATA, "generated", run_name)
    # 同名批次已经有图了 → 追加时间戳，绝不覆盖旧批次（旧的就是"上一步"的记录）
    if os.path.isdir(outdir) and any(f.endswith(".png") for f in os.listdir(outdir)):
        run_name += "_" + datetime.now().strftime("%H%M%S")
    # 防重复：同一实验已有一批在生成中就不叠加
    for rn, p in list(_GEN_JOBS.items()):
        if rn.startswith(tag + "_") and p.poll() is None:
            return {"ok": False, "error": f"这个实验已有一批在生成中（{rn}），等它出完再说"}

    cmd = [sys.executable, os.path.join("scripts", "30_generate.py"),
           "--model", "diffusion", "--ckpt", os.path.relpath(ckpt, C.ROOT),
           "--n", str(n), "--batch", "16", "--ddim-steps", str(ddim),
           "--out", run_name, "--seed", str(_clamp(params.get("seed", 1234), 0, 2**31 - 1, 1234))]
    cfg = C.read_config(tag) or {}
    if cfg.get("model_type"):
        cmd += ["--model-type", str(cfg["model_type"])]

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    flags = 0
    if hasattr(subprocess, "DETACHED_PROCESS"):
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    if hasattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB"):   # 同上：脱离父 job，别被连坐
        flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
    logf = open(os.path.join(C.LOGS, "gen_" + run_name + ".log"), "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=C.ROOT, env=env, stdout=logf,
                                stderr=subprocess.STDOUT, creationflags=flags)
    except OSError as exc:
        return {"ok": False, "error": f"启动出图失败：{exc}"}
    _GEN_JOBS[run_name] = proc
    log(f"已启动出图：{run_name} n={n} ddim={ddim} PID={proc.pid}")
    return {"ok": True, "run": run_name, "pid": proc.pid, "n": n,
            "hint": "约 30~60 秒后出现在「皮肤产出」列表最前面"}


def stop_training() -> dict:
    tag = C.pick_active_run()
    st = C.run_state(tag)
    pid = st.get("pid")
    if not pid:
        # 心跳里没有 PID：退回到「本服务启动过的进程」
        for t, p in list(_LAUNCHED.items()):
            if p.poll() is None:
                pid, tag = p.pid, t
                break
    if not pid:
        return {"ok": False, "error": "没有找到正在运行的训练进程"}
    try:
        r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, text=True, timeout=20, errors="replace")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"停止失败：{exc}"}
    ok = r.returncode == 0
    log(f"已请求停止训练：tag={tag} PID={pid} 结果={'成功' if ok else r.stderr.strip()}",
        "warn" if not ok else "info")
    _LAUNCHED.pop(tag, None)
    return {"ok": ok, "pid": pid, "tag": tag, "output": (r.stdout or r.stderr or "").strip()}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "drmage-monitor/1.0"

    # ---- 工具 --------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict | None = None) -> None:
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

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str)
                   .encode("utf-8"), "application/json; charset=utf-8")

    def _file(self, path: str) -> None:
        if not os.path.isfile(path):
            self._json({"error": "not found", "path": path}, 404)
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith(("javascript", "json")):
            ctype += "; charset=utf-8"
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError as exc:
            self._json({"error": str(exc)}, 500)
            return
        self._send(200, body, ctype)

    def log_message(self, fmt, *args):        # 关掉逐请求刷屏
        pass

    # ---- 路由 --------------------------------------------------------
    def do_GET(self) -> None:                 # noqa: N802
        try:
            self._route_get()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            log(f"GET {self.path} 出错：{type(exc).__name__}: {exc}", "error")
            log(traceback.format_exc(limit=4), "error")
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            except Exception:
                pass

    def _route_get(self) -> None:
        u = urlparse(self.path)
        path = unquote(u.path)
        q = parse_qs(u.query)

        def one(k, d=None):
            return (q.get(k) or [d])[0]

        if path in ("/", "/index.html"):
            self._file(os.path.join(STATIC, "index.html"))
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            self._file(os.path.join(STATIC, rel.replace("/", os.sep)))
            return
        if path == "/api/stream":
            self._sse()
            return

        with STATE_LOCK:
            # 注意：``qtag`` 与 ``tag`` 要分开。``tag`` 是「请求指定，否则用当前活跃」，
            # 而判断「是否需要按指定 tag 现算」必须和 **STATE 里缓存的那个 tag** 比。
            # 早先是拿 qtag 去比 tag，两者相等时永远走缓存 → 指定历史实验也会
            # 返回当前活跃实验的数据（实测踩过）。
            qtag = one("tag")
            st_tag = STATE.get("tag")
            tag = qtag or st_tag
            overview = dict(STATE.get("overview") or {})
            system = dict(STATE.get("system") or {})
            artifacts = dict(STATE.get("artifacts") or {})
            events = dict(STATE.get("events") or {})

        if path == "/api/bootstrap":
            self._json({
                "project": {"name": C.PROJECT_NAME, "desc": C.PROJECT_DESC,
                            "root": C.ROOT, "webui": C.WEBUI},
                "face_index": C.face_index(),
                "runs": C.list_runs(),
                "active_tag": tag,
                "thresholds": C.THRESHOLDS,
                "server": {"started_at": STATE["started_at"], "port": self.server.server_port,
                           "python": sys.version.split()[0], "pid": os.getpid()},
                "tick_seconds": TICK_SECONDS,
            })
            return
        if path == "/api/overview":
            if qtag and qtag != st_tag:
                self._json(build_overview(qtag))      # 查看历史实验
            else:
                self._json(overview)
            return
        if path == "/api/system":
            self._json({"now": system, "history": C.history(),
                        "host": C.host_info(), "thresholds": C.THRESHOLDS})
            return
        if path == "/api/series":
            self._json(build_series(tag, _clamp(one("points", MAX_POINTS), 50, 20000,
                                                MAX_POINTS)))
            return
        if path == "/api/artifacts":
            if qtag and qtag != st_tag:
                self._json(C.artifacts_overview(qtag))  # 查看历史实验的产物
            else:
                self._json(artifacts)
            return
        if path == "/api/generated_items":
            res = C.generated_items(one("run", "") or "",
                                    _clamp(one("limit", 48), 1, 400, 48),
                                    _clamp(one("offset", 0), 0, 100000, 0))
            if res is None:
                self._json({"error": "unknown run", "run": one("run", "")}, 404)
            else:
                self._json(res)
            return
        if path == "/api/reports":
            self._json(C.reports())
            return
        if path == "/api/logs":
            self._json(C.read_log(tag, one("name", "train.log"),
                                  _clamp(one("lines", 400), 20, 5000, 400),
                                  one("level") or None, one("q") or None))
            return
        if path == "/api/log_files":
            self._json({"tag": tag, "files": C.list_log_files(tag)})
            return
        if path == "/api/events":
            self._json(C.EventEngine.snapshot(_ENGINE,
                                              int(one("since", 0) or 0)))
            return
        if path == "/api/server_log":
            lines = _clamp(one("lines", 200), 20, 3000, 200)
            items = []
            if os.path.isfile(SERVER_LOG):
                for line in open(SERVER_LOG, "r", encoding="utf-8",
                                 errors="replace").read().splitlines()[-lines:]:
                    m = re.match(r"^\[([\d\-]+ ([\d:]+))\] \[(\w+)\s*\] (.*)$", line)
                    if m:
                        items.append({"time": m.group(2), "level": m.group(3).lower(),
                                      "msg": m.group(4), "raw": line})
                    elif line.strip():
                        items.append({"time": "", "level": "info", "msg": line,
                                      "raw": line})
            self._json({"items": items})
            return
        if path.startswith("/media/"):
            self._media(path[len("/media/"):])
            return
        self._json({"error": "unknown endpoint", "path": path}, 404)

    def _media(self, rel: str) -> None:
        """只在白名单目录里取文件，并做路径穿越检查。"""
        rel = posixpath.normpath("/" + rel).lstrip("/")
        parts = rel.split("/")
        root = None
        for prefix in sorted(MEDIA_ROOTS, key=len, reverse=True):
            pp = prefix.split("/")
            if parts[:len(pp)] == pp:
                root = MEDIA_ROOTS[prefix]
                rel = "/".join(parts[len(pp):])
                break
        if root is None:
            self._json({"error": "media root not allowed"}, 403)
            return
        full = os.path.abspath(os.path.join(root, rel.replace("/", os.sep)))
        if not full.startswith(os.path.abspath(root)):
            self._json({"error": "path traversal blocked"}, 403)
            return
        self._file(full)

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q: list = []
        with _SSE_LOCK:
            _SSE_CLIENTS.append(q)
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            # 首帧立刻推一次，避免前端等 2 秒
            with STATE_LOCK:
                first = {"overview": STATE.get("overview") or {},
                         "events": STATE.get("events") or {},
                         "server": {"uptime_s": round(time.time() - STATE["started_at"], 1),
                                    "ticks": STATE.get("ticks", 0),
                                    "errors": STATE.get("errors", 0)}}
            self.wfile.write(("data: " + json.dumps(first, ensure_ascii=False,
                                                    default=str) + "\n\n").encode("utf-8"))
            self.wfile.flush()
            idle = 0
            while True:
                if q:
                    while q:
                        self.wfile.write(q.pop(0))
                    self.wfile.flush()
                    idle = 0
                else:
                    time.sleep(0.4)
                    idle += 1
                    if idle % 30 == 0:            # 每 12 秒发一次注释保活
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with _SSE_LOCK:
                if q in _SSE_CLIENTS:
                    _SSE_CLIENTS.remove(q)

    def do_POST(self) -> None:                # noqa: N802
        try:
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json({"ok": False, "error": "请求体不是合法 JSON"}, 400)
                return
            # ⚠️ 这四段必须是**一条 if/elif 链**。曾经 train/start 是独立的 if，
            # 执行完会继续往下判断，最终掉进 else → 对调用方返回 404
            # ——**训练其实已经启动了，面板却报「启动失败」**。
            if u.path == "/api/train/start":
                if not body.get("confirm"):
                    self._json({"ok": False, "error": "缺少确认标记 confirm=true"}, 400)
                    return
                res = start_training(body)
            elif u.path == "/api/skins/generate":
                if not body.get("confirm"):
                    self._json({"ok": False, "error": "缺少确认标记 confirm=true"}, 400)
                    return
                res = generate_skins(body)
            elif u.path == "/api/train/stop":
                if not body.get("confirm"):
                    self._json({"ok": False, "error": "缺少确认标记 confirm=true"}, 400)
                    return
                res = stop_training()
            else:
                self._json({"ok": False, "error": "unknown endpoint"}, 404)
                return
            self._json(res, 200 if res.get("ok") else 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            log(f"POST {self.path} 出错：{type(exc).__name__}: {exc}", "error")
            log(traceback.format_exc(limit=4), "error")
            try:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
            except Exception:
                pass


_ENGINE = C.EventEngine()


def main() -> int:
    ap = argparse.ArgumentParser(description="drmage-1.0-flash-preview 训练监控平台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-monitor", action="store_true",
                    help="只起服务，不启动采集线程（调试用）")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    log("=" * 72)
    log(f"训练监控平台启动 | 项目 {C.PROJECT_NAME} · 根目录 {C.ROOT}")
    log(f"监听 http://{args.host}:{args.port}/  | 事件库已有 "
        f"{len(_ENGINE.events)} 条 | Python {sys.version.split()[0]}")

    if not args.no_monitor:
        t = threading.Thread(target=monitor_loop, args=(_ENGINE,), daemon=True)
        t.start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("收到中断信号，正在退出…")
    finally:
        C.save_history()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
