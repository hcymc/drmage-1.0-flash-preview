"""collect.py — 训练监控平台的采集层（纯标准库，零第三方依赖）。

设计原则
--------
1. **只读项目文件**。唯一写入是 ``webui/`` 自己目录下的
   ``events.jsonl`` / ``system_history.json`` / ``server.log``，
   绝不碰 ``logs/``、``models/``、``data/`` 里的任何东西。
2. **不依赖 psutil / fastapi**。本机 pip 出网不稳定，所以系统指标全部走
   ``ctypes`` 调 Windows API（内存 / 磁盘 / CPU / 进程存活），
   GPU 走 ``nvidia-smi`` 查询。
3. **训练状态以心跳为准**。训练脚本每个 epoch（以及每 100 步）会写
   ``logs/<tag>/heartbeat.json``（由训练进程自己写，所以 pid 一定是训练进程）。
   判断「在跑 / 已停 / 卡死」用的是 **PID 是否存活**（精确、瞬时），
   心跳时间戳只作为辅助。
4. **中文日志是解析出来的，不是原文照抄**。日志原文里混着缩写
   （``mse=0.02800``、``vram_peak=4.49GB``、``step=81400``），
   这里按已知格式生成「人话」描述，同时保留 ``raw`` 供核对。
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from ctypes import wintypes
from datetime import datetime

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
WEBUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEBUI)
LOGS = os.path.join(ROOT, "logs")
MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")
EXPORTS = os.path.join(ROOT, "exports")
EVENTS_PATH = os.path.join(WEBUI, "events.jsonl")
HISTORY_PATH = os.path.join(WEBUI, "system_history.json")

PROJECT_NAME = "drmage-1.0-flash-preview"
PROJECT_DESC = "Minecraft 皮肤生成 · 实验预览（DDPM · classic 骨架 · 8GB 消费级显卡可训）"

#: 本发布**只面向这一个实验**。监控、训练控制、产物浏览都锁定在它上面，
#: 不再做旧仓库里的「多实验扫描 / 切换」——那套逻辑服务于本地迭代期的
#: 几十个对照实验，对使用本发布的人只有干扰。
SINGLE_TAG = "diff_v2_masked"

# ---------------------------------------------------------------------------
# Windows 系统指标（ctypes，无第三方依赖）
# ---------------------------------------------------------------------------


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _memory() -> dict:
    st = _MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return {"total_gb": 0, "used_gb": 0, "percent": 0}
    total = st.ullTotalPhys
    avail = st.ullAvailPhys
    return {
        "total_gb": round(total / 1024 ** 3, 2),
        "used_gb": round((total - avail) / 1024 ** 3, 2),
        "avail_gb": round(avail / 1024 ** 3, 2),
        "percent": round(st.dwMemoryLoad, 1),
    }


def _disk_free(drive: str) -> dict:
    free = ctypes.c_ulonglong(0)
    total = ctypes.c_ulonglong(0)
    tfree = ctypes.c_ulonglong(0)
    ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(drive), ctypes.byref(free), ctypes.byref(total),
        ctypes.byref(tfree))
    if not ok or total.value == 0:
        return {"error": "unavailable"}
    return {
        "total_gb": round(total.value / 1024 ** 3, 1),
        "free_gb": round(tfree.value / 1024 ** 3, 1),
        "used_gb": round((total.value - tfree.value) / 1024 ** 3, 1),
        "percent": round((total.value - tfree.value) / total.value * 100, 1),
    }


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


def _ft(v: "_FILETIME") -> int:
    return (v.dwHighDateTime << 32) | v.dwLowDateTime


class CpuSampler:
    """CPU 占用率只能由两次采样的差值算出来，所以需要持有上一次的读数。

    ``GetSystemTimes`` 返回三个**自开机以来的累计值**：``idle`` / ``kernel`` / ``user``。
    关键事实：**``kernel`` 已经包含 ``idle``**。所以

        总时间 = kernel + user
        忙时间 = 总时间 − idle
        占用率 = (kernel + user − idle) / (kernel + user)

    这个 API 的 argtypes 必须显式声明（否则 ctypes 会按默认 int 处理指针参数，
    行为不可靠）。
    """

    def __init__(self) -> None:
        self._prev = None
        self._lock = threading.Lock()
        fn = ctypes.windll.kernel32.GetSystemTimes
        fn.argtypes = [ctypes.POINTER(_FILETIME)] * 3
        fn.restype = wintypes.BOOL

    def sample(self) -> float:
        def raw():
            idle, kern, user = _FILETIME(), _FILETIME(), _FILETIME()
            if not ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return None
            return (_ft(idle), _ft(kern), _ft(user))

        cur = raw()
        if cur is None:
            return 0.0
        with self._lock:
            prev, self._prev = self._prev, cur
        if prev is None:
            # 第一次采样没有参照值。直接返回 0.0 会让面板刚启动时显示「CPU 0%」，
            # 看起来像坏了；这里等 150ms 再采一次，拿一个真实读数。
            time.sleep(0.15)
            cur2 = raw()
            if cur2 is None:
                return 0.0
            with self._lock:
                prev, self._prev = cur, cur2
            cur = cur2
        d_idle = cur[0] - prev[0]
        d_kern = cur[1] - prev[1]
        d_user = cur[2] - prev[2]
        # **`kernel` 时间已经包含 `idle`**，所以系统总时间是 `kernel + user`，
        # 忙 = 总 − idle。占用率 = 忙 / 总。
        #
        # 早先写成 `total = kernel + user - idle; 1 - idle/total`（等于 idle 减两次），
        # 实测本机 kernel 增量 109.4M、idle 100.2M → total 只有 21.7M，算出 −361%
        # 被钳成 0，于是面板上 CPU **永远是 0.0%**，看起来像「读不到」。
        total = d_kern + d_user
        if total <= 0:
            return 0.0
        busy = total - d_idle
        return round(max(0.0, min(100.0, busy / total * 100.0)), 1)


_CPU = CpuSampler()


def _boot_uptime() -> float:
    ms = ctypes.windll.kernel32.GetTickCount64()
    return round(ms / 1000.0, 0)


def _gpu() -> list[dict]:
    """nvidia-smi 查询。这是本机唯一可靠的 GPU 读数来源。"""
    fields = ("index,name,memory.total,memory.used,utilization.gpu,"
              "temperature.gpu,power.draw,power.limit,fan.speed,"
              "clocks.current.sm,clocks.max.sm")
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in (r.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        def num(i):
            try:
                return float(parts[i])
            except (ValueError, IndexError):
                return None
        mt, mu = num(2), num(3)
        out.append({
            "index": parts[0], "name": parts[1],
            "mem_total_mb": mt, "mem_used_mb": mu,
            "mem_percent": round(mu / mt * 100, 1) if mt and mu is not None else None,
            "util_percent": num(4), "temp_c": num(5),
            "power_w": num(6), "power_limit_w": num(7), "fan_percent": num(8),
            "clock_mhz": num(9), "clock_max_mhz": num(10),
            "driver": None,
        })
    if out:
        try:
            v = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                "--format=csv,noheader"], capture_output=True,
                               text=True, timeout=5)
            out[0]["driver"] = (v.stdout or "").strip().splitlines()[0]
        except Exception:
            pass
    return out


def _gpu_processes() -> list[dict]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in (r.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            out.append({"pid": int(parts[0]), "mem_mb": float(parts[1] or 0)})
    return out


def pid_alive(pid: int | None) -> bool:
    """进程存活判断：瞬时、精确，不需要枚举进程表。

    ``OpenProcess`` + ``WaitForSingleObject(0)``：拿到句柄说明进程存在；
    等到信号说明已退出。这比「心跳时间戳是否新鲜」可靠得多——
    一个 epoch 要 250 秒，中间日志完全不动，靠时间戳判断会误报卡死。
    """
    if not pid:
        return False
    SYNCHRONIZE = 0x00100000
    h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
    if not h:
        return False
    try:
        WAIT_TIMEOUT = 0x00000102
        return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == WAIT_TIMEOUT
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def _processes() -> list[dict]:
    """用 tasklist 拿 python 进程的内存占用（tasklist 在本机可用，wmic 被禁用）。"""
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe",
                            "/FO", "CSV", "/NH"],
                           capture_output=True, text=True, timeout=10, errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in (r.stdout or "").splitlines():
        if "python" not in line.lower():
            continue
        p = [x.strip('"') for x in line.split('","')]
        if len(p) < 5:
            continue
        try:
            mem = int(p[4].replace(",", "").replace(" K", "")) / 1024
        except ValueError:
            mem = 0
        out.append({"pid": int(p[1]), "mem_mb": round(mem, 1)})
    return out


def host_info() -> dict:
    import platform
    drives = []
    for d in "CDEFGHIJKL":
        p = f"{d}:\\"
        if os.path.isdir(p):
            info = _disk_free(p)
            info["drive"] = p
            drives.append(info)
    proj_drive = os.path.splitdrive(ROOT)[0] + "\\"
    return {
        "hostname": platform.node(),
        "user": os.environ.get("USERNAME", ""),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "boot_uptime_s": _boot_uptime(),
        "drives": drives,
        "project_drive": proj_drive,
    }


def system_snapshot() -> dict:
    gpus = _gpu()
    procs = _processes()
    info = host_info()
    proj_drive = info["project_drive"]
    proj_disk = next((d for d in info["drives"] if d["drive"] == proj_drive), {})
    return {
        "ts": time.time(),
        "time": datetime.now().strftime("%H:%M:%S"),
        "cpu_percent": _CPU.sample(),
        "memory": _memory(),
        "disk_project": proj_disk,
        "disk_root": _disk_free("C:\\"),
        "gpus": gpus,
        "python_procs": sorted(procs, key=lambda x: -x["mem_mb"])[:6],
        "loader": [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None,
    }


# ---------------------------------------------------------------------------
# 训练状态
# ---------------------------------------------------------------------------

def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def list_runs() -> list[dict]:
    """扫描 ``logs/*/``，把每个训练 tag 的概要列出来（最近活跃的排前面）。

    本发布锁定单实验（``SINGLE_TAG``）：只列它，其它目录一律忽略。
    """
    runs = []
    if not os.path.isdir(LOGS):
        return runs
    for name in sorted(os.listdir(LOGS)):
        if name != SINGLE_TAG:
            continue
        d = os.path.join(LOGS, name)
        if not os.path.isdir(d) or name in ("samples", "atlas", "__pycache__"):
            continue
        log = os.path.join(d, "train.log")
        met = os.path.join(d, "metrics.jsonl")
        hb = os.path.join(d, "heartbeat.json")
        if not (os.path.isfile(log) or os.path.isfile(met)):
            continue
        runs.append({
            "tag": name,
            "has_log": os.path.isfile(log),
            "has_metrics": os.path.isfile(met),
            "has_heartbeat": os.path.isfile(hb),
            "mtime": max(_mtime(log), _mtime(met), _mtime(hb)),
        })
    runs.sort(key=lambda r: -r["mtime"])
    return runs


_METRICS_CACHE: dict[str, tuple] = {}


def read_metrics(tag: str) -> list[dict]:
    """读 ``logs/<tag>/metrics.jsonl``。按 (mtime, size) 缓存，避免每 2 秒重解析。"""
    path = os.path.join(LOGS, tag, "metrics.jsonl")
    if not os.path.isfile(path):
        return []
    try:
        st = os.stat(path)
    except OSError:
        return []
    key = (st.st_mtime, st.st_size)
    cached = _METRICS_CACHE.get(tag)
    if cached and cached[0] == key:
        return cached[1]
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    _METRICS_CACHE[tag] = (key, rows)
    return rows


def read_config(tag: str) -> dict:
    """从 metrics.jsonl 的第一行 ``event=config`` 里取训练配置。"""
    for rec in read_metrics(tag)[:3]:
        if rec.get("event") == "config":
            # ``arch`` 必须带出去：它是判断「新旧架构」的唯一可靠来源。
            # 丢掉它之后，下游只能靠猜指标名或 args 里有没有 ``cond``，
            # 结果把 PIDiff 的 cond_dim 误算成 0。
            return {"args": rec.get("args") or {}, "device": rec.get("device"),
                    "time": rec.get("time"), "arch": rec.get("arch")}
    return {}


def _cond_dim(args: dict, arch: str) -> int:
    """条件向量维度。

    **不能只看 ``args["cond"]``**：那个开关是旧架构 argparse 的产物，
    PIDiff 的脚本里根本没有它 → 会把 PIDiff 的 cond_dim 误算成 0。
    PIDiff 恒为 36（3 通道 × 12 维基）再按需加 4 维 overlay 位。

    【修复】``args["cond"]`` 是 **bool**，而 Python 里 bool 是 int 的子类——
    旧写法 ``isinstance(v, int) and v > 0`` 会把 ``True`` 直接当维度返回，
    面板条件维度一栏显示 ``true`` 而不是 36。先排除 bool 再判数值。
    """
    v = args.get("cond_dim")
    if isinstance(v, int) and not isinstance(v, bool) and v > 0:
        return v
    if args.get("cond"):
        return 36
    if arch == "pidiff":
        return 40 if args.get("ov_bits") else 36
    return 0


def run_state(tag: str | None) -> dict:
    """把心跳 + 指标合成「训练状态」给前端用。

    状态机：``idle`` 没跑过 / ``running`` 在跑 / ``finished`` 正常结束 /
    ``stale`` 进程没了但没写结束标记（异常中断）/ ``stopped`` 被手动停止。
    """
    if not tag:
        return {"tag": None, "status": "idle", "status_cn": "无训练记录"}
    d = os.path.join(LOGS, tag)
    hb = _read_json(os.path.join(d, "heartbeat.json"))
    cfg = read_config(tag)
    args = cfg.get("args", {})
    arch = cfg.get("arch") or "ddpm"
    metrics = read_metrics(tag)
    hist = [m for m in metrics if not m.get("event")]

    status, pid, alive, hb_age = "stopped", None, False, None
    if hb:
        pid = hb.get("pid")
        alive = pid_alive(pid)
        hb_age = round(time.time() - hb.get("ts", 0), 1)
        if hb.get("finished"):
            status = "finished"
        elif alive:
            status = "running"
        else:
            status = "stale"

    steps = [m["step"] for m in hist if "step" in m]
    last = hist[-1] if hist else {}
    # 【兼容性】损失字段名随架构变：旧架构写 ``mse``，PIDiff 写 ``ce``。
    # 写死 ``mse`` 的后果：PIDiff 实验的 KPI「训练损失」永远显示 —，
    # mse_best / mse_delta 全空，看起来像训练根本没在跑。
    def _loss_of(rec: dict):
        for k in ("mse", "ce"):
            v = rec.get(k)
            if isinstance(v, (int, float)):
                return v
        return None

    loss_name = "训练损失 CE" if arch == "pidiff" else "训练损失 MSE"
    mses = [(m["step"], v) for m in hist if (v := _loss_of(m)) is not None]
    vrams = [m["vram_peak_gb"] for m in hist if m.get("vram_peak_gb")]
    epochs = [m["epoch"] for m in hist if "epoch" in m]

    cur_step = max(steps) if steps else (hb or {}).get("step", 0) or 0
    # **重要**：metrics.jsonl 里的 epoch 是 0 起编号（首条 ``epoch=0, step=2200``
    # 对应 train.log 里的「epoch 1/60」）。所以「已跑完的轮数」= max(epoch)，
    # 而不是 max(epoch)+1 —— 直接用 max 会在最后少算一轮。
    # 心跳里的 ``epochs_done`` 是训练脚本自己算的 1 基完成轮数，优先采用。
    epochs_done_metrics = max(epochs) if epochs else 0
    epochs_done = (hb or {}).get("epochs_done") or epochs_done_metrics
    # 续训会把目标轮数往上加（本发布的历史日志就是：首段 config 写 200，
    # 实际续到 350）。心跳里的 ``epochs`` 来自**最后一次**启动，所以两者取大者，
    # 否则进度条会出现「687400 / 440000 = 156%」这种荒谬读数。
    target_epochs = max(args.get("epochs") or 0, (hb or {}).get("epochs") or 0)

    # 每轮步数：用「每个 epoch 的最大 step」的差分中位数。
    # 不能用 ``max(step)//epoch``：最后一个 epoch 通常只跑了一部分，
    # 那样算出来会偏大（实测 2235 vs 真值 2200）。
    ep_max: dict[int, int] = {}
    for m in hist:
        if m.get("epoch") is not None and m.get("step") is not None:
            ep_max[m["epoch"]] = max(ep_max.get(m["epoch"], 0), m["step"])
    ks = sorted(ep_max)
    diffs = [ep_max[ks[i]] - ep_max[ks[i - 1]] for i in range(1, len(ks))]
    if diffs:
        sd = sorted(diffs)
        steps_per_epoch = sd[len(sd) // 2]
    else:
        # 【踩过的坑】新实验刚起步时 metrics.jsonl 可能一条记录都没有，
        # ``max(steps)`` 直接抛 ValueError —— 而 run_state 在**每 2 秒的采集循环**里，
        # 一抛整个 overview 就不更新，前端表现为「日志/档案全都没加载」。
        # 所以这里必须对空序列兜底。
        steps_per_epoch = ((max(steps) if steps else 0) // max(epochs_done, 1)) or None

    # 每个 epoch 的耗时（用于 ETA），取最近 5 个 epoch 的中位数
    ep_secs = []
    by_epoch = {}
    for m in hist:
        if m.get("epoch") is not None and m.get("elapsed_s") is not None:
            by_epoch[m["epoch"]] = m["elapsed_s"]
    seq = sorted(by_epoch.items())
    for i in range(1, len(seq)):
        ep_secs.append(seq[i][1] - seq[i - 1][1])
    recent = ep_secs[-5:]
    epoch_seconds = round(sum(recent) / len(recent), 1) if recent else (
        (hb or {}).get("epoch_seconds"))

    best = min((v for _, v in mses), default=None)
    # 优先用心跳里的损失：**训练正常结束后心跳仍带着最后一次读数**，
    # 而短实验（--max-steps 只有几十步）根本不会产生 metrics 记录。
    # 早先写成「只有 running 才读心跳」，导致结束事件里显示「最终 MSE None」。
    # PIDiff 的心跳把 ce 也写在 ``mse`` 字段里，所以这一处无需区分架构。
    mse = None
    if hb and hb.get("mse") is not None:
        mse = hb["mse"]
    if mse is None and mses:
        mse = mses[-1][1]
    prev_mse = mses[-2][1] if len(mses) >= 2 else None

    eta_s = None
    if status == "running" and target_epochs and epoch_seconds:
        # 保守估计：把「正在进行中的这一轮」按完整一轮计入剩余量
        remain_epochs = max(0, target_epochs - epochs_done)
        eta_s = remain_epochs * epoch_seconds
    total_steps = (int(steps_per_epoch * target_epochs)
                   if (steps_per_epoch and target_epochs) else None)

    return {
        "tag": tag,
        "status": status,
        "status_cn": {"running": "训练中", "finished": "已完成", "stale": "异常中断",
                      "stopped": "已停止", "idle": "空闲"}[status],
        "pid": pid, "pid_alive": alive, "heartbeat_age_s": hb_age,
        "heartbeat": hb,
        "step": cur_step, "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "epochs_done": epochs_done,
        "epoch": (min(epochs_done + 1, target_epochs) if target_epochs else epochs_done + 1),
        "epochs": target_epochs,
        "progress": round(cur_step / total_steps, 4) if total_steps else None,
        "mse": mse, "mse_prev": prev_mse, "mse_best": best,
        "loss_name": loss_name, "arch": arch,
        "mse_delta": (round(mse - prev_mse, 6) if (mse is not None and prev_mse is not None)
                      else None),
        "lr": args.get("lr") or (hist[-1].get("lr") if hist else None),
        "vram_gb": vrams[-1] if vrams else (hb or {}).get("vram_gb"),
        "elapsed_s": (hist[-1].get("elapsed_s") if hist else None),
        "epoch_seconds": epoch_seconds,
        "eta_s": eta_s,
        "records": len(hist),
        "args": args,
        "model": {
            "params_m": None,
            # in_ch：旧架构看 ``--channels``；PIDiff 恒为 4（RGB + alpha）
            "in_ch": args.get("channels") or (4 if arch == "pidiff" else 3),
            "T": args.get("timesteps"), "schedule": args.get("schedule"),
            "cond_dim": _cond_dim(args, arch),
            "base": args.get("base"), "batch": args.get("batch"),
            "mask_loss": args.get("mask_loss"),
            "alpha_mode": args.get("alpha_mode") or "template",
        },
        "device": cfg.get("device"),
        "log_mtime": _mtime(os.path.join(d, "train.log")),
        "metrics_mtime": _mtime(os.path.join(d, "metrics.jsonl")),
    }


# ---------------------------------------------------------------------------
# 日志 → 中文结构化
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")

# 【兼容性】epoch 行的损失字段名**随架构变**：
#   旧架构（21_train_diffusion）→ ``epoch N/M | 123.4s | mse=0.012 | step=S | vram_peak=V GB``
#   PIDiff（40_train_pidiff）    → ``epoch N/M | 123.4s | step=S | vram_peak=V GB``（**无损失段**）
# 原来写死 ``mse=([\d.]+)`` 会让 PIDiff 的每一轮都匹配不上，
# 日志页里「第 N 轮完成」那条整条消失，看起来像训练卡住了。中间段做成可选组。
_EPOCH_RE = re.compile(
    r"^epoch\s+(\d+)/(\d+)\s*\|\s*([\d.]+)s\s*\|\s*"
    r"(?:(\w+)=([\d.]+)\s*\|\s*)?step=(\d+)\s*\|\s*vram_peak=([\d.]+)GB")
_EPOCH_D_RE = re.compile(
    r"^epoch\s+(\d+)/(\d+)\s*\|\s*([\d.]+)s\s*\|\s*d=([\d.]+)\s+g=([\d.]+)")
_BATCH_RE = re.compile(r"^batch\s+(\d+)\s*\|\s*(\d+)\s*step/epoch\s*\|\s*目标\s*(\d+)\s*epoch"
                       r"\s*\|\s*AMP=(\w+)")
_UNET_RE = re.compile(r"^UNet\s+([\d.]+)M\s*\|\s*in_ch=(\d+)\s*\|\s*T=(\d+)\s*\|\s*"
                      r"schedule=(\w+)\s*\|\s*cond_dim=(\d+)")
# PIDiff 的模型摘要行字段与旧架构完全不同：
#   旧: ``UNet 11.9M | in_ch=3 | T=1000 | schedule=cosine | cond_dim=36``
#   新: ``PIDiffUNet 13.66M | logits=768 | T=1000 | schedule=cosine | aux=0.0 | cond_dropout=0.0``
_PIDIFF_UNET_RE = re.compile(
    r"^\s*PIDiffUNet\s+([\d.]+)M\s*\|\s*logits=(\d+)\s*\|\s*T=(\d+)\s*\|\s*"
    r"schedule=(\w+)(?:\s*\|\s*aux=([\d.]+))?(?:\s*\|\s*cond_dropout=([\d.]+))?")

# 训练内探针。PIDiff 用的是「相邻精确相等 / ≥8px 同色块 / 色熵 / 唯一色 / adj_le8」
# 这套口径，与旧架构的 vis_gen_unique_colors / real_flat_frac 不是一回事。
_PROBE_RE = re.compile(
    r"^\s*\[probe:(\w+)\]\s+step=(\d+)\s+adj_eq\s+([\d.]+)\s+≥8px\s+([\d.]+)\s+"
    r"色熵\s+([\d.]+)\s+唯一色\s+(\d+)\s+adj_le8\s+([\d.]+)")

# PIDiff 的 cond 维度只出现在这一行：
# ``train (105605, 4, 64, 64) · cond_dim=36 · 105605 张``
_COND_RE = re.compile(r"cond_dim=(\d+)")

_SKEL_RE = re.compile(r"^骨架\s*(\w+)[：:]\s*train\s+(\d+)\s*/\s*val\s+(\d+)")
_DATA_RE = re.compile(r"^train\s+\(([\d,]+),\s*(\d+),\s*(\d+),\s*(\d+)\)\s*\|\s*"
                      r"val\s+\(([\d,]+)")
_RESUME_RE = re.compile(r"^从\s+(\S+)\s*续训[：:]\s*epoch\s+(\d+)\s+step\s+(\d+)")
_SMOKE_RE = re.compile(r"^到达\s+max_steps=(\d+)")

_LEVEL_HINT = [
    (re.compile(r"(Traceback|Error|Exception|错误|失败|异常)", re.I), "error"),
    (re.compile(r"(\[!\])|警告|注意|warn", re.I), "warn"),
    (re.compile(r"(完成|结束|成功|全绿|OK$)", re.I), "success"),
]


def _human(line: str) -> tuple[str, str]:
    """把一行日志翻成「人话」。返回 ``(级别, 中文描述)``。"""
    m = _EPOCH_RE.match(line)
    if m:
        ep, tot, sec, mkey, mval, step, vram = m.groups()
        # 旧架构带 ``mse=`` 段；PIDiff 的 epoch 行没有损失段（损失看 metrics.jsonl）
        loss = f" · {mkey.upper()} {mval}" if mkey else ""
        return "info", (f"第 {ep}/{tot} 轮完成 · 用时 {sec} 秒{loss} · "
                        f"已训练 {int(step):,} 步 · 峰值显存 {vram} GB")
    m = _EPOCH_D_RE.match(line)
    if m:
        ep, tot, sec, d, g = m.groups()
        return "info", (f"第 {ep}/{tot} 轮完成 · 用时 {sec} 秒 · "
                        f"判别器损失 {d} · 生成器损失 {g}")
    m = _BATCH_RE.match(line)
    if m:
        b, spe, tot, amp = m.groups()
        return "info", (f"批大小 {b} · 每一轮 {int(spe):,} 步 · 共 {tot} 轮 · "
                        f"混合精度 {'开启' if amp.lower() == 'true' else '关闭'}")
    m = _UNET_RE.match(line)
    if m:
        p, ch, T, sch, cd = m.groups()
        return "info", (f"模型 UNet {p}M 参数 · 输入 {ch} 通道 · 扩散步数 {T} · "
                        f"噪声调度 {sch} · 条件维度 {cd}")
    m = _PIDIFF_UNET_RE.match(line)
    if m:
        p, logits, T, sch, aux, cd = m.groups()
        extra = f" · 辅助损失权重 {aux}" if aux else ""
        extra += f" · 条件丢弃 {cd}" if cd else ""
        return "info", (f"模型 PIDiffUNet {p}M 参数 · 码本 logits {logits} 维 · "
                        f"扩散步数 {T} · 噪声调度 {sch}{extra}")
    m = _PROBE_RE.match(line)
    if m:
        path, step, aeq, bb, ent, uc, ale8 = m.groups()
        name = {"hard": "全程硬量化", "late": "末段硬量化（可交付路径）"}.get(path, path)
        return "info", (f"采样探针（{name}）· 第 {int(step):,} 步 · "
                        f"相邻精确相等 {aeq} · ≥8px 同色块 {bb} · 色熵 {ent} · "
                        f"唯一色 {uc} · 相邻差≤8 灰阶 {ale8}")
    m = _SKEL_RE.match(line)
    if m:
        sk, tr, va = m.groups()
        return "info", f"骨架筛选 {sk} · 训练集 {int(tr):,} 张 · 验证集 {int(va):,} 张"
    m = _DATA_RE.match(line)
    if m:
        tr, ch, h, w, va = m.groups()
        return "info", f"数据集就位 · 训练 {tr} 张 · 验证 {va} 张 · 每张 {h}×{w}×{ch}"
    m = _RESUME_RE.match(line)
    if m:
        path, ep, step = m.groups()
        return "info", f"从断点续训 {os.path.basename(path)} · 起点第 {ep} 轮 / 第 {int(step):,} 步"
    if "训练结束" in line:
        return "success", "训练正常结束"
    if line.startswith("==="):
        return "info", f"阶段：{line.strip('= ')}"
    for pat, lv in _LEVEL_HINT:
        if pat.search(line):
            return lv, line
    return "info", line


def read_log(tag: str, name: str = "train.log", lines: int = 300,
             level: str | None = None, keyword: str | None = None) -> dict:
    """读日志并逐行结构化。``level``/``keyword`` 在服务端过滤，前端只管展示。"""
    path = os.path.join(LOGS, tag, name)
    if not os.path.isfile(path):
        return {"tag": tag, "name": name, "exists": False, "items": [], "total": 0}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.read().splitlines()
    except OSError:
        return {"tag": tag, "name": name, "exists": False, "items": [], "total": 0}

    items = []
    for raw in raw_lines:
        if not raw.strip():
            continue
        m = _TS_RE.match(raw)
        if m:
            ts, body = m.group(1), m.group(2)
        else:
            ts, body = "", raw
        level_cn, msg = _human(body)
        if level and level_cn != level:
            continue
        if keyword and keyword.lower() not in raw.lower():
            continue
        items.append({"ts": ts, "level": level_cn, "msg": msg, "raw": raw})
    total = len(items)
    return {"tag": tag, "name": name, "exists": True, "total": total,
            "items": items[-lines:]}


def list_log_files(tag: str) -> list[dict]:
    d = os.path.join(LOGS, tag)
    out = []
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith((".log", ".jsonl", ".json")):
            p = os.path.join(d, f)
            out.append({"name": f, "size": os.path.getsize(p),
                        "mtime": _mtime(p)})
    return out


# ---------------------------------------------------------------------------
# 产物清单
# ---------------------------------------------------------------------------

def _grid_step(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def list_sample_grids(tag: str) -> list[dict]:
    d = os.path.join(LOGS, "samples", tag)
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        if f.endswith(".png"):
            p = os.path.join(d, f)
            out.append({
                "name": f, "step": _grid_step(f), "mtime": _mtime(p),
                "url": f"/media/logs/samples/{tag}/{f}",
                "final": f == "final.png",
            })
    out.sort(key=lambda x: (x["step"], x["mtime"]))
    return out


def list_checkpoints(tag: str) -> list[dict]:
    d = os.path.join(MODELS, tag)
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        if f.endswith(".pt"):
            p = os.path.join(d, f)
            out.append({"name": f, "size_mb": round(os.path.getsize(p) / 1e6, 1),
                        "mtime": _mtime(p)})
    out.sort(key=lambda x: -x["mtime"])
    return out


def all_checkpoints() -> list[dict]:
    """断点清单。本发布只扫 ``models/<SINGLE_TAG>/``——阶段一预训练权重
    ``models/diff_v1/`` 不进面板（它是复现链条的中间产物，见 docs/REPRODUCE.md）。
    """
    out = []
    d = os.path.join(MODELS, SINGLE_TAG)
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith(".pt"):
            p = os.path.join(d, f)
            out.append({"tag": SINGLE_TAG, "name": f, "path": f"models/{SINGLE_TAG}/{f}",
                        "size_mb": round(os.path.getsize(p) / 1e6, 1),
                        "mtime": _mtime(p)})
    out.sort(key=lambda x: -x["mtime"])
    return out


def list_skins() -> list[dict]:
    """皮肤产出：``exports/<集合>/skin_*.png``，带该集合的指标 CSV（如有）。"""
    out = []
    if not os.path.isdir(EXPORTS):
        return out
    for group in sorted(os.listdir(EXPORTS)):
        d = os.path.join(EXPORTS, group)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.startswith("skin_") and f.endswith(".png"))
        if not files:
            continue
        meta = {}
        csvp = os.path.join(d, "_metrics.csv")
        if os.path.isfile(csvp):
            try:
                import csv as _csv
                with open(csvp, "r", encoding="utf-8") as fh:
                    for row in _csv.DictReader(fh):
                        meta[row.get("file", "")] = row
            except OSError:
                pass
        items = []
        for f in files:
            p = os.path.join(d, f)
            m = meta.get(f, {})
            items.append({
                "name": f, "size_kb": round(os.path.getsize(p) / 1024, 1),
                "mtime": _mtime(p),
                "url": f"/media/exports/{group}/{f}",
                "metrics": {k: m.get(k) for k in
                            ("block_share", "mean_sat", "n_colors", "face_colors",
                             "visible_ratio", "dom_hue_deg") if m.get(k) is not None},
            })
        preview = os.path.join(d, "_preview.png")
        out.append({
            "group": group, "count": len(items), "items": items,
            "preview": f"/media/exports/{group}/_preview.png" if os.path.isfile(preview) else None,
            "mtime": max((i["mtime"] for i in items), default=0),
        })
    out.sort(key=lambda g: -g["mtime"])
    return out


def list_generated() -> list[dict]:
    """``data/generated/<run>/``：每次「拿权重出图」的**原始产出**（未筛选）。

    这是「最新皮肤」的来源：``exports/`` 里的是人工挑过的精选集，
    而这里每个文件夹都是一次生成的全量结果，按**最新出图时间**降序。
    """
    d = os.path.join(DATA, "generated")
    if not os.path.isdir(d):
        return []
    out = []
    for g in sorted(os.listdir(d)):
        gd = os.path.join(d, g)
        if not os.path.isdir(gd) or g.startswith("_"):
            continue
        pngs = [f for f in os.listdir(gd)
                if f.endswith(".png") and not f.startswith("contact_sheet")]
        if not pngs:
            continue
        sheet = os.path.join(gd, "contact_sheet.png")
        # 用**图片自己的 mtime** 而不是文件夹 mtime：文件夹 mtime 会被新建子项刷新，
        # 排序会飘；图片 mtime 才是「这批图是什么时候出的」。
        newest = max((_mtime(os.path.join(gd, f)) for f in pngs), default=0)
        out.append({
            "run": g, "n_png": len(pngs),
            "contact_sheet": f"/media/data/generated/{g}/contact_sheet.png"
                             if os.path.isfile(sheet) else None,
            "mtime": newest or _mtime(gd),
        })
    out.sort(key=lambda x: -x["mtime"])
    return out


def latest_ckpt_step(tag: str) -> int:
    """当前实验最新断点对应的步数：优先心跳，其次最新样本快照，都没有就 0。

    给「按当前断点出图」的批次命名用（``<tag>_step097400``），
    这样面板上「选当前实验的哪一步」就是选不同的批次文件夹。"""
    hb = _read_json(os.path.join(LOGS, tag, "heartbeat.json")) or {}
    if hb.get("step"):
        return int(hb["step"])
    grids = list_sample_grids(tag)
    if grids:
        return int(grids[-1].get("step") or 0)
    return 0


def generated_items(run: str, limit: int = 48, offset: int = 0) -> dict | None:
    """某个生成文件夹里的图片清单（按名称排序，稳定；前端按需拉取，避免每次轮询都搬全量）。"""
    if not run or "/" in run or "\\" in run or run.startswith("."):
        return None
    gd = os.path.join(DATA, "generated", run)
    if not os.path.isdir(gd):
        return None
    pngs = sorted(f for f in os.listdir(gd)
                  if f.endswith(".png") and not f.startswith("contact_sheet"))
    total = len(pngs)
    page = pngs[offset: offset + max(1, limit)]
    items = [{
        "name": f, "size_kb": round(os.path.getsize(os.path.join(gd, f)) / 1024, 1),
        "mtime": _mtime(os.path.join(gd, f)),
        "url": f"/media/data/generated/{run}/{f}",
        "metrics": {},
    } for f in page]
    return {"run": run, "total": total, "offset": offset, "items": items}


# ---------------------------------------------------------------------------
# 诊断报告（项目里已有的一堆 JSON 结论，集中读出来展示）
# ---------------------------------------------------------------------------

_REPORT_FILES = {
    "atlas_stats": ("logs/atlas_stats.json", "图集与结构标定（真实数据实测目标值）"),
    "struct_metrics": ("logs/struct_metrics.json", "逐区域真实 vs 生成对照"),
    "color_audit": ("logs/color_audit.json", "唯一色数与色块覆盖审计"),
    "semantic_diversity": ("logs/semantic_diversity.json", "抗噪声语义多样性"),
    "seam_bleed": ("logs/seam_bleed.json", "UV 跨缝/边缘比值"),
    "capacity_probe": ("logs/capacity_probe.json", "容量探针（泛化间隙）"),
    "clean_report": ("logs/clean_report.json", "数据清洗报告"),
    "validation_dcgan_v1": ("logs/validation_dcgan_v1.json", "DCGAN 验证报告"),
}


def reports() -> dict:
    out = {}
    for key, (rel, desc) in _REPORT_FILES.items():
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if os.path.isfile(p):
            out[key] = {"title": desc, "path": rel, "mtime": _mtime(p),
                        "data": _read_json(p)}
    # 数据集规模（只读 shape，不载入）
    try:
        import numpy as np
        splits = {}
        for s in ("train", "val", "test"):
            f = os.path.join(DATA, "processed", f"{s}.npy")
            if os.path.isfile(f):
                a = np.load(f, mmap_mode="r")
                splits[s] = {"shape": list(a.shape),
                             "size_gb": round(os.path.getsize(f) / 1e9, 2)}
        out["dataset"] = {"title": "数据集规模", "data": splits}
    except Exception:
        pass
    # 文档
    docs = []
    dd = os.path.join(ROOT, "docs")
    if os.path.isdir(dd):
        for f in sorted(os.listdir(dd)):
            if f.endswith(".md"):
                p = os.path.join(dd, f)
                docs.append({"name": f, "size_kb": round(os.path.getsize(p) / 1024, 1),
                             "mtime": _mtime(p)})
    out["docs"] = {"title": "设计与诊断文档", "data": docs}
    return out


def face_index() -> dict:
    """把图集的权威面坐标导给前端（前端不硬编码 UV 布局）。"""
    try:
        import sys
        lib = os.path.join(ROOT, "scripts", "lib")
        if lib not in sys.path:
            sys.path.insert(0, lib)
        from skinatlas import face_index as _fi
        return {n: list(v) for n, v in _fi().items()}
    except Exception as exc:      # 前端有兜底布局
        return {"__error__": str(exc)}


# ---------------------------------------------------------------------------
# 事件引擎（监控告警 + 系统消息）
# ---------------------------------------------------------------------------

LEVEL_ORDER = {"error": 0, "warn": 1, "success": 2, "info": 3}

#: 阈值（前端也展示，便于用户知道告警口径）
THRESHOLDS = {
    "gpu_temp_warn": 80,
    "gpu_temp_error": 86,
    "gpu_mem_warn": 90,
    "disk_free_warn_gb": 15,
    "cpu_warn": 95,
    "mse_rise_warn": 3,
    "heartbeat_grace_s": 600,
}


class EventEngine:
    """对比相邻两次快照的差异，产生「系统消息 / 监控告警」。

    只有**状态发生变化**时才产生事件（而不是每 2 秒重复刷），
    所以事件流是可以直接读的：每一条都对应一个真实发生的变化。
    """

    def __init__(self, maxlen: int = 800):
        self.events: deque = deque(maxlen=maxlen)
        self.prev: dict = {}
        self._load()
        self._seq = 0

    # -- 持久化：事件历史跨重启保留 -------------------------------------
    def _load(self) -> None:
        if not os.path.isfile(EVENTS_PATH):
            return
        try:
            with open(EVENTS_PATH, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            self._seq = max((e.get("id", 0) for e in self.events), default=0)
        except OSError:
            pass

    def _persist(self, ev: dict) -> None:
        try:
            with open(EVENTS_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def add(self, level: str, code: str, title: str, detail: str = "",
            source: str = "monitor", silent_persist: bool = False) -> dict:
        self._seq += 1
        ev = {"id": self._seq, "ts": time.time(),
              "time": datetime.now().strftime("%H:%M:%S"),
              "date": datetime.now().strftime("%Y-%m-%d"),
              "level": level, "code": code, "title": title,
              "detail": detail, "source": source}
        self.events.append(ev)
        if not silent_persist:
            self._persist(ev)
        return ev

    # -- 主循环 ---------------------------------------------------------
    def tick(self, overview: dict, system: dict, artifacts: dict) -> list[dict]:
        new = []
        p = self.prev
        tr = overview.get("train", {})
        tag = tr.get("tag")

        # 第一次 tick：报服务启动
        if not p:
            self.add("info", "service_start", "监控平台已启动",
                     f"项目目录 {ROOT} · 已载入 {len(self.events)} 条历史事件",
                     source="service")
            if tag:
                self.add("info", "run_attach", f"已接入训练记录 {tag}",
                         f"当前状态：{tr.get('status_cn')} · 第 {tr.get('epoch')}/"
                         f"{tr.get('epochs')} 轮 · 第 {tr.get('step'):,} 步")
            p.update(self._base(tr, system, artifacts))
            return list(self.events)[-40:]

        # ---- 训练状态迁移 ----
        if tag and tr.get("status") != p.get("status"):
            old, cur = p.get("status"), tr.get("status")
            if cur == "running":
                self.add("success", "train_start", f"训练已启动：{tag}",
                         f"PID {tr.get('pid')} · 目标 {tr.get('epochs')} 轮 · "
                         f"批大小 {tr.get('model', {}).get('batch')} · "
                         f"输入通道 {tr.get('model', {}).get('in_ch')}"
                         + (" · 掩码损失已开启" if tr.get("model", {}).get("mask_loss")
                            else ""),
                         source="train")
            elif cur == "finished":
                self.add("success", "train_finish", f"训练正常结束：{tag}",
                         f"共 {tr.get('epoch')} 轮 / {tr.get('step'):,} 步 · "
                         f"最终 MSE {tr.get('mse')}", source="train")
            elif cur == "stale":
                self.add("error", "train_died", f"训练进程已消失：{tag}",
                         f"PID {tr.get('pid')} 不再存活，且没有写入结束标记 → "
                         f"判定为异常中断（最后到第 {tr.get('step'):,} 步）",
                         source="train")
            elif cur == "stopped":
                self.add("warn", "train_stopped", f"训练已停止：{tag}",
                         f"停在 第 {tr.get('epoch')} 轮 / 第 {tr.get('step'):,} 步",
                         source="train")

        # ---- 新的 epoch ----
        if tag and tr.get("epoch") and tr.get("epoch") != p.get("epoch") \
                and tr.get("status") in ("running", "stopped", "stale"):
            ep = tr["epoch"]
            mse = tr.get("mse")
            delta = tr.get("mse_delta")
            txt = f"第 {ep}/{tr.get('epochs')} 轮完成"
            if mse is not None:
                txt += f" · MSE {mse:.5f}"
            if delta is not None:
                txt += f"（{'↓' if delta < 0 else '↑'}{abs(delta):.5f}）"
            if tr.get("epoch_seconds"):
                txt += f" · 本轮 {tr['epoch_seconds']} 秒"
            if tr.get("vram_gb"):
                txt += f" · 峰值显存 {tr['vram_gb']} GB"
            self.add("info", "epoch", txt,
                     f"累计第 {tr.get('step'):,} 步"
                     + (f" · 预计剩余 {_fmt_dur(tr['eta_s'])}" if tr.get("eta_s") else ""),
                     source="train")

        # ---- 最优 MSE ----
        best = tr.get("mse_best")
        if best is not None and p.get("mse_best") is not None and best < p["mse_best"] - 1e-9:
            self.add("success", "best_mse", f"新的最优 MSE：{best:.5f}",
                     f"较此前最优 {p['mse_best']:.5f} 改善 "
                     f"{(p['mse_best'] - best) / max(p['mse_best'], 1e-9) * 100:.2f}%",
                     source="train")

        # ---- MSE 连续上升 ----
        rise = p.get("rise_streak", 0)
        if tr.get("mse_delta") is not None and tr.get("mse_delta") > 0:
            rise += 1
        else:
            rise = 0
        p["rise_streak"] = rise
        if rise == THRESHOLDS["mse_rise_warn"]:
            self.add("warn", "mse_rise", f"MSE 连续 {rise} 轮上升",
                     f"当前 {tr.get('mse')} · 可能是学习率过大或数据分布异常，建议查看曲线",
                     source="train")

        # ---- 新产物 ----
        if artifacts.get("grids") and artifacts["grids"] != p.get("grids"):
            g = artifacts.get("latest_grid") or {}
            self.add("info", "sample_grid", f"新的样本快照（第 {g.get('step'):,} 步）",
                     "训练脚本用固定噪声采样，同一格在不同快照间可直接对比进度",
                     source="artifacts")
        if artifacts.get("skins") and artifacts["skins"] != p.get("skins"):
            self.add("success", "skins_export", f"皮肤产出已更新（共 {artifacts['skins']} 张）",
                     "在「皮肤产出」面板可预览 2D 人物拼合视图",
                     source="artifacts")
        ck = artifacts.get("ckpt_mtime")
        if ck and p.get("ckpt_mtime") and abs(ck - p["ckpt_mtime"]) > 1:
            self.add("info", "checkpoint", "已保存新断点",
                     f"{artifacts.get('ckpt_latest')} · "
                     f"{artifacts.get('ckpt_size_mb')} MB", source="artifacts")

        # ---- 系统阈值 ----
        gpus = system.get("gpus") or []
        if gpus:
            g0 = gpus[0]
            temp = g0.get("temp_c")
            if temp is not None:
                if temp >= THRESHOLDS["gpu_temp_error"] and p.get("gpu_temp_lv") != "error":
                    self.add("error", "gpu_temp", f"显卡温度过高：{temp:.0f}°C",
                             "已超过 86°C，长时间高温会降频并影响训练稳定性",
                             source="system")
                    p["gpu_temp_lv"] = "error"
                elif THRESHOLDS["gpu_temp_warn"] <= temp < THRESHOLDS["gpu_temp_error"] \
                        and p.get("gpu_temp_lv") not in ("warn", "error"):
                    self.add("warn", "gpu_temp", f"显卡温度偏高：{temp:.0f}°C",
                             "超过 80°C 告警线，注意机箱散热", source="system")
                    p["gpu_temp_lv"] = "warn"
                elif temp < THRESHOLDS["gpu_temp_warn"] - 3:
                    p["gpu_temp_lv"] = None
            mp = g0.get("mem_percent")
            if mp is not None and mp >= THRESHOLDS["gpu_mem_warn"] \
                    and p.get("gpu_mem_lv") != "warn":
                self.add("warn", "gpu_mem", f"显存占用 {mp:.0f}%",
                         f"已用 {g0.get('mem_used_mb', 0):.0f} / "
                         f"{g0.get('mem_total_mb', 0):.0f} MB，接近上限", source="system")
                p["gpu_mem_lv"] = "warn"
            elif mp is not None and mp < THRESHOLDS["gpu_mem_warn"] - 5:
                p["gpu_mem_lv"] = None

        disk = system.get("disk_project") or {}
        free = disk.get("free_gb")
        if free is not None:
            if free < THRESHOLDS["disk_free_warn_gb"] and p.get("disk_lv") != "warn":
                self.add("warn", "disk", f"项目盘剩余空间不足：{free} GB",
                         f"{disk.get('drive')} 仅剩 {free} GB，模型断点与数据集可能写不下",
                         source="system")
                p["disk_lv"] = "warn"
            elif free > THRESHOLDS["disk_free_warn_gb"] + 5:
                p["disk_lv"] = None

        cpu = system.get("cpu_percent") or 0
        if cpu >= THRESHOLDS["cpu_warn"] and p.get("cpu_lv") != "warn":
            self.add("info", "cpu_high", f"CPU 占用 {cpu:.0f}%",
                     "可能有其它任务在抢资源", source="system")
            p["cpu_lv"] = "warn"
        elif cpu < 80:
            p["cpu_lv"] = None

        # ---- 心跳丢失（进程还在但久不写心跳）----
        if tr.get("status") == "running" and tr.get("heartbeat_age_s") is not None:
            grace = THRESHOLDS["heartbeat_grace_s"]
            if tr["heartbeat_age_s"] > grace and p.get("hb_lv") != "warn":
                self.add("warn", "heartbeat_stale",
                         f"心跳已 {_fmt_dur(tr['heartbeat_age_s'])} 未更新",
                         f"进程仍在，但超过 {_fmt_dur(grace)} 没有写入训练进度 → "
                         f"可能卡在采样/数据读取", source="train")
                p["hb_lv"] = "warn"
            elif tr["heartbeat_age_s"] < grace / 2:
                p["hb_lv"] = None

        p.update(self._base(tr, system, artifacts))
        return new

    @staticmethod
    def _base(tr: dict, system: dict, art: dict) -> dict:
        return {
            "status": tr.get("status"), "epoch": tr.get("epoch"),
            "step": tr.get("step"), "mse_best": tr.get("mse_best"),
            "grids": art.get("grids"), "skins": art.get("skins"),
            "ckpt_mtime": art.get("ckpt_mtime"),
        }

    def snapshot(self, since_id: int = 0) -> dict:
        items = [e for e in self.events if e.get("id", 0) > since_id]
        counts = {"info": 0, "success": 0, "warn": 0, "error": 0}
        for e in self.events:
            counts[e.get("level", "info")] = counts.get(e.get("level", "info"), 0) + 1
        return {"items": items[-200:], "counts": counts,
                "total": len(self.events),
                "last_id": self.events[-1]["id"] if self.events else 0}


def _fmt_dur(sec) -> str:
    if sec is None:
        return "—"
    sec = int(sec)
    if sec < 60:
        return f"{sec} 秒"
    if sec < 3600:
        return f"{sec // 60} 分 {sec % 60} 秒"
    return f"{sec // 3600} 小时 {sec % 3600 // 60} 分"


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def artifacts_overview(tag: str | None) -> dict:
    grids = list_sample_grids(tag) if tag else []
    ck = list_checkpoints(tag) if tag else []
    skins = list_skins()
    return {
        "grids": len(grids), "latest_grid": grids[-1] if grids else None,
        "all_grids": grids[-60:],
        "checkpoints": ck,
        "checkpoints_all": all_checkpoints(),
        "ckpt_mtime": ck[0]["mtime"] if ck else 0,
        "ckpt_latest": ck[0]["name"] if ck else None,
        "ckpt_size_mb": ck[0]["size_mb"] if ck else None,
        "skins": sum(g["count"] for g in skins),
        "skin_groups": skins,
        "generated": list_generated(),
    }


_LAST_GOOD_RUN = {"tag": None}


def pick_active_run() -> str | None:
    """选「当前关注的训练」。

    旧仓库是在跑的优先、否则取最近活跃的（多实验竞争注意力）；
    本发布只有一个实验，直接返回 ``SINGLE_TAG``（logs 里有它的目录就盯它）。
    """
    if os.path.isdir(os.path.join(LOGS, SINGLE_TAG)):
        return SINGLE_TAG
    runs = list_runs()
    return runs[0]["tag"] if runs else None


# ---- 系统历史环（给前端画系统曲线）----------------------------------------

_HISTORY: deque = deque(maxlen=1800)      # 2 秒一条 → 1 小时
_HIST_LOCK = threading.Lock()


def _load_history() -> None:
    d = _read_json(HISTORY_PATH)
    if isinstance(d, list):
        for x in d[-900:]:
            _HISTORY.append(x)


def push_history(snap: dict) -> None:
    g = (snap.get("gpus") or [{}])[0]
    with _HIST_LOCK:
        _HISTORY.append({
            "t": snap["ts"],
            "cpu": snap.get("cpu_percent"),
            "ram": (snap.get("memory") or {}).get("percent"),
            "gpu_util": g.get("util_percent"),
            "gpu_mem": g.get("mem_percent"),
            "gpu_temp": g.get("temp_c"),
            "gpu_power": g.get("power_w"),
        })


def history() -> list[dict]:
    with _HIST_LOCK:
        return list(_HISTORY)


def save_history() -> None:
    with _HIST_LOCK:
        data = list(_HISTORY)[-900:]
    try:
        tmp = HISTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, HISTORY_PATH)
    except OSError:
        pass


_load_history()


if __name__ == "__main__":       # 便于命令行自检
    import json as _j
    print("ROOT =", ROOT)
    print("runs =", _j.dumps(list_runs(), ensure_ascii=False, indent=1)[:600])
    t = pick_active_run()
    print("active =", t)
    print(_j.dumps(run_state(t), ensure_ascii=False, indent=1)[:1400])
    print(_j.dumps(system_snapshot(), ensure_ascii=False, indent=1)[:900])
    print(_j.dumps(artifacts_overview(t), ensure_ascii=False, indent=1)[:600])
