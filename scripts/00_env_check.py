#!/usr/bin/env python
"""00_env_check.py — 环境自检。

在动手之前先确认「这台机器到底能不能干这件事」，并把结论落盘成
``logs/env_report.json`` / ``logs/env_report.md``，作为可复现性证据的一部分。

检查项
------
* Python / PyTorch / CUDA 可用性，GPU 型号与**当前空闲显存**（不是标称显存）
* 关键依赖：PIL / numpy / sklearn / tqdm / imagehash / cv2
* 磁盘余量
* **传输层实测**：分别用 ``requests`` 与 ``curl --ssl-no-revoke`` 打同一批 URL，
  给出各自成功率。这一项是本机最关键的坑——见 README「环境备注」。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

PROBE_URLS = [
    ("mineskin_api", "https://api.mineskin.org/v2/skins?size=4"),
    ("textures_cdn", "https://textures.minecraft.net/texture/"
                     "292009a4925b58f02c77dadc3ecef07ea4c7472f64e0fdc32ce5522489362680"),
    ("mojang_api", "https://api.mojang.com/users/profiles/minecraft/Notch"),
    ("github_api", "https://api.github.com/rate_limit"),
    ("huggingface", "https://huggingface.co/api/datasets/vajdaad4m/minecraft-skin"),
    ("mc_heads", "https://mc-heads.net/skin/Notch"),
    ("crafatar", "https://crafatar.com/skins/069a79f444e94726a5befca90e38aaf5"),
]


def deps() -> dict:
    out = {}
    for m in ["PIL", "numpy", "sklearn", "tqdm", "imagehash", "cv2", "requests", "aiohttp"]:
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "ok")
        except Exception:
            out[m] = None
    return out


def gpu() -> dict:
    info: dict = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            info["vram_total_gb"] = round(total / 1024 ** 3, 2)
            info["vram_free_gb"] = round(free / 1024 ** 3, 2)
            info["vram_free_mib"] = round(free / 1024 ** 2)
    except Exception as e:
        info["torch_error"] = repr(e)
    return info


def disk() -> dict:
    out = {}
    for letter in "CDEFGH":
        p = f"{letter}:\\"
        if os.path.exists(p):
            try:
                u = shutil.disk_usage(p)
                out[letter] = round(u.free / 1024 ** 3, 1)
            except OSError:
                pass
    return out


def transport() -> dict:
    """对同一批 URL 用两种通道各打 3 次，比较成功率。"""
    res: dict = {}
    try:
        import requests
        import urllib3
        urllib3.disable_warnings()
    except Exception:
        requests = None

    for name, url in PROBE_URLS:
        r_ok = r_n = c_ok = c_n = 0
        last_err = ""
        if requests is not None:
            for _ in range(3):
                r_n += 1
                try:
                    resp = requests.get(url, timeout=20, verify=False,
                                        headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code < 500:
                        r_ok += 1
                except Exception as e:
                    last_err = f"{type(e).__name__}:{str(e)[:60]}"
                time.sleep(0.2)
        for _ in range(3):
            c_n += 1
            p = subprocess.run(["curl", "-sS", "--ssl-no-revoke", "-o", os.devnull,
                                "-w", "%{http_code}", "-m", "20", "-A", "Mozilla/5.0", url],
                               capture_output=True, text=True)
            code = p.stdout.strip()
            if code.isdigit() and int(code) < 500:
                c_ok += 1
        res[name] = {"requests": f"{r_ok}/{r_n}", "curl_ssl_no_revoke": f"{c_ok}/{c_n}",
                     "sample_error": last_err}
    return res


def main() -> None:
    os.makedirs(LOGS, exist_ok=True)
    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "deps": deps(),
        "gpu": gpu(),
        "disk_free_gb": disk(),
    }
    print("== 依赖 ==", json.dumps(rep["deps"], ensure_ascii=False))
    print("== GPU ==", json.dumps(rep["gpu"], ensure_ascii=False))
    print("== 磁盘剩余 (GB) ==", json.dumps(rep["disk_free_gb"], ensure_ascii=False))
    print("== 传输层实测（各 3 次）…")
    rep["transport"] = transport()
    for k, v in rep["transport"].items():
        print(f"   {k:16s} requests={v['requests']:5s} curl={v['curl_ssl_no_revoke']:5s} "
              f"{v['sample_error'][:60]}")

    with open(os.path.join(LOGS, "env_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)

    md = ["# 环境自检报告", "", f"生成时间：{rep['generated_at']}", "",
          f"- Python：`{rep['python'].splitlines()[0]}`",
          f"- 解释器：`{rep['executable']}`", "",
          "## 依赖", "", "| 包 | 版本 |", "|---|---|"]
    for k, v in rep["deps"].items():
        md.append(f"| {k} | {v or '**缺失**'} |")
    g = rep["gpu"]
    md += ["", "## GPU / PyTorch", "", "| 项 | 值 |", "|---|---|"]
    for k, v in g.items():
        md.append(f"| {k} | {v} |")
    md += ["", "## 磁盘剩余", "", "| 盘 | 剩余 GB |", "|---|---|"]
    for k, v in rep["disk_free_gb"].items():
        md.append(f"| {k} | {v} |")
    md += ["", "## 传输层实测", "",
           "对同一批 URL 各打 3 次，比较两条通道的成功率：", "",
           "| 目标 | `requests` | `curl --ssl-no-revoke` | 典型报错 |", "|---|---|---|---|"]
    for k, v in rep["transport"].items():
        md.append(f"| {k} | {v['requests']} | {v['curl_ssl_no_revoke']} | `{v['sample_error'][:70]}` |")
    md += ["", "> 结论决定数据采集用哪条通道。详见 README「环境备注」。"]
    with open(os.path.join(LOGS, "env_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("报告 -> logs/env_report.md")


if __name__ == "__main__":
    main()
