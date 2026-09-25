"""transport.py — 可靠的 HTTP 传输层。

为什么不用 requests / aiohttp
------------------------------
本机 Python 的 HTTPS 出站**不稳定**：沙箱侧存在 TLS 中间层，会把握手数据流截断，
表现为 ``SSLError(142, '[ASN1: NOT_ENOUGH_DATA] not enough data')``。实测对
huggingface.co / api.github.com / api.mojang.com / textures.minecraft.net /
mc-heads.net / crafatar.com / api.mineskin.org 七个站点，普通 requests 请求
**0/4 全部失败**（同一批 URL 用 curl 请求 4/4 成功）。``verify=False`` 无法绕过。

因此本模块统一以 ``curl --ssl-no-revoke``（Windows Schannel 后端，作用见下）作为
传输实现：
  * ``--ssl-no-revoke``  绕开证书吊销检查。不加会报
                         ``CRYPT_E_NO_REVOCATION_CHECK (0x80092012)``。
  * ``--parallel``       批量并发下载，实测 ~17.8 req/s。
  * ``--retry``          传输层自动重试。

对外暴露三层接口：
  * ``http_get`` / ``http_get_json``  —— 单请求，带重试与 UA 轮换
  * ``fetch_many``                    —— 并发批量落盘，返回逐条结果
  * ``FetchLedger``                   —— 失败 URL 台账 + 断点续传状态
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

CURL = "curl"

#: 轮换用的 User-Agent 池
UA_POOL: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
)


def pick_ua(rng: random.Random | None = None) -> str:
    return (rng or random).choice(UA_POOL)


def _base_args(ua: str, timeout: int) -> list[str]:
    return [CURL, "-sS", "--ssl-no-revoke", "--compressed", "-A", ua, "-m", str(timeout)]


# --------------------------------------------------------------------------
# 单请求
# --------------------------------------------------------------------------

def http_get(url: str, *, retries: int = 4, timeout: int = 40,
             ua: str | None = None, backoff: float = 1.6) -> bytes | None:
    """GET 一个 URL 返回字节；全部重试失败返回 None。"""
    last_err = ""
    for attempt in range(retries):
        args = _base_args(ua or pick_ua(), timeout) + ["-L", "--fail", url]
        p = subprocess.run(args, capture_output=True)
        if p.returncode == 0 and p.stdout:
            return p.stdout
        last_err = (p.stderr or b"").decode("utf-8", "replace")[:200]
        if attempt < retries - 1:
            time.sleep(backoff ** attempt * 0.5 + random.random() * 0.4)
    return None


def http_get_json(url: str, *, retries: int = 4, timeout: int = 40,
                  ua: str | None = None) -> dict | None:
    raw = http_get(url, retries=retries, timeout=timeout, ua=ua)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# 并发批量下载
# --------------------------------------------------------------------------

@dataclass
class FetchResult:
    url: str
    path: str
    ok: bool
    bytes: int = 0
    error: str = ""


def fetch_many(entries: Sequence[tuple[str, str]], *, max_parallel: int = 12,
               timeout: int = 40, retries: int = 2, rotate_ua: bool = True,
               create_dirs: bool = True) -> list[FetchResult]:
    """并发下载 ``[(url, dest_path), ...]``。

    用 ``curl --parallel`` 单进程并发，避免起 N 个进程的开销。
    完成后逐条检查落盘文件，并**自行清理 0 字节 / 残缺文件**——这样上层可以
    无脑把 ``ok=False`` 的条目重新入队，天然支持断点续传。

    Returns:
        与输入等长的 :class:`FetchResult` 列表（顺序一致）。
    """
    if not entries:
        return []

    if create_dirs:
        for _, dest in entries:
            d = os.path.dirname(dest)
            if d:
                os.makedirs(d, exist_ok=True)

    cfg_lines: list[str] = []
    for url, dest in entries:
        safe_url = url.replace('"', "%22")
        safe_dest = os.path.abspath(dest).replace("\\", "/").replace('"', "%22")
        cfg_lines.append(f'url = "{safe_url}"')
        cfg_lines.append(f'output = "{safe_dest}"')
        cfg_lines.append('write-out = "%{http_code} %{size_download}\\n"')

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(entries[0][1])) or ".",
                            f".curlcfg_{os.getpid()}_{int(time.time()*1000)%100000}.txt")
    os.makedirs(os.path.dirname(cfg_path) or ".", exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(cfg_lines) + "\n")

    args = [CURL, "-sS", "--ssl-no-revoke", "--compressed",
            "-A", pick_ua() if rotate_ua else UA_POOL[0],
            "-m", str(timeout),
            "--parallel", "--parallel-max", str(max_parallel),
            "--retry", str(retries), "--retry-delay", "1", "--retry-connrefused",
            "-K", cfg_path.replace("\\", "/")]
    try:
        subprocess.run(args, capture_output=True)
    finally:
        try:
            os.remove(cfg_path)
        except OSError:
            pass

    # 以落盘结果为准（比解析 curl stdout 更可靠：并发时 stdout 顺序不保证）
    results: list[FetchResult] = []
    for url, dest in entries:
        if os.path.isfile(dest):
            size = os.path.getsize(dest)
            if size > 0:
                results.append(FetchResult(url, dest, True, size))
                continue
            os.remove(dest)  # 0 字节残留，清掉等重试
        results.append(FetchResult(url, dest, False, 0, "empty_or_missing"))
    return results


# --------------------------------------------------------------------------
# 台账 / 断点续传
# --------------------------------------------------------------------------

class FetchLedger:
    """失败 URL 台账 + 已见集合，用于断点续传与去重。

    * ``failed_urls.txt``  —— 逐行 ``时间戳\\tURL\\t原因``（需求硬性要求）
    * ``seen.txt``         —— 已成功处理的键（哈希 / URL），用于重启后跳过
    * ``state.json``       —— 任意游标类续传状态
    """

    def __init__(self, log_dir: str, state_dir: str):
        self.log_dir = log_dir
        self.state_dir = state_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)
        self.failed_path = os.path.join(log_dir, "failed_urls.txt")
        self.seen_path = os.path.join(state_dir, "seen.txt")
        self.state_path = os.path.join(state_dir, "state.json")
        self._seen: set[str] = set()
        self._load_seen()
        self._fh_failed = open(self.failed_path, "a", encoding="utf-8")

    def _load_seen(self) -> None:
        if os.path.isfile(self.seen_path):
            with open(self.seen_path, "r", encoding="utf-8") as fh:
                self._seen = {ln.strip() for ln in fh if ln.strip()}

    def seen(self, key: str) -> bool:
        return key in self._seen

    def mark_seen(self, *keys: str) -> None:
        new = [k for k in keys if k and k not in self._seen]
        if not new:
            return
        self._seen.update(new)
        with open(self.seen_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(new) + "\n")

    def record_failure(self, url: str, reason: str) -> None:
        self._fh_failed.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{url}\t{reason}\n")
        self._fh_failed.flush()

    def load_state(self) -> dict:
        if os.path.isfile(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def save_state(self, state: dict) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, self.state_path)

    def close(self) -> None:
        try:
            self._fh_failed.close()
        except OSError:
            pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
