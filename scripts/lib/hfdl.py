"""hfdl.py — HuggingFace 大文件并行分片下载器（curl 传输层，支持断点续传）。

本机 Python 的 TLS 出站不可靠（见 transport.py 的说明），所以一律以 ``curl
--ssl-no-revoke`` 作为传输层。HF 现在会 302 到 xethub 的 CAS，curl ``-L``
可以正常跟随，且 Range 请求在整个重定向链上都被保留。

核心思路
--------
把远端文件切成固定大小的分片，用 N 个线程各自 ``curl -r a-b`` 抓一个分片，
直接 seek 写到目标文件的对应偏移。每个分片落在一个临时文件里，写完立刻删除。
完成的分片记进 ``<dest>.state.json``，所以中断后重启只补没下完的部分。

实测（本机，hf-mirror.com）：单连接 ~1.9 MB/s，8 并发可到 ~7.5 MB/s；
12 并发反而掉到 2.6 MB/s（被限速）。所以默认 ``workers=8``。
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import subprocess
import threading
import time

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

#: HF 官方域名；本机实测 hf-mirror.com 快约 1 倍，可用 ``MIRROR`` 覆盖
HF_HOST = "https://huggingface.co"
MIRROR = "https://hf-mirror.com"


def _curl_base(url: str) -> list[str]:
    return ["curl", "-sS", "--ssl-no-revoke", "-L", "-A", UA, url]


def remote_size(url: str, timeout: int = 40) -> int:
    """HEAD 取文件大小（跟随重定向）。失败返回 -1。"""
    cmd = ["curl", "-sS", "--ssl-no-revoke", "-L", "-I", "-A", UA, "-m", str(timeout), url]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout + 20)
    size = -1
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())   # 取最后一个（重定向后真实的）
            except ValueError:
                pass
    return size


class ParallelDownloader:
    def __init__(self, workers: int = 8, chunk_mb: int = 16, timeout: int = 120,
                 log=print):
        self.workers = workers
        self.chunk = int(chunk_mb * 1024 * 1024)
        self.timeout = timeout
        self.log = log
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def download(self, url: str, dest: str, expect_size: int | None = None) -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        state_path = dest + ".state.json"

        size = expect_size if expect_size and expect_size > 0 else remote_size(url)
        if size <= 0:
            self.log(f"  [x] 无法取得大小: {url}")
            return False

        if os.path.exists(dest) and os.path.getsize(dest) == size and not os.path.exists(state_path):
            self.log(f"  [=] 已完整，跳过 ({size/1e6:.1f}MB): {os.path.basename(dest)}")
            return True

        st = {"url": url, "size": size, "done": []}
        if os.path.exists(state_path):
            try:
                old = json.load(open(state_path, encoding="utf-8"))
                if old.get("url") == url and old.get("size") == size:
                    st = old
            except Exception:
                pass

        n_chunks = (size + self.chunk - 1) // self.chunk
        done = set(st.get("done") or [])
        todo = [i for i in range(n_chunks) if i not in done]

        # 预分配目标文件
        if not os.path.exists(dest) or os.path.getsize(dest) != size:
            with open(dest, "wb") as fh:
                if size > 0:
                    fh.truncate(size)

        if not todo:
            self.log(f"  [=] 全部分片已完成: {os.path.basename(dest)}")
            self._finish(state_path)
            return True

        self.log(f"  [>] {os.path.basename(dest)}  {size/1e6:.1f}MB  分片 {len(todo)}/{n_chunks}  "
                 f"并发 {self.workers}")
        t0 = time.time()
        written = [0]
        fail = []

        def work(idx: int):
            a = idx * self.chunk
            b = min(a + self.chunk, size) - 1
            tmp = f"{dest}.part{idx:05d}"
            for attempt in range(1, 6):
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                cmd = ["curl", "-sS", "--ssl-no-revoke", "-L", "-A", UA, "-m", str(self.timeout),
                       "-r", f"{a}-{b}", "-o", tmp, url]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True,
                                       errors="replace", timeout=self.timeout + 30)
                except subprocess.TimeoutExpired:
                    r = None
                if r is not None and os.path.exists(tmp):
                    got = os.path.getsize(tmp)
                    if got == b - a + 1:
                        with open(tmp, "rb") as src, open(dest, "r+b") as dst:
                            dst.seek(a)
                            dst.write(src.read())
                        os.remove(tmp)
                        with self._lock:
                            done.add(idx)
                            written[0] += got
                            st["done"] = sorted(done)
                            json.dump(st, open(state_path, "w", encoding="utf-8"))
                        return True
                time.sleep(min(2 ** attempt, 12))
            with self._lock:
                fail.append(idx)
            return False

        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(work, todo))

        if fail:
            self.log(f"  [x] 有 {len(fail)} 个分片失败，下次重跑会续传: {fail[:8]}")
            return False

        self._finish(state_path)
        dt = time.time() - t0
        self.log(f"  [v] 完成 {os.path.basename(dest)}  {size/1e6:.1f}MB  {dt:.1f}s  "
                 f"({size/1e6/max(dt,0.001):.2f} MB/s)")
        return True

    def _finish(self, state_path: str):
        try:
            if os.path.exists(state_path):
                os.remove(state_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    def hf_url(self, repo: str, filename: str, kind: str = "datasets",
               mirror: bool = True) -> str:
        host = MIRROR if mirror else HF_HOST
        return f"{host}/{kind}/{repo}/resolve/main/{filename}"


def download(url: str, dest: str, workers: int = 8, chunk_mb: int = 16, log=print) -> bool:
    """便捷入口。"""
    return ParallelDownloader(workers=workers, chunk_mb=chunk_mb, log=log).download(url, dest)
