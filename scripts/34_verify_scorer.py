#!/usr/bin/env python
"""_verify_scorer_v2.py — 评分器 v2（学习型）的验收复跑。

复现用户实测的两轮大候选抽奖（新旧评分并排对比）：
  A: n=64  seed=479184 ddim=226 K=32
  B: n=128 seed=479184 ddim=108 K=32
两轮都是 final_k=4。对每轮输出：
  * 旧排名 top-4（rule_total，即 v1 上线的排序键）
  * 新排名 top-4（learned P(real)，v2 排序键）
  * 指定关注候选（val 条件 #15 等）的新旧名次
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "http://127.0.0.1:8851"
WATCH = {15, 47}          # val 条件下标：#15 = 用户展示的粉皮肤候选


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def run_lot(n_pool: int, seed: int, ddim: int, k: int) -> dict:
    op = {"n_pool": n_pool, "batch": min(n_pool, 128), "final_k": 4}
    j = post("/api/generate", {
        "cond": dict(op, mode="lottery"), "ddim": ddim, "seed": seed,
        "quant": {"mode": "adaptive", "k": k}, "alpha": "retrieval"})
    if "error" in j:
        raise SystemExit(f"拒绝：{j['error']}")
    for _ in range(600):
        time.sleep(1)
        job = get("/api/job?id=" + j["job"])
        if job["state"] == "done":
            return job
        if job["state"] == "error":
            raise SystemExit("任务失败：" + str(job.get("error")))
    raise SystemExit("超时")


def report(tag: str, job: dict) -> None:
    items = job["items"]
    old_rank = {it["idx"]: r for r, it in enumerate(
        sorted(items, key=lambda x: -x["score_parts"]["rule_total"]), 1)}
    new_rank = {it["idx"]: r for r, it in enumerate(
        sorted(items, key=lambda x: -x["score"]), 1)}
    print(f"\n===== {tag}（{len(items)} 候选）=====")
    print("旧 top-4（规则分 v1）:")
    for it in sorted(items, key=lambda x: -x["score_parts"]["rule_total"])[:4]:
        print(f"  cond#{it['real_index']:>3}  规则 {it['score_parts']['rule_total']:.4f}"
              f"  → 新分 {it['score']:.4f}（新名次 {new_rank[it['idx']]}）")
    print("新 top-4（判别器 v2）:")
    for it in items[:4]:
        print(f"  cond#{it['real_index']:>3}  新分 {it['score']:.4f}"
              f"  P(real) 排名（旧名次 {old_rank[it['idx']]}，规则 {it['score_parts']['rule_total']:.4f}）")
    for w in WATCH:
        hit = [it for it in items if it["real_index"] == w]
        if hit:
            it = hit[0]
            print(f"  关注 cond#{w}: 旧名次 {old_rank[it['idx']]}/{len(items)}"
                  f"（规则 {it['score_parts']['rule_total']:.4f}）→ "
                  f"新名次 {new_rank[it['idx']]}/（新分 {it['score']:.4f}）")
    print(f"  summary: {json.dumps(job['lottery'], ensure_ascii=False)[:200]}")


def main() -> None:
    report("A: n=64 seed=479184 ddim=226 K=32", run_lot(64, 479184, 226, 32))
    report("B: n=128 seed=479184 ddim=108 K=32", run_lot(128, 479184, 108, 32))
    print("\nVERIFY_SCORER_V2_DONE")


if __name__ == "__main__":
    main()
