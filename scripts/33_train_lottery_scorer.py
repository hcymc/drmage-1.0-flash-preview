#!/usr/bin/env python
"""33_train_lottery_scorer.py — 抽奖模式的学习型评分器（真实 vs 生成 判别器）。

为什么不用手搓特征
------------------
手搓五组件（色块 / 镜像 / 脸部 / 颜色 / 贴合）有**方向性偏置**：
平涂皮肤天然拿满结构分，精细设计反而被「色块不够大 / 脸部相邻相等率低」
扣分——实测一张细节丰富的优质生成图 0.695 分输给一张纯色西装 0.947。
「这张图整体像不像真的 Minecraft 皮肤」是一个无法枚举规则的高维直觉，
所以直接学：**小 CNN 判别器**，真实皮肤 vs 生成皮肤，评分 = P(real)。

防止捷径（本项目一贯的口径纪律）
--------------------------------
判别器最容易学的不是「质量」而是「预处理差异」，所以两侧样本过
**完全相同**的后处理管线：

* 生成侧：检索真实 mask（mask_bank）+ 自适应量化 + 破洞修补；
* 真实侧：同样过 自适应量化（K 逐张随机 ∈ {16,32,48,64}，让判别器对
  量化档位不敏感）+ 破洞修补；
* 两侧的 alpha 都是真实 mask 分布（生成侧来自 mask 库检索）；
* 类别均衡：n_real = n_gen；验证集（val 侧）与训练集（train 侧）严格分离。

用法
----
    python scripts/33_train_lottery_scorer.py generate --n-gen 8000   # 生成缓存（~15 分钟）
    python scripts/33_train_lottery_scorer.py train                   # 训练 + AUC 验证（~3 分钟）
    python scripts/33_train_lottery_scorer.py all --n-gen 8000

产物
----
* ``models/lottery_scorer/latest.pt``   判别器权重 + 验证 AUC + 配置
* ``models/lottery_scorer/gen_train.npy`` / ``gen_val.npy``   生成侧缓存（uint8 HWC）
* ``models/lottery_scorer/real_train.npy`` / ``real_val.npy`` 真实侧缓存（同管线预处理）
* ``logs/lottery_scorer/train.log``     训练日志
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "webui", "infer"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
ANNOT = os.path.join(ROOT, "labels", "annotations.jsonl")
OUT = os.path.join(ROOT, "models", "lottery_scorer")
LOGS = os.path.join(ROOT, "logs", "lottery_scorer")
CKPT = os.path.join(ROOT, "models", "diff_v2_masked", "ema.pt")

QUANT_K_CHOICES = (16, 32, 48, 64)


# ---------------------------------------------------------------------------
# 公共预处理：量化 + 修补（真实侧 / 生成侧同一条管线）
# ---------------------------------------------------------------------------

def _postprocess(arr_hwc: np.ndarray, k: int) -> np.ndarray:
    """对一张 (64,64,4) uint8 皮肤做与推理完全一致的后处理。

    顺序必须与 ``engine.generate`` 相同：**先修补、后量化**——推理侧补进
    调色板的颜色会被量化吸附，两侧口径差一步，判别器就会学到顺序差异。
    """
    import skinheal
    out = arr_hwc.copy()
    out, _ = skinheal.heal_image(out)
    vis = out[..., 3] >= 128
    if k > 0:
        from quantize import quantize_u8
        u8 = np.clip(np.rint(out[..., :3].astype(np.float32)), 0, 255).astype(np.uint8)
        u8 = quantize_u8(u8, vis, "per", k, models_dir=os.path.join(ROOT, "models"), seed=0)
        out[..., :3] = u8
    out[out[..., 3] < 128, :3] = 0
    return out


# ---------------------------------------------------------------------------
# 步骤一：生成缓存
# ---------------------------------------------------------------------------

def cmd_generate(n_gen: int, batch: int, seed: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    import torch
    from labelset import build_cond_matrix
    import engine as E

    model, cond_dim, args, meta = E.load(os.path.relpath(CKPT, ROOT))
    dev = E.device()
    print(f"载入 {meta['rel']} | params {meta['params_m']}M | cond_dim={cond_dim}",
          flush=True)

    def _gen_split(split: str, n: int, sids_key: str) -> np.ndarray:
        with open(os.path.join(DATA, f"{sids_key}.json"), encoding="utf-8") as fh:
            sids = json.load(fh)
        rng = np.random.default_rng(seed if split == "train" else seed + 1)
        pick = rng.choice(len(sids), size=min(n, len(sids)), replace=False)
        sids_pick = [sids[i] for i in pick]
        cond = build_cond_matrix(ANNOT, sids_pick).astype(np.float32)
        arrs = np.zeros((len(sids_pick), 64, 64, 4), dtype=np.uint8)
        t0 = time.time()
        done = 0
        for s0 in range(0, len(sids_pick), batch):
            s1 = min(s0 + batch, len(sids_pick))
            cc = cond[s0:s1]
            a_np = E.make_alpha(s1 - s0, "retrieval", seed + done)
            x = model.sample(s1 - s0, device=dev,
                             cond=torch.from_numpy(cc).to(dev),
                             ddim_steps=50, eta=0.0, amp=True)
            x = ((x.clamp(-1, 1) + 1) / 2).float().cpu().permute(0, 2, 3, 1).numpy()
            a2d = a_np.reshape(s1 - s0, 64, 64)
            for i in range(s1 - s0):
                arr = np.zeros((64, 64, 4), dtype=np.uint8)
                arr[..., :3] = (np.clip(x[i], 0, 1) * 255).round().astype(np.uint8)
                arr[..., 3] = np.where(a2d[i] > 0.5, 255, 0).astype(np.uint8)
                k = int(rng.choice(QUANT_K_CHOICES))
                arrs[s0 + i] = _postprocess(arr, k)
            done = s1
            print(f"  [{split}] {done}/{len(sids_pick)}  {time.time()-t0:.0f}s", flush=True)
        return arrs

    def _real_split(split: str, n: int, seed_off: int) -> np.ndarray:
        arr_all = np.load(os.path.join(DATA, f"{split}.npy"), mmap_mode="r")
        rng = np.random.default_rng(seed + seed_off)
        pick = rng.choice(len(arr_all), size=min(n, len(arr_all)), replace=False)
        out = np.zeros((len(pick), 64, 64, 4), dtype=np.uint8)
        t0 = time.time()
        for j, i in enumerate(pick):
            a = np.asarray(arr_all[i]).transpose(1, 2, 0)      # CHW → HWC
            k = int(rng.choice(QUANT_K_CHOICES))
            out[j] = _postprocess(a, k)
            if (j + 1) % 1000 == 0:
                print(f"  [{split}-real] {j+1}/{len(pick)}  {time.time()-t0:.0f}s", flush=True)
        return out

    n_val = max(1000, n_gen // 6)
    for name, arr in (("gen_train", _gen_split("train", n_gen, "train_sids")),
                      ("gen_val", _gen_split("val", n_val, "val_sids")),
                      ("real_train", _real_split("train", n_gen, 2)),
                      ("real_val", _real_split("val", n_val, 3))):
        np.save(os.path.join(OUT, f"{name}.npy"), arr)
        print(f"saved {name}.npy {arr.shape}", flush=True)
    print("GENERATE_DONE", flush=True)


# ---------------------------------------------------------------------------
# 步骤二：训练判别器
# ---------------------------------------------------------------------------

def build_model() -> "torch.nn.Module":
    import torch
    import torch.nn as nn

    def sn(ci, co, s=2):
        # 谱归一化卷积（无 BN）：BN 会把逐通道统计在批内平均掉，
        # 恰好抹平「局部伪影」这类最关键的判别信号；GAP 同罪——
        # 第一次训练（BN+GAP）val AUC=0.49（=瞎猜），训练损失却在降，
        # 说明模型只能靠记忆 train 集、局部信号全被平均没。改为
        # 谱归一化 + 全空间展平，让 4×4 上的局部结构直接进分类头。
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ci, co, 3, s, 1)), nn.LeakyReLU(0.2, True))

    class Disc(nn.Module):
        """输入 (B,4,64,64) ∈ [0,1]；输出 real logit。"""

        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(
                sn(4, 32),      # 32×32
                sn(32, 64),     # 16×16
                sn(64, 128),    # 8×8
                sn(128, 128),   # 4×4
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.LeakyReLU(0.2, True),
                nn.Linear(256, 1))

        def forward(self, x):
            return self.head(self.feat(x)).squeeze(-1)

    return Disc()


def _load_pair(name: str) -> np.ndarray:
    return np.load(os.path.join(OUT, f"{name}.npy"), mmap_mode="r")


def cmd_train(epochs: int, batch: int, lr: float, seed: int) -> None:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(LOGS, exist_ok=True)
    logf = open(os.path.join(LOGS, "train.log"), "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    gen_tr, real_tr = _load_pair("gen_train"), _load_pair("real_train")
    gen_va, real_va = _load_pair("gen_val"), _load_pair("real_val")
    n = min(len(gen_tr), len(real_tr))
    say(f"train: gen {len(gen_tr)} / real {len(real_tr)} → 用 {n} 对 | "
        f"val: gen {len(gen_va)} / real {len(real_va)}")

    model = build_model().to(device)
    say(f"判别器参数 {sum(p.numel() for p in model.parameters())/1e3:.0f}K")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    def epoch_step(split_gen, split_real, n_iter, train: bool) -> float:
        rng = np.random.default_rng(seed + (0 if train else 999))
        gi = rng.choice(len(split_gen), size=n_iter * batch // 2, replace=len(split_gen) < n_iter * batch // 2)
        ri = rng.choice(len(split_real), size=n_iter * batch // 2, replace=len(split_real) < n_iter * batch // 2)
        model.train(train)
        tot, nb = 0.0, 0
        for s0 in range(0, len(gi), batch // 2):
            g = np.asarray(split_gen[gi[s0:s0 + batch // 2]])
            r = np.asarray(split_real[ri[s0:s0 + batch // 2]])
            x = np.concatenate([r, g], axis=0).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            # 标签平滑（real=0.9）：判别器置信度过高会过早饱和
            y = torch.cat([torch.full((len(r),), 0.9), torch.zeros(len(g))]).to(device)
            xt = torch.from_numpy(x).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logit = model(xt)
                loss = lossf(logit.float(), y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += float(loss.detach()) * len(y)
            nb += len(y)
        return tot / max(nb, 1)

    best_auc, best_state = 0.0, None
    for ep in range(epochs):
        tr_loss = epoch_step(gen_tr, real_tr, n // batch, train=True)
        # 验证 AUC（rank-based，免依赖 sklearn）
        with torch.no_grad():
            def scores(arr):
                out = []
                for s0 in range(0, len(arr), 256):
                    x = np.asarray(arr[s0:s0 + 256]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        out.append(torch.sigmoid(model(torch.from_numpy(x).to(device)).float()).cpu().numpy())
                return np.concatenate(out)
            sr, sg = scores(real_va), scores(gen_va)
        y = np.concatenate([np.ones(len(sr)), np.zeros(len(sg))])
        p = np.concatenate([sr, sg])
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(p) + 1)
        auc = (ranks[y == 1].sum() - len(sr) * (len(sr) + 1) / 2) / (len(sr) * len(sg))
        say(f"epoch {ep+1}/{epochs} | train_loss {tr_loss:.4f} | val AUC {auc:.4f} | "
            f"real score med {np.median(sr):.3f} | gen score med {np.median(sg):.3f} | "
            f"gen spread p10-p90 [{np.quantile(sg, .1):.3f}, {np.quantile(sg, .9):.3f}]")
        if auc >= best_auc:
            best_auc, best_state = auc, {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    with torch.no_grad():
        def scores(arr):
            out = []
            for s0 in range(0, len(arr), 256):
                x = np.asarray(arr[s0:s0 + 256]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out.append(torch.sigmoid(model(torch.from_numpy(x).to(device)).float()).cpu().numpy())
            return np.concatenate(out)
        sr, sg = scores(real_va), scores(gen_va)
    auc = best_auc

    ck = {"model": model.state_dict(), "config": {"epochs": epochs, "batch": batch, "lr": lr,
                                                  "quant_k_choices": list(QUANT_K_CHOICES)},
          "val_auc": auc,
          "val_stats": {"real_median": float(np.median(sr)), "gen_median": float(np.median(sg)),
                        "gen_p90": float(np.quantile(sg, 0.9)), "gen_p10": float(np.quantile(sg, 0.1))},
          "args": {"n_gen": int(len(gen_tr))}}
    os.makedirs(OUT, exist_ok=True)
    torch.save(ck, os.path.join(OUT, "latest.pt"))
    say(f"saved {os.path.join(OUT, 'latest.pt')} | AUC {auc:.4f}")
    say("TRAIN_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["generate", "train", "all"])
    ap.add_argument("--n-gen", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.step in ("generate", "all"):
        cmd_generate(a.n_gen, a.batch, a.seed)
    if a.step in ("train", "all"):
        cmd_train(a.epochs, a.batch, a.lr, a.seed)


if __name__ == "__main__":
    main()
