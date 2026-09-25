# REPRODUCE.md —— 从零复现指南

> 本仓库附带**处理好的数据集**（`data/processed/`）与全部训练产物，
> 所以「复现」有三档，按需选择：
>
> | 档 | 需要什么 | 耗时 |
> |---|---|---|
> | A. 只想出图 | 权重（已附带） | 0 |
> | B. 从数据集重训 | 附带的数据集 + 8GB GPU | 阶段一 ~1.5h + 阶段二 ~21h（RTX 2080S 实测） |
> | C. 连数据集一起重建 | 三源原始数据 + 全链路脚本 | 数小时（网络 + CPU） |

> **诚实声明**：本文给出的是与发布权重**同配置、同口径**的复现配方。
> 由于数据抽取顺序 / cuDNN 非确定性等因素，重新训练**不会得到逐位相同**
> 的权重；发布在 `models/` 里的权重就是原始训练产物本身（训练日志
> `logs/diff_v2_masked/` 与之一一对应，可交叉验证）。

---

## 1. 环境

实测环境（其余相近版本大概率可用）：

| 项 | 值 |
|---|---|
| OS | Windows 10 x64（webui 用了 Win32 API；训练/生成脚本本身跨平台） |
| Python | 3.10 |
| torch | 2.2.0+cu121 |
| numpy / pillow | 1.26.4 / 12.2.0 |
| GPU | RTX 2080 SUPER 8GB，训练峰值显存 **4.6GB**（batch 48 + AMP） |
| 磁盘 | 数据集约 2.2GB，权重约 0.5GB，日志与快照约 0.1GB |

```bash
pip install -r requirements.txt
python scripts/00_env_check.py     # 环境自检：GPU / CUDA / 依赖 / 磁盘
```

---

## 2. 档 A：出图（用发布权重）

见根 README §2。要点：出图用 **`ema.pt`**（EMA 权重明显更干净）；
`latest.pt` 里的原始权重仅供对比 / 续训。推理侧 alpha 默认从
`models/mask_bank.npz` 检索真实 mask（与训练分布一致），缺库会自动回落
到 i.i.d. 合成模板并打印警告——回落会显著变差，别删 mask_bank。

---

## 3. 档 B：从附带数据集重训（两阶段）

发布权重是**两阶段**训练的产物：

```
diff_v1（阶段一：无掩码损失的基础 DDPM）
   └─ resume ─→ diff_v2_masked（阶段二：+ 掩码损失）= 发布权重
```

为什么两阶段：阶段一的 alpha 通道由模型自己回归，实测 overlay 区域被学成
暗色（历史 bug）；阶段二起改用「RGB 目标 + 掩码损失」，把 alpha 从学习
目标里彻底拿掉（3 通道模型，alpha 由真实 mask 库外供）。

### 阶段一 · diff_v1（历史：跑到 ~81,400 步 / 37 epoch 被用作阶段二初始化）

```bash
python scripts/21_train_diffusion.py \
    --tag diff_v1 --epochs 40 --batch 48 --base 64 \
    --t-dim 256 --timesteps 1000 --schedule cosine --ddim-steps 50 \
    --channels 3 --cond --model-type classic \
    --no-mask-loss --seed 42
```

* 与历史 config 唯一的出入是 `--epochs`：原始记录是 60（即目标 60 轮），
  但实际在 37 轮（81,400 步）时被手动停止并转作阶段二初始化。
  想严格对齐历史，用 `--epochs 60` 并在第 37 轮手动停止，效果等价。
* `--no-mask-loss` 是刻意的：这就是阶段一与阶段二的**全部本质差异**。

### 阶段二 · diff_v2_masked（发布权重）

```bash
python scripts/21_train_diffusion.py \
    --tag diff_v2_masked --epochs 276 --batch 48 --base 64 \
    --t-dim 256 --timesteps 1000 --schedule cosine --ddim-steps 50 \
    --channels 3 --cond --model-type classic \
    --resume models/diff_v1/latest.pt --seed 42
```

* 历史实际过程：从 `diff_v1/latest.pt`（81,400 步）续训，首次目标 200 轮，
  之后又两次上调目标（最终 350），在 687,400 步 / 312 轮手动停止冻结。
  `--epochs 276` = 37 + 276 ≈ 313 轮，一步到位复刻总步数量级
  （每轮 2,200 步 × 48 batch）。
* 训练会自动写 `logs/diff_v2_masked/metrics.jsonl`（每 100 步一条）、
  `heartbeat.json`（监控面板据此判断状态）、每 2,200 步一张固定噪声快照
  （`logs/samples/diff_v2_masked/`，同格跨步可直接对比进步）。
* 每 20 轮落一个 `epoch_XXXX.pt` 全量断点；`latest.pt` / `ema.pt` 每轮刷新。

### 也可以从 webui 面板启动

`python webui/server.py` → 顶栏「训练控制」：预设选
**阶段二 · diff_v2_masked 官方配方**（tag 填 `diff_v2_masked`，
断点填 `models/diff_v1/latest.pt`）或**阶段一 · 预训练**。
面板会拒绝用没有 CUDA 的解释器启动训练（CPU 静默训练慢一个数量级，
且日志看不出异常——这是历史上真实踩过的坑）。

---

## 4. 档 C：数据集从零重建

> 三源数据合计约 25 万张原始皮肤，完整链路的实测数字与设计动机
> 不在本发布展开（属于上游项目），这里只给**可照抄的命令序列**。

```bash
# 1) 拉取 HuggingFace 两个公开集合（走镜像，8 路并发分片）
python scripts/01b_fetch_hf.py --all --workers 8

# 2) 三源摄取：只写宽表 CSV，不落任何中间图片
python scripts/02a_ingest_mineskin.py  --quality simple --model-gate strict
python scripts/02b_ingest_hf.py        --target 90000 --quality simple --model-gate strict
python scripts/02c_ingest_captioned.py --target 60000 --quality simple --model-gate strict

# 3) 清洗去重（纯 CSV）+ 标注（同样不读像素）
python scripts/02_clean.py --strict
python scripts/10_label.py base

# 4) 构建数据集（唯一一趟读像素）
python scripts/04_dataset.py

# 5) 构建 alpha mask 库（推理 / 训练可视化检索用）
python scripts/25_build_mask_bank.py --split train --limit 40000
```

产物核对（发布附带的数据集即由此产出）：

| 文件 | 形状 / 规模 |
|---|---|
| `data/processed/train.npy` | (105,605, 4, 64, 64) uint8，约 1.73GB |
| `data/processed/val.npy` / `test.npy` | (13,200, 4, 64, 64) |
| `*_cond.npy` | (N, 36) float32 条件矩阵 |
| `*_sids.json` | 行号 → 样本 ID（`sha256` 前缀），用于与 manifest 对齐 |
| `data/clean_manifest.csv` | 132,090 行宽表（去重后定稿） |
| `labels/annotations.jsonl` | 132,090 条四层标注 |
| `models/mask_bank.npz` | (40000, 4096) uint8 + (40000, 4) 部位位 |
| `models/alpha_rates.json` | 逐面 alpha 覆盖率（三态频率模型） |

数据源与许可见 `MODEL_CARD.md` §训练数据。

---

## 5. 生成与验证

```bash
# 批量生成（带逐样本统计与 contact sheet）
python scripts/30_generate.py --model diffusion \
    --ckpt models/diff_v2_masked/ema.pt --n 256 --batch 16 \
    --ddim-steps 50 --seed 1234 --out repro_run

# 条件生成 + 调色板量化（逐图自适应 k-means，K=64 实测最贴真实分布）
python scripts/30_generate.py --model diffusion --ckpt models/diff_v2_masked/ema.pt \
    --n 64 --cond-spec '{"tone":"dark","model_type":"classic"}' \
    --quantize 64 --out repro_dark

# 五关验证（对某个生成 run：格式 / 结构 / alpha / 分布对齐 / 记忆检查）
python scripts/31_validate.py --run repro_run
```

### 冒烟验收（本发布打包前实测通过）

```bash
python -m compileall -q scripts webui                 # 全部可编译
python scripts/30_generate.py --model diffusion --ckpt models/diff_v2_masked/ema.pt \
    --n 8 --batch 8 --ddim-steps 50 --seed 42 --out smoke_ema
#   → uv_valid 8/8，1.9s（RTX 2080S）
python scripts/21_train_diffusion.py --tag _smoke --epochs 1 --batch 8 --base 16 \
    --t-dim 64 --channels 3 --cond --sample-every 0 --max-steps 3
#   → 3 步训练 + 断点落盘，跑完删掉 models/_smoke logs/_smoke
python webui/infer/server.py --port 8851              # /api/meta + 同步生成接口
python webui/server.py --port 8849                    # /api/bootstrap 单实验锁定
```

---

## 6. 踩坑提示（都是历史真实发生过的）

* **用错 conda 环境**：`pytorch` 环境装的是 CPU 版 torch，训练会**静默跑 CPU**。
  训练前先 `python -c "import torch;print(torch.cuda.is_available())"`。
* **EMA 与原始权重**：出图一律用 `ema.pt`；`build_diffusion(prefer_ema=True)`
  已修复为「checkpoint 里带 EMA shadow 就优先用它」（旧版这是个死参数）。
* **别删 `models/mask_bank.npz`**：缺库时 alpha 回落 i.i.d. 模板，
  分布错配会让第二层出椒盐 / 穿孔（训练喂的是真实 alpha）。
* **webui 用 `python webui/server.py` 前台起**，别在 shell 里 `&` 后台化
  （Windows 下子进程可能被作业对象连坐回收；面板起训练时已做 breakaway 处理）。
* 训练进度看 `logs/<tag>/metrics.jsonl` / `heartbeat.json`，**不是** train.log。
