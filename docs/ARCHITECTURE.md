# ARCHITECTURE.md —— 模型与代码结构

> 面向想改代码的人。读完应该能回答：一张皮肤是怎么生成的、
> 条件向量每一位是什么、损失在哪算、alpha 从哪来。
> 训练快照与推理管线的差异消融（第二层收益的来源）见
> `docs/PIPELINE_ABLATION.md`。

---

## 1. 端到端数据流

```
                      ┌─ 训练 ──────────────────────────────────────┐
data/processed/train.npy (N,4,64,64) uint8, memmap
  │  取 batch → [:, :3] 除 127.5 减 1 → x0 (RGB, [-1,1])
  │  取 batch → [:, 4] ≥128 → a_mask（可见性掩码，0/1）
  │  labels/annotations.jsonl → build_cond_matrix → cond (36)
  ▼
SmallUNet(xt, t, cond)  ──预测噪声──→  masked MSE（只算 a_mask 内的像素）
  ▲                                        │
  │  q_sample(x0, t, ε)                    └→ AdamW + AMP + 梯度裁剪 + EMA
  │
  └─ t ~ U(1..1000)，cosine schedule

                      └─ 推理 ──────────────────────────────────────┘
cond (36, spec 或 val 真实抽样)
  │
  ├─ models/mask_bank.npz ──检索──→ alpha mask (64,64) ∈ {0,1}
  ▼
DDIM 50 步（η=0，EMA 权重） → x (RGB, [-1,1])
  │
  ├─ 可选：逐图自适应 k-means 量化（K=64 实测最贴真实分布）
  ▼
RGB × alpha → 64×64 RGBA PNG（alpha 严格 0/255，透明区 RGB 归零）
```

## 2. 关键设计

### 2.1 alpha 与内容解耦（3 通道模型）

模型只学 RGB（`--channels 3`）；alpha 在训练时表现为**掩码**（损失只在
可见像素上算），在推理时表现为**检索到的真实 mask**。这样：

* 根除了「alpha 学成半透明雾 / overlay 灌黑」两类历史 bug
  （后者正是 `diff_v2_masked` 相对 `diff_v1` 的修复点：`--mask-loss`）；
* 推理侧 mask 必须与训练侧 alpha **同分布**——所以从 4 万张真实皮肤
  建 mask 库（`models/mask_bank.npz`）做检索，而不是逐像素 i.i.d. 合成
  （合成模板只有逐面覆盖率对、空间结构为零，实测会让第二层出椒盐/穿孔）。

### 2.2 条件注入

36 维条件向量经两层 MLP 投影后在 UNet 的每个分辨率上做 **FiLM 式调制**
（scale/shift）；同一套 `models.SmallUNet` 同时支持无条件模式（`--cond` 不开）。

> 注：本发布的权重训练时 FiLM 层尚未加入代码，加载时这些键缺失、被
> 零初始化（等价于无 FiLM）；`build_diffusion` 对此有显式校验
> ——只放行 `.film.` 缺键，其它缺键直接报错，避免静默错配出坏图。

### 2.3 掩码损失与 EMA

* `masked_loss`：MSE 只在 `a_mask` 内的像素上求均值。全图口径会把
  55% 的透明像素也算进去，overlay 区域（有效可见像素只有 base 的 1/6.7）
  的梯度被稀释——掩码是修复「第二层学不到东西」的前提。
* EMA（decay 0.9995，前 1000 步 warmup）：采样与交付一律用 EMA 权重；
  `30_generate.build_diffusion(prefer_ema=True)` 会优先读 checkpoint 里
  的 EMA shadow（含 `latest.pt`——旧版这里是个死参数，已修复）。

### 2.4 采样与后处理

* DDIM 50 步、η=0、固定种子；无 CFG（本权重训练时没开条件 dropout，
  cfg 必须 = 1，UI 已按此锁定）。
* 可选逐图自适应 k-means 量化（`lib/quantize.py::quantize_u8`，mode=per）：
  K=64 时唯一色 64 对真实 66、饱和 0.356 对 0.353，是最贴真实的后处理档位；
  全局调色板路线实测过冲且偏灰，已不作为默认。

## 3. 条件向量 36 维布局（`lib/labelset.py::vector_from_record`）

| 段 | 维度 | 含义 |
|---|---|---|
| 0:12 | 12 | 色相直方图（12 桶，可见像素） |
| 12 | 1 | 色相均值（deg/360） |
| 13:15 | 2 | 饱和度 / 明度均值 |
| 15:19 | 4 | 中性 / 暗 / 亮 / 高饱和像素占比 |
| 19:23 | 4 | tone one-hot（`TONE_ORDER`：dark/mid/bright/gray） |
| 23:26 | 3 | 饱和档 one-hot（`SAT_ORDER`） |
| 26:31 | 5 | 复杂度档 one-hot（`CPX_ORDER`，全图纹理强度） |
| 31 | 1 | 用过 overlay（全局存在位） |
| 32 | 1 | 透明像素占比 |
| 33:36 | 3 | 骨架 one-hot（`MODEL_ORDER`；训练集 100% classic） |

推理入口：`--cond-spec '{"tone":"dark",...}'`（`vector_from_spec`），
或从 val 集抽真实条件（推理 webui 的「真实抽样」模式）。

## 4. 代码地图

```
scripts/
  21_train_diffusion.py   训练主脚本（两阶段配方见 docs/REPRODUCE.md §3）
  30_generate.py          批量生成 + manifest/contact sheet；build_diffusion()
                          是权重加载的唯一实现（webui 推理端按路径复用它）
  31_validate.py          五关验证（格式/结构/alpha/分布/记忆）
  33_train_lottery_scorer.py  抽奖评分器训练（真实 vs 生成判别器，docs/LOTTERY.md §2）
  34_verify_scorer.py     评分器验收复跑（新旧排名对照）
  25_build_mask_bank.py   从训练集建真实 alpha mask 库
  02_clean / 10_label / 04_dataset / 02x_ingest / 01b_fetch_hf   数据管线
  lib/
    models.py             SmallUNet / DDPM（loss / masked_loss / sample / predict_x0；
                          sample 支持 x_T 预置噪声与 fp16 autocast）
    labelset.py           条件向量构建与 spec 解析（36 维布局的权威定义）
    skinatlas.py          UV 图集几何（face_index / overlay 盒 / mask 库 / 覆盖率）
    skinuv.py             皮肤 UV 校验、骨架判别（classic/slim 双判据）、质量三维
    trainutil.py          checkpoint / EMA 保存、心跳、样本网格、种子
    quantize.py           调色板量化（per-image k-means / global）
    losses.py             结构统计量（训练可视化探针用）
    manifest / labeling / sourceio / hfdl / transport               数据管线支撑
webui/
  server.py + collect.py + static/     训练监控面板（锁定 diff_v2_masked 单实验；
                                       Win32 API 读系统指标，纯标准库）
  infer/server.py + engine.py + geom.py + skin_score.py + skinheal.py + static/
                                       推理实验台（engine 复用 30_generate 的采样链；
                                       skin_score = 学习型评分器 + 规则分诊断；
                                       skinheal = 破洞修补；geom 把 UV 图集贴到
                                       可旋转 3D 人形）
```

## 5. 复现性与口径

* 训练日志 `logs/diff_v2_masked/metrics.jsonl` 每 100 步一条（mse / lr /
  vram / 可见率），`heartbeat.json` 为监控面板的状态来源；
  275 张固定噪声快照（同格跨步对比）在 `logs/samples/diff_v2_masked/`。
* 权重键名：`latest.pt` = `{model, opt, ema, epoch, gstep, args, rng}`；
  `ema.pt` = `{model: EMA shadow, gstep, epoch, args}`。`args` 记录了
  全部训练开关，`build_diffusion` 从它回读结构（in_ch / schedule / base），
  换权重不用改代码。
* 全部指标数字的测量口径（取样集 / 相邻边定义 / 判据分母）见
  `docs/MODEL_CARD.md` §3——引用任何数字前先看口径。
