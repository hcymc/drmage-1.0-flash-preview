# data/ —— 数据目录说明

本目录随发布附带**处理好的最终数据集**，开箱即可训练/推理，无需重跑采集管线。
文件的生成方式与完整重建命令见 `docs/REPRODUCE.md` §4。

| 文件 | 说明 |
|---|---|
| `processed/{train,val,test}.npy` | `(N,4,64,64)` uint8 CHW RGBA：train 105,605 / val 13,200 / test 13,200 |
| `processed/*_cond.npy` | `(N,36)` float32 条件向量（布局见 `docs/ARCHITECTURE.md` §3） |
| `processed/*_sids.json` | 行号 → 样本 ID（sha256 前缀），与 manifest 对齐用 |
| `processed/meta.json` | 数据集构建参数 |
| `clean_manifest.csv` | 去重后 132,090 张的宽表清单（三源合并） |

数据的上游来源与许可见 `docs/MODEL_CARD.md` §2/§6：
`MihaiPopa-1/minecraft-skins-1.1m-deduped-64x64`（Apache-2.0）、
`summykai/minecraft-skins-captioned-900k`（MIT）、MineSkin API 自采。

只想推理、不打算重训的话，整个 `data/` 都可以删——推理只需要 `models/` 里的
权重、mask 库与 `webui/`。
