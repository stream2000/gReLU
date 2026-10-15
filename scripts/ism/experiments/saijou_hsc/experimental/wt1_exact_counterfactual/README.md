# WT1 exact counterfactual：闭项源码说明

Saijou/WT1 已由用户在 2026-10-13 确认收尾。这是历史 experimental 工作的最后版本保留，不是新实验入口的推荐，也不代表重新验证了结果。

## 保存范围与来源

本次只保存以下三个源文件，保持与共享主工作区逐字节一致：

- `run_all_tracks_readout.py`：保留 `--weights`，模型加载和运行元数据使用所选 checkpoint。
- `analyze_all_tracks_readout.py`：保留 `--run`，读写指定的运行目录。
- `compare_checkpoints.py`：保存原来未跟踪的两个 checkpoint 读出比较脚本。

来源为 `/work2/Users/qijun/gReLU-replication` 的 `alphagenome` 工作区，基准 commit `9fbb943b56d9ffe1ee8956f87f89eb4b7ce2ad8d`，读取时间 `2026-10-13T12:45:21.199202+00:00`。前两个文件包含用户未提交改动；第三个文件未跟踪。源码读取时的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `run_all_tracks_readout.py` | `573bbbf7e9973e5557b12232f7836e64d3dedc745938c49b6a1ddeb34c9ba723` |
| `analyze_all_tracks_readout.py` | `0c07ac16aded15c2e3b6002662c57b2d8403b072e98c22f81b80c31176e7fd69` |
| `compare_checkpoints.py` | `60e0168e32f213c336b3eda6fd640b33e2ccfb293f157dfe95d4d43f8402da52` |

同一个提交仅另增本说明；未修改算法、阈值、坐标、读出定义或结果。主工作区仍保留原文件与未提交状态，本地保存不等于已经合入主工作区。任务为 task #11；来源核对、精确 patch 和验证报告位于该任务 worktree 的 `runs/preserve-20261013/`，生成证据不纳入代码提交。

## 固定输入与历史接口

默认运行需要从仓库根目录解释这些相对路径。单独复制本目录不能提供所有依赖：

| 脚本/阶段 | 固定依赖或输入 | 输出 |
|---|---|---|
| all-track 推理 | `experiments/ism/20260918_1349_wt1_exact_counterfactual/prepared/mutation_manifest.tsv`；脚本中固定的 FASTA/Mdk 几何/读出；AlphaGenome 权重及本仓库模型接口 | `sums_shard<N>.npz`（alt/ref/mutation_id/readouts）、`channels.tsv`、`run_metadata.json` |
| all-track 分析 | `--run` 下的 NPZ/channels；`src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_mouse.parquet`；固定 ROOT 下 `prepared/variant_design.tsv`，部分情况回读 `analysis/panel_effects.tsv` | `variant_track_effects.parquet`、`track_summary.tsv`、`sample_level_tpm_filtered.tsv`、`sample_level_counts.tsv` |
| sample-level 分析 | `experiments/ism/20261003_hsc_ag_training_data_similarity/prepared/expr_cpm.parquet` 的 Mdk 行 | 上述 sample-level 表；变量名和阈值沿用原脚本，本文未重新校验 expression 单位 |
| checkpoint 比较 | 两次分析的 `variant_track_effects.parquet` 和 `sample_level_counts.tsv`；默认 fold_0 / distilled 的目录名称 | `effect_agreement.tsv`、`sample_level_counts_side_by_side.tsv` |

`--weights` 不会改写固定 manifest/基因窗口，`--run` 也不会把 ROOT、track metadata 或 expression 表一起切换。两个运行必须使用相同 mutation manifest、readout、channel/metadata 口径才有可比较的意义。输入大文件、权重、NPZ、表和图不随本提交复制；历史索引/归档任务需要把这些依赖一起定位，不能因 track-similarity 项目不再推进而删除它的关联 expression 表。

两个 checkpoint 的历史用途是比较 fold_0 与 distilled release 对同组编辑的读出差异；这些名称及解释沿用源脚本，本文未独立核查权重版本或得出新的研究结论。

## 未验证边界

本次验证仅包括源码 hash/patch 对应、静态解析/编译及 Git diff 检查，没有导入并运行这些模块，没有推理、分析、Motif 扫描、作图或运行 pytest。

以下原有 experimental 边界随源码保留，未修复：

- 比较脚本使用 inner merge，尚无完整 keyset/1:1/运行元数据一致性校验；样本 counts 将 samples 数量纳入 join key。
- 空集、点数不足及 constant 向量的相关/拟合行为未验证；现有摘要不得替代输入完整性审查。
- 分析按文件名收集 NPZ shards；分片完整性和固定输入版本需由使用者另核对。默认路径下存在文件不代表本次已经验证数据。

将来如需重跑，由新任务明确模型、输入版本、环境和坐标契约，再由 runner 安排作业；本说明没有授予运行/迁移/对外发布权限。本地代码保存与主工作区合入分开审查。
