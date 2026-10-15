# 1-bp 表征未提高严格召回，WT1 仍不支持 HSC 特异调控

更新日期：2026-09-14
证据截止：2026-09-13

## 执行摘要

- 历史 AlphaGenome `res1/bin32` LoRA 微调已完整跑完：3 GPU、20 epochs、约 4.73M 可训练参数、墙钟时间约 63.4 小时；最终 `val_pearson_log1p=0.8143`。这里的 “1 bp” 指 **AlphaGenome trunk 的 1-bp embedding/feature resolution**，监督标签和最终输出仍是 **32-bp bins**，不是 1-bp 预测。
- 在同一份九基因、TSS ±3 kb、严格 shuffle manifest 上，从完成的 feature parquet 直接比较 res128 与 res1 后，95 阈值下单 AlphaGenome 四细胞 Profile 的 Canonical 召回保持 **2/7 → 2/7**，HSC 召回保持 **3/7 → 3/7**。这不支持“更细分辨率带来总体准确率提升”。
- 启动子 WT1 `wt1_0001`（Mdk 转录坐标 −1300..−1290）仍是阴性：Canonical `54.37 → 51.03`；HSC `S_RNA` 排名虽从 `7.95 → 16.73`，但 HSC 优先性原始差值仍为负（`−0.0119 → −0.00230`）。
- 末端外显子/剪接邻近 WT1 `wt1_0002`（+2260..+2270）维持高 Canonical 分数（`96.89 → 96.19`），但四个细胞均高、驱动细胞不是稳定的 HSC。几何交换与原始模型轨道拆解表明它主要反映末端外显子、剪接和 3′ 端结构敏感性，不能作为 WT1 motif 调控证据。

## 1. 范围与口径

本文汇总三个已完成且互相衔接的实验层次：

1. 2026-07-31 至 2026-08-03 完成的 AlphaGenome `res1/bin32` LoRA 训练；
2. 2026-09-11 完成的九基因 6-kb res1 AlphaGenome ISM；
3. 2026-09-13 从底层 feature parquet 重新计算的 res128–res1 Profile 召回，以及 Mdk/WT1 冻结候选重评分。

分辨率比较不使用网页快照。网页的 Canonical、AG/Borzoi 双模型分数与 original Method 3 属于冻结的旧快照和另一套指标，不能替代单 AlphaGenome 的 res128–res1 对比。本文也不把扫描背景百分位解释为 p 值、FDR、体内结合证据或因果效应。

### WT1 坐标合同（沿用已批准清单）

- 物种/组装：mouse, mm10。
- 基因/转录本：Mdk-201 / ENSMUST00000028672。
- 链与报告方向：负链；正文 offset 均按 5′→3′ 转录方向报告。
- 锚点/窗口：Mdk-201 TSS，九基因扫描为 TSS ±3,000 bp。
- 基因组坐标：0-based half-open；括号内给出等价 1-based closed 坐标。

| 实例 | 转录坐标 | mm10 基因组坐标 | 结构位置 | 本文角色 |
|---|---:|---|---|---|
| `wt1_0001` | −1300..−1290 | `chr2:[91,933,588,91,933,598)`（91,933,589–91,933,598） | TSS 上游启动子 | WT1 motif 主检验 |
| `wt1_0002` | +2260..+2270 | `chr2:[91,930,028,91,930,038)`（91,930,029–91,930,038） | 最后一个外显子、距 TES 约 200 bp | WT1 标签的结构/剪接邻近对照 |

坐标与候选定义见 [`candidate_instances.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/candidate_instances.tsv)；完整解释见 [`mdk_motif_ism_and_wt1_comprehensive_report.md`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/mdk_motif_ism_and_wt1_comprehensive_report.md)。

## 2. “1-bp 分辨率”实验实际完成了什么

| 项目 | 已完成配置/结果 |
|---|---|
| AlphaGenome trunk | `resolution=1`，1-bp embedding/feature resolution |
| 监督与输出 | `bin_size=32`，196,608-bp label window 输出 6,144 个 32-bp bins |
| 微调 | LoRA active，20 epochs，3-GPU DDP，约 4.73M / 455M 参数可训练 |
| 训练时间 | 2026-07-31 15:43:28 至 2026-08-03 07:08:33 JST，约 63.4 小时 |
| 最终记录 | epoch 19：`val_loss=4.590697`，`val_pearson_log1p=0.814332` |
| 最高记录 | epoch 16：`val_pearson_log1p=0.817718` |
| 最佳 checkpoint | epoch 19 |

直接证据：

- 运行入口与参数：[`run_alphagenome_res1_bin32.sh`](../src/ft-scripts/run_alphagenome_res1_bin32.sh)、[`hparams.yaml`](../runs/alphagenome_split_chr10_chr11_seq524288_label196608_bin32_res1_poisson_multinomial_lora_active_h512x1/version_1/hparams.yaml)
- 开始/结束时间和启动参数：[`ag_res1_bin32_launcher.log`](../experiments/logs/ag_res1_bin32_launcher.log)
- 3-GPU DDP、epoch 19 完成和 checkpoint：[`ag_lora_res1_bin32_3gpu_20260731_154328.log`](../experiments/logs/ag_lora_res1_bin32_3gpu_20260731_154328.log)
- 逐 epoch 指标：[`version_1/metrics.csv`](../runs/alphagenome_split_chr10_chr11_seq524288_label196608_bin32_res1_poisson_multinomial_lora_active_h512x1/version_1/metrics.csv)
- 最佳权重：[`epochepoch=19.ckpt`](../runs/alphagenome_split_chr10_chr11_seq524288_label196608_bin32_res1_poisson_multinomial_lora_active_h512x1/checkpoints/epochepoch=19.ckpt)

### 训练层不能回答“精度是否提高”

旧 `res128/bin128` 运行的 `val_pearson` 没有有限值，且没有记录 `val_pearson_log1p`；`val_loss` 和 `val_mse` 又随 bin 数、每 bin 计数和损失几何变化，不能与 res1 数值直接比较。因此 `0.8143` 只能证明 res1 运行自身收敛，不能证明它优于 res128。

旧训练记录见 [`res128 version_0/metrics.csv`](../runs/alphagenome_split_chr10_chr11_seq524288_label196608_bin128_res128_poisson_multinomial_lora_active_h512x1/version_0/metrics.csv) 和 [`res128 version_1/metrics.csv`](../runs/alphagenome_split_chr10_chr11_seq524288_label196608_bin128_res128_poisson_multinomial_lora_active_h512x1/version_1/metrics.csv)。因此本轮性能判断以同 manifest 的下游 ISM 召回为主。

## 3. 同一 6-kb manifest 上的直接 Profile 比较

### 3.1 可比性与验收

两次运行使用完全相同的 prepared-manifest SHA256：

```text
25af624a39b2955b824d921e1458330e6ef721897072002900a1f42205dfd822
```

比较只读取完成的 AlphaGenome FT feature parquet，不使用网页 JSON、Borzoi 或 original Method 3。两个分辨率均覆盖 26,922 个中心和 107,688 条四细胞 Profile 记录；数值有限、键唯一、区间成对，最终校验状态为 `ok`。详见 [`validation_summary.json`](../experiments/ism/20260913_profile_resolution_recall/validation_summary.json) 和方法说明 [`README.md`](../experiments/ism/20260913_profile_resolution_recall/README.md)。

原始输入：

- res128：[`combined_mutation_features.parquet`](../experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle/runs/alphagenome_finetuned/features/combined_mutation_features.parquet)
- res1：[`gpu0 parquet`](../experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1/shards/gpu0/features/combined_mutation_features.parquet)、[`gpu1 parquet`](../experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1/shards/gpu1/features/combined_mutation_features.parquet)、[`gpu2 parquet`](../experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1/shards/gpu2/features/combined_mutation_features.parquet)、[`gpu3 parquet`](../experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1/shards/gpu3/features/combined_mutation_features.parquet)

### 3.2 召回没有整体提升

95 阈值下：

| 指标 | res128 | res1 | 结论 |
|---|---:|---:|---|
| Canonical（四细胞最大值的全局 rank） | 2/7 | 2/7 | 不变 |
| HSC | 3/7 | 3/7 | 不变 |
| Mac | 2/7 | 3/7 | +1 |
| LSEC | 2/7 | 2/7 | 不变 |
| Chol | 2/7 | 3/7 | +1 |

90 阈值下 Canonical 从 `3/7 → 4/7`，新增的是 Col1a1 SMAD（`84.01 → 92.86`）。这说明细胞轨道之间发生了分数重分配，但严格阈值总体召回没有改善。完整阈值曲线见 [`threshold_recall.tsv`](../experiments/ism/20260913_profile_resolution_recall/threshold_recall.tsv)。

七个正控的 Canonical 区间分数：

| 正控 | res128 | res1 | Δ |
|---|---:|---:|---:|
| Acta2 SRF/CArG | 99.95 | 99.96 | +0.01 |
| Col1a1 SMAD3/4 | 84.01 | 92.86 | +8.85 |
| Col1a2 SMAD3 | 50.29 | 52.62 | +2.34 |
| Col1a1 AP-1 | 92.84 | 93.33 | +0.49 |
| Col1a2 AP-1 | 69.45 | 62.50 | −6.94 |
| Timp1 AP-1 | 99.95 | 99.96 | +0.01 |
| Col1a1 SP1/KLF6 | 88.21 | 80.98 | −7.23 |

逐区间四细胞分数、富集和驱动细胞见 [`interval_comparison.tsv`](../experiments/ism/20260913_profile_resolution_recall/interval_comparison.tsv)；逐中心结果见 [`profile_center_scores.tsv`](../experiments/ism/20260913_profile_resolution_recall/profile_center_scores.tsv)。

### 3.3 HSC-only `S_RNA` 是另一套问题，不与 Canonical 混用

按冻结的 Mdk `S_RNA` 契约只看 HSC gene-body 区域中位效应时，正控严格 `>95` 召回为 **1/7 → 1/7**，分数 3 升、4 降：Acta2 `91.71 → 94.87`、Col1a2 AP-1 `52.13 → 63.89`，而 Col1a1 三个邻近区间共同下降；Timp1 维持高分 `99.53 → 99.46`。

这与 Canonical 的 `2/7 → 2/7` 不矛盾：Canonical 允许 Mac/LSEC/Chol 驱动，`S_RNA` 只回答 HSC 响应。完整表和伪计数/基线校正说明见 [`positive_control_score_comparison.tsv`](../experiments/ism/20260913_positive_control_res1_rescore/positive_control_score_comparison.tsv) 与 [`README.md`](../experiments/ism/20260913_positive_control_res1_rescore/README.md)。

## 4. WT1：两个位点给出不同但一致的否定性结论

### 4.1 启动子 `wt1_0001`：更细分辨率没有救回 WT1 motif

| 指标 | res128 | res1 | 解释 |
|---|---:|---:|---|
| Profile Canonical | 54.37 | 51.03 | 仍未召回 |
| HSC Profile peak percentile | 37.96 | 46.58 | 有小幅上升，但远低于 95 |
| `S_RNA` percentile | 7.95 | 16.73 | 仍为极弱 HSC 响应 |
| HSC 优先性原始差值 `C` | −0.01187 | −0.00230 | 仍为负 |
| `S_HSC` percentile | 40.80 | 65.02 | 排名上升但联合判据仍失败 |

因此 res1 只使它“没那么反 HSC”，没有把它变成 HSC 特异候选。Profile 数据见 [`interval_comparison.tsv`](../experiments/ism/20260913_profile_resolution_recall/interval_comparison.tsv)，冻结 Mdk 判据重评分见 [`score_comparison_res128_vs_res1.tsv`](../experiments/ism/20260913_mdk_motif_ism_res1_rescore/score_comparison_res128_vs_res1.tsv)。

### 4.2 末端外显子 `wt1_0002`：高分稳定，但不是 WT1 调控证据

| Profile percentile | res128 | res1 |
|---|---:|---:|
| Canonical | 96.89 | 96.19 |
| HSC | 96.95 | 96.14 |
| Mac | 96.40 | 97.81 |
| LSEC | 91.08 | 97.03 |
| Chol | 98.63 | 98.11 |

四细胞均高，且 Canonical 驱动细胞在两次运行中都是 Chol；这更符合广泛的末端外显子/剪接结构敏感性，而不是 HSC 特异 WT1 开关。

几何交换实验进一步拆解了早期“original 42.57% vs fine-tuned 91.86%”的表面反差：该位点位于 TSS 下游约 2.26 kb，超出 original `tss_1024bp` 读出。对齐读出几何后：

- AlphaGenome original：TSS 几何 `44.17`，gene-body `59.62`，TES/3′ `86.60`，单轨最高 `98.60`；
- Borzoi original：TSS 几何 `60.54`，gene-body `82.32`，TES/3′ `85.83`，单轨最高 `96.38`；
- original 和 fine-tuned 的中位 signed effect 均为正，剩余差异随轨道测定化学（total RNA、polyA/CAGE、10x 3′ pseudobulk）明显分层。

所以原始模型并非“没有看到”该位点；早期差距主要混入读出几何与 assay chemistry。具体数据见 [`wt1_geometry_swap_summary.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/wt1_geometry_swap_summary.tsv)、逐窗口数据 [`wt1_geometry_swap_scores.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/wt1_geometry_swap_scores.tsv) 和校验 [`wt1_geometry_swap_validation.json`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/wt1_geometry_swap_validation.json)。

### 4.3 附近最强信号是剪接结构响应，不是 motif 命中

Mdk 冻结候选中的 `splice_prior_2175_2189` 在 res1 下更强：

- HSC `S_RNA` 原始效应 `0.1898 → 0.3363`，排名 `98.29 → 99.75`；
- 但 HSC 优先性 `C` 从 `+0.0508 → −0.0361`，`S_HSC` 排名从 `98.69 → 5.16`；
- res1 下四细胞效应接近：HSC 0.336、Mac 0.383、LSEC 0.380、Chol 0.315。

这说明更细的模型加强了广泛的剪接邻近响应，同时削弱了“它是 HSC 特异信号”的解释。完整分析见 [`res1_rescore_analysis.md`](../experiments/ism/20260913_mdk_motif_ism_res1_rescore/res1_rescore_analysis.md)。

## 5. 对 Mdk–WT1 假说的当前结论边界

### 已由工程验收支持

- res1 LoRA 训练完成，checkpoint、日志和 metrics 均存在；
- 九基因 res1 ISM 四个 shard 均 `status=ok`，核心值无非有限值、无重复 feature key、reference 可确定复现；
- res128 与 res1 使用同一 manifest，Profile 比较通过唯一键、有限值和区间配对校验；
- r5 motif-ISM 工程状态为 `ok`，六类 TF motif × 两个 FT backend 的 `S_RNA` 和联合 HSC 判据均为 `0/12` 通过。

四个 res1 shard 的验收文件：[`gpu0`](../experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1/shards/gpu0/validation_summary.json)、[`gpu1`](../experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1/shards/gpu1/validation_summary.json)、[`gpu2`](../experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1/shards/gpu2/validation_summary.json)、[`gpu3`](../experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1/shards/gpu3/validation_summary.json)。r5 总验收见 [`validation_summary.json`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/validation_summary.json)，生物学分析见 [`biological_analysis.md`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/biological_analysis.md)。

### 模型内部证据支持的最窄结论

1. res1 没有改善严格 Canonical 或 HSC 正控召回，变化主要是细胞和位点间的对比度重分配。
2. `wt1_0001` 在 res128 和 res1 中都弱，不支持它是 Mdk 的主导 HSC motif。
3. `wt1_0002` 和邻近剪接区域可以很强，但其跨细胞、跨读出几何和 assay chemistry 的行为更符合末端外显子/3′ 端结构敏感性。
4. 当前结果不支持把任何已检验 WT1 位点称为 Mdk 的 HSC 特异调控开关。

### 当前证据不能支持

- 不能把百分位排名当作 p 值、FDR 或独立重复统计；背景窗口互相重叠。
- 不能从模型内部 ISM 推断体内 WT1 结合、因果调控或 SNV 准确性。
- 不能把 `val_pearson_log1p=0.8143` 当作相对 res128 的提升，因为旧 run 缺少可比验证指标。
- 没有独立的 chr11 locked test，也没有 1-bp 输出层面的专项评估。
- 不能用 `wt1_0002` 的高分为 `wt1_0001` 背书；两者处在不同结构环境，必须分别报告。

## 6. 最终判断

本轮 1-bp 表征实验的主要价值是完成了更细的 32-bp 输出模型，并暴露了哪些信号对分辨率和细胞差值稳定、哪些不稳定；它没有带来严格正控召回的整体提升。对于 WT1，启动子 motif 持续阴性，末端外显子位点持续高分但呈泛细胞和结构性特征。当前最稳妥的结论仍是：**模型识别到 Mdk 的末端外显子/剪接敏感区域，但没有识别到一个可支持 HSC 特异 Mdk 调控叙事的 WT1 motif。**

> 数据使用提醒：不要引用 `experiments/ism/20260913_profile_resolution_recall_pre_dedup_invalid/`；该目录是输入路径去重修复前的审计废稿。正式结果只使用 [`experiments/ism/20260913_profile_resolution_recall/`](../experiments/ism/20260913_profile_resolution_recall/README.md)。
