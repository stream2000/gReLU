# Saijou 单细胞 ↔ AlphaGenome 训练源数据：细胞相似度执行计划（v2）

日期：2026-10-03（Asia/Tokyo）

工作区：`/work2/Users/qijun/gReLU-replication`

状态：仅计划。本文件在本地通过 sshfs 写成，写作时没有服务器环境；所有路径与命令在执行前都要在 bagpipe 上核实。

v2 变更：v1 计划用 AlphaGenome 自己的预测生成“基因 × 轨道”矩阵，现已删除这一步。改为直接下载 AlphaGenome 训练所用的公开源数据（ENCODE、FANTOM5 的现成基因级定量表），和我们的单细胞数据一起放进一个预训练好的细胞嵌入模型里比较。全程不跑 AlphaGenome、不训练任何模型。

## 0. 问题

只回答一件事：**AlphaGenome 训练数据里的哪些小鼠样本，在原始表达层面和 Saijou 的 HSC（以及 Mac、LSEC、Chol）最像？HSC 和这些样本的相对相似度有多高？**

用途：为“原模型的哪条轨道可以当 HSC 的替代读出”提供数据依据。此前的原模型 ISM 面板是人工挑的，见 `src/grelu/interpret/ism/model_adapters/alphagenome_original.py::build_alphagenome_mouse_reference_panel`。

不回答：模型预测是否正确，以及 Wt1 是否调控 Mdk。

## 1. 数据来源

### 1.1 AlphaGenome 小鼠表达轨道对应的源数据

轨道元数据：`src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_mouse.parquet`

| 头 | 有效轨道 | 去重后样本（ontology × assay） | 来源 | 现成基因级表 |
|---|---:|---:|---|---|
| rna_seq | 173（其余为填充） | 91 | ENCODE（polyA plus 117 / total 56） | 每个 ENCODE 实验都有 `gene quantifications` TSV（RSEM，含 TPM 和 expected_count） |
| cage | 188 | 95 | FANTOM5（hCAGE / LQhCAGE） | `https://fantom.gsc.riken.jp/5/datafiles/reprocessed/mm10_latest/extra/CAGE_peaks_expression/mm10_fair+new_genes_phase1and2_counts.osc.txt.gz`（19 MB，基因 × 样本计数；同目录还有 TPM 版） |

- **ENCODE 映射**：元数据里没有文件 accession，需要用 `ontology_curie` + `assay_title` + 物种查 ENCODE API。例如：`https://www.encodeproject.org/search/?type=File&output_type=gene+quantifications&assembly=mm10&status=released&biosample_ontology.term_id=<CURIE>&assay_title=<assay>&format=json&limit=all`。同一样本有多个重复时取 released 文件的均值，并记录所有 accession。
- **FANTOM5 映射**：基因表的列名带 FANTOM 样本库 ID 和描述。用 `ontology_curie` 对 FANTOM5 SDRF 中的 cell/tissue ontology 进行匹配，匹配不上的先手工核对，再写入映射表。
- 我们关心的样本里，训练数据中**有**骨髓巨噬细胞和巨噬细胞（RNA + CAGE）、肝细胞（CAGE）、整肝、胚胎成纤维（RNA）、成纤维（CAGE）、主动脉平滑肌（CAGE）、骨髓间充质干细胞（CAGE）；**没有** HSC、LSEC、胆管细胞、内皮细胞。
- 可选扩展：同目录 FANTOM5 还有很多 AlphaGenome 没用到的样本。只比较 AlphaGenome 用到的那部分，避免回答偏题。

### 1.2 Saijou 单细胞

| 情况 | 做法 |
|---|---|
| 找到计数矩阵（h5ad / rds / CellRanger 输出） | 主路线，可细分 HSC_1、HSC_2 和时间点 |
| 只有伪 bulk bigWig `/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig/{hsc,mac,lsec,chol}.CPM.mapq10.bw` | 退路：按基因 3′ 端 1 kb 外显子窗口取信号，当作 4 个伪 bulk 样本；不能区分亚群，报告里写明 |

在 bagpipe 上查找：

```bash
find /work2 /work -maxdepth 6 \( -name '*.h5ad' -o -name '*.rds' -o -name 'filtered_feature_bc_matrix*' -o -name '*.loom' \) 2>/dev/null | grep -i -E 'saij|hsc|taa|liver'
```

找不到就向 Saijo 组索要带 HSC_1/HSC_2 和时间点标签的 Seurat 或 AnnData 对象。

### 1.3 AlphaGenome Atlas

不需要。Atlas 是人类全基因组 SNV 的预计算效应库，和“小鼠细胞相似度”无关。只有以后要在人类 MDK 同源区段查已有变异效应时，才考虑用它。

## 2. 方法

全部使用预训练权重，零样本推理，不微调。

| 方法 | 用途 | 说明 |
|---|---|---|
| M0 中心化表达相关（基线，必须做） | 不依赖大模型 | log1p(CPM)；每个基因跨所有样本做 z-score；在 Saijou HVG ∪ 标记基因上算 Spearman。不做中心化会被管家基因主导 |
| M1 UCE（主方法） | 原生支持小鼠，不需要同源映射 | snap-stanford/UCE，33 层预训练权重，`species=mouse`，输出 1280 维嵌入 |
| M2 scGPT（可选，稳健性） | 第二种架构 | whole-human 预训练权重；小鼠基因按一对一同源映射到人类，记录映射保留率 |

### 2.1 bulk 样本如何进单细胞模型

单细胞基础模型是在稀疏计数上训练的。常用做法是先把 bulk 下采样成“伪细胞”：

- 对每个 bulk 样本的计数向量归一化成概率 p，从 Multinomial(N, p) 抽 100 个伪细胞。N 取 Saijou 细胞 UMI 中位数（只有 bigWig 时用 5,000）。固定随机种子。
- Saijou 端：真实细胞直接嵌入后按群取均值；另外每个群也抽 100 个伪细胞，走同样的流程，用来衡量伪细胞化本身带来的偏移。
- 两边放进同一次 UCE 运行，用同一份基因词表。

### 2.2 相似度与相对分数

- 每个 bulk 样本取伪细胞嵌入均值，与每个 Saijou 细胞群均值求余弦相似度，得到 S[q, s]。
- **相对分数（主要报告）**：
  - 样本特异度：对每个 bulk 样本 s，在 Saijou 的 4（或 5）个群之间做 z-score，`spec[q, s] = (S[q,s] − mean_q' S[q',s]) / sd_q' S[q',s]`。这能抵消 10x 3′ 单细胞与 bulk 之间的平台差异。
  - 群内排名：对每个群，把全部 bulk 样本按 S 排成百分位。
- 稳定性：伪细胞重抽 20 次，报告排名的 95% 区间。M0 另外做 200 次基因标签打乱，得到零分布。
- 可视化：bulk 样本与 Saijou 细胞放在同一张 UMAP 上，各画一张图，按来源和组织着色。

### 2.3 对照（通过后才解读）

| 对照 | 预期 | 未通过时 |
|---|---|---|
| Mac | 骨髓巨噬细胞、巨噬细胞样本排在 Mac 前 5%，且特异度最高 | 方法不可靠，停止解读 |
| 肝细胞/整肝 | 不应对 HSC 表现出最高特异度 | 记录 |
| 伪细胞化 | Saijou 群伪细胞化后与原始嵌入均值的余弦 > 0.9 | 以 M0 为主 |
| 标记基因 | 被选为 HSC 最像的样本在原始表中 Lrat、Des、Pdgfrb、Reln、Dcn、Colec11 较高，Ptprc、Pecam1 较低 | 记录为疑点 |

## 3. 输出（遵守 ANALYSIS_HARNESS.md）

代码：`scripts/ism/experiments/saijou_hsc/experimental/track_similarity/`（实验性，不导入 canonical 工作流）

数据：`experiments/ism/20261003_hsc_ag_training_data_similarity/`

| 文件 | 键 | 内容 |
|---|---|---|
| `prepared/ag_sample_map.tsv` | ag_track_index, output_type | 与 ENCODE accession / FANTOM 样本 ID 的映射、匹配方式 |
| `prepared/bulk_gene_counts.parquet` | gene_id × sample_id | 下载的原始基因级计数 |
| `analysis/similarity_long.tsv` | method, query_group, sample_id | S、rank_pct、spec、重抽区间 |
| `analysis/top_samples_by_query.tsv` | method, query_group, rank | 每个群前 20 个样本 |
| `analysis/controls.tsv` | control_id | §2.3 结果 |
| `analysis/validation_summary.json` | — | 映射数、基因交集数、种子、权重版本、同源保留率 |
| `figures/umap_joint.png` | — | 联合嵌入图 |

## 4. 怎么解读（先冻结）

- 某个样本被视为 **HSC 最像**：HSC 群内排名前 5%，`spec[HSC] > 1`，且 M0 与 M1 都满足；同时 Mac 对照通过。
- 若满足条件的都是成纤维、平滑肌、间充质类：说明此前人工面板的方向基本对，可以用排序结果替换人工挑选。
- 若没有任何样本满足：结论是“AlphaGenome 小鼠训练数据中没有和 HSC 足够相似的样本”。这说明原模型的细胞类型读出不能用来判断 HSC 特异性，本身就是有效结论。
- 这只是表达层面的相似，不代表模型在这些轨道上学到了 HSC 的调控语法。

## 5. 风险

1. 平台差异：10x 3′ 单细胞 vs bulk 全长 RNA-seq / CAGE。相对分数和伪细胞对照可以缓解，但不能完全消除。CAGE 测的是 TSS 计数，和单细胞 3′ 计数差别更大，RNA 和 CAGE 两组分开报告。
2. 分布外：bulk 伪细胞不是真实细胞。M1 与 M0 冲突时以 M0 为准，并报告冲突。
3. 状态缺失：训练数据里没有纤维化或恢复期 HSC，最像的样本也不包含 Wt1 被诱导的 rHSC 状态。
4. 映射：FANTOM5 与 AlphaGenome 元数据靠 ontology 匹配，可能一对多。映射表必须保留，便于复核。

## 6. 执行顺序

E0 找单细胞矩阵 → E1 下载并映射 ENCODE / FANTOM5 基因表 → E2 伪细胞 → E3 UCE（及可选 scGPT）嵌入 → E4 相似度、对照、图 → 交给用户解读。

本计划不涉及基因组坐标编辑或 ISM。只有退路方案（从 bigWig 取 3′ 窗口）会用到坐标：mm10，`/work/Database/Database_fromDocker/Referencedata_mm10/gtf_chrUCSC/chr.gtf`，0-based half-open，按转录方向取最后 1 kb 外显子。走这条路前需按 AGENTS.md 请用户确认。
