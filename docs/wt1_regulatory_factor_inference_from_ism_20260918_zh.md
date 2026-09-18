# 当前 ISM 没有验证 Mdk 的局部 WT1 开关，但显著收窄了 Wt1 修复网络假说

更新日期：2026-09-18
ISM 证据截止：2026-09-13
背景来源：用户提供的 `20260914_SAIJOU_research.pdf` 逻辑链摘要；按用户要求，本报告不再独立读取或逐页核对 PDF。

## 执行摘要

- 用户提供的研究逻辑链支持把 **Wt1 视为 rHSC 修复程序的候选上游调控因子**：Wt1 在恢复期上调，被 CellOracle 从候选网络中筛出，并与 CDI 共表达模块共同指向 Mdk 等恢复相关基因。但这条链中，CellChat、CellOracle 和 CDI 仍是计算推断；五配体 AAV 实验证明配体组合有修复作用，却没有直接证明 Wt1 必需、充分或直接结合 Mdk。
- 当前 ISM 检验的是一个更窄的命题：**Mdk ±3 kb 内已识别的 WT1 motif 是否是直接、HSC 特异的顺式调控开关**。在这个层级，结果是否定的。启动子位点 `wt1_0001` 在 AlphaGenome/Borzoi、original/fine-tuned、res128/res1 和五种读出几何下都弱；没有通过 HSC RNA、HSC 特异性或 original 多轨道主判据。
- 末端外显子位点 `wt1_0002` 虽有 96+ 的 Profile Canonical 分数，但四个细胞均高，驱动细胞是 Chol；original AlphaGenome 和 Borzoi 也在 gene-body/TES 读出中复现。其强度依赖读出窗口和轨道类型；polyA+ 与 10x 3′ 的一致性支持 RNA 加工/捕获假说，但 CAGE（5′ cap/TSS assay）也很高，说明机制可能更广。最稳妥的结论是 **末端外显子/3′ 邻近区域的广泛序列敏感性**，而不是 WT1–HSC 特异调控。
- 因而，当前 ISM **没有提供“Wt1 是重要调控因子”的正向新证据**，但提供了重要的机制排除证据：不能再用这两个局部 motif，特别是 `wt1_0002`，作为“Wt1 直接开启 Mdk”的依据。
- 当前最合理的工作模型是：**Wt1 仍可能在 rHSC 状态或更广泛的修复网络中发挥作用，但其作用可能通过远端增强子、其他靶基因或间接调控实现；Mdk 附近这两个已测试位点不是已验证的直接 WT1 开关。**

## 1. 本报告要回答的问题

本报告区分三个经常被混在一起的问题：

1. **状态关联层**：Wt1 是否与恢复期 rHSC 状态相关？
2. **网络调控层**：Wt1 是否可能位于 Mdk、Hgf、Igf1、Vegfc、Ngf 等修复配体程序的上游？
3. **直接顺式机制层**：Wt1 是否通过 Mdk 附近一个具体 WT1 motif 直接控制 Mdk，且该作用在 HSC 中具有特异性？

用户提供的研究链条主要覆盖前两个层级；我们的 ISM 主要检验第三个层级。只有先分开这三个问题，才能正确回答“ISM 是否帮助推断 Wt1 是重要调控因子”。

## 2. 用户提供的研究逻辑链：支持什么，不支持什么

以下内容按用户提供的论文逻辑摘要整理，并非本报告重新从 PDF 提取。

| 逻辑阶段 | 提供的证据 | 可以支持 | 尚不能支持 |
|---|---|---|---|
| TAA 纤维化与消退时间序列 | 对照、纤维化、恢复 1 周和 4 周的非实质细胞 scRNA-seq | 存在可追踪的细胞状态和表达动态 | 某个 TF 是修复的因果驱动者 |
| HSC_2/rHSC 的识别 | HSC_2 在纤维化和恢复期富集；CellChat 预测恢复 1 周 outgoing strength 最高 | HSC_2 是活跃的候选通讯细胞群 | CellChat 边代表真实配体释放或因果通讯 |
| 修复配体程序 | HSC_2 高表达 Mdk、Hgf、Igf1、Vegfc、Ngf 等 | rHSC 具有与组织修复一致的分泌表达谱 | 这些配体由 Wt1 直接调控 |
| 上游 TF 逆推 | CellOracle 从 32 个候选中结合动态筛至 Wt1、Mitf、Trp53 | Wt1 是值得优先验证的网络候选 | Wt1 必需、充分或直接结合 Mdk |
| CellOracle 与 CDI 交叉 | Wt1/Mitf 预测靶基因与 CDI 模块交集，包含 Mdk、Ncam1、Gas7 等 | 两套计算证据对同一网络模块形成收敛 | 交集等同于直接 TF–DNA 结合或因果边 |
| 五配体 AAV 实验 | Igf1、Mdk、Vegfc、Ngf、Hgf 混合物减少胶原并改善肝功能 | 下游配体组合具有修复能力 | Wt1 控制这五个配体；Wt1 是该表型的必要“总开关” |

### 2.1 这条链最强的结论

最强且不过度外推的结论是：

> Wt1 是恢复期 rHSC 修复网络中一个由表达动态和网络推断共同支持的高优先级候选调控因子。

### 2.2 这条链仍缺失的因果环节

要从“候选调控因子”提升为“核心总开关”，至少还需要直接证明：

- 操纵 Wt1 会一致改变 rHSC 的五配体程序；
- Wt1 在相关状态下实际结合这些靶基因的调控区域；
- Wt1 的作用对于恢复表型是必要或充分的；
- 五配体能够在 Wt1 缺失背景下救援表型，从而把 Wt1 与下游配体放在同一条因果链上。

当前 ISM 只能触及其中第二点的一小部分：候选局部 DNA 序列是否具有模型预测的功能效应。

## 3. ISM 实际检验了哪一个环节

研究推理链可简化为：

```text
恢复期细胞状态
    ↓ 观察/关联
rHSC 配体表达程序
    ↓ CellOracle/CDI 推断
Wt1 候选上游网络
    ↓ 尚需直接验证
Wt1 结合具体顺式元件
    ↓ 序列功能
Mdk 等靶基因输出变化
    ↓ 生理表型
纤维化消退
```

当前 saturation ISM 主要测试倒数第三条箭头：改变 Mdk 附近的候选序列后，模型预测的 Mdk RNA 或多轨道输出是否变化。它不直接操纵 Wt1 蛋白，不测 Wt1 占据，也不包含恢复期 Wt1 浓度作为显式条件。

因此：

- ISM 阳性最多说明“该序列对模型输出敏感”，还需要 motif 特异对照才能归因于 WT1；
- ISM 阴性只削弱“这个具体局部位点是开关”，不能否定 Wt1 在远端或间接网络中的作用；
- original AlphaGenome/Borzoi 是序列到多轨道预测模型，其高分不能自动解释成 WT1 蛋白结合。

## 4. 实验与评分范围

### 4.1 坐标与扫描合同

- 物种/组装：mouse, mm10。
- 基因/转录本：Mdk-201 / ENSMUST00000028672。
- 链：负链；offset 按 5′→3′ 转录方向报告。
- TSS：`chr2:91932297`（0-based）。
- 扫描窗口：TSS ±3,000 bp，基因组半开区间 `chr2:[91929297,91935297)`。
- 扫描规模：2,974 个可变中心、8,922 个严格单核苷酸组成保持的 10-bp shuffle；每个中心 3 个确定性替换。

候选位点：

| 实例 | 转录坐标 | 0-based half-open | 位置 | 角色 |
|---|---:|---|---|---|
| `wt1_0001` | −1300..−1290 | `chr2:[91933588,91933598)` | TSS 上游启动子 | 冻结 r5 的 WT1 主检验位点 |
| `wt1_0002` | +2260..+2270 | `chr2:[91930028,91930038)` | 最后一个外显子/3′UTR，距 TES 约 200 bp | 后续 WT1/结构上下文审计位点 |

候选定义见 [`candidate_instances.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/candidate_instances.tsv)。

### 4.2 四类 backend

- AlphaGenome fine-tuned：Saijou 四细胞 10x 3′ scRNA-seq pseudobulk head。
- Borzoi fine-tuned：同一四细胞任务。
- AlphaGenome original：curated mouse RNA/CAGE/染色质等多轨道。
- Borzoi original：独立架构和独立多轨道面板。

### 4.3 本报告使用的四类分数

| 分数 | 回答的问题 | 不能解释为 |
|---|---|---|
| `S_RNA` | HSC gene-body 绝对效应在同几何背景中的百分位 | p 值、FDR 或 HSC 特异性 |
| `S_HSC` | `HSC − max(Mac,LSEC,Chol)`；严格门要求 `C>0`、`S_HSC>95`、`S_RNA>95` | 直接 WT1 结合 |
| original `S_MULTI` | original 模型 output/local 多轨道组合响应 | 与 fine-tuned 原始幅度可直接比较 |
| Profile Canonical | 单 AlphaGenome 四细胞百分位最大值的全局 rank | HSC 特异性或 motif 身份 |

全部百分位都是扫描背景内的相对排名，不是独立样本统计显著性。

## 5. `wt1_0001`：启动子 WT1 候选在所有主要检验中为阴性

### 5.1 冻结 r5 主判据

| Backend | 指标 | 百分位 | 是否通过 |
|---|---|---:|---|
| AlphaGenome FT | `S_RNA` | 7.95 | 否 |
| AlphaGenome FT | `S_HSC` | 40.80；`C=-0.0119` | 否 |
| Borzoi FT | `S_RNA` | 19.55 | 否 |
| Borzoi FT | `S_HSC` | 55.15；`C=-0.00566` | 否 |
| AlphaGenome original | `S_MULTI` | 83.09 | 否 |
| Borzoi original | `S_MULTI` | 22.84 | 否 |

原始数据见 [`region_scores.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/region_scores.tsv)。

### 5.2 读出几何稳健性

`wt1_0001` 在四个 backend × 五种读出几何中的中位百分位约为 15–47。更换 TSS、gene-body、TES、proximal 或 observed-peak 读出，没有把它变成稳定高分位点。这构成一个有价值的阴性内对照：几何交换不会自动使所有 WT1 候选升高。

完整结果见 [`wt1_geometry_swap_summary.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/wt1_geometry_swap_summary.tsv)。

### 5.3 res1 没有救回该位点

- `S_RNA`：`7.95 → 16.73`，仍不通过；
- `S_HSC`：`40.80 → 65.02`，但 `C` 仍为负（`−0.0119 → −0.00230`）；
- Profile Canonical：`54.37 → 51.03`；
- HSC Profile peak percentile：`37.96 → 46.58`。

数据见 [`score_comparison_res128_vs_res1.tsv`](../experiments/ism/20260913_mdk_motif_ism_res1_rescore/score_comparison_res128_vs_res1.tsv) 和 [`interval_comparison.tsv`](../experiments/ism/20260913_profile_resolution_recall/interval_comparison.tsv)。

### 5.4 对 Wt1 假说的含义

该结果直接削弱的是：

> `wt1_0001` 是 Mdk 的主要、HSC 特异、直接 WT1 顺式开关。

它不排除未扫描到的远端 WT1 元件，也不排除 Wt1 通过其他靶基因或间接网络影响 Mdk。

## 6. `wt1_0002`：高分更符合末端外显子/3′ 邻近上下文，不能作为 WT1 证据

### 6.1 四细胞 Profile 均高，但不是 HSC 特异

| Profile percentile | res128 | res1 |
|---|---:|---:|
| HSC | 96.95 | 96.14 |
| Mac | 96.40 | 97.81 |
| LSEC | 91.08 | 97.03 |
| Chol | 98.63 | 98.11 |
| Canonical | 96.89 | 96.19 |

Canonical 在两个分辨率中均由 Chol 驱动。这里“驱动”仅表示四个细胞中 Chol 最高；其他三个细胞也同样处于高百分位。这个模式支持泛细胞敏感性，而不是 HSC 优先性。

### 6.2 早期 original–fine-tuned 比较存在读出几何混杂

冻结 r5 的评分代码对两类 backend 使用了不同窗口：

- fine-tuned 使用 `gene_body_output_clipped`；
- original 硬编码为 `tss_1024bp`。

`wt1_0002` 位于 TSS 下游约 2.26 kb，落在 TSS 1,024-bp 读出之外。因此早期“AlphaGenome original 42.57 vs fine-tuned 91.86”不能解释为微调学到了新的 WT1 语法。错的是比较口径，而不是已经保存的模型输出。

### 6.3 对齐几何后，两个 original 模型都复现 3′ 响应

| Backend | TSS 1024 bp | Gene body | TES/3′ 1024 bp | Gene-body 中位 signed effect |
|---|---:|---:|---:|---:|
| AlphaGenome original | 44.17 | 59.62 | 86.60 | +0.00393 |
| AlphaGenome FT | 53.94 | 93.83 | 92.48 | +0.05993 |
| Borzoi original | 60.54 | 82.32 | 85.83 | +0.00520 |
| Borzoi FT | 53.67 | 90.30 | 83.94 | +0.02482 |

这说明 `wt1_0002` 的响应并非 fine-tuning 后才出现；两个独立 original 模型都在覆盖末端外显子/TES 的读出中看到信号。

### 6.4 AlphaGenome original 的轨道拆解显示显著的轨道与测定类型依赖

在 AlphaGenome original 的 gene-body 几何下：

| 代表轨道 | 类型 | 百分位 |
|---|---|---:|
| liver CAGE | 5′ capped RNA ends/TSS 相关 | 98.60 |
| liver polyA+ RNA-seq | polyA 富集 RNA | 95.12 |
| liver CAGE | 5′ capped RNA ends/TSS 相关 | 94.44 |
| liver total RNA | total RNA | 59.62 |
| liver total RNA | total RNA | 30.90 |

肝脏 polyA/CAGE 子集的中位百分位为 94.44，接近 AlphaGenome FT gene-body 的 93.83，但不能把两者统称为 3′ 测定化学：CAGE 捕获的是加帽 RNA 的 5′ 端。polyA+ 高、total RNA 较弱，加上微调目标是 10x 3′ scRNA-seq pseudobulk，说明 RNA 文库选择和 3′ 捕获可能贡献了响应；与此同时 CAGE 也高，提示模型响应可能涉及更广泛的转录/局部序列表征。该结果支持“轨道和读出依赖”，但不足以单独锁定为 3′ RNA 加工机制。

微调后 RNA 信号仍然很强并不矛盾。fine-tuned head 预测的正是四个细胞的 10x 3′ RNA profile；训练标签把 Mdk 的 RNA 丰度及其沿基因的空间分布投射到输出 bin 上。位于末端外显子附近的序列被扰动后，共享 trunk 的局部表示发生变化，四个 RNA head 都可能把它转化为 gene-body/TES 输出变化。由于 10x 3′ 读出在转录本 3′ 端具有明显的捕获几何，这种响应可以在微调后被保留或放大。它说明模型学到了“该局部序列与 RNA profile 相关”，但不等于模型已经证明了剪接、polyA、RNA 稳定性中的某一个真实机制；四细胞均高且 Chol 最高，反而不符合 HSC 特异 WT1 开关的预期。

逐轨道结果见 [`wt1_geometry_swap_scores.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/wt1_geometry_swap_scores.tsv)，汇总与校验见 [`wt1_geometry_swap_summary.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/wt1_geometry_swap_summary.tsv) 和 [`wt1_geometry_swap_validation.json`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/wt1_geometry_swap_validation.json)。

### 6.5 高百分位不等于巨大或一致下降的表达效应

AlphaGenome FT 的 gene-body 中位绝对 log2 效应为 0.0720，中位 signed effect 为 +0.0599；Borzoi FT 分别为 0.0503 和 +0.0248。按平滑后的模型输出比例近似，signed effect 约对应 +4.2% 和 +1.7%，而不是 93% 或 96% 的表达变化。

因此：

- 96+ 是在扫描背景中的相对排名；
- 分数使用绝对效应，不能推断所有细胞表达下降；
- aggregate signed effect 略为正，与“破坏 WT1 motif 后 Mdk 输出下降”的简单方向性预期不一致。

### 6.6 结构/加工解释的证据强度

证据强地支持：

- `wt1_0002` 不是 HSC 特异信号；
- 其强度依赖读出是否覆盖 3′ 区域；
- 原始 AlphaGenome/Borzoi 已经能够复现；
- 其信号随轨道和测定类型显著变化；polyA+/10x 3′ 的一致性支持 RNA 加工/捕获假说，而高 CAGE 表明机制不应被缩窄为单一 3′ 加工过程。

证据只中等程度支持：

- 该区域可能与末端外显子、3′ 端或 RNA 加工有关。

当前证据不能确定：

- `wt1_0002` 本身是剪接二核苷酸或 polyA signal；
- 该扰动一定导致剪接失败、RNA 降解或转录本失效；
- 响应由 WT1 motif identity 而不是周围 GC-rich/结构序列驱动。

真正的 `splice_prior_2175_2189` 位于另一个相距约 70–80 bp 的区间，不能与 `wt1_0002` 合并为同一个位点。

## 7. 全部 Mdk motif-ISM 结果也没有形成 WT1 正证据

r5 对六类 TF motif 在两个 fine-tuned backend 上进行主判据评分：

- FT `S_RNA`：0/12 通过；
- 严格 HSC 联合判据：0/12 通过；
- original `S_MULTI`：0/12 通过。

这不是“模型完全没有局部序列响应”。最强且跨模型一致的信号来自剪接邻近区域，而非已归因的 TF motif。完整分析见 [`biological_analysis.md`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/biological_analysis.md)；工程验收见 [`validation_summary.json`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/validation_summary.json)。

res1 下，`splice_prior_2175_2189` 的 HSC `S_RNA` 从 98.29 升至 99.75，但 HSC 优先性 `C` 从 +0.0508 变为 −0.0361，`S_HSC` 从 98.69 降至 5.16。四细胞绝对效应接近，进一步支持其为泛细胞结构响应，而非 HSC 特异 TF 开关。详见 [`res1_rescore_analysis.md`](../experiments/ism/20260913_mdk_motif_ism_res1_rescore/res1_rescore_analysis.md)。

## 8. 当前 ISM 是否帮助推断 Wt1 是重要调控因子

答案取决于“帮助”指哪一个层级。

| 推断层级 | 当前证据 | ISM 的贡献 | 当前判断 |
|---|---|---|---|
| Wt1 与恢复期 rHSC 状态相关 | 用户提供的表达动态 | ISM 不直接测试 | 仍然成立，待原始研究证据支持 |
| Wt1 是 rHSC 配体网络候选上游 | CellOracle + CDI 收敛 | ISM 不足以否定远端/间接网络 | 仍是合理候选，但不是已证实总开关 |
| Wt1 直接调控 Mdk | 计算网络边 | ISM 检查局部 sequence-to-output 效应 | 未获得正向支持 |
| `wt1_0001` 是直接 HSC 开关 | 局部 PWM hit | 多模型、多分辨率均阴性 | 不支持 |
| `wt1_0002` 是直接 HSC 开关 | 高 Canonical rank | 泛细胞、3′ 几何/化学依赖、original 复现 | 不支持；高分不能归因于 WT1 |
| Wt1 是修复表型的必要/充分因子 | 五配体 AAV 有效 | ISM 不测试蛋白扰动或动物表型 | 尚未建立 |

### 8.1 ISM 提供的真正增量

ISM 的价值不是把 Wt1 从候选提升为核心因子，而是：

1. 排除了两个容易被高分误导的局部解释；
2. 证明早期 original 与 fine-tuned 的表面差异受到读出几何混杂，并显示明显的轨道/测定类型依赖；
3. 将“Wt1 网络假说”与“具体 WT1 motif 机制”解耦；
4. 指出下一步必须直接测试 Wt1 蛋白、结合和多个独立顺式元件。

因此，当前 ISM 是 **有信息量的机制收窄结果**，但不是 Wt1 重要性的正向验证。

## 9. 修订后的工作假说

### 9.1 仍然合理的网络级假说

> Wt1 可能参与建立或维持恢复期 rHSC 的修复性转录状态，并直接或间接影响 Mdk、Hgf、Igf1、Vegfc、Ngf 等配体程序。

该假说与用户提供的表达动态、CellOracle 和 CDI 证据相容，也没有被当前局部 ISM 排除。

### 9.2 当前不再支持的局部机制

> Wt1 通过 `wt1_0001` 或 `wt1_0002` 直接、HSC 特异地开启 Mdk。

`wt1_0001` 缺乏功能响应；`wt1_0002` 的高响应更符合 3′ 端上下文且缺乏 HSC 特异性和 motif-loss 因果对照。

### 9.3 `wt1_0002` 的窄假说

> `wt1_0002` 标记了一个位于末端外显子、对读出窗口和轨道类型敏感的区域；RNA 加工/3′ 捕获是候选解释之一，但其确切机制及 WT1 motif 身份均未被证明具有功能意义。

## 10. 建立 Wt1 因果链所需的下一步实验

### 10.1 直接操纵 Wt1

在恢复期 rHSC/HSC_2 中进行 Wt1 knockdown、CRISPRi/KO 或过表达，测量：

- Mdk、Hgf、Igf1、Vegfc、Ngf 的表达；
- rHSC 状态和细胞通讯输出；
- 纤维化消退相关表型。

这是判断 Wt1 必要性/充分性的首要实验。

### 10.2 验证实际占据

在相关 HSC 状态中进行 Wt1 CUT&RUN、CUT&Tag 或 ChIP-seq，并与开放染色质和恢复期增强子标记联合分析。只有看到 Wt1 在 Mdk 或其他核心靶基因调控区域的状态相关占据，才能把网络边推进为直接结合假说。

### 10.3 做 motif identity 的匹配反事实

对于多个独立 WT1 候选位点，设计：

- WT1-PWM loss 编辑；
- WT1-PWM preserving、相同中心/改变碱基数/局部组成的编辑；
- 不改变 WT1 PWM 但扰动 3′ 结构上下文的编辑；
- splice/polyA preserving 与 disrupting 编辑。

若只有 WT1-loss 在 Wt1 高表达的 HSC 状态中稳定降低目标输出，才支持 motif-specific 因果归因。

### 10.4 扩展到远端调控区

当前 ±3 kb 扫描不能覆盖远端增强子。应结合 HSC ATAC、H3K27ac、染色质接触或 enhancer–gene linking，扫描与 Mdk 及五配体连接的远端候选元件。

### 10.5 分离转录调控与 RNA 加工

对 `wt1_0002` 使用：

- total RNA 与 3′ RNA-seq 平行读出；
- nascent RNA；
- isoform/splicing ratio；
- 3′ RACE 或 polyA-site usage；
- minigene/terminal-exon reporter。

这些实验可以判断高分来自转录、剪接、3′ 端加工、稳定性还是测量偏好。

### 10.6 将 Wt1 与五配体表型连接起来

理想的因果设计是：

```text
Wt1 loss
  → 五配体程序下降
  → 修复能力受损
  → 五配体补充能够救援
```

这比“五配体本身有效”更能证明 Wt1 位于该修复程序的上游。

## 11. 最终结论

用户提供的研究逻辑链足以把 Wt1 放在恢复期 rHSC 网络的高优先级候选位置，但不足以单独确立 Wt1 是必要、充分且直接控制 Mdk 的“总开关”。当前 ISM 对这个问题的贡献是局部和否定性的：

- `wt1_0001` 在全部主要模型和评分中为阴性；
- `wt1_0002` 虽高分，但更符合泛细胞的末端外显子/3′ 邻近区域敏感性，不能归因于 WT1；
- 六类 Mdk TF motif 没有形成严格主判据阳性；
- 更细的 res1 模型也没有把 WT1 motif 变成可信的 HSC 特异元件。

因此，**当前 ISM 没有帮助我们正向证明 Wt1 是重要调控因子；它帮助我们否定了“这两个 Mdk 局部 WT1 motif 就是直接开关”的过度简化机制。** Wt1 的网络级重要性仍然是合理且值得直接实验检验的假说，但其证据必须来自 Wt1 操纵、实际结合、远端元件和配体救援，而不能来自 `wt1_0002` 的高百分位。

## 12. 主要数据与可复核入口

- r5 生物学分析：[`biological_analysis.md`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/biological_analysis.md)
- r5 工程校验：[`validation_summary.json`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/validation_summary.json)
- 候选实例与坐标：[`candidate_instances.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/candidate_instances.tsv)
- 冻结主分数：[`region_scores.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/region_scores.tsv)
- original 视图分数：[`original_view_scores.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/original_view_scores.tsv)
- WT1 几何交换汇总：[`wt1_geometry_swap_summary.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/wt1_geometry_swap_summary.tsv)
- WT1 几何交换逐轨道数据：[`wt1_geometry_swap_scores.tsv`](../experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5/wt1_geometry_swap_scores.tsv)
- res128–res1 Mdk 重评分：[`score_comparison_res128_vs_res1.tsv`](../experiments/ism/20260913_mdk_motif_ism_res1_rescore/score_comparison_res128_vs_res1.tsv)
- res128–res1 Profile 区间比较：[`interval_comparison.tsv`](../experiments/ism/20260913_profile_resolution_recall/interval_comparison.tsv)
- Profile 阈值召回：[`threshold_recall.tsv`](../experiments/ism/20260913_profile_resolution_recall/threshold_recall.tsv)
- 1-bp/32-bp 模型历史和训练证据：[`recent_wt1_and_1bp_resolution_results_20260914_zh.md`](recent_wt1_and_1bp_resolution_results_20260914_zh.md)

> 数据使用提醒：不要引用 `experiments/ism/20260913_profile_resolution_recall_pre_dedup_invalid/`。正式 Profile 分辨率结果仅使用 `experiments/ism/20260913_profile_resolution_recall/`。
