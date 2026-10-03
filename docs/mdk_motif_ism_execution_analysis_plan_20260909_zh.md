# MDK motif–ISM：执行与独立分析交接计划

日期：2026-09-09（Asia/Tokyo）

工作区：`/work2/Users/qijun/gReLU-replication`

版本：`mdk-motif-ism-v1`

状态：计划已落盘；尚未执行本计划的数据处理、评分或生物学分析。

## 0. 使用方式与责任边界

用户将分别指定执行模型和分析模型。本文件不启动代理，不启动推理。

| 角色 | 负责 | 不负责／不得擅自做 |
|---|---|---|
| 执行模型 E | 核实来源、登记候选、实现冻结评分、计算、测试、交付 TSV/JSON 与执行记录 | 不调参追求阳性；不输出最终生物学结论；不只交最高分案例 |
| 分析模型 A | 独立检查契约和数据包，逐例解释 AG、BZ 与各自 original/FT 结果，写最终讨论 | 不静默修改评分、候选或数据；不通过重新挑轨道/读出来改善结果 |
| 用户 | 指定运行模型，决定范围变更、新推理或额外实验 | 坐标契约已确认，无需在相同范围重复询问 |

顺序：E0 核实 → E1 冻结来源与候选 → E2 标准化 → E3 评分 → E4 motif 证据 → E5 可选 SNV 桥接 → E6 验收交接 → A0 独立验收 → A1–A4 科学分析。

执行模型交付的是“计算与证据包”，不是已经成立的机制故事。分析模型可以返回具体修正单；任何修正均产生新版本并保留旧版，不允许覆盖后假装从未变更。

## 1. 科学问题与不允许偷换的结论

主问题：在 Saijou 小鼠肝星状细胞背景下，现有模型的 Mdk 局部序列扰动响应是否与原生 motif 破坏相关，而不是仅反映一般多碱基扰动、剪接敏感性或参考信号问题？

辅助问题：

1. 该响应是否在 FT 模型中优先影响 HSC？
2. 同一模型家族的 original 模型是否已有 RNA/CAGE 或局部调控响应？
3. 在可比的数据交集内，SNV 与 10-bp shuffle 分别提供什么证据？

背景依据：`docs/project_scientific_question_audit_zh.md`。HSC 指 hepatic stellate cells，不是造血干细胞。当前是 HSC/mac/LSEC/chol 四条聚合 RNA 输出，没有独立疾病时间点；不能推断纤维化到恢复的动态机制。

以下说法不成立：

- shuffle 效应大于 SNV，因此模型没有 SNV 精度；
- 输出 bin 是 32/128 bp，因此模型不可能有单碱基敏感性；
- score >95，因此具有 95% 置信度或 FDR <5%；
- PWM family 命中，因此该具体 TF 已结合；
- original 没有响应、FT 有响应，因此微调创造了新调控元件；
- HSC 优先，因此该元件只在 HSC 有效；
- 两模型一致、三个 shuffle 一致，因此获得了独立生物重复。

## 2. 已获用户确认的坐标契约

| 字段 | 固定值 |
|---|---|
| 物种／组装 | Mus musculus / mm10 |
| 基因／转录本 | Mdk / Mdk-201 / ENSMUST00000028672 |
| 链与报告方向 | 负链；offset 沿转录方向增加 |
| TSS 锚点 | chr2:91932297，项目采用的 0-based 锚点 |
| 研究窗口 | TSS −3000..+3000 bp；基因组半开区间 chr2:[91929297,91935297) |
| 权威 | `scripts/ism/experiments/saijou_hsc/configs/provided_nine_gene_transcripts.tsv`，老板指定转录本并经 mm10 注释核对 |
| 允许的后续操作 | 用户启动执行阶段后，在此范围内复用序列和现有结果，定位 motif、整理注释、评分和形成分析表 |

采用已验证 manifest 的 `edit_start/edit_end` 和 `variant_offset_from_tss_transcription_bp`，不要重新猜测负链中心、端点或人类启动子相对坐标。基因组 REF/ALT 与转录方向显示序列需明确区分。

转录本、assembly、窗口或坐标解释发生变化时，停止相关分支并向用户说明。其他基因的既有正对照可以读取其验证表作为历史证据，不在本计划中新增其坐标扫描。

## 3. 输入与实现路由

### 3.1 执行前必读

1. `AGENTS.md`、`experiments/Progress.md`。
2. `scripts/ism/experiments/saijou_hsc/WORKFLOW_INDEX.md`。
3. `scripts/ism/experiments/saijou_hsc/ANALYSIS_HARNESS.md`。
4. 本计划及实际目录中的更具体 `AGENTS.md`（如存在）。

从 `source activate.sh` 进入环境，核验 `grelu.__file__` 指向本 checkout。保留既有 dirty-worktree 修改，不提交、不重启训练或服务。

### 3.2 数据来源（2026-09-09 讨论中核对，E 必须重新验证）

- 主数据：`experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle/`。
  - `prepared/`：mutation manifest、genes、readouts、坐标审计、排除窗口。
  - `runs/{alphagenome_finetuned,alphagenome_original,borzoi_finetuned,borzoi_original}/`。
  - 当前记录为 Mdk 2,974 个可变中心、8,922 条编辑，10 bp / 2 bp stride / 3 replacements；22 个请求中心因不可组成保持替换而排除。
  - 四个 backend 均有 `status=ok` 的运行验证记录；这不是本次新分析已通过。
- 旧 SNV：`experiments/ism/saijou_hsc_ag_tss1kb_full/Mdk/` 与 `experiments/ism/saijou_hsc_borzoi_tss1kb_full/Mdk/`。
  - 各记录 1,024 bp、3,072 个 SNV，主要是 HSC 读出，不能推定有完整四细胞或 original 配对。
  - 旧 TES 锚点 91929804，新 TES 锚点 91929827，差 23 bp。
  - metadata 中 FT checkpoint 名称分别为 AG epoch 19、BZ epoch 39；名字相同不替代完整可比性检查。
- 历史 Method 3：`experiments/ism/original_multitrack_score_20260728/analysis/exploration_summary/frozen_score_contract.json`。
- 历史评分比较：同目录下 `method_comparison.tsv` 与该实验 `result_summary.md`。
- 既有 motif 参数：主数据 `analysis/motif_validation_summary.json`。

旧候选、旧高分区和旧 motif-loss 表只作既往探索来源，不当独立验证真值。当前主数据中的跨模型 `importance_score` 也不能直接充当本计划的单模型评分。

### 3.3 复用代码与新代码位置

- 读出效应：`src/grelu/interpret/ism/readouts.py`。
- 效应汇总：`scripts/ism/experiments/saijou_hsc/tools/effect_summary.py`。
- 原模型评分：`experimental/original_multitrack_score/score_calculations.py` 及相关入口（相对于 Saijou 工作流目录）。
- motif 原型：`annotate_saijou_candidate_motifs.py`、`tools/genomics.py`。
- 四种历史 JS 示例：`experimental/interactive_workbench/SCORE_JS_EXAMPLES.md`。

新增原型放在 `scripts/ism/experiments/saijou_hsc/experimental/mdk_motif_ism/`，复用现有读出/扫描/校验工具；不要复制一套 prepare/run/analyze/report 工作流，不让 canonical 模块导入 experimental 模块。

建议执行输出根目录为 `experiments/ism/20260909_mdk_motif_ism_v1/`。如果已存在，读取来源并选择明确的新版本目录，不覆盖。当前写计划阶段不创建该结果目录。

## 4. 评分契约：仅三个主评分，不加总、不投票

本节具体化上一轮的推荐方案。S_RNA、S_HSC 是新的单模型区域实现，S_MULTI 复用冻结 Method 3 的中心统计并规范区域校准。不得声称三者已经通过独立生物学验证。

### 4.1 共同效应量

对模型 m、轨道 t、中心 i、替换 r，在固定读出内：

```text
e[m,t,i,r] = log2((sum(ALT) + n*alpha) / (sum(REF) + n*alpha))
alpha = 1.0 per bin；n = 该读出的实际 bin 数
a[m,t,i] = median_r(abs(e[m,t,i,r]))
d[m,t,i] = median_r(e[m,t,i,r])
```

沿用代码对预测负值的处理，并保留负值计数。`log2fc_ratio_of_sums` 与 `log2fc_mean` 不可混用；每 bin 的 pseudocount 与 sum 的 pseudocount 不可混用。

主 FT 读出固定 `gene_body_output_clipped`；TES/3′ 为独立敏感性结果，不能取两者的最大值。跨尺度比较需要同 checkpoint、输入上下文、实际读出 bin、效应公式及轨道；不满足时标明不可比。

### 4.2 区间、背景和百分位的实施约定

以下为本计划新增的可重复性约定，不是历史实现已经具备的行为；E 在读取候选 ISM 响应前将其写入 `score_contract.json`：

1. 单位为具体 motif instance 或独立结构位点的固定半开区间。C(R) 是现有 edit footprint 与该区间相交的全部中心，按半开区间判断，不用含糊的 ±5 近似代替 manifest。
2. 背景在同一 Mdk ±3 kb 扫描范围、同 backend/track/readout 内生成。保持与候选一致的相对扫描网格、区间物理宽度、中心数与缺失覆盖模式，不跨缺口把有效行拼接成连续窗口。
3. 用候选同一统计量计算背景值。排除与全部预登记研究实例的编辑支持区重叠的背景；排除规则由序列和坐标决定，不看 ISM 大小。输出每个背景成员和每一步排除原因。
4. 主背景是“扫描区间比较背景”，不是已证明无功能的中性集合。GC、CpG、序列复杂度、motif 保留等匹配控制属于独立归因分析，不能与排名背景混称。
5. 百分位采用 `100*(N_less + 0.5*N_equal)/N_background`；相等容差固定 1e-12，less/equal 分类互斥。空背景返回 null，不返回 0；全相等应返回 50。
6. 同宽有效背景少于 100 时，输出原始统计量和背景数，主百分位及通过状态为 null / insufficient_background。100 是工程上的最低覆盖约定，不是统计独立样本数；不自动放宽匹配条件。
7. 所有百分位仅表示本次范围内排名。背景窗口相互重叠，不能用其行数当独立样本量推导 p 值。

历史 helper 的 `rolling(len(subset))` 是行数匹配，不保证真实同宽；应通过新实验层 wrapper/纯函数解决并单测，保留历史结果与其版本。新旧校准差异需记录，不能把新输出冒称逐值复现旧分数。

### 4.3 S_RNA：目标 RNA 区域效应（主判据）

```text
E[m,t,R] = median_i_in_C(R)(a[m,t,i])
S_RNA[m,t,R] = percentile_same_geometry(E[m,t,R])
```

- FT 主轨道为 HSC；另外三细胞保留同样的 E、signed effect 与参考信号。
- original 可对预先登记的 RNA/CAGE 轨道单独计算，不混 assay，不在观察响应后选择最佳轨道作为“主读出”。轨道登记依据 metadata 的细胞、assay、链、参考覆盖，而非突变响应。
- 不施加未经校准的 `1/(1+10*MAD)` 权重。输出三替换的值、MAD、正/负/零数量；不同 motif 破坏程度导致的差异不自动视为噪声。
- `S_RNA>95` 仅为区域响应排名通过；同时输出 E、signed log2FC、REF/ALT 与绝对变化，不把微小效应包装成强生物学作用。

### 4.4 S_HSC：目标细胞相对优先性（仅 FT）

```text
c[m,i] = a[m,HSC,i] - max(a[m,mac,i], a[m,lsec,i], a[m,chol,i])
C[m,R] = median_i_in_C(R)(c[m,i])
S_HSC[m,R] = percentile_same_geometry(C[m,R])
HSC_priority_rank_pass = C>0 AND S_HSC>95 AND S_RNA[HSC]>95
```

采用“其他细胞最大值”而非历史 JS 的“其他细胞均值”，并删除 AG/BZ minimum。这两个变更必须写入版本差异。

所有细胞须具有相同中心和替换集合、固定读出及可解释的参考预测。缺少任一必要细胞时不可评价；参考信号近零或 pseudocount 支配时添加质量警告，不能只依据 contrast 给出细胞优先结论。保留原始幅度与参考信号，必要时检查 alpha=0.5/2 的敏感性，不据此重选主 alpha。

original 无匹配四细胞面板时该项为 not_applicable。S_HSC 不通过不否定 motif 有效，只是不支持 HSC 优先。

### 4.5 S_MULTI：原模型多模态区域支持（仅 original）

每个模型单独复用冻结 Method 3：

1. 每轨道按三替换中位数取绝对及 signed effect；按固定 track_group 汇总，再在 gene/view/group 内排名。
2. output view：RNA/CAGE，冻结读出为 TSS 1024 bp 的 ratio-of-sums。
3. local_regulatory view：ATAC/DNase/histone/TF，冻结读出为突变所在 128-bp bin 及相邻各一 bin 中最大绝对 signed log2FC。实际 bin 网格和降采样规则由历史实现与 profile metadata 核实。
4. 每视图取第三高 group percentile；在物理 ±4 bp 中取空间中位数，再得到该视图的基因内中心排名 V_view(i)。
5. `U(i)=max(V_output(i), V_local(i))`。
6. `S_MULTI(R)=percentile_same_geometry(median_i_in_C(R)(U(i)))`。另保存两个视图按各自 spatial_support 计算的同宽区域分数。

不足三个有效组的视图不可用，不降低 k。任一必需视图不可用时不生成双视图总分；仍可报告可用视图，明确不是完整 Method 3。不同 assay/组织的组不当独立重复。

轨道组定义在看响应前冻结。BZ 的 HSC CAGE 可以作为独立有名轨道展示，但不以单轨替代三组规则。FT 没有局部调控头，不套用此分数。

### 4.6 阈值与不得自动计算的“总成功率”

- 三项区域分数统一主阈值严格 `>95`，阈值 90/99 只作预登记敏感性展示。
- 不计算三分平均、AG/BZ 最小值或 original/FT 差分总分。
- 不以三个评分“任一过线”判断 motif 成功。它们回答不同问题，适用模型也不同。
- 未设跨 assay 的通用绝对 log2FC 生物学阈值；不得执行时自创 0.1/0.2 等门槛。
- 没有足够独立阴性/阳性控制，不估计生物学准确率、FDR 或最佳评分排名。

## 5. 执行模型 E 的任务清单

### E0：输入盘点与可比性审计

- [ ] 记录 branch、commit、dirty 状态、环境、命令、时间与软件版本。
- [ ] 校验四 backend 的 mutation identity、REF/ALT、坐标、缺失键、重复键、非有限值与已有校验记录。
- [ ] 对主 manifest/metadata/配置做 hash；大型原始文件至少记录路径、大小、mtime，说明未做全文件 hash 的范围。
- [ ] 对照 checkpoint、实际输入窗口、输出网格、读出、pseudocount 与轨道 metadata；original/FT 的生物学可比性不能只靠同一 gene/readout 名称。
- [ ] 核验原模型 HSC CAGE 的细胞名称和链方向；不得把造血 HSC 或错链轨道当肝星状细胞。
- [ ] 输出 `source_inventory.tsv`、`comparability_audit.tsv`、`execution_manifest.json`。只读旧数据，不修改来源。

### E1：在查看新评分之前登记候选和版本

候选来源为以下八类 TF motif 假说加核心启动子、剪接结构两类位点；目标约十个可解释实例，不保证每类都有命中，也不强制正好十个。

| 类别 | 优先检验的依据／边界 | 起始来源（需 E 核实） |
|---|---|---|
| SP/KLF GC box | 人 MDK SP1 结合/定点突变；小鼠与 TF 身份不可直接外推 | https://pmc.ncbi.nlm.nih.gov/articles/PMC4310735/ |
| WT1 | 人 MDK 的结合与抑制证据，破坏未必导致下降 | https://pubmed.ncbi.nlm.nih.gov/8950987/ |
| RAR/RXR RARE | 人 MDK 视黄酸响应；必须核实小鼠实例 | https://www.jstage.jst.go.jp/article/biochemistry1922/117/4/117_4_845/_article/-char/en |
| HIF1A/ARNT HRE | 非 HSC 背景 Mdk 启动子功能突变证据 | https://pubmed.ncbi.nlm.nih.gov/15197188/ |
| NF-kB/RELA | MDK 诱导关系不自动证明具体小鼠位点 | https://link.springer.com/article/10.1186/1755-8794-1-6 |
| AP-1 | HSC 背景相关，不是本窗口已验证的 Mdk 位点 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3362071/ |
| SMAD3/4 | TGF-beta/HSC 假说；保持 TF-gene 与位点证据分离 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3362071/ |
| TCF21-associated E-box | Tcf21 去活化与 Mdk 上调关系，不证明直接结合 | https://onlinelibrary.wiley.com/doi/full/10.1002/hep.30965 |
| 核心启动子/起始元件 | 由指定转录本和原生序列定义；不预设 TATA | 本地指定转录本/注释 |
| 剪接供体/受体邻近位点 | 用于识别结构性解释，不能计入 TF-motif 阳性分母 | 本地指定转录本/注释 |

- [ ] 使用可用的文献检索技能，优先本地 Paperbase；来源不足再查原始论文。保留物种、细胞、实验类型、直接位点或仅关系证据。
- [ ] 建立 `hypothesis_registry.tsv`，先登记 family/PWM/来源/预期方向（未知写 unknown）。然后仅基于参考序列和注释定位，生成 `candidate_instances.tsv`，不先看 model response 再选 motif。
- [ ] 输出全部合格命中；若只展示约十个，按文献证据等级、参考 motif 匹配及覆盖的预登记规则选取，未入选实例仍保留。预期方向不依据结果回填。
- [ ] 同一坐标的多个相似 PWM 合并为一个 family 实例并保留 aliases；同一 family 不同坐标仍是不同实例。记录嵌套/重叠实例，不能将其当独立 motif 证据重复计数。
- [ ] 已受历史 ISM 选择影响的实例标 `prior_ism_selected=true`；不能称严格盲选/独立 holdout。此前整个 Mdk 已被探索，换模型执行不使数据重新独立。
- [ ] 类别无命中时保留 `no_reference_hit`；不随意降 motif 阈值以凑十个。
- [ ] 将评分、PWM版本、背景、轨道、候选、方向和控制匹配规则写成 `score_contract.json`，记录时间与 hash，之后才进入 E2/E3。存在必须改变科学含义的冲突时先交问题单给用户。

### E2：标准化读出与编辑级数据

- [ ] 生成长表 `edit_effects.tsv`；按 Mdk 分块读取大型 parquet，不无差别加载九基因所有 profile。
- [ ] 保留每一条替换，不能只留 median；验证三个 replacement index 和实际不同 ALT 的数量，重复 ALT 不当独立重复。
- [ ] 统一 ratio-of-sums；必须包含 `n_bins/ref_sum/alt_sum/pseudocount`、signed effect、absolute delta、参考质量字段。
- [ ] 生成 `center_effects.tsv`，按模型/轨道/读出/中心汇总中位数、MAD、方向计数；四细胞 join 为严格一对一。
- [ ] 对旧结果重构读出时验证精度：若只剩有损粗粒度 log2FC，不能声称精确恢复原生分辨率。保留重构方法与误差，无法匹配则中止相应桥接分支。

### E3：三个评分与背景

- [ ] 实现第 4 节三个分数的纯函数，先做合成测试，再计算实际数据。
- [ ] 为每个实例保存背景成员 `background_windows.tsv`，含几何/覆盖、纳入排除原因和未排名统计量。
- [ ] 生成 `region_scores.tsv` 与 `original_view_scores.tsv`，保留原始 E/C/U、百分位、背景数、适用状态与阈值敏感性。
- [ ] 不读取跨 AG/BZ combined score 作为输入；原模型 historical Method 3 的中心实现先做既有数据重现核验，区域背景变更另列。
- [ ] 将低参考信号、替换异质性、single-center dominance 和视图数量不足写为旗标，不把质量警告隐藏在总分内。

### E4：独立 motif 破坏证据和对照（不输入 E3 的分数）

- [ ] 参考扫描沿用 JASPAR2024 vertebrates、p<=1e-4；阈值为匹配筛选，不是 TF结合/ISM显著性。
- [ ] 保留旧 `native_loss` 兼容标签：差值>=5、同中心至少2/3替换支持。注明这只是历史工具标签，不是新的因果判定。
- [ ] 固定同一个原生 motif instance/PWM/strand，对 REF 和 ALT 计算未阈值截断的连续分数；未显著匹配不填为真实0。旧工具取 edit-overlap 最大 hit 且缺失填0，不能直接用作固定实例连续破坏量。
- [ ] 另查编辑是否新建或强化其他 motif、是否同时破坏同 family 的另一实例、是否影响剪接。固定实例损失与附近 gain 分开列。
- [ ] 输出 `motif_edit_evidence.tsv`，含连续 delta、兼容损失标签、实际改变碱基数、GC/CpG、重叠motif/剪接、gain 信息。
- [ ] 优先寻找同中心、相同实际改变碱基数的 loss/preserved 配对。可先定义 loss 为未截断分数减少>=5，preserved 为参考和替换均通过匹配门槛且绝对分差<1，中间状态为 ambiguous；这是工作分类而非已校准结合阈值，须在 E1 写入契约，不能事后改变。
- [ ] 相邻中心的次级匹配必须预登记距离和序列匹配规则，并单独列出，不冒充同中心反事实。若匹配不足，记录 `no_matched_preserving_control`，不生成新突变填补。
- [ ] 输出逐配对 `motif_control_pairs.tsv` 与描述性汇总 `motif_response_evidence.tsv`：独立中心数、不同编辑数、方向和 loss/preserved 效应差。预期方向未知时用绝对效应差并明确是双向敏感性；已知激活/抑制方向则保留方向性对比。
- [ ] 不把多个相邻中心乘三个替换当独立 n。不默认跑普通 t-test、普通逐行 bootstrap 或输出未经设计的 p/FDR。证据不足时留原始表供 A 判读。

E3 不受 motif-loss 权重影响；E4 仍共享序列背景和模型数据，因此“独立检验条件”不等于获得了独立湿实验或消除了所有混杂。

### E5：SNV 桥接（可选，不阻塞主 shuffle 交付）

- [ ] 仅使用现有 1024-bp SNV 覆盖与当前实例的交集。
- [ ] 核验 checkpoint、输入序列、REF、轨道、读出实际 bin 和公式；特别处理 23-bp TES差异。
- [ ] 能由现有输出可靠对齐时，输出 `snv_bridge.tsv`，逐碱基、逐 ALT 显示 motif 核心/非核心响应及连续 PWM变化；否则输出不可比较原因和缺失字段。
- [ ] 不把单碱基效应求和等同真实多碱基效果；不以 shuffle 幅度更大证明定位更准确。没有等位基因实验真值时，结果名称使用“尺度响应比较”，不用“SNV准确率验证”。
- [ ] 需要新推理、扩展扫描或新模型时，列在 `blocked_or_deferred.tsv` 等用户决定，本计划不自动授权。

### E6：验收与交付，完成后停止

- [ ] 合成测试覆盖：零效应、三替换单异常值、方向相反、HSC与另一细胞同强、对照更强、背景ties/空集合、缺失中心、负链半开区间、少于三组、两视图适用性、motif阈值截断和重叠实例。
- [ ] saved-artifact 校验覆盖：schema/主键/一对一join、非有限值、replacement完整性、REF一致性、候选全覆盖、背景几何、所有 null 的 reason、无跨模型混分。
- [ ] 测试不得硬编码“某个 motif 必须阳性”；通过门槛是计算/接口正确，不是生物学故事符合预期。
- [ ] 至少抽查若干原始编辑到 region score 的全链条，记录来源和计算；若数值由128-bp float16重构，另报告误差，不能默认为原始精度。
- [ ] 输出 `validation_summary.json`、测试日志和 `execution_handoff.md`；明确主分析可用、局部分支不可用、历史验证未重跑三个不同状态。
- [ ] `git diff --check`，仅记录本任务改变；不提交代码或生成物。按仓库规则写简洁 Progress 记录，不在本任务中擅自压缩历史。
- [ ] 不生成最终科学结论，交包后停止，等待用户交给 A。

## 6. 两阶段之间的最小数据接口

所有新文件位于执行输出根目录。必须附 `schema.json`，逐列说明类型、单位、枚举、主键、nullable 和 reason；文本空值用明确约定，数值缺失不得用0代替。

| 文件 | 最小主键／内容 |
|---|---|
| `score_contract.json` | 版本、候选/PWM/轨道/背景冻结hash、公式、阈值、历史差异 |
| `source_inventory.tsv` | source_id；路径、metadata、hash/大小、核验时间 |
| `comparability_audit.tsv` | comparison_id；模型/尺度/读出可比性、依据、限制 |
| `hypothesis_registry.tsv` | hypothesis_id；来源、物种/细胞、证据级别、family、预期方向 |
| `candidate_instances.tsv` | instance_id；区间/链、PWM aliases、类别、先验来源、prior_ism_selected、覆盖 |
| `edit_effects.tsv` | backend + mutation_id + track_id + readout_id；完整REF/ALT测量及效应 |
| `center_effects.tsv` | backend + center_id + track_id + readout_id；a/d/MAD/replicates |
| `background_windows.tsv` | instance_id + background_id + backend + score_id + track/readout；范围、覆盖、原始统计、纳入原因 |
| `region_scores.tsv` | instance_id + backend + score_id + track/readout；raw_statistic、score、background_n、rank_pass、status/reason |
| `original_view_scores.tsv` | instance_id + backend + view；区域分数、group数、驱动组及signed effects |
| `motif_edit_evidence.tsv` | instance_id + pwm_id + mutation_id；连续REF/ALT PWM、loss/gain、混杂字段 |
| `motif_control_pairs.tsv` | pair_id + backend + track/readout；双方mutation_id、匹配层级、差异 |
| `motif_response_evidence.tsv` | instance_id + backend + track/readout；覆盖/对照数量、方向、描述性效应差、限制 |
| `snv_bridge.tsv` | instance_id + backend + mutation_id + track/readout；可选对齐后的SNV测量 |
| `blocked_or_deferred.tsv` | item_id；阻塞分支、原因、已有替代、需要用户决定的动作 |
| `validation_summary.json` | engineering_status、analysis_ready_scope、检查结果、未运行项、warning/error |
| `execution_handoff.md` | 精确运行命令、输入输出、测试、偏差、未解决项；不含最终机制结论 |

建议状态：`ok`、`not_applicable`、`insufficient_background`、`incomplete_coverage`、`no_reference_hit`、`no_matched_preserving_control`、`incomparable_readout`、`low_reference_warning`、`validation_failed`。区分质量旗标与主状态；`rank_pass` 可为 null，不将不可评价编码为 False。

有可选分支未完成但主表验证通过时，交付可以完成，必须限定 `analysis_ready_scope`。键错配、坐标冲突、读出混用、来源不明或分数计算错误时，主分析不得通过。

## 7. 分析模型 A 的任务清单

### A0：独立验收，先判数据可用范围

- [ ] 先读本计划、`score_contract.json`、`execution_handoff.md`、`validation_summary.json` 与 `schema.json`，再读结果。
- [ ] 复核公式、背景、轨道、候选是否按冻结版本执行；尤其检查 max-other vs mean-other、AG/BZ合并、95阈值、物理同宽、缺口处理。
- [ ] 独立抽查每类分数至少一个可用实例；以长表重算关键统计，不靠 execution prose 认定正确。可做只读重算，不静默写回数据。
- [ ] 对 blocking 问题给 E 一个精确修正单；非阻塞限制进入最终报告。工程status=ok不代表生物学假说成立。

### A1：逐实例判断“是否真的测试了 motif”

按以下层级给结论，分开记录，不能只打一列 effective=true：

| 层级 | 可报告结论 | 必须看的证据 |
|---|---|---|
| 未充分测试 | 无法评价，不是生物学阴性 | reference hit、覆盖、实际破坏、质量状态 |
| 区域响应未恢复 | 当前模型/读出未见突出区域响应 | S_RNA或对应original视图、原始效应量、实际测试充分性 |
| 区域敏感，归因未定 | 有高排名效应，但不知是否来自目标motif | score、方向、缺失控制/重叠motif/剪接 |
| 支持motif相关响应 | 实际破坏与响应有关联，替代解释得到一定约束 | 匹配loss/preserved、连续PWM变化、位置与方向；证据仍为模型内 |
| HSC优先 | 在上层基础上进一步支持HSC相对优势 | C>0、S_HSC>95、S_RNA>95及四细胞参考质量 |
| 局部调控响应 | original局部视图支持，但不等于RNA输出支持 | S_MULTI两视图与具体轨道 |

若多个TF共用PWM，最多说family；若只存在关系证据，不用“已知精确阳性位点”表述。剪接位点与TF-motif分开统计。

### A2：AG 完整讨论，再 BZ 完整讨论

每个模型家族独立完成：

1. 参考预测与读出可解释性。
2. 全部候选清单的三个分数适用性、原始效应、方向与独立motif证据。
3. original与FT同一编辑的响应对照；RNA/CAGE/染色质分别解释，不能将百分位差当fine-tuning增益。
4. 至多三个代表案例：支持假说、未恢复/反向、结构或一般转录解释。每类先按E1登记的生物证据优先级选择，平局按instance_id；类别为空直接报告，不硬凑三个。
5. 与SNV桥接的证据和限制；若不可比，明确停止该结论，不用相邻图形替代定量对齐。

可以最后附一段AG/BZ一致与分歧的描述性总结，但不生成混合评分、投票阳性或统一效应量。

### A3：回到科学问题与反证

- [ ] 分别回答“有模型敏感性”“有motif相关性”“有HSC优先性”三层，禁止压成“ISM有效”。
- [ ] 主动讨论：GC/CpG/复杂度、shuffle实际改动数、新建motif、剪接、低参考信号、轨道不匹配、重叠窗口依赖、候选历史选择偏差。
- [ ] 评分未过线但原始效应有方向时，区分“未达到预登记排名标准”与“完全无响应”。
- [ ] 约十个不同证据级别的实例不能估计模型总体准确率；不同适用范围的三个评分不是同一测试集上的三个分类器。
- [ ] 缺少匹配反事实时不宣布motif归因成立；没有独立等位基因真值时不宣布SNV级精度差。
- [ ] 若需要新推理或湿实验，最后只提出最小、可证伪的下一步及其信息增益，不执行。

### A4：最终交付

- 输出根目录下写 `final_analysis.md`，结构为：一句话结论 → 数据可用范围 → AG逐例 → BZ逐例 → motif/SNV尺度结论 → 替代解释 → 最小下一步。
- 写 `instance_interpretations.tsv`：instance_id/backend、测试充分性、区域响应、motif归因、HSC优先、主要反证、证据表行键、结论等级；不能改写执行表。
- 如生成图表，renderer只消费已验证接口；效应图用signed log2FC、分数图单独标rank，缺口不连线、参考信号可见、未测试位点明确标注。参考profile图所需的数据由E提供验证接口，不让报告层读取raw parquet重算科学结果。
- HTML/PDF仅在用户要求或交付需要时生成，canonical留在该输出目录；最终PDF另放便利副本到`/home/fqijun/report`，不替代canonical。
- 附 `analysis_review.md`，说明独立复核项、结论不确定性及执行修正记录。

## 8. 不在此次执行范围内

- 新的全窗口模型推理、训练、checkpoint替换、增加original轨道推理。
- 新的motif-breaking/preserving/rescue突变推理，或扩展SNV窗口。
- 修改canonical历史输出、旧Workbench数据/服务、其他人的dirty修改。
- 根据结果调整候选、评分权重、窗口、轨道、阈值以提高阳性率。
- 调用远程部署、共享上传、自动发消息、git commit。

出现上述需求时，E/A均应写入缺口或下一步建议，由用户另行授权。

## 9. 可直接复制的交接指令

### 发给执行模型 E

> 请执行 `/work2/Users/qijun/gReLU-replication/docs/mdk_motif_ism_execution_analysis_plan_20260909_zh.md` 中 E0–E6。先读仓库规则。坐标契约已经用户确认，范围不变无需重复询问。只处理现有结果：登记候选、冻结契约、标准化、计算三个单模型评分与独立motif证据、可比时接入旧SNV、测试并交付第6节数据包。AG/BZ不混分。不得启动新推理，不得调参追阳性，不得生成最终生物学结论。无法评价的分支明确留空和原因，不伪装成阴性。交付时给出绝对输出路径、命令、验收状态和未解决项，然后停止，供另一个模型分析。

### 发给分析模型 A

> 请阅读 `/work2/Users/qijun/gReLU-replication/docs/mdk_motif_ism_execution_analysis_plan_20260909_zh.md`，对执行模型交付的输出根目录（由用户附上实际绝对路径）完成 A0–A4。先独立核查schema、冻结契约和关键计算，再分析全部实例，先AG后BZ，并分别讨论original/FT。不要静默改数据、候选或评分；发现阻塞错误先提交修正单。重点区分区域敏感、motif归因和HSC优先，避免把shuffle幅度大解释成SNV精度差。交付final_analysis.md、instance_interpretations.tsv和analysis_review.md，不启动额外推理。
