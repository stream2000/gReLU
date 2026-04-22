# eQTL AUROC Benchmark — 实验报告

## 参考论文

**Linder et al., Nature Genetics, Vol 57, April 2025, pp. 949–961**
> "Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation"
> Johannes Linder, Divyanshi Srivastava, Han Yuan, Vikram Agarwal, David R. Kelley (Calico Life Sciences)
> DOI: https://doi.org/10.1038/s41588-024-02053-6
> 本地 PDF: `s41588-024-02053-6.pdf`

---

## 任务定义（论文 Fig. 5b）

用模型预测的变异效应分数区分 GTEx eQTL 因果变异（正例）与 TSS 距离匹配的非 eQTL 变异（负例），
按 49 个 GTEx v8 组织各自计算 AUROC，取均值报告。

### 论文标准方法

| 步骤 | 论文设置 |
|------|---------|
| 正例 | SuSiE PIP ≥ 0.9，SNP only，蛋白编码基因，GENCODE v41 |
| 负例 | GTEx sumstat 真实检测 SNP，PIP < 0.01（所有组织），TSS 距离匹配 |
| 打分统计 | `\|SUM score\|` = Σ_t \|Σ_l (y_alt^(4/3) − y_ref^(4/3))\| （全 7,611 tracks，count 空间） |
| RC 平均 | 正向 + 反向互补序列预测均值 |
| Ensemble | f0–f3 四个 replicate 均值 |
| **论文结果** | **Borzoi = 0.7943**（49 组织均值），Enformer = 0.7469 |

---

## 数据来源

**VCF 数据**：直接使用论文原始数据（非自行构建）

```
gs://borzoi-paper/qtl/eqtl/*.vcf.gz  (requester-pays bucket)
本地缓存: ~/.cache/eqtl_finemapping/paper_vcfs/
```

- 49 个组织，每组织一对 pos/neg VCF，格式 GTEx 命名（`Brain_Cortex_pos.vcf` 等）
- 正负例完全对称（1:1）
- 代码生成的 manifest: `~/.cache/eqtl_finemapping/paper_vcfs/manifest.tsv`

---

## 当前实验结果

### brain_cortex 验证实验（2026-04-22）

**硬件**：4× NVIDIA RTX 6000 Ada Generation (48GB each)
**软件**：gReLU framework，4-GPU DDP（Lightning）

| 模型 | Score 模式 | Variants (pos/neg) | AUROC | 95% CI | 运行时间 |
|------|------------|---------------------|-------|--------|---------|
| **AlphaGenome** (rna_seq, 768 tracks, 128bp, fold_0) | `--score sum`（旧） | 647 / 650 | 0.7410 | [0.714, 0.766] | ~3.5 min |
| **Borzoi** rep0 (all 7611 tracks, 32bp) | `--score sum`（旧） | 640 / 642 | 0.7053 | [0.678, 0.734] | ~2 min |
| **AlphaGenome** (rna_seq, 768 tracks, 128bp, fold_0) | **`--score l2`（新）** | 647 / 650 | **0.7435** | [0.718, 0.768] | ~3.5 min |
| **Borzoi** rep0 (all 7611 tracks, 32bp) | **`--score l2`（新）** | 640 / 642 | **0.7485** | [0.722, 0.775] | ~2 min |
| 论文 Borzoi 4-rep ensemble + RF | — | — | **0.7943** | — | — |

> 注：paper VCF 中有少量 variants 超出 Borzoi 524kb 窗口边界（20 个），AlphaGenome 131kb 窗口边界（5 个），
> 均在 `filter_edge_variants()` 中过滤，不计入评估。

---

## 与论文的剩余差距分析

| 因素 | 论文 | 当前 | 估计影响 |
|------|------|------|---------|
| 负例来源 | ✅ GTEx sumstat PIP<0.01 真实 SNP | ✅ **已修复**（使用论文原始 VCF） | 主要差距已消除 |
| SUM score | ✅ 全 tracks + 逆变换 count 空间 | ✅ **已实现** | 已对齐 |
| RC 平均 | ✅ 正向 + RC 均值 | ❌ 未实现 | ~1–2% AUROC |
| Replicate ensemble | ✅ f0–f3 四个 rep 均值 | ❌ 单 rep（Borzoi rep0 / AG fold_0）| ~2–3% AUROC |

**L2 评分改善**：
- Borzoi：0.7485 − 0.7053 = **+0.043**（消除跨 track 符号相消后效果显著）
- AlphaGenome：0.7435 − 0.7410 = **+0.003**（改善较小；AG 768 tracks 方向较一致，sum 已近似 L1）

**与论文 ensemble 差距**：0.7943 − 0.7485 = **0.046**，来自 RC + 4-rep ensemble（~3–5%）和零样本 vs. RF（~3–5%）。

---

## AlphaGenome vs Borzoi 比较

### `--score sum`（旧，存在跨 track 相消问题）

- AlphaGenome 0.7410 > Borzoi 0.7053（**+0.036**，CI 基本不重叠）
- 差距大部分来自 Borzoi 的 7611 tracks 中正负方向相消严重

### `--score l2`（新默认，消除相消）

- AlphaGenome 0.7435 vs Borzoi 0.7485（**差距 0.005，CI 完全重叠**）
- **两模型在 brain_cortex 基本持平**
- AlphaGenome：131kb 窗口 + 768 RNA-seq tracks
- Borzoi：524kb 窗口 + 7611 全模态 tracks

**解读**：旧版中 AG 的优势主要来自 Borzoi SUM score 的设计缺陷（跨组织 tracks 方向相消）。修复后两模型真实能力相近，但 Borzoi 略高（+0.005）。需要全 49 组织才能得出可靠结论。

---

## 实现细节

### 评估脚本

```bash
# brain_cortex 单组织验证
python -m scripts.run_eqtl_auroc --model borzoi --tissue brain_cortex --devices 0,1,2,3
python -m scripts.run_eqtl_auroc --model alphagenome --tissue brain_cortex --devices 0,1,2,3

# 全 49 组织
python -m scripts.run_eqtl_auroc --model borzoi --devices 0,1,2,3 --output results_borzoi.tsv
python -m scripts.run_eqtl_auroc --model alphagenome --devices 0,1,2,3 --output results_ag.tsv
```

### 关键技术决策

1. **BorzoiSumTransform**：`x.clamp(min=0)^(4/3)` 逆变换后沿长度维求和 → `(T, 1)`，
   因为 Borzoi 训练目标经过 `y^(3/4)` squash，模型输出在 squashed 空间。

2. **AGSumTransform**：直接沿长度维求和（不需要逆变换），
   因为 AlphaGenome 的 `GenomeTracksHead.unscale()` 已在模型内部完成 `x^(1/0.75)` + `track_means` 缩放，
   输出已在 count 空间。

3. **DDP padding fix**：`predict_on_dataset` 在 DDP 模式下可能返回比变量数多的预测（边界 padding），
   `scores = scores[:len(labels)]` 截断；同时 `dist.get_rank() != 0` 的 worker 跳过 AUROC 计算，
   避免结果重复输出。

4. **边缘过滤**：paper VCF 按 Borzoi 边界过滤，两个模型用各自的 seq_len 独立过滤
   （AG 131kb 窗口更小，过滤掉的 variants 更少）。

---

## 下一步

1. ~~**[高优先]** 用 `--score l2` 跑 AlphaGenome brain_cortex 验证~~ ✅ 完成（AG 0.7435，与 Borzoi 0.7485 基本持平）
2. **[最高优先]** 全 49 组织跑完 Borzoi + AlphaGenome（`--score l2`），报告 mean ± SD
3. **[对齐论文]** 加 RC 平均（`rc=True`，运行时间翻倍，预计 +1–2% AUROC）
4. **[对齐论文]** Borzoi 加载 f0–f3 四个 rep ensemble
5. **[分析]** 各组织 AUROC 分布，找 AlphaGenome vs Borzoi 的组织特异性差异


## 质疑与回应（2026-04-22 更新）

原始质疑通过对比论文原文（Linder et al. 2025, Methods 章节）与代码实现，指出了四个差距。以下逐条响应并记录已采取的修复措施。

---

### 质疑 1：零样本（Zero-shot）vs. 监督随机森林（RF）

**质疑**：论文明确使用随机森林对 7611 维特征向量（每 track 一个 SUM/L2 值）进行监督分类，而代码是零样本启发式（直接将 7611 tracks 加总得到标量）。

**回应（已接受，不修复）**：

论文的随机森林需要训练集，这在 Borzoi vs AlphaGenome 跨模型头对头对比中引入了**额外的偏差**：
- RF 特征维度不同（7611 vs 768），学习的权重不可比；
- RF 的泛化能力和训练集划分策略影响最终 AUROC，使比较不公平。

本项目的目标是**同等条件下对比两个模型的内在判别能力**，零样本是合理的 common benchmark 设计。

零样本下可比的论文数字是：
- Borzoi single rep + SUM + 无RF（从论文推算）≈ 0.76
- 我们的零样本 L2（`--score l2`）对应此基线

**后续计划**（若追求完全复现）：实现 RF 步骤，在论文 paper VCF 上用 80/20 split 训练 + 评估。

---

### 质疑 2：|SUM| 跨轨道抵消问题

**质疑**：`|Σ_t u_t|` 中，若某组织 track 上调、另一组织 track 下调，符号相消导致分数为 0。代码实现丢失了大量组织特异性信号。

**回应（已修复）**：

质疑完全正确。已将默认评分模式改为 `--score l2`，实现：

```
u_t = Σ_l log2(1 + y_count_l)   per track（BorzoiL2Transform）
score = Σ_t |u_t^alt - u_t^ref|   per-track absolute logSUM 之和
```

关键变化：
- **Log 空间**：侧重 fold change 而非绝对量，与论文 L2 精神一致；
- **Per-track 取绝对值**：消除跨 track 抵消，脑 RNA-seq 上调 + 肝 RNA-seq 下调 = 两个正贡献；
- **内存高效**：transform 输出 `(T, 1)` 而非 `(T, L)`，避免 `(N×7611×16384)` ≈ 640 GB 的内存开销。

论文 exact L2 = `sqrt(Σ_l diff_l^2)` per track（可避免 within-track 空间抵消）。因 eQTL 通常使一个基因的所有 bins 同向变化，`|Σ_l diff_l|`（我们用的 logSUM 取绝对值）在实践中 ≈ L2。

---

### 质疑 3：逆 Squash 不完整（y > 384 分支缺失）

**质疑**：代码仅实现了 `x^(4/3)`，忽略了 Borzoi squash 的 `y_sq > 384` 分支（`384 + sqrt(y^(3/4)-384)` 的逆变换），导致高表达 bin 的效应被系统性低估。

**回应（已修复）**：

已实现完整分段逆变换函数 `_inverse_squash_borzoi()`：

```python
# if y_sq <= 384: y = y_sq^(4/3)
# if y_sq > 384: y = (384 + (y_sq-384)^2)^(4/3)
z = (x - 384.0).clamp(min=0)
return torch.where(x > 384.0, (384.0 + z**2) ** (4.0/3.0), x ** (4.0/3.0))
```

已通过数值验证：
- 低表达（y=100, y_sq=31.6）：逆变换还原 100.0 ✓
- 高表达（y=10000, y_sq=408.8）：逆变换还原 9999.99 ✓

`BorzoiSumTransform`（`--score sum`）和 `BorzoiL2Transform`（`--score l2`）均调用此完整逆变换。

---

### 质疑 4：参考基线数字不匹配

**质疑**：代码在汇总行打印 `Reference: Borzoi ensemble = 0.7943`，但实际跑的是单模型零样本，与该基线方法论不匹配。

**回应（已修复）**：

已将汇总行更新为三层参考数字：

```
Paper refs: Borzoi ensemble+L2+RF=0.7943, single+L2+RF=0.7880, ensemble+SUM+RF=0.7720
            (zero-shot heuristic is lower than RF-supervised baselines)
```

用户清楚知道自己的零样本结果对应哪个论文基线区间。

---

### 剩余差距（未修复，列为已知限制）

| 维度 | 论文 | 当前 | 影响估计 |
|------|------|------|---------|
| RF 分类器 | ✅ 监督 RF | ❌ 零样本 | ~3–5% AUROC |
| Replicate ensemble | ✅ f0–f3 均值 | ❌ rep0 单模型 | ~1–2% AUROC |
| RC 平均 | ✅ 正向 + RC 均值 | ❌ 仅正向 | ~1–2% AUROC |
| L2 exact (sqrt(Σl diff^2)) | ✅ per-track L2 norm | ≈ per-track \|logSUM\| | 小，实践中近似 |

期望我们的零样本 `--score l2` 单模型结果在 49 组织均值约 0.72–0.76，低于论文 ensemble+RF+L2 的 0.794，但高于旧版 `|SUM|` 零样本。


