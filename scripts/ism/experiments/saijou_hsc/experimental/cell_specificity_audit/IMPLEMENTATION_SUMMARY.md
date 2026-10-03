# Saijou 细胞特异性大规模审计：实现说明

## 1. 目的

本次工作将原先基于少量对照窗口的细胞特异性检查，扩展为固定验证集上的大规模 bin-level 输出审计。核心问题是：微调模型的四条 Saijou 轨道，是否保留了观测标签中的细胞间差异，以及模型是否会在观测为静默的区域产生假阳性信号。

本审计只评价模型输出层的特异性，不据此直接证明 embedding 坍缩、灾难性遗忘或其具体机制。

## 2. 与上一个对话基础设施的衔接

上一个对话已经建立了以下基础设施：

- 四轨道观测值与预测值的归一化熵特异性、中心化 L2 对比度和 Top-2 log2 margin；
- dominant-track identity、macro recall、confusion matrix；
- Pearson/Spearman 轨道间相关性；
- 按验证窗口进行 cluster bootstrap；
- all bins、observed-active 和训练集 P90/P95/P99 等 mask；
- 九个已保存对照基因的窗口均值敏感性分析；
- 8 个窗口的 GPU smoke test，用于检查 checkpoint、shape、推理和输出链路。

这些内容被保留并复用。本次工作没有把九个对照基因的结果当作全验证集结论，而是在同一套指标和输出契约上增加大规模 Peak 分层与阴性对照。

## 3. Peak 定义与防泄露设计

Peak 选择不使用预测值。训练标签先计算：

1. 四轨道总信号的 train P90；
2. 每条轨道各自的 train P95；
3. Top-2 margin 使用的正值 train-label epsilon。

这些阈值随后固定应用于验证集。

验证 bin 首先满足总信号达到 train P90，成为 `peak_candidate`。再根据达到各轨道 train P95 的轨道数分层：

- `shared_peak`：4 条轨道全部达到各自 train P95；
- `cell_specific_peak`：1–2 条轨道达到各自 train P95；
- `other_peak_0or3of4`：0 条或 3 条轨道达到阈值，作为边界层保留，不悄悄丢弃。

因此，Peak 分层不会根据预测结果挑选有利区域，也不会把 3/4 或 0/4 的候选隐含地混入主要结论。

## 4. 阴性对照

从验证标签中寻找四条轨道总信号为零的 silent bins，使用固定随机种子进行全验证集抽样。抽取数量与 `shared_peak + cell_specific_peak` 的 primary positive 数量相同；本次验证集有足够的 silent bins，因此实现了严格等量匹配。

阴性对照只用于检查模型是否在观测静默区域产生假阳性，不参与 Peak 定义，也不查看预测值后再筛选。

## 5. 32 bp 到 128 bp 的处理

Borzoi e39 checkpoint 的原生输出是 32 bp bins，而本次方案要求以 128 bp bin 为主要审计粒度。因此：

- 复用已完成的 32 bp 全验证集预测缓存；
- 将连续四个 32 bp 预测 bin 求和，得到 128 bp prediction；
- observed 不使用简单聚合后的 32 bp 标签，而是读取同一验证 split 的 canonical 128 bp validation label cache；
- 所有 Peak 阈值均从对应的 128 bp training label cache 重新计算。

这样保证预测与观测都在 128 bp 粒度比较，同时避免独立标签缓存的 bin 边界差异被忽略。

## 6. 数据形状与内部表示

128 bp canonical cache 的数组布局是：

```text
train_labels.npy: (11201 windows, 4 tracks, 1536 bins)
val_labels.npy:   (662 windows, 4 tracks, 1536 bins)
```

模型输出和原始标签在 cache 中使用 `(window, track, bin)` 布局。为了让指标函数始终把最后一维解释为四条轨道，进入分析前会转成：

```text
(window, track, bin) -> (window, bin, track)
```

随后把前两维展平成 bin-level rows：

```text
(window, bin, track) -> (N rows, 4 tracks)
N = 662 * 1536 = 1,016,832
```

每个 row 仍然保留 `window_idx` 和 `bin_idx`，用于窗口级 bootstrap 和结果追溯；没有把这些索引转换成新的染色体坐标。

## 7. 阈值计算算法

令训练标签矩阵为 `X_train[n, t]`，其中 `n` 是训练集中的一个 bin，`t` 的顺序固定为：

```text
[hsc, mac, lsec, chol]
```

首先计算每个 bin 的四轨道总信号：

```text
s[n] = X_train[n, hsc]
     + X_train[n, mac]
     + X_train[n, lsec]
     + X_train[n, chol]
```

然后只在训练标签上计算：

- `q_total_p90 = quantile(s, 0.90)`：Peak candidate 的总信号门槛；
- `q_total_p95` 和 `q_total_p99`：作为额外敏感性 mask；
- `q_track_p95[t] = quantile(X_train[:, t], 0.95)`：每条轨道自己的高信号门槛；
- `epsilon = quantile({X_train[n,t] > 0}, 0.01)`：Top-2 margin 的固定正值 epsilon。

这些值被保存到 `validation_summary.json` 的 `thresholds_train_only` 字段，并且只计算一次。验证集只接受这些阈值，不重新估计分位数。

这样做的关键点是：验证标签只用于判断某个固定 bin 是否属于预先定义的 signal strata，而不是在验证集上重新寻找“最适合模型”的 Peak；预测值完全不参与任何 mask 构造。

## 8. Peak mask 的逐 bin 算法

对于一个验证 bin 的四轨道观测向量 `x = [x_1, x_2, x_3, x_4]`，先计算：

```text
total = sum(x)
is_peak_candidate = (total >= q_total_p90)
k = number of tracks t satisfying x_t >= q_track_p95[t]
```

然后按 `k` 分类：

```text
shared_peak          = is_peak_candidate and k == 4
cell_specific_peak   = is_peak_candidate and 1 <= k <= 2
other_peak_0or3of4   = is_peak_candidate and (k == 0 or k == 3)
```

三类正向 Peak mask 互斥，并且满足：

```text
peak_candidate
    = shared_peak
    + cell_specific_peak
    + other_peak_0or3of4
```

`other_peak_0or3of4` 不进入主要 shared-vs-cell-specific 对比，但必须保留并报告，以便检查 Peak candidate 的总数是否闭合，避免只报告有利的两类。

## 9. Silent negative 的抽样算法

阴性对照的候选池定义为：

```text
silent_pool = {validation bins | total observed signal == 0}
```

primary positive 数量为：

```text
m = count(shared_peak) + count(cell_specific_peak)
```

使用固定 seed `20260825`，从 `silent_pool` 中无放回随机抽取 `m` 个 bin：

```text
n_sample = min(m, len(silent_pool))
silent_negative = seeded_sample(silent_pool, n_sample, replace=False)
```

本次 `len(silent_pool)` 足够大，因此 `n_sample == m == 85,329`。抽样是全验证集范围的固定随机抽样，而不是按预测强度筛选；每个被抽到的 row 会保留其 `window_idx` 和 `bin_idx`。

这个设计回答的是另一个问题：模型是否在标签完全静默的区域产生非零且具有轨道差异的输出。由于 observed silent vector 是全零，观测端的三项特异性指标都应为零；预测端若明显大于零，就表示存在可量化的背景假阳性。

## 10. 三项特异性指标

所有指标都分别对 observed 和 predicted 计算，最后才计算 `predicted - observed`。因此，gap 不是把预测和观测先混合后计算，而是两个独立特异性测量之间的差。

### 10.1 归一化熵特异性

对非零向量 `x`，先归一化为轨道比例：

```text
p_t = x_t / sum_j(x_j)
```

计算 Shannon entropy：

```text
H(p) = -sum_t p_t * log(p_t)
```

四条轨道的最大熵为 `log(4)`，因此特异性定义为：

```text
S_entropy(x) = 1 - H(p) / log(4)
```

解释：

- 四轨道完全均匀时，`S_entropy = 0`；
- 只有一条轨道有信号时，`S_entropy = 1`；
- 四轨道越集中在少数轨道，特异性越高；
- 全零向量被显式定义为 0，而不是产生 NaN。

### 10.2 Centered L2 contrast / total

令四轨道均值为 `mean(x)`，定义：

```text
S_L2(x) = sqrt(sum_t (x_t - mean(x))^2) / sum_t(x_t)
```

它也是 scale-invariant 的：整体把四条轨道同时乘以同一个正数，不改变结果。均匀向量和全零向量的结果定义为 0；一条轨道独占信号时达到四维 one-hot 的最大对比度。

### 10.3 Top-2 log2 margin

将四轨道按信号从高到低排序，取最大值 `top` 和第二大值 `second`：

```text
S_margin(x) = log2((top + epsilon) / (second + epsilon))
```

`epsilon` 来自训练标签正值的 1% 分位数，不从预测或验证集重新估计。这样可以避免第二大轨道为零时出现无穷大，同时保持 margin 对 top-vs-second 差异敏感。

并列最大值的 margin 为 0。全零向量也被定义为 0。

## 11. Top-cell identity 与相关性补充指标

除了连续特异性分数，审计还保留离散方向信息：

1. 用 observed 的 `argmax` 定义该 bin 的 dominant track；
2. observed 有并列最大值时，不把它当作可靠的真实 cell identity，并单独计数；
3. predicted 使用稳定的 `argmax`；
4. 输出 top accuracy、macro recall 和四轨道 confusion matrix；
5. predicted 的并列最大值单独计数，避免把方向不确定性误当作正确预测。

同时对 observed 和 predicted 的轨道两两组合计算 Pearson/Spearman 相关。相关性只作为补充诊断，不能单独等价于“特异性坍缩”：四条轨道都随总体信号变化时，相关性可能很高，但轨道比例仍然可能过于平均。

## 12. Window-cluster bootstrap 算法

每个 bin 的 gap 记为：

```text
g[w,b] = S_predicted[w,b] - S_observed[w,b]
```

由于同一 validation window 内的 bins 共享输入背景和序列上下文，不能把全部 bin 当作相互独立的样本。实现中以 `window_idx` 作为 cluster：

1. 先计算每个窗口在当前 mask 下的 gap 总和与 bin 数；
2. 从参与该 mask 的窗口集合中有放回抽取同样数量的窗口；
3. 用被抽中窗口的总 gap 除以被抽中窗口的总 bin 数，得到一次 bootstrap mean；
4. 重复 2,000 次；
5. 取 bootstrap 分布的 2.5% 和 97.5% 分位数作为 95% CI。

点估计仍然是所有选中 bin 的总体均值；bootstrap 只用于表达窗口层面的不确定性。不同 mask 的有效窗口数可以不同，因为某些窗口没有落入对应 Peak strata。

## 13. 完成的全量审计

审计对象为 Borzoi LoRA e39 的固定 chr10/chr11 validation split：

- 662 个 validation windows；
- 每个 window 1,536 个 128 bp bins；
- 共 1,016,832 个 bin-level rows；
- 训练阈值定义的 Peak candidate：111,878；
- `shared_peak`：23,750；
- `cell_specific_peak`：61,579；
- `other_peak_0or3of4`：26,549；
- 等量 `silent_negative`：85,329。

每个主要分层都计算 observed、predicted 和 predicted-minus-observed gap，并使用 validation window 作为 bootstrap cluster，避免把同一窗口内的 bins 当作完全独立样本。

## 14. 主要结果

以下 gap 均为 `predicted - observed`：

| 分层 | Entropy gap | Centered L2 gap | Top-2 log2 margin gap |
|---|---:|---:|---:|
| `shared_peak` | -0.0623 [-0.0715, -0.0545] | -0.0987 [-0.1090, -0.0893] | -0.3532 [-0.4173, -0.2962] |
| `cell_specific_peak` | -0.3273 [-0.3488, -0.3054] | -0.2923 [-0.3055, -0.2788] | -1.5372 [-1.7662, -1.3309] |
| `silent_negative` | +0.1017 [+0.0976, +0.1055] | +0.2543 [+0.2480, +0.2601] | +0.3689 [+0.3520, +0.3854] |

这组结果支持两个输出层现象：

1. 预测特异性在 shared Peak 上下降，在 cell-specific Peak 上下降更明显；
2. silent negative 中观测特异性为零，但预测出现非零轨道差异，说明存在可量化的假阳性输出。

这些结果仍然是输出审计证据，不能单独定位问题来自 Head、LoRA、归一化状态、损失函数还是 trunk representation。

## 15. 输出产物

主要 128 bp 结果位于：

`experiments/validation/cell_specificity_audit/borzoi_lora_e39/full_val_bin128/`

其中包括：

- `validation_summary.json`：数据、阈值、分层计数、缓存来源和运行契约；
- `summary.tsv`：各 mask 和指标的均值、gap、bootstrap CI；
- `stratified_bins.tsv.gz`：shared、cell-specific 和 silent negative 的可审计明细；
- `bin_metrics.tsv.gz`：全 1,016,832 个 bin 的完整指标表；
- `calibration_deciles.tsv`、`pairwise_correlations.tsv`、`top_cell_confusion.tsv`；
- `specificity_gap_by_mask.png` 和 `specificity_calibration_deciles.png`；
- 128 bp 粒度的 `observed.npy` 和 `predicted.npy`。

## 16. 代码与验证

实现仍放在 experimental 分析目录：

- `metrics.py`：训练阈值、Peak 分层、静默对照抽样、bin 聚合和基础指标；
- `run_cell_specificity_audit.py`：checkpoint/cache 匹配、预测缓存复用、32→128 聚合、全量分析和结果写出；
- `tests/test_saijou_cell_specificity_audit.py`：指标边界、train-only mask、cache layout、bin 聚合、抽样复现性和 synthetic collapse 检查。

最终测试结果为 8 个测试通过；语法检查和 `git diff --check` 通过。当前工作仍属于 experimental audit，没有将其提升为 canonical Saijou workflow，也没有新增或转换基因组坐标文件。
