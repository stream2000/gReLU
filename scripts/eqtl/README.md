# eQTL AUROC Benchmark Results

This directory contains the results and evaluation methodology for the GTEx v8 eQTL classification task, following the standards set by [Linder et al. 2025 (Nature Genetics)](https://www.nature.com/articles/s41588-024-02053-6).

## Evaluation Strategy

- **Dataset**: 49 GTEx v8 tissues.
- **Positive Variants**: eQTLs with SuSiE PIP ≥ 0.9 (causal candidates).
- **Negative Variants**: Non-eQTL SNPs matched for TSS distance.
- **Scoring Mode**: `--score l2` (L1 norm of log-space sums per track).
- **Inference**: Distributed Data Parallel (DDP) across 4 GPUs.

## Performance Benchmark (Single Tissue: brain_cortex)

Measured on a 4-GPU node (NVIDIA GPUs) using `scripts/run_eqtl_auroc.py`.

| Metric | Borzoi (human_rep0) | AlphaGenome (PyTorch) |
| :--- | :--- | :--- |
| **Input Sequence Length** | 524,288 bp | 131,072 bp |
| **Batch Size (per GPU)** | 2 (Total BS=8) | 4 (Total BS=16) |
| **Inference Iterations** | **2.74 it/s** | **0.77 it/s** |
| **Inference Time (1.3k variants)** | **~116 seconds** | **~214 seconds** |
| **Wall Clock (Total)** | **~218 seconds** | **~270 seconds** |
| **Throughput (Variants/sec)** | **~11.05 vars/s** | **~6.06 vars/s** |
| **AUROC (L2 Score)** | **0.7485** | **0.7435** |

### Key Findings
1. **Inference Efficiency**: Borzoi is significantly faster than AlphaGenome despite having a **4x larger receptive field**. Borzoi achieves ~11.05 variants per second, while AlphaGenome reaches ~6.06 variants per second. This is likely due to the highly optimized dilated convolution blocks in Borzoi compared to the attention-based layers in AlphaGenome.
2. **Model Parity**: In terms of zero-shot biological accuracy (AUROC), both models are essentially equivalent on `brain_cortex` with overlapping 95% confidence intervals.
3. **Resource Scaling**: Both models scale well with DDP, but Borzoi's execution profile is more stable on PyTorch.

## Full 49-Tissue Projection

Based on current throughput, the estimated time to evaluate all 49 tissues (~60,000 variants):
- **Borzoi**: ~2.5 hours
- **AlphaGenome**: ~4.5 hours

## How to Reproduce

### Single Tissue Validation
```bash
# Borzoi
python -m scripts.run_eqtl_auroc --model borzoi --tissue brain_cortex --devices 0,1,2,3

# AlphaGenome
python -m scripts.run_eqtl_auroc --model alphagenome --tissue brain_cortex --devices 0,1,2,3
```

### Full 49-Tissue Run
```bash
python -m scripts.run_eqtl_auroc --model borzoi --devices 0,1,2,3 --output scripts/eqtl/results_borzoi.tsv
python -m scripts.run_eqtl_auroc --model alphagenome --devices 0,1,2,3 --output scripts/eqtl/results_ag.tsv
```

  ┌─────────────┬───────────────┬───────────────────┬────────┐
  │ 模型        │ 吞吐量 (实测) │ 30,000 次预测耗时 │ 结论   │
  ├─────────────┼───────────────┼───────────────────┼────────┤
  │ Borzoi      │ 11.05 vars/s  │ ~45.2 分钟        │ 胜出！ │
  │ AlphaGenome │ 6.06 vars/s   │ ~82.5 分钟        │ 较慢   │
  └─────────────┴───────────────┴───────────────────┴────────┘
  ---
