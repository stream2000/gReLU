# eQTL AUROC Benchmark

基于 [Linder et al. 2025 (Nature Genetics)](https://www.nature.com/articles/s41588-024-02053-6) 论文标准，
对 Borzoi / AlphaGenome 进行 GTEx v8 eQTL 分类评估。

## 脚本

| 文件 | 用途 |
|---|---|
| `run_eqtl_auroc.py` | 主评估脚本：加载模型 → 变异效应预测 → AUROC (zero-shot 或 RF) |
| `run_all_tissues.sh` | 批量运行 49 个组织 |

## 数据

- **论文官方 VCF**: `~/.cache/eqtl_finemapping/paper_vcfs/`（49 组织，来自 `gs://borzoi-paper/qtl/eqtl/`）
- **输出目录**: `results/eqtl/`（gitignored）

## 快速开始

```bash
source activate.sh

# 单组织，单 GPU（推荐，无 DDP 风险）
python scripts/eqtl/run_eqtl_auroc.py --model borzoi --tissue brain_frontal_cortex_ba9 --rf --devices 0

# 多 GPU（DDP bug 已修复）
python scripts/eqtl/run_eqtl_auroc.py --model borzoi --tissue brain_frontal_cortex_ba9 --rf --devices 0,1,2,3

# 全 49 组织
bash scripts/eqtl/run_all_tissues.sh
```

## DDP Bug（2026-05-01 修复）

`src/grelu/lightning/__init__.py` 中 `predict_on_dataset` 的 all_gather 在不同 rank 预测数不一致时
（如 459 vs 458）创建了不同大小的 buffer，导致 NCCL 静默污染特征矩阵。

**修复后验证**: brain_frontal_cortex_ba9 单卡 AUROC=0.7444，四卡 AUROC=0.7442，特征矩阵完全一致。

## 相关文档

- `docs/plan&reports/eqtl_auroc_report.md` — 技术决策：SUM vs L2 score、逆变换、RC 平均等
- `docs/plan&reports/eqtl_evaluation_guide.md` — eQTL 评估方法论完整指南
- `docs/plan&reports/borzoi_paper_summary.md` — Borzoi 论文摘要
