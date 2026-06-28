# Borzoi Saijou Fine-Tuning Runbook

## Setup

Run from the repository root:

```bash
cd /home/fqijun/python/gReLU-replication
source activate.sh
```

Generate or refresh mouse splits:

```bash
python src/ft-scripts/make_borzoi_splits_mouse.py
```

LoRA mode uses the Scooby-compatible PEFT fork with `nn.Conv1d` support:

```bash
python -m pip install \
  'peft @ git+https://github.com/lauradmartens/peft.git@f00283c5ef35bead9bdc68113797c4619a2cc06d'
```

Default input paths:

```text
bigWig dir: /work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig
mm10 FASTA: /work/Database/Database_fromDocker/Referencedata_mm10/genome.fa
split dir: dataset_store/splits
mmap cache dir: dataset_store/mmap_caches
```

## Quick Test

Use the 1/50-size split:

```bash
python src/ft-scripts/train_borzoi.py \
  --split_name split_chr10_chr11_tiny50 \
  --finetune_mode lora \
  --devices 0 \
  --num_workers 2 \
  --max_epochs 1
```

For a setup-only LoRA smoke check:

```bash
python src/ft-scripts/train_borzoi.py \
  --split_name split_chr10_chr11_tiny50 \
  --finetune_mode lora \
  --devices cpu \
  --num_workers 0 \
  --max_epochs 1 \
  --dry_run
```

## Full Three-GPU Trial

Check GPU availability first:

```bash
nvidia-smi
```

Example using physical GPUs 1, 3, and 0 as logical devices 0, 1, and 2:

```bash
CUDA_VISIBLE_DEVICES=1,3,0 python src/ft-scripts/train_borzoi.py \
  --split_name split_chr10_chr11 \
  --finetune_mode lora \
  --devices 0,1,2 \
  --num_workers 2 \
  --max_epochs 1
```

Head-only remains available as a baseline:

```bash
CUDA_VISIBLE_DEVICES=1,3,0 python src/ft-scripts/train_borzoi.py \
  --split_name split_chr10_chr11 \
  --finetune_mode headonly \
  --devices 0,1,2 \
  --num_workers 2 \
  --max_epochs 1
```

The first full run builds:

```text
dataset_store/mmap_caches/split_chr10_chr11__log1p_mse__bin32/
  train_labels.npy
  val_labels.npy
  manifest.json
```

After the cache exists, reruns reuse it unless `--force_rebuild` is supplied.

## Expected Tensor Shapes

The script performs a shape check before training:

```text
x:    (1, 4, 524288)
y:    (1, 4, 6144)
yhat: (1, 4, 6144)
```

## Outputs

Training outputs are written under:

```text
runs/borzoi_<split_name>_<target_mode>_lora/
runs/borzoi_<split_name>_<target_mode>_headonly/
```

Checkpoints are written under:

```text
runs/borzoi_<split_name>_<target_mode>_lora/checkpoints/
runs/borzoi_<split_name>_<target_mode>_headonly/checkpoints/
```

Resume a stopped LoRA run from the latest checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,3 python src/ft-scripts/train_borzoi.py \
  --split_name split_chr10_chr11 \
  --finetune_mode lora \
  --devices 0,1,2 \
  --num_workers 2 \
  --max_epochs 20 \
  --lr 1e-5 \
  --checkpoint_path runs/borzoi_split_chr10_chr11_log1p_mse_lora/checkpoints/last.ckpt
```

## Notes

- Labels are pre-aggregated to 32 bp bins before caching.
- Label caches are mmap-compatible `.npy` files, not pickled Dataset objects.
- Dataset workers lazy-open FASTA and label mmap handles in their own process.
- LoRA mode uses `lauradmartens/peft` for `nn.Conv1d` support, following the
  Scooby dependency choice. Default LoRA targets are Borzoi trunk convs,
  transformer `to_q` / `to_v`, and FFN dense layers; separable conv layers are
  left untouched.
- Keep `num_workers` conservative for multi-GPU trials; total worker count is
  approximately `num_gpus * num_workers`.
