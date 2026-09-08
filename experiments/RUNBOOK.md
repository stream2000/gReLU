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

## Positive-Control Plots

Generate the fixed positive-control validation layout with the original wet
track PDF on the left and observed-vs-predicted LoRA overlay on the right:

```bash
CUDA_VISIBLE_DEVICES=2 python src/ft-scripts/plot_borzoi_positive_controls.py \
  --checkpoint runs/borzoi_split_chr10_chr11_poisson_multinomial_lora/checkpoints/epochepoch=19.ckpt \
  --out_dir experiments/validation/borzoi_lora_poisson_multinomial_epoch19_controls_raw_cpm
```

Main outputs:

```text
experiments/validation/<run>/per_track_metrics.csv
experiments/validation/<run>/coordinate_qc.csv
experiments/validation/<run>/positive_controls_observed_vs_lora_poisson_multinomial_contact_sheet.png
experiments/validation/<run>/summary/positive_controls_original_left_lora_poisson_multinomial_overlay_right_contact_sheet.png
experiments/validation/<run>/summary/positive_controls_original_left_lora_poisson_multinomial_overlay_right.pdf
```

The left panel uses the PDFs in `referrence/validation/`; the script shells out
to system `pdftoppm` for first-page rendering. `coordinate_qc.csv` parses each
PDF title with `pdftotext` and checks that the right-side overlay uses the same
gene, transcript, strand, chromosome, start, and end.

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

## NTv3 frozen-profile MVP (2026-09-08)

Current user override: a single randomly chosen seed43 on GPUs0/1/2 using DDP,
per-rank batch4 and accumulation2 (effective batch24), validation batch1.
The old three-independent-seed service is stopped; artifacts are preserved.
Current service: `ntv3-seed43-ddp-b4-20260908.service`; outputs:
`experiments/validation/20260908_1032_ntv3_seed43_ddp_b4/`.
Batch benchmarking and DDP resume qualification are saved under
`experiments/validation/20260908_1030_ntv3_single_ddp/`.
The runner now requires `--seed`, `--batch_size`, `--accumulate_grad_batches`
and `--checkpoint_path`; evaluation requires `--seed` and processes that seed
only. See `plan/ntv3_mvp_three_stage_design.md` for the active contract and commands.

Uses the existing mouse/mm10 Saijou split and Poisson-multinomial bin32 labels:
524288 input, central 196608 output, four tracks in hsc/mac/lsec/chol order.
The frozen trunk is 7,692,475 parameters; the local head is 86,148 parameters.
Official external code is pinned separately from the weights. The adapter uses
the final post-skip deconv tensor, **not** the post-GELU MLM projection input.
No shared environment changes or embedding cache are required.

Validated entry points (run `source activate.sh` first):

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python src/ft-scripts/train_ntv3.py \
  --revision c57a813117f0f90142098f81cc912b3357c9ecd1 \
  --dry_run --out /absolute/new-experiment/stage1.json

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python src/ft-scripts/train_ntv3.py \
  --revision c57a813117f0f90142098f81cc912b3357c9ecd1 \
  --max_epochs 1 --out /absolute/new-experiment/tiny50.json
```

Full training uses `--split_name split_chr10_chr11 --max_epochs 40 --seed 43`,
per-rank batch size 4, accumulation 2, lr 3e-4, bf16, val-loss early stopping
with patience 3. `--checkpoint_path` restores optimizer and training state.
Outputs must be new paths; matching existing label-cache manifests are required.
The CLI does not silently regenerate train/val caches.

Inspect the current durable pipeline with:

```bash
systemctl --user status ntv3-seed43-ddp-b4-20260908 --no-pager
```

`pipeline_status.json` is the run state; `seed*/progress.json` is updated per
epoch. Only `final_validation_summary.json` plus validated metric TSVs indicates
completion. `run_ntv3_mvp.py` waits for the single DDP training run,
then `finish_ntv3.py` locks its selected checkpoint hash before creating experiment-local test
labels and evaluating chr11. A negative result does not trigger retraining.
