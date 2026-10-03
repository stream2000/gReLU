# NTv3-8M frozen-profile MVP: active execution contract

Updated: 2026-09-08. Status: implementation and qualification passed; full training
is running. This document supersedes the original three-independent-seed design.
The original text remains locally under
`experiments/archive/ntv3/20260908_superseded_three_stage_design.md` (not committed).

## Fixed scientific configuration

- Mouse/mm10; reuse `split_chr10_chr11`: 11,201 train / 662 validation / 619 test.
- Existing BED coordinates, forward reporting order; no new transcript/TSS definition.
- Input 524,288 bp; central 196,608 bp; 6,144 non-overlapping 32-bp bins.
- Track order: `hsc, mac, lsec, chol` from `saijou_tasks.py`.
- Frozen `InstaDeepAI/NTv3_8M_pre`, weight revision
  `c57a813117f0f90142098f81cc912b3357c9ecd1`.
- External `InstaDeepAI/ntv3_base_model` code revision
  `0ecff3637f0d3ba5b686d1095083218157c2ca34`.
- Final post-skip deconv embedding, not the post-GELU MLM projection input.
  Official selective output `embeddings_deconv_7` matches `hidden_states[-1]`
  exactly in the real interface qualification.
- Frozen trunk: 7,692,475 parameters; fixed local head: 86,148 parameters,
  29-bin receptive field. Pool only after the full-context backbone.
- Reuse `BorzoiMmapSeqDataset -> LightningModel`; train RC augmentation remains,
  validation/test are forward-only. No new trainer or embedding cache.

## Training decisions

The user cancelled repeated seeds and the cost-confirmation gate. Seed43 was
randomly drawn without selecting on metrics. GPUs0/1/2 jointly train this single
model through DDP; GPU3 and unrelated processes are left alone.

- Per-rank batch4, accumulation2, effective batch24; validation batch1.
- Adam, lr3e-4, Poisson-multinomial total_weight0.2, bf16, clipping1.0.
- Maximum40 epochs, val-loss early stopping with patience3, matching the historical
  full Borzoi training budget. Do not claim that all40 epochs must execute.
- Resume weights and optimizer from seed43's completed epoch0; the interrupted
  partial epoch1 is rerun. Earlier best checkpoint remains eligible by val loss.
- Old seed17/29/43 artifacts are preserved, but no multi-seed stability claim is made.

Full-window benchmarks of per-card batch1/2/4/8/16 measured peak allocated memory
of approximately2.88/5.71/11.38/22.72/45.41GB and similar throughput near7.8 samples/s.
Three-rank tiny50 batch4/8 had median batch times0.517/1.032s. Batch4 was selected
for comparable throughput, lower memory and slightly lower observed pilot wall time;
this is not evidence of a large speedup from increasing batch size.

## Entry points

| File | Responsibility |
|---|---|
| `src/grelu/model/trunks/ntv3.py` | Pinned offline loading, strict token mapping, final embeddings, crop/pool, freezing |
| `src/grelu/model/heads.py` | Local four-track probe |
| `src/grelu/model/models.py` | `NTv3PretrainedProfileModel` registration |
| `src/ft-scripts/train_ntv3.py` | Single-window qualification, existing-stack training, DDP audit, resume validation |
| `src/ft-scripts/benchmark_ntv3_batch.py` | Real-window memory/throughput measurement without optimizer updates |
| `src/ft-scripts/run_ntv3_mvp.py` | Durable one-seed DDP training and automatic evaluation sequencing |
| `src/ft-scripts/finish_ntv3.py` | Checkpoint lock, test labels, per-track metrics and validation summary |

The official rotary cos/sin caches are derived runtime state: omit them from saved
state and clear them before strict reload. Learned weights and other buffers stay
in the frozen checksum. Hyperparameters must reflect effective training settings.

## Running and checking

Use `source activate.sh`. The gated weight and code snapshots must already exist;
runtime loading is offline. Credentials never enter commands, checkpoints or Git.

Current service: `ntv3-seed43-ddp-b4-20260908.service`.
Current output: `experiments/validation/20260908_1032_ntv3_seed43_ddp_b4/`.

```bash
systemctl --user status ntv3-seed43-ddp-b4-20260908 --no-pager
```

To launch an equivalent new run after qualifying the same batch configuration:

```bash
python src/ft-scripts/run_ntv3_mvp.py \
  --root /absolute/new-output-directory \
  --tiny50_result experiments/validation/20260908_1030_ntv3_single_ddp/tiny_b4_final/training_seed43.json \
  --revision c57a813117f0f90142098f81cc912b3357c9ecd1 \
  --seed 43 --max_epochs 40 --batch_size 4 --accumulate_grad_batches 2 \
  --checkpoint_path experiments/validation/20260908_0943_ntv3_full_mvp/seed43/checkpoints/last.ckpt
```

Do not run a second copy against the current output directory. New outputs must
not exist. Existing train/validation label-cache manifests and shapes are checked;
cache mismatch fails instead of silently rebuilding labels.

## Qualification and completion

Real single-window, tiny50, three-rank DDP, optimizer resume, strict checkpoint
reload and frozen-state qualification have passed. Batch/DDP evidence is under
`experiments/validation/20260908_1030_ntv3_single_ddp/`; initial evidence is under
`experiments/validation/20260908_0940_ntv3_8m_pre_mvp/`.

```bash
python -m pytest -q --no-cov tests/test_ntv3_model.py tests/test_ntv3_training.py tests/test_heads.py tests/test_models.py
git diff --check
```

`pipeline_status.json` and per-epoch `seed43/progress.json` describe live state.
After successful training, the runner invokes `finish_ntv3.py --seed 43` to lock
the best checkpoint hash before materializing experiment-local test labels and
evaluating the existing chr11 split. Final metrics use exact sequential evaluation
(training-time DDP validation may pad the distributed sampler).

Completion requires `final_validation_summary.json`, validated metric keys and
finite loss/MSE, plus `metrics_by_seed.tsv` and `aggregate_metrics.tsv`. The latter
contains a single seed, not an estimate of cross-seed uncertainty. Constant-predictor
Pearson is undefined and retained as NaN. Negative scientific results are preserved;
test metrics never trigger retraining. No claim that bulk supervision is harmful
or that this single baseline establishes superiority over Borzoi is warranted.
