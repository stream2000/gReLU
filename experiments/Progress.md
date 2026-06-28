# gReLU Replication Progress

## 2026-06-28

- Re-audited the Borzoi head-only training path before extending the run to
  15 epochs. Patched DDP logging in `src/grelu/lightning/__init__.py` to use
  `sync_dist=True` for train/val/test losses and metrics, and added
  `--disable_early_stopping` to `src/ft-scripts/train_borzoi.py` so the 15-epoch
  request is not cut short by the old patience-3 callback.

- Restarted head-only training from
  `runs/borzoi_split_chr10_chr11_log1p_mse_headonly/checkpoints/last-v1.ckpt`
  in tmux session `borzoi_headonly_resume15` on physical GPUs `0,1,3`:
  `CUDA_VISIBLE_DEVICES=0,1,3 python src/ft-scripts/train_borzoi.py --split_name split_chr10_chr11 --finetune_mode headonly --devices 0,1,2 --num_workers 2 --max_epochs 15 --checkpoint_path runs/borzoi_split_chr10_chr11_log1p_mse_headonly/checkpoints/last-v1.ckpt --disable_early_stopping`.
  Log:
  `experiments/logs/borzoi_headonly_split_chr10_chr11_3gpu_resume15_synced_20260628_101745.log`.
  Startup confirmed `early_stopping: False`, 3 DDP ranks, restored optimizer
  state, and 7,684 trainable head parameters out of 171,279,652 total.
  The run completed at epoch 14 / max_epochs 15 with best checkpoint
  `runs/borzoi_split_chr10_chr11_log1p_mse_headonly/checkpoints/epochepoch=14.ckpt`;
  final/best synced `val_loss=0.010852`, `val_mse=0.010852`, and
  `train_loss_epoch=0.012292`.

- Generated latest positive-control prediction comparisons for head-only epoch
  14 versus LoRA epoch 19 on raw CPM scale. Outputs:
  `experiments/validation/borzoi_headonly_epoch14_controls_raw_cpm/` and
  `experiments/validation/borzoi_headonly_epoch14_vs_lora_epoch19_raw_cpm/`.
  In the direct observed/head-only/LoRA overlay, LoRA had lower four-track
  gene-window MSE at 9/9 loci and lower expected-track MSE at 7/7 loci with an
  expected marker track, but both models still show diluted cell-type
  specificity and shared peak behavior.

- Re-audited the poor head-only positive-control plots. Label cache alignment
  checked against live bigWig reads for train/val sampled windows and matched
  exactly. Found a real freeze-logic issue: setting `requires_grad=False` on
  the Borzoi embedding did not freeze BatchNorm running stats. Existing
  head-only epoch 14 has 11 embedding BatchNorm counters advanced to 61,612,
  and LoRA epoch 19 has the same issue with counters at 74,681. Patched
  `grelu.lightning.LightningModel` and `train_borzoi.py` so future head-only
  and LoRA runs keep embedding BatchNorm stats in eval mode during training.

- Added `src/ft-scripts/run_borzoi_style_retrain.sh`, an overnight sequential
  runner that trains LoRA first and then head-only from fresh Borzoi pretrained
  initialization using `target_mode=poisson_multinomial`, `total_weight=0.2`,
  max 20 epochs, and early stopping patience 3. Started it in tmux session
  `borzoi_style_retrain` on physical GPUs `0,1,3`. LoRA log:
  `experiments/logs/borzoi_split_chr10_chr11_poisson_multinomial_lora_3gpu_20260628_170023.log`;
  head-only log will be
  `experiments/logs/borzoi_split_chr10_chr11_poisson_multinomial_headonly_3gpu_20260628_170023.log`.
  Full `poisson_multinomial` label cache was built successfully; LoRA DDP
  initialized with 44 targets and initial synced `val_loss=8.06947`.

## 2026-06-26

- Borzoi LoRA audit: current checkout has Borzoi head-only fine-tuning scripts
  but no in-repo `LoRA` / `peft` implementation. Pulled references to
  `/tmp/codex_refs/scooby` and `/tmp/codex_refs/lauradmartens-peft`; Scooby uses
  a PEFT fork with `nn.Conv1d` LoRA support. For gReLU Borzoi, a matched target
  set should adapt selected trunk convs plus transformer `to_q` / `to_v` and
  FFN dense layers while leaving separable depthwise/pointwise layers alone.

- Installed `peft @ git+https://github.com/lauradmartens/peft.git` into
  `grelu_dev` and changed `src/ft-scripts/train_borzoi.py` to support
  `--finetune_mode lora`. LoRA mode uses PEFT on the Borzoi embedding and
  trains the adapter parameters plus the new four-track head; head-only remains
  available as `--finetune_mode headonly`.

- LoRA dry run passed with
  `python src/ft-scripts/train_borzoi.py --split_name split_chr10_chr11_tiny50 --finetune_mode lora --devices cpu --num_workers 0 --max_epochs 1 --dry_run`.
  It matched 44 Borzoi trunk targets, used rank 8 / alpha 16, reported
  1,306,084 trainable parameters out of 172,578,052 total, and produced
  `yhat` shape `(1, 4, 6144)` matching the cached target.

- Manual tiny-split LoRA backward/optimizer smoke passed on CPU. One batch from
  `split_chr10_chr11_tiny50` produced loss `0.04677`; 44 target modules created
  88 trainable LoRA tensors, 43 tensors had nonzero gradients after backward,
  and 3 of the first 8 sampled LoRA tensors changed after one Adam step.

- Started an overnight full-split LoRA run in tmux session `borzoi_lora_3gpu`
  using physical GPUs `0,1,3` and excluding occupied GPU 2:
  `CUDA_VISIBLE_DEVICES=0,1,3 python src/ft-scripts/train_borzoi.py --split_name split_chr10_chr11 --finetune_mode lora --devices 0,1,2 --num_workers 2 --max_epochs 20 --lr 1e-5`.
  Logs are in `experiments/logs/borzoi_lora_split_chr10_chr11_3gpu_20260626_003701.log`;
  Lightning DDP initialized all 3 ranks.

- The 3-GPU LoRA run reached completed epoch 11 with best `val_loss=0.006214`
  before stopping during epoch 12 from a DataLoader `KeyError: '>'` while
  converting a fetched sequence. Full train/val interval scans found no literal
  bad FASTA windows, so `borzoi_mmap_dataset.py` was hardened to read
  `pyfaidx` `.seq`, validate sequence length, and map non-ACGTN characters to
  `N`. Added `--checkpoint_path` to `train_borzoi.py` and resumed from
  `runs/borzoi_split_chr10_chr11_log1p_mse_lora/checkpoints/last.ckpt` in tmux
  session `borzoi_lora_3gpu`; DDP restored all states successfully.

## 2026-06-25

- Set up `/home/fqijun/python/gReLU-replication` as the clean
  `origin/alphagenome` workspace. `activate.sh` uses the shared `grelu_dev`
  conda environment while prioritizing this checkout's `src/`, and `import
  grelu` resolves to this repo. This branch has AlphaGenome support but not the
  separate ISM/MREG stack from `origin/ism` or the dirty original
  `/home/fqijun/python/gReLU` worktree.

- Borzoi/Saijou data profile: four CPM-normalized 1 bp bigWigs (`hsc`, `mac`,
  `lsec`, `chol`) live under
  `/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig/`.
  Full `split_chr10_chr11` has 11,201 train / 662 val / 619 test windows; the
  1/50-size `split_chr10_chr11_tiny50` has 225 train / 14 val / 13 test
  windows and is useful for smoke tests.

- Cache/memory decision: the reference pickle path stores large in-memory
  Python arrays and scales poorly with many DataLoader workers. Tiny50 probes
  showed raw 1 bp label mmap reduced `num_workers=2` total PSS from about
  1.91 GB to 0.76 GB, and 32 bp binned-label mmap reduced the train label cache
  to 21.1 MB. Detailed probe notes remain in ignored `experiments/` docs.

- Production fine-tuning scripts now live under `src/ft-scripts/`:
  `borzoi_mmap_dataset.py`, `make_borzoi_splits_mouse.py`, `train_borzoi.py`,
  with the runbook at `experiments/RUNBOOK.md`. The current path pre-aggregates
  labels to 32 bp, stores them as mmap-compatible `.npy` files, lazy-opens
  FASTA/labels in workers, and still uses gRELU's `LightningModel` /
  `train_on_dataset()`.

- Full `split_chr10_chr11` one-epoch head-only Borzoi fine-tune completed on
  physical GPUs 1/3/0. It uses pretrained Borzoi weights with the embedding
  frozen and a new 4-task head trained on `hsc/mac/lsec/chol`; final metrics
  were `val_loss=0.04035`, `val_mse=0.04062`, and
  `train_loss_epoch=0.06174`. gRELU's default `val_pearson` logged `nan`, but
  posthoc val-set Pearson over all bins was `0.04145` globally and
  `hsc=0.07048`, `mac=0.04078`, `lsec=-0.03077`, `chol=0.11252` per track.
  Cache files are 1.1 GB train / 63 MB val, and
  checkpoints are under
  `runs/borzoi_split_chr10_chr11_log1p_mse_headonly/checkpoints/`.

- Positive-control locus validation for the epoch-19 LoRA checkpoint used the
  wet-observation PDFs in `referrence/validation/` and wrote overlays/metrics
  to `experiments/validation/borzoi_lora_epoch19_controls/`. Test-chromosome
  loci `Cd68` and `Col1a1` matched observed top mean tracks (`mac`, `hsc`) and
  showed high gene-window correlations, but several train-chromosome loci show
  amplitude or cell-type-specificity problems (`Pcdh17`, `Epcam`, `Trem2`), so
  this is partial validation rather than a clean biological pass.
