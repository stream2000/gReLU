# gReLU Replication Progress

## 2026-06-25

- Set up `/home/fqijun/python/gReLU-replication` as the clean
  `origin/alphagenome` workspace. `activate.sh` uses the shared `grelu_dev`
  conda environment while prioritizing this checkout's `src/`, and `import
  grelu` resolves to this repo. This branch has AlphaGenome support but not the
  separate ISM/MREG stack from `origin/ism` or the dirty original
  `/home/fqijun/python/gReLU` worktree.

- Added the Borzoi/Saijou task notes and data profile under `experiments/`.
  The data is four CPM-normalized 1 bp bigWigs (`hsc`, `mac`, `lsec`, `chol`)
  in `/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig/`.
  Full `split_chr10_chr11` is 11,201 train / 662 val / 619 test windows; a
  1/50-size `split_chr10_chr11_tiny50` is available with 225 train / 14 val /
  13 test windows. `referrence/train_borzoi.py` now accepts CLI args including
  `--split_name`, so small tests can use `--split_name split_chr10_chr11_tiny50`.

- Added memory probes for the current pickle cache and mmap alternatives,
  without starting Borzoi. On tiny50, the current train pickle is 787.5 MB and
  contains 1 bp labels; cached-load `num_workers=2` reached about 1.91 GB total
  PSS and ~209 MB `Private_Dirty` per worker. A raw 1 bp label mmap version
  keeps the same `(225, 4, 196608)` label shape but drops the same probe to
  about 0.76 GB total PSS and ~62 MB `Private_Dirty` per worker. A separate
  32 bp binned-label mmap prototype reduces the train label cache to 21.1 MB
  and the worker probe to about 0.57 GB total PSS. Detailed numbers are in
  `experiments/cache_worker_memory_probe_result.md`,
  `experiments/raw_mmap_worker_memory_probe_result.md`, and
  `experiments/mmap_worker_memory_probe_result.md`.
