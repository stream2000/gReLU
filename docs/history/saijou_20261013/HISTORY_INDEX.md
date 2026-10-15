# Saijou HSC / WT1 — closed 2026-10-13

The user closed the Saijou HSC ISM case and the WT1 exact-counterfactual
follow-up on 2026-10-13. Nothing here is an active task. Use this page to find
the code, data, and evidence; do not extend these workflows.

## Where things are

| What | Location |
| --- | --- |
| Code (frozen in place) | `scripts/ism/experiments/saijou_hsc/` — see `WORKFLOW_INDEX.md` there |
| Last uncommitted WT1 edits | commit `a1e4d46` (`--weights`, `--run`, `compare_checkpoints.py`, README) |
| Run artifacts (~190 GB, 75 entries) | `/work3/Users/qijun/gReLU-history/saijou-20261013/ism/<name>/` |
| Old artifact paths | `experiments/ism/<name>` still holds the original copy on `/work2`; the symlink swap (`migrate_saijou.sh swap`) has not been run as of 2026-10-15 |
| Mapping | `artifact_locations.tsv` next to this file |
| Not migrated | `experiments/ism/20261003_hsc_ag_training_data_similarity/` (WT1 TPM sign test reads its `prepared/expr_cpm.parquet`) |
| Ignored legacy code | `scripts/ism/experiments/saijou_hsc/archive/` (main checkout only, not in Git) |
| Workbench | nested repo `experimental/interactive_workbench` pinned at `751b3be`; served from bagpipe on port 8900 |
| Old repo copies (bagpipe) | 16 untracked/dirty files saved in `/work2/Users/qijun/grelu-wt/runner-12/runs/preservation-20261013/` |

The `/work3` copy is the canonical location; the symlinks only keep old
scripts, tests, and docs working. Inside the copy, the four
`20260911_*_ag_res1` links into `20260903_*` are relative.

## Keep in core

Not part of the closed case:

- `src/grelu/interpret/ism/` (generic ISM runner, profiles, shards, validation).
- `src/grelu/model/` AlphaGenome/Borzoi/NTv3 interfaces and Lightning metrics.
- `src/ft-scripts/train_{alphagenome,borzoi}.py`: the fine-tuned model
  adapters import their LoRA helpers at setup, so they can't be archived
  until those helpers are pulled out into a shared module.

## Known debts (for whoever reuses this code)

- Paths are part of the API: scripts locate the repo with `parents[4..6]`,
  some run from the checkout root with relative `experiments/ism/...` paths,
  and tests path-load files under `experimental/`. Moving directories means
  updating all of them together.
- Resume accepts an old output whose row count matches, without confirming it
  came from the same checkpoint, manifest, track order, or pseudocount
  (`gene_runner.py` resume path, `shards.py` `PROVENANCE_KEYS`). Add a run
  fingerprint before relying on resume again.
- `compare_checkpoints.py` uses a plain inner merge and has no guards for
  empty or constant inputs.
- There are two motif-family label sets (`tools/genomics.py` and
  `tools/mdk/analyze_mdk_motif_disruption.py`). They encode different
  historical conventions; don't merge them.

## Source reports (2026-10-13, Raft agents, read-only)

- Archive inventory (task #8): scout workspace on bandoneon,
  `~/.raft/agents/a94aa805-…/reports/task-8/` (directory, code, and reference
  inventories, plus a proposed directory map).
- Code audit since 2026-07-01 (task #9):
  `/work2/Users/qijun/grelu-wt/builder-9/runs/audit-20261013/REPORT.md`.
- bagpipe workspace inventory (task #10):
  `/work2/Users/qijun/grelu-wt/runner-10/runs/inventory-20261013/REPORT.md`.
