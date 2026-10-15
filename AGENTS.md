# gReLU Clustering Agent Guide

Last updated: 2026-10-15.

Checkout:

```text
/work2/Users/qijun/gReLU-clustering
```

This repository starts a new phase from the saturation ISM infrastructure of
`gReLU-replication`. It was cloned from that repository's `alphagenome` branch
(commit `8409dfe`) on 2026-10-15 and trimmed to the ISM core; Git history is
kept.

## Start here

```bash
cd /work2/Users/qijun/gReLU-clustering
git status --short
source activate.sh
python - <<'PY'
import grelu
print(grelu.__file__)
PY
```

The expected import is
`/work2/Users/qijun/gReLU-clustering/src/grelu`. `activate.sh` is the single
environment entry point: it activates the shared user-home `grelu_dev` conda
environment and prepends this checkout's `src/` and the AlphaGenome submodule
to `PYTHONPATH`. Without it, `import grelu` can resolve to another checkout.

## What is here

- `src/grelu/interpret/ism/`: generic saturation ISM runner, mutation sets,
  profiles/readouts, shards, validation, and model adapters for the original
  (pretrained) AlphaGenome and Borzoi models.
- `src/grelu/model/`: model trunks (AlphaGenome, Borzoi, NTv3, ...).
- `src/alphagenome_pytorch/`: submodule; its own `CLAUDE.md` applies there.
- `tests/`: library tests plus the ISM core tests
  (`test_saturation_ism_core.py`, `test_ism_*.py`, `test_saijou_targeted_*.py`).

There is no experiment runner script yet. New phase entry points go under
`scripts/` and build on `grelu.interpret.ism`.

## History (read-only)

The Saijou HSC / WT1 phase is closed. Its code, docs, training scripts, and
reports remain in `/work2/Users/qijun/gReLU-replication` (frozen, branch
`alphagenome`); start from `docs/history/saijou_20261013/HISTORY_INDEX.md`
there. Run artifacts are on `/work3/Users/qijun/gReLU-history/saijou-20261013/`.
The fine-tuned (LoRA) Saijou adapters and checkpoints were not carried over;
they live in that checkout (`src/grelu/interpret/ism/model_adapters/*_finetuned.py`,
`src/ft-scripts/`, `runs/`). Read from these locations; don't write into them.

## Coordinate approval gate

This is a hard review gate. Before any state-changing action that creates,
changes, transforms, scans, annotates, or plots genomic coordinates:

1. Present a concise coordinate contract to the user.
2. Include only relevant fields: species/assembly, gene and transcript,
   coordinate convention, strand/reporting orientation, anchor/window, and
   authority source.
3. State the proposed operation and wait for explicit approval before editing
   manifests, running inference, producing annotations, or rendering plots.

Read-only investigation needed to build the contract is allowed before
approval. Once a contract is approved, reuse it without repeated clarification.
Clarify again only if a contract field changes, sources conflict, or a new
coordinate interpretation is required. Pure propagation or re-export of an
already validated coordinate manifest does not require another approval.

## Layering rules

- Treat scan width, ranking profile, thresholds, and report layout as
  parameters or versioned configuration, not new parallel workflows.
- Keep inference, reusable analysis, validated TSV/JSON interfaces, and report
  rendering as separate layers. Renderers consume validated outputs; they do
  not recompute results from raw model files.
- Unvalidated investigations start in an `experimental/<topic>/` directory and
  move into core only with a metric contract, schema checks, synthetic tests,
  and saved-artifact validation.
- Don't make paths part of the API: locate inputs through arguments or config,
  not `Path(__file__).parents[n]` or checkout-relative paths.

## Known debts in the ISM core

- Resume accepts an old output whose row count matches, without confirming it
  came from the same checkpoint, manifest, track order, or pseudocount
  (`gene_runner.py` resume path, `shards.py` `PROVENANCE_KEYS`). Add a run
  fingerprint before relying on resume.

## Artifacts

Large inputs, checkpoints, and run outputs belong on `/work3` (`/work2` is
shared and nearly full); `experiments/`, `runs/`, and `dataset_store/` are
ignored by Git. For a final PDF deliverable, also place a clearly named copy in
`/home/fqijun/report`.

## Verification and commits

Use validation proportional to the change and run focused tests for the touched
entry points. Always run `git diff --check` before a commit. Commit only the
intended scope; no generated outputs or cache files. Existing dirty-worktree
changes belong to the user.

## Task-specific documentation

- `SETUP_GUIDE.md`: AlphaGenome environment and inference setup.
