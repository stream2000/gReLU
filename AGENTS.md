# gReLU Replication Agent Guide

Last updated: 2026-06-29.

This checkout is the user's fork. The active shared checkout lives at:

```text
/work2/Users/qijun/gReLU-replication
```

`/home/fqijun/python/gReLU-replication` is expected to be a symlink to that
shared path on configured hosts.

It is currently on the `alphagenome` branch. This branch contains the
AlphaGenome integration code, including:

```text
src/grelu/model/trunks/alphagenome.py
src/alphagenome_pytorch/
tests/test_alphagenome_inference.py
```

The separate `ism` branch has additional ISM/MREG/DIC work that has not been
merged here yet. Do not assume `src/grelu/interpret/ism/` or the MREG ISM
example runners exist in this checkout until that merge/cherry-pick is done.

## Environment

Code and environment are intentionally separate concerns. Prefer the shared
checkout path for code, but choose the conda environment per host and workload.
Shared conda environments can be slow for Python startup/import because imports
touch many small package and metadata files on shared storage.

Default interactive setup:

```bash
cd /work2/Users/qijun/gReLU-replication
source activate.sh
```

`activate.sh` activates a known working environment and prepends this checkout's
`src/` to `PYTHONPATH`, so Python imports should resolve to:

```text
/work2/Users/qijun/gReLU-replication/src/grelu
```

For long training runs or plotting scripts on hosts where the shared env starts
slowly, it is acceptable and often preferable to use the host-local env while
keeping the shared repo on `PYTHONPATH`:

```bash
cd /work2/Users/qijun/gReLU-replication
source /work/miniconda3/etc/profile.d/conda.sh
conda activate grelu_dev
export PYTHONPATH=/work2/Users/qijun/gReLU-replication/src:${PYTHONPATH:-}
```

Do not assume the shared `/work/gReLU/env` is always the best runtime just
because the source checkout is shared. Verify `import grelu; print(grelu.__file__)`
when switching hosts or shells.

## Basic Checks

After activation, use these commands for a quick sanity check:

```bash
python - <<'PY'
import grelu
print(grelu.__file__)
PY

python -m pytest -q tests/test_alphagenome_inference.py
```

For broader branch checks, inspect existing docs first:

```text
SETUP_GUIDE.md
scripts/README.md
scripts/eqtl/README.md
```

## Progress Notes

Maintain a short `experiments/Progress.md` in this checkout. Update it after
important usable results or branch-integration decisions. The Borzoi fine-tuning
runbook lives at `experiments/RUNBOOK.md`.

Record:

- date/time
- branch or commit being tested
- exact command
- model/checkpoint path when relevant
- important output path
- result or failure summary
- caveats and next action

Keep entries brief. If a later run corrects the same result, update the latest
relevant entry instead of adding a correction-only note.

## Current Branch Discussion

This checkout is intended to be a clean AlphaGenome-side replication workspace.
Use it to test the AlphaGenome branch separately from the original dirty ISM
worktree at:

```text
/home/fqijun/python/gReLU
```

Known branch gap as of 2026-06-25:

- `origin/alphagenome` has AlphaGenome model/trunk support.
- `origin/ism` adds the ISM framework, TF context utilities, MREG/DIC scripts,
  reports, and tests.
- the original local `ism` branch also has unpushed commits and uncommitted
  MREG/ISM changes.

Before moving ISM work into this checkout, decide whether to cherry-pick only
the AlphaGenome-relevant trunk/runner fixes or merge the full ISM branch.
