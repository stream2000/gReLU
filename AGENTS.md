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

`src/grelu/interpret/ism/` now exists here: the saturation ISM core and the
Saijou HSC plugin have landed. The MREG plugin and the DIC work have not, so do
not assume the MREG example runners exist until that merge/cherry-pick is done.
See `docs/saturation_ism_framework_design.md` for the intended layering.

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

## Report Artifacts

Keep canonical report sources, analysis tables, figures, and the primary PDF in
their project or experiment output directory. When a final deliverable includes
a PDF, also copy that PDF to `/home/fqijun/report` (`~/report`) with a clear,
descriptive filename so it is easy to find. Do not move or replace the canonical
project artifact when making this convenience copy.

## Saijou Analysis Harness

For Saijou HSC ISM work, read
`scripts/ism/experiments/saijou_hsc/WORKFLOW_INDEX.md` and
`scripts/ism/experiments/saijou_hsc/ANALYSIS_HARNESS.md` before adding or
moving analysis code. The maintained research baseline is the canonical
nine-gene workflow plus the focused audited Mdk workflow.

Keep model inference, reusable analysis, versioned analysis tables, and report
rendering as separate layers. Report renderers must consume the validated
TSV/JSON analysis interface rather than recomputing from raw parquet. Keep HTML
and CSS in external templates rather than embedding large documents in Python.

Unvalidated investigations belong under
`scripts/ism/experiments/saijou_hsc/experimental/<topic>/`. They are research
progress, not canonical evidence, and canonical or retained workflows must not
import them. Promote code from `experimental/` only after its metric contract,
schema checks, synthetic tests, and saved-artifact validation are documented.

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
