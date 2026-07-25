# gReLU Replication Agent Guide

Last updated: 2026-07-25.

This is the concise operating guide for the shared checkout:

```text
/work2/Users/qijun/gReLU-replication
```

`/home/fqijun/python/gReLU-replication` may be a symlink to the same checkout.
Verify the live branch and worktree instead of relying on an older branch
description.

## Start here

```bash
cd /work2/Users/qijun/gReLU-replication
git status --short
source activate.sh
python - <<'PY'
import grelu
print(grelu.__file__)
PY
```

The expected import is:

```text
/work2/Users/qijun/gReLU-replication/src/grelu
```

`activate.sh` is the single environment entry point. It defaults to the
user-home `grelu_dev` environment and prepends this checkout's `src/` to
`PYTHONPATH`. Use its supported host overrides instead of duplicating conda
activation in launchers.

## Repository scope

- AlphaGenome integration lives under `src/grelu/model/trunks/alphagenome.py`
  and `src/alphagenome_pytorch/`.
- The reusable saturation ISM core lives under `src/grelu/interpret/ism/`.
- Saijou model-case workflows live under
  `scripts/ism/experiments/saijou_hsc/`.
- Do not assume MREG or DIC workflows exist in this checkout; inspect before
  referring to code from another worktree or branch.
- Existing dirty-worktree changes belong to the user. Preserve them and stage
  only files that belong to the current task.

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

The purpose of this gate is to prevent silent coordinate assumptions, not to
trigger broad or repetitive coordinate checking.

## Saijou workflow rules

Read these before adding or moving Saijou analysis code:

```text
scripts/ism/experiments/saijou_hsc/WORKFLOW_INDEX.md
scripts/ism/experiments/saijou_hsc/ANALYSIS_HARNESS.md
```

The maintained baseline is the canonical nine-gene workflow plus the focused
audited Mdk workflow.

- Treat scan width, transcript authority, ranking profile, motif threshold, and
  report layout as parameters or versioned configuration, not new parallel
  workflows.
- Keep inference, reusable analysis, validated TSV/JSON interfaces, and report
  rendering as separate layers.
- Report renderers consume validated analysis outputs; they do not recompute
  scientific results from raw model files.
- New unvalidated investigations belong under `experimental/<topic>/`.
  Promote reusable logic only after its metric contract, schema checks,
  synthetic tests, and saved-artifact validation are documented.

See `docs/saturation_ism_framework_design.md` for the framework layering.

## Progress and artifacts

`experiments/Progress.md` is the current-state handoff and rolling two-month
history. Keep the most recent fourteen calendar days at the established detail
level, then retain days 15 through 60 as compact weekly summaries. Detailed
entries record the branch/commit, command, checkpoint, output, validation,
caveat, and next action when relevant; prefer a few meaningful entries per day
over batch-level noise.

Run Progress compaction once every seven days. Preserve detail that has aged
beyond fourteen days under `experiments/archive/progress/`, update the weekly
summaries, and record the compaction time, covered range, archive path,
detailed cutoff, and next scheduled compaction in the active file. If nothing
aged out, record a no-op compaction instead of rewriting history. Do not delete
history less than two months old, prematurely compress the two-week detailed
window, or impose a fixed line limit. Deletion or further compression beyond
two months requires explicit user approval.

Canonical analysis tables, figures, and PDFs stay in their experiment output
directory. For a final PDF deliverable, also place a clearly named convenience
copy in `/home/fqijun/report` without replacing the canonical artifact.

## Verification and commits

Use validation proportional to the change. The basic AlphaGenome check is:

```bash
source activate.sh
python -m pytest -q tests/test_alphagenome_inference.py
```

For Saijou changes, run the focused tests listed by `WORKFLOW_INDEX.md` and
validate saved TSV/JSON interfaces. Always run `git diff --check` before a
commit.

Commit only the intended scope. Do not include generated experiment outputs,
cache files, unrelated user changes, or local Progress archives.

## Documentation map

- `SETUP_GUIDE.md`: AlphaGenome environment and inference setup.
- `experiments/RUNBOOK.md`: Borzoi Saijou fine-tuning.
- `scripts/README.md`: benchmark/script overview.
- `scripts/ism/experiments/saijou_hsc/WORKFLOW_INDEX.md`: Saijou stage order.
- `scripts/ism/experiments/saijou_hsc/ANALYSIS_HARNESS.md`: analysis contracts.
- `experiments/Progress.md`: current validated state and open decisions.
