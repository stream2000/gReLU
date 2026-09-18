# gReLU Replication Agent Guide

Last updated: 2026-09-18.

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
- Inspect the live tree before referring to code from another worktree or
  branch.
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

`docs/saturation_ism_framework_design.md` records framework and migration
rationale. Use the live workflow index, tests, and code as implementation
authority.

## Progress and artifacts

`experiments/Progress.md` is the status-sensitive handoff, not required reading
for every task. Read it before reporting current project state or changing an
active experiment. Record only material, validated results and link detailed
experiment-local summaries. Its own `History and update rule` section is the
authority for retention and compaction; do not turn unrelated tasks into
Progress maintenance.

Canonical analysis tables, figures, and PDFs stay in their experiment output
directory. For a final PDF deliverable, also place a clearly named convenience
copy in `/home/fqijun/report` without replacing the canonical artifact.

## Verification and commits

Use validation proportional to the change. Run focused tests for the touched
entry points; for Saijou changes, also validate saved TSV/JSON interfaces.
Always run `git diff --check` before a commit.

Commit only the intended scope. Do not include generated experiment outputs,
cache files, unrelated user changes, or local Progress archives.

## Task-specific documentation

Load only the branch relevant to the task; do not preload this entire map.

- `SETUP_GUIDE.md`: AlphaGenome environment and inference setup.
- `experiments/RUNBOOK.md`: Saijou fine-tuning and NTv3 MVP operations.
- `scripts/README.md`: benchmark/script overview.
- `docs/project_scientific_question_audit_zh.md`:
  historical question/scope audit with an evidence cutoff of 2026-07-28; do
  not use it as the current evidence summary.
