# gReLU Replication Agent Guide

Last updated: 2026-10-15.

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
- The closed Saijou model case is frozen under
  `scripts/ism/experiments/saijou_hsc/` (see below).
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

## Saijou (closed 2026-10-13)

The Saijou HSC case and WT1 follow-up are closed. Their code is frozen under
`scripts/ism/experiments/saijou_hsc/`; their run artifacts live on `/work3`,
and `experiments/ism/<name>` symlinks point there. Start from
`docs/history/saijou_20261013/HISTORY_INDEX.md`. Don't add new work to these
workflows; build new phases on the generic core in `src/grelu/interpret/ism/`.

The layering rules from that case still apply to new analysis code:

- Treat scan width, ranking profile, thresholds, and report layout as
  parameters or versioned configuration, not new parallel workflows.
- Keep inference, reusable analysis, validated TSV/JSON interfaces, and report
  rendering as separate layers. Renderers consume validated outputs; they do
  not recompute results from raw model files.
- Unvalidated investigations start in an `experimental/<topic>/` directory and
  move into core only with a metric contract, schema checks, synthetic tests,
  and saved-artifact validation.

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
directory. Large inputs, checkpoints, and closed runs belong on `/work3`
(`/work2` is shared and nearly full). For a final PDF deliverable, also place a clearly named convenience
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
