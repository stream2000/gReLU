# Shared Saijou Borzoi LoRA delivery

This branch is the DL-pillar shareable snapshot of the Saijou four-cell-type Borzoi fine-tuning workflow. The source code is in `src/ft-scripts/`; the reviewed one-epoch integration entry point is `src/ft-scripts/run_saijou_lora_smoke.sh`.

## Runtime and data boundary

The project root contains two deliberately separate shared resources:

- `runtime/grelu_dev`: read-only symbolic link to the existing `grelu_dev` Conda environment. The launcher sets `GRELU_CONDA_ENV` before sourcing `activate.sh`, so the shared checkout's `src/` is imported while the developer environment is not modified.
- `data/`: a single reviewable inventory of all runtime data dependencies and generated smoke artifacts. The four original bigWigs and mm10 reference files are linked rather than duplicated. The Borzoi state dict is copied here with group-read permission because the original cache is private to one user.

All users need read and execute access through the `nakatolab` group. They should not write to `runtime/` or the linked source data. Caches, profile files, logs, checkpoints, and metrics are created under `data/`.

## Approved smoke profile

`data/profiles/example_smoke_v1/` contains one train, validation, and test input interval. It is intentionally too small for any scientific conclusion. Its immutable coordinate contract is stored in `manifest.json` and includes assembly, coordinate convention, orientation, label/input windows, source authority, and exact intervals.

The run uses one epoch, one physical GPU selected at runtime, zero DataLoader workers, batch size one, and a hard 180-second cap. It uses LoRA only; no full-parameter fine-tuning path is exposed by this launcher.

## Review checklist

1. Confirm `data/README.md` lists four bigWigs, mm10 reference files, the Borzoi state dict, and the profile.
2. Read `skills/saijou-borzoi-lora-smoke/SKILL.md` for execution and acceptance criteria.
3. Run the launcher and inspect its generated cache manifest, CSV metrics, and checkpoint under `data/`.
4. Treat a timeout, missing dependency, non-finite tensor, or absent checkpoint as a failed smoke test rather than changing the profile silently.
