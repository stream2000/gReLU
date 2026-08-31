---
name: saijou-borzoi-lora-smoke
description: Run and review the shared Saijou four-track Borzoi LoRA smoke test; use for a bounded reproducibility check, not scientific model assessment.
---

# Saijou Borzoi LoRA smoke test

Use this skill to show that the shared code, read-only data dependencies, model weights, and `grelu_dev` runtime can complete one bounded LoRA fine-tuning run. It is an integration check, not a result suitable for biological interpretation or model comparison.

## Review first

- Read `../../docs/saijou_lora_shared_delivery.md`.
- Inspect `../../../data/README.md` and `../../../data/profiles/example_smoke_v1/manifest.json`.
- Do not change the approved example profile. A new genomic profile needs a separately approved coordinate contract.

The approved profile uses mm10 BED 0-based half-open coordinates, reference-forward reporting, and no transcript selection. It has one 524,288-bp input window each for train/validation/test; labels are the centered 196,608-bp windows, aggregated to 32-bp bins.

## Run

From the repository root, verify a free GPU with `nvidia-smi`, then run:

```bash
GPU=<free physical GPU> src/ft-scripts/run_saijou_lora_smoke.sh
```

The launcher selects a GPU automatically only when `GPU` is unset, caps the run at 180 seconds, uses exactly one epoch, and writes all generated artifacts under `../data/`. It reads the four original bigWigs and the mm10 reference through the reviewed `data/` inventory and uses the local, group-readable Borzoi weights instead of downloading model data.

## Verify

Success requires exit status zero plus all of the following:

- `../data/profiles/example_smoke_v1/manifest.json` matches the approved contract.
- `../data/caches/` contains the 32-bp mmap labels for the profile.
- `../data/runs/borzoi_example_smoke_v1_log1p_mse_lora/` contains CSV metrics and a checkpoint.
- The launcher output reports LoRA targets, finite input/label/prediction shapes, and a best or last checkpoint path.

If the 180-second cap is reached, treat it as an unsuccessful integration check; retain logs, do not silently enlarge the profile or time budget, and report the limiting stage.
