# Saijou targeted-ISM engineering structure

This note records the maintained code boundaries for the Saijou four-model
workflow. Scientific output formats and CLI flags are treated as compatibility
contracts.

## Layering

- `src/grelu/interpret/ism/mutations.py`: mutation manifest generation and
  reference-validated equal-length edits.
- `src/grelu/interpret/ism/readouts.py`: genomic readout-to-bin mapping and
  REF/ALT summary metrics.
- `src/grelu/interpret/ism/model_adapters/`: model loading and normalized
  `[sequence, track, bin]` predictions. Shared one-hot encoding and public track
  selection live in `utils.py`.
- `src/grelu/interpret/ism/profiles.py`: profile aggregation, streaming float16
  log2FC storage, profile index/metadata, loading, and ALT reconstruction.
- `src/grelu/interpret/ism/validation.py`: file provenance, expected-row
  geometry, finite-value checks, and feature-key uniqueness.
- `src/grelu/interpret/ism/gene_runner.py`: one-gene reference prediction,
  mutation batching, feature extraction, and streaming profile writes.
- `src/grelu/interpret/ism/runner.py`: run-level selection, geometry, and
  artifact combination. It does not own model setup or shard merging.
- `src/grelu/interpret/ism/shards.py`: provenance-checked shard selection and
  atomic per-gene/final artifact assembly.
- `scripts/ism/experiments/saijou_hsc/run_saijou_targeted_ism.py`: CLI and the
  four Saijou model choices only. It should not acquire model-independent
  profile or validation logic.
- `scripts/ism/experiments/saijou_hsc/analyze_saijou_all_genes_10bp_scan.py`:
  experiment orchestration. Transcript/splice annotation lives in
  `tools/genomics.py`; the preregistered control and candidate policy lives in
  `tools/candidates.py`.
- `scripts/ism/experiments/saijou_hsc/tools/report_*.py`: report input loading,
  summary validation, static figures, and declarative narrative specification.
- `scripts/ism/experiments/saijou_hsc/build_saijou_all_genes_10bp_report.py`:
  summary-table generation, artifact assembly, and `--finalize-html` print
  pagination. These stay together because they form one report operation.
- `scripts/ism/experiments/saijou_hsc/tools/mdk/`: specialist Mdk background,
  motif, specificity, and focused-report utilities outside the canonical path.

## Stable artifact contracts

Each canonical run keeps:

- `track_manifest.tsv` and `model_metadata.json`;
- per-gene and combined feature tables;
- per-gene native REF profiles;
- 128-bp mutation log2FC profiles, profile indices, and metadata;
- `sequence_contexts.tsv` and `validation_summary.json`.

The profile loader and validators consume these files by name. Changing names,
track order, bin aggregation, pseudocount semantics, or feature keys requires a
format migration and a regression comparison against existing canonical runs.

## Engineering decisions

- Scientific validation is not treated as optional defensive programming.
  Deterministic REF checks, finite values, complete row counts, profile index
  order, splice exclusion, and ALT reconstruction remain hard gates.
- Model-specific checkpoint reconstruction stays inside adapters; the generic
  runner does not branch on AlphaGenome versus Borzoi.
- Candidate-selection policy remains experiment-specific because its fixed
  controls and Mdk ranges are biological preregistration, not general ISM
  infrastructure.
- Report narrative remains separate from figure rendering inside the report
  tools package. Static plotting is reusable, while biological interpretation
  is intentionally explicit rather than hidden in a generic framework.
- Historical report builders are retained when they reproduce a distinct
  delivered analysis. Duplicate helpers are removed, but old scientific
  deliverables are not deleted merely because a newer report exists.
- Superseded implementations are moved to the Git-ignored local `archive/`;
  the tracked `WORKFLOW_INDEX.md` records each replacement. Active code may not
  import from the archive.

## Regression checks

```bash
source activate.sh
python -m pytest -q \
  tests/test_saturation_ism_core.py \
  tests/test_ism_model_adapter_utils.py \
  tests/test_ism_validation.py \
  tests/test_ism_shards.py \
  tests/test_saijou_targeted_profile_storage.py \
  tests/test_saijou_targeted_runner.py \
  tests/test_saijou_all_genes_hotspot_analysis.py

python scripts/ism/experiments/saijou_hsc/validate_saijou_all_genes_10bp_pipeline.py \
  --root experiments/ism/saijou_all_genes_10bp_scan \
  --require-original
```

For candidate-selection refactors, compare `candidate_segments.tsv` and
`candidate_centers.tsv` cell-for-cell against the pre-refactor outputs. For
profile storage refactors, compare reconstruction maxima for all four canonical
runs, not only synthetic unit tests.

## Canonical entry point

`scripts/ism/experiments/saijou_hsc/WORKFLOW_INDEX.md` is the short operating
index. It identifies the eight maintained stages, the distinct Mdk audit flow,
and the ignored archive. New experiment scripts should not duplicate runner,
profile, annotation, motif-context, or report-assembly helpers already listed
there.
