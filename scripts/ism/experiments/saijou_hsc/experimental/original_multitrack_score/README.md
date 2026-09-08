# Original-model multitrack score experiment

This directory is an unpromoted Saijou experiment.  It reuses the validated
three-gene strict-shuffle manifest and compares a label-free score developed on
original AlphaGenome with an unchanged transfer to original Borzoi.

## Responsibility boundaries

1. `prepare_three_gene_full_scan.py` selects the complete `Mdk`, `Col1a1`
   and `Acta2` 3-kb manifests.  It is the only preparation entry point.
2. `analyze_group_balanced_consensus.py`,
   `analyze_spatial_group_support.py` and
   `analyze_modality_view_score.py` preserve the progressive AlphaGenome
   exploration with names that describe each calculation.
3. `freeze_alphagenome_score_contract.py` compares the stages, freezes the
   modality-view score and records why it was selected.
4. `analyze_borzoi_score_transfer.py` applies that contract without refitting.
5. `prepare_score_region_motif_annotation.py` stages either model's regions
   for the canonical motif annotator.
6. `build_original_multitrack_score_report.py` reads validated tables and
   renders the report; it does not recompute scores from raw model output.

The non-CLI modules have one responsibility each:

- `experiment_config.py`: experiment paths, AlphaGenome run names and benchmark
  intervals.
- `profile_features.py`: validated track-manifest and saved-profile readers.
- `score_calculations.py`: path-free spatial, window, region and modality-view
  calculations.

The earlier sparse 40-bp benchmark preparer was removed after the experiment
switched to the complete three-gene 3-kb manifest.  Its stopped partial run is
not an input to any analysis here.

## Metric boundary

- `importance_score > 95` means the within-gene top five percent for the
  relevant method; it is not a probability, p-value or FDR.
- Formal control recall uses the percentile of a same-width window.
- Any-center and per-view recall are diagnostic only.
- Signed log2FC and native motif loss are reported separately from rank.

Promotion into the maintained workflow still requires the gates in
`../../ANALYSIS_HARNESS.md`.
