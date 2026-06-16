# ISM Scripts

This directory contains the runnable scripts for the current local
AlphaGenome/ISM work. The formal LC-DIC motif batch experiment is the current
maintained path. Older exploratory material is kept under ignored `docs/` or
`legacy/` directories and should not be treated as the active workflow.

For the full architecture, data contracts, and extension boundaries, read:

```text
experiment/code_design.md
```

For the formal LC-DIC motif run commands, parameters, shard assignments and
artifact boundary, read:

```text
experiment/lc_dic_motif_runbook.md
```

For the existing HC-DIC report inventory and rerun notes, read:

```text
experiment/hc_dic_runbook.md
```

## Current LC-DIC motif batch pipeline

These scripts implement the completed formal experiment.

| File | Current role |
|---|---|
| `scan_lc_dic_motifs.py` | Strictly scans literature-guided TF motifs around audited LC-DIC Pol2 summits. It filters to real Pol2 peaks, keeps motifs inside the same peak and summit +/-100 bp, and writes motif candidate tables. |
| `make_lc_motif_batch_inputs.py` | Converts strict motif candidates into generic batch `sites.tsv`, `tracks.tsv`, and optional TSS readout tables. It is the LC-DIC motif-specific adapter before generic mutation preparation. |
| `make_dic_tss_readouts.py` | Maps intragenic DIC sites to a containing-gene proxy TSS and builds 4 kb named readouts when the TSS window fits inside the AlphaGenome context. |
| `prepare_batch_ism.py` | Thin CLI over `grelu.interpret.ism.batch.prepare_batch_manifests`. It validates sites/tracks and writes prepared site, mutation, readout, track, and QC manifests. |
| `run_batch_1d_features.py` | Runs resumable AlphaGenome REF/ALT 1D inference from prepared manifests, reduces predictions to bounded 128 bp ChIP summaries, and writes matched-control-adjustable feature tables. |
| `analyze_batch_population.py` | Generic entry point for grouped batch-ISM population analysis. It currently delegates to `analyze_real_ctcf_group_features.py`. |
| `analyze_real_ctcf_group_features.py` | Loads real-checkpoint feature shards, validates provenance, builds feature representations, checks shard confounding, and writes clustering reports. Despite the historical name, it is used by the current LC-DIC motif batch analysis. |
| `analyze_ctcf_population_clustering.py` | Shared clustering/statistics implementation used by `analyze_real_ctcf_group_features.py`, including PCA, stability, ARI comparisons, and optional contact-summary merging. |
| `analyze_lc_motif_effects.py` | LC-DIC motif-specific statistical consumer. It summarizes local and host-TSS effects by motif family, tests family differences, computes local-to-TSS correlations, performs response-cluster enrichment, and ranks candidates. |
| `build_lc_motif_pdf_report.py` | Builds the static chart assets and HTML/PDF-ready report for the formal LC-DIC motif batch experiment. |

## Existing HC-DIC population run

The HC-DIC population analysis is an existing local-control experiment, not the
current formal LC motif experiment. It uses the same generic batch
infrastructure from prepared manifests:

| File | HC-DIC role |
|---|---|
| `prepare_batch_ism.py` | Recreates `data/DICs/batch_ism_prepared/` from existing `data/DICs/batch_ism_inputs/sites.tsv` and `tracks.tsv`. HC sites use `ctcf_pwm_disruption`. |
| `run_batch_1d_features.py` | Reruns HC-only AlphaGenome 1D inference with `--site-class hc_dic` and deterministic site shards. |
| `analyze_batch_population.py` | Reruns HC-only clustering/statistics with `--site-class hc_dic`. |
| `analyze_real_ctcf_group_features.py` | Produces the existing `outputs/dic_batch_analysis_hc/report.html` clustering report and shard-confounding checks. |
| `analyze_ctcf_population_clustering.py` | Shared clustering/statistics implementation used by the HC report. |

The active committed scripts support rerunning from existing batch input or
prepared manifests. Regenerating `sites.tsv` directly from raw
`HC-DIC.modified.bed` would require adding a small adapter script; keep that
separate from the generic runner.

## Existing single-site helpers

These scripts predate the batch motif pipeline but are still tracked because
they support the MREG/single-locus workflow.

| File or directory | Current role |
|---|---|
| `run_tf_context.py` | Single-site AlphaGenome/TF-context runner used by earlier MREG and CTCF experiments. |
| `summarize_center_track_effects.py` | Utility for summarizing centered track effects from earlier single-site or small-run outputs. |
| `examples/mreg/` | MREG single-locus examples and plotting/analysis helpers. Use this for the validated TSS/HC-DIC/LC-DIC MREG case, not for the formal LC-DIC motif population run. |

## Ignored historical material

The following paths are intentionally ignored by `.gitignore`:

```text
scripts/ism/docs/
scripts/ism/legacy/
```

They contain historical v1.2 notes and old exploratory scripts. Do not revive
them into the active workflow unless a future task explicitly asks for that
older path.

## Removed from the active top level

The old top-level CTCF/MVP/contact helper scripts were removed from the
working tree for the current commit because they are not part of the formal
LC-DIC motif 1D experiment described in `experiment/code_design.md`. Contact
maps are the next experiment and should be implemented as a separate pass that
reuses the prepared mutation manifests.
