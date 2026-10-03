# Saijou HSC ISM workflow index

This directory keeps only the code needed for the current biological question:
which TSS-proximal bases or motif families drive the Saijou HSC prediction,
with matched AlphaGenome/Borzoi and original/fine-tuned comparisons.

## Canonical nine-gene workflow

Run these stages in order:

1. `prepare_saijou_all_genes_10bp_scan.py` prepares the nine genes, readouts,
   and deterministic strict-composition 10-bp edits.
2. `run_saijou_targeted_ism.py` is the single inference CLI for fine-tuned and
   original AlphaGenome/Borzoi backends. `tools/finalize_saijou_sharded_run.py`
   merges per-gene shards when a run is distributed.
3. `analyze_saijou_all_genes_10bp_scan.py` scores fine-tuned results, annotates
   transcript/splice context, and selects candidates. Its maintained default is
   the HSC profile; `--analysis-profile cross_model_track` selects the generic
   fixed-readout score without creating another analysis entry point.
4. `annotate_saijou_candidate_motifs.py` scans candidate edits against motifs.
   `--annotation-profile native_loss` applies the conservative native motif-loss
   gate used by the generic score.
5. `compare_saijou_candidate_original_models.py` compares the same selected
   edits across original and fine-tuned models.
6. `analyze_mdk_shuffle_manifest_robustness.py` compares the current Mdk scan
   with the independent broad Mdk background.
7. `build_saijou_all_genes_10bp_report.py` creates summary tables and the
   portable report artifact. After canonical HTML packaging, run the same
   entry point with `--finalize-html REPORT.html` before PDF export.
8. `validate_saijou_all_genes_10bp_pipeline.py --require-original` is the
   final completeness, storage, and profile-reconstruction gate.

The top level intentionally exposes only these eight entry points plus this
index. Reusable or specialist code lives under `tools/`:

- `tools/genomics.py`: transcript/splice annotation, locus geometry, edit
  contexts, and motif-family normalization.
- `tools/candidates.py`: preregistered controls and candidate-selection policy.
- `tools/report_*.py`: report data, figures, specification, and summary-table
  helpers. Artifact assembly and print finalization stay in the report entry
  point because they are part of one delivery operation.
- `tools/build_nine_gene_browser_report.py`: thin CLI for the four cell-specific
  two-panel Browser PDFs. Its helper consumes validated analysis TSVs plus saved
  reference profiles; it never recomputes scores from raw feature parquet.
- `tools/finalize_saijou_sharded_run.py`: distributed-run assembly.
- `tools/effect_summary.py`: shared signed-effect and fixed-readout summaries
  used by both the nine-gene report and focused Mdk analysis.
- `tools/region_importance.py`: shared model/track calibration, cross-model
  conjunction score, region calling, and positive-control threshold audit.
- `tools/cell_browser_report.py`: shared observed/reference/log2FC data and
  four-cell two-panel PDF renderer.
- `tools/harness.py`: path, schema, key, finiteness, and JSON contracts.
- `tools/mdk/`: focused Mdk audit tools outside the canonical nine-gene path.

- `tools/manifests.py`: the strict-shuffle mutation manifest schema, its two
  mutation-id shapes, and the excluded-window table.
- `configs/provided_nine_gene_transcripts.tsv`: exact boss-PDF transcript/TSS
  authority, checked against the mm10 GTF by the canonical preparer.
- `configs/positive_control_registry.tsv`: registered wet/reported TF--gene
  intervals and coordinate-evidence policy.

Neither sequence editing nor the manifest schema is per-preparer code. The three
strict-shuffle preparers — `prepare_saijou_all_genes_10bp_scan.py`,
`tools/mdk/prepare_mdk_specificity_background.py` and
`tools/mdk/prepare_saijou_targeted_loci.py` — take scan geometry and
replacements from `grelu.interpret.ism.mutations` (`AnchoredScan`,
`anchored_scan_centers`, `scan_anchored_strict_shuffles`) and their 24-column
rows from `tools/manifests.py`. They own only their locus vocabulary and how
their edits are chosen. A new preparer should call both rather than re-derive
centers, re-implement the strand-aware offset mapping, re-handle homopolymer
exclusion, or restate the manifest columns.

One deliberate exception: `tools/genomics.py::add_genomic_annotation` mirrors
`AnchoredScan.genomic_center` instead of calling it, because
`analyze_saijou_all_genes_10bp_scan.py` imports that module without putting
`src/` on `sys.path` and must not gain a `grelu` dependency for one expression.

See `ANALYSIS_HARNESS.md` for the required analysis/report boundary and the
promotion rules for new experimental work.

## Wider TSS scan preset

The boss-transcript TSS +/-1500-bp run is the canonical workflow with different
parameters, not a separate workflow:

```bash
root=experiments/ism/saijou_nine_gene_tss_3kb_strict_shuffle_pdf_transcripts

python scripts/ism/experiments/saijou_hsc/prepare_saijou_all_genes_10bp_scan.py \
  --transcript-authority scripts/ism/experiments/saijou_hsc/configs/provided_nine_gene_transcripts.tsv \
  --half-window-bp 1500 \
  --out-dir "${root}/prepared"

# Run or shard both fine-tuned models with run_saijou_targeted_ism.py, then use
# tools/finalize_saijou_sharded_run.py exactly as in the canonical workflow.

python scripts/ism/experiments/saijou_hsc/analyze_saijou_all_genes_10bp_scan.py \
  --root "${root}" \
  --analysis-profile cross_model_track \
  --transcript-authority scripts/ism/experiments/saijou_hsc/configs/provided_nine_gene_transcripts.tsv

python scripts/ism/experiments/saijou_hsc/annotate_saijou_candidate_motifs.py \
  --root "${root}" \
  --annotation-profile native_loss \
  --pthresh 1e-4

python scripts/ism/experiments/saijou_hsc/tools/build_nine_gene_browser_report.py \
  --root "${root}" \
  --half-window-bp 1500
```

Only the two configuration TSVs are preset-specific. Inference, shard
finalization, analysis, motif annotation, and report entry points are shared.

## Retained Mdk audit workflow

The focused Mdk robustness analysis remains supported because Mdk is the
primary biological target and the nine-gene report consumes its independent
background evidence:

- `tools/mdk/prepare_saijou_targeted_loci.py`
- `tools/mdk/prepare_mdk_specificity_background.py`
- `tools/mdk/analyze_saijou_targeted_original_comparison.py`
- `tools/mdk/analyze_mdk_cell_specificity_deep_dive.py`
- `tools/mdk/analyze_mdk_motif_disruption.py`
- `tools/mdk/build_mdk_cell_specificity_report.py`
- `tools/mdk/analyze_mdk_audited_effects.py`: raw feature artifacts to audited
  TSV/JSON tables for the focused single-gene discussion.
- `tools/mdk/build_mdk_audited_report.py`: report-only renderer; it reads the
  audited tables and external templates under `tools/mdk/templates/`.

These scripts may read the earlier
`experiments/ism/saijou_targeted_original_comparison/` results, but inference
still goes through the canonical `run_saijou_targeted_ism.py` CLI.

## Experimental work

Unvalidated investigations live under `experimental/<topic>/` and are not
part of the maintained baseline. Nothing in the canonical or retained Mdk
workflows may import from `experimental/`. Promotion requires a declared metric
contract, schema/finite/key checks, synthetic tests, saved-artifact validation,
and an update to this index.

## Ignored archive

The local `archive/` directory is intentionally ignored by Git. It contains
superseded implementations retained only for forensic recovery. Nothing in the
canonical or Mdk workflows may import from it.

| Archived file | Former purpose | Current replacement |
|---|---|---|
| `run_saijou_hsc_ag_ism.py` | Monolithic one-model SNV/early shuffle runner | `run_saijou_targeted_ism.py` plus model adapters and core runner |
| `run_saijou_hsc_borzoi_ism.py` | Eight-line Borzoi wrapper around the monolith | `run_saijou_targeted_ism.py --model borzoi_finetuned` |
| `rebuild_saijou_hsc_ag_ism_combined.py` | Rebuild early per-gene combined tables | canonical combined artifacts and `tools/finalize_saijou_sharded_run.py` |
| `compare_saijou_hsc_ism_models.py` | Early three-gene AG/Borzoi comparison | nine-gene candidate analysis and original-model comparison |
| `analyze_known_motif_recovery.py` | Early Acta2/Col1a1 motif-control analysis | `annotate_saijou_candidate_motifs.py` and report control table |
| `build_saijou_hsc_ism_pdf_report.py` | First broad-region PDF builder | canonical nine-gene portable report |
| `build_saijou_hsc_ism_audit_pdf.py` | One-off audit PDF builder | `docs/saijou_ism_engineering.md` and pipeline validator |
| `build_saijou_targeted_report.py` | Earlier six-locus original-vs-fine report | nine-gene report plus retained Mdk-specific report |
| `random_base_replacement/` | Completed composition-breaking replacement prototype and one-off reports | archived until a second use justifies a core random-replacement generator and canonical analysis profile |

To inspect an archived implementation, read it in place. Restore it to the
active directory only if a current workflow needs behavior that is not already
covered by the canonical core and its regression tests.
