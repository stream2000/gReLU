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
   transcript/splice context, and selects two candidates per gene.
4. `annotate_saijou_candidate_motifs.py` scans candidate edits against motifs.
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
- `tools/finalize_saijou_sharded_run.py`: distributed-run assembly.
- `tools/mdk/`: focused Mdk audit tools outside the canonical nine-gene path.

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

These scripts may read the earlier
`experiments/ism/saijou_targeted_original_comparison/` results, but inference
still goes through the canonical `run_saijou_targeted_ism.py` CLI.

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

To inspect an archived implementation, read it in place. Restore it to the
active directory only if a current workflow needs behavior that is not already
covered by the canonical core and its regression tests.
