# MREG Examples

## Directory overview

| Script | Purpose | Status |
|---|---|---|
| `make_mreg_ctcf_example.py` | Legacy MVP: create a single CTCF motif perturbation near MREG TSS | Preserved |
| `run_mreg_ctcf_mvp.sh` | Legacy MVP runner script | Preserved |
| `run_tf_context_multires_center.py` | Legacy runner: multi-resolution centered track extraction | Preserved |
| `plot_mreg_ctcf_multires_topology.py` | Legacy plot: topology views for multires output | Preserved |
| `run_mreg_ctcf_contact_multiscale.py` | Legacy runner: contact-map multi-scale analysis | Preserved |
| `plot_mreg_ctcf_contact_multiscale.py` | Legacy plot: contact-map views | Preserved |
| **`prepare_mreg_three_region_pilot.py`** | Phase 0–1 manifest preparation and DIC-candidate validation | Fixed; DIC candidates still need review |
| **`run_mreg_three_region_chromatin.py`** | Phase 2–3 AlphaGenome 128 bp ChIP inference | Coordinate-frame bug fixed; not rerun |
| **`analyze_mreg_three_region_chromatin.py`** | Phase 4 metrics and response matrices | Current legacy run restricted to TSS positive control |
| **`plot_mreg_three_region_chromatin.py`** | Phase 4 visualization | Current legacy run emits TSS-only figure |

## New three-region pilot (2026-06-10)

The teacher's latest question asks whether AlphaGenome can detect differential
responses when mutations are placed at MREG TSS, HC-DIC, and LC-DIC regions.
The current run has only completed the TSS technical positive control. DIC
local effects, distal TSS effects, matched-control adjustment, and three-region
comparison are not complete. The intended workflow follows the
[exploration plan](../../../../agent-doc/plan&reports/ctcf_alphagenome/20260610_1543_mreg_tss_hc_lc_dic_ism_exploration_plan.md).

The original cross-region outputs are archived under
`invalidated/legacy_coordinate_frame/` and must not be cited.

### Key design decisions

1. **128 bp ChIP only** — no 1 bp, no RNA-seq, no contact maps (Phase 1 scope).
2. **TSS + HC-DIC use CTCF motif disruption**; LC-DIC uses Pol2 peak-center perturbation.
3. **Named readout regions** — not just mutation-centered windows.
4. **Per-track REF + ALT + DELTA** — no cross-track averaging before visualization.
5. **Long-form parquet output** — no full 1 Mb × all-tracks persistence.

### Rerun commands

These commands describe the corrected workflow. They have not been used to
replace the current legacy model run.

```bash
cd /home/fqijun/python/gReLU
source activate.sh

# Phase 0–1: Prepare manifests (requires curated regions input)
python scripts/ism/examples/mreg/prepare_mreg_three_region_pilot.py \
  --regions agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/inputs/mreg_curated_regions.tsv \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --gtf /home/fqijun/.local/share/genomes/hg38/hg38.annotation.gtf \
  --output-dir agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/prepared

# Phase 2: Smoke test (single TSS mutation)
python scripts/ism/examples/mreg/run_mreg_three_region_chromatin.py \
  --mutations agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/prepared/mreg_mutation_manifest.tsv \
  --readout-regions agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/prepared/mreg_readout_regions.tsv \
  --track-registry agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/prepared/mreg_track_registry.tsv \
  --mutation-id mreg_tss_00_ctcf_snv \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --device 0 \
  --output-dir agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/run_tss_smoke

# Phase 3 is blocked until HC-DIC and LC-DIC coordinates are confirmed.

# Re-analyze the current legacy run; this intentionally emits TSS-only results.
python scripts/ism/examples/mreg/analyze_mreg_three_region_chromatin.py \
  --run-dir agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/run \
  --output-dir agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/analysis

python scripts/ism/examples/mreg/plot_mreg_three_region_chromatin.py \
  --analysis-dir agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/analysis \
  --run-dir agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/run \
  --prepared-dir agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/prepared \
  --output-dir agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/figures
```

### DIC identification algorithm

The paper algorithm for identifying HC-DIC and LC-DIC sites is documented in:
`agent-doc/paper/nakato2022_dic_identification_algorithm.md`

### Curated regions input format

The `--regions` input must be a TSV with at minimum:

```text
region_role	chrom	start	end	summit	strand	source	review_status
tss	chr2	216013551	216013751	216013551	-	local_gtf	confirmed
hc_dic	chr2	215950000	215960000	215955000	.	nakato_supplement	confirmed
lc_dic	chr2	216020000	216030000	216025000	.	nakato_supplement	confirmed
```

`region_role` must be one of: `tss`, `hc_dic`, `lc_dic`. DIC rows are skipped
unless `review_status` is `confirmed` and their coordinates satisfy the gene-body
and gene-end-distance checks.

### Output directory structure

```text
agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/
├── inputs/
│   └── mreg_curated_regions.tsv
├── prepared/
│   ├── mreg_region_manifest.tsv
│   ├── mreg_mutation_manifest.tsv
│   ├── mreg_readout_regions.tsv
│   ├── mreg_track_registry.tsv
│   ├── sequence_contexts.tsv
│   └── preparation_qc.json
├── run/
│   ├── mreg_chromatin_profiles.parquet
│   ├── scored_intervals.tsv
│   ├── resolved_tracks.tsv
│   ├── run_config.json
│   ├── timing.tsv
│   └── provenance.json
	├── analysis/
	│   ├── validation_status.json
	│   ├── mreg_chromatin_metrics.tsv
	│   ├── mreg_control_adjusted_metrics.tsv  # header only in current run
	│   ├── mreg_response_matrix.tsv           # header only in current run
	│   ├── result_summary.md
	│   └── experimental_report.md
	├── invalidated/
	│   └── legacy_coordinate_frame/           # audit only; do not cite
	└── figures/
	    └── mreg_tss_positive_control_ref_alt_delta.png
```

### Historical examples

The older scripts (`make_mreg_ctcf_example.py`, `run_tf_context_multires_center.py`,
`plot_mreg_ctcf_multires_topology.py`, and contact-map variants) are preserved for
reference but are **outside** the current three-region pilot workflow. Their paths
and output assumptions remain those of the original examples.

### Not in scope (this phase)

- 1 bp resolution output
- RNA-seq, ATAC/DNase/CAGE heads
- Contact maps
- Population-level multi-locus analysis
- Full track clustering
