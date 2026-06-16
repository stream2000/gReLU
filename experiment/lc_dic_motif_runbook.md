# LC-DIC motif batch ISM runbook

Last updated: 2026-06-16

This runbook records the commands and parameters for the completed formal
LC-DIC Pol2-anchored motif batch ISM experiment. It complements
`experiment/code_design.md`: the design document explains architecture; this
file explains how the current formal artifacts were produced and how to
reproduce them.

## 1. Scope and formal artifact boundary

The formal run is the Pol2-anchored LC-DIC motif experiment only.

Formal input/output prefixes:

```text
outputs/lc_dic_motif_scan_pol2_0.90/
data/DICs/lc_motif_pol2_batch_inputs/
data/DICs/lc_motif_pol2_batch_prepared/
outputs/lc_motif_pol2_1d_shard0/
outputs/lc_motif_pol2_1d_shard1/
outputs/lc_motif_pol2_1d_shard2/
outputs/lc_motif_pol2_1d_shard3/
outputs/lc_motif_pol2_tss_shard0/
outputs/lc_motif_pol2_tss_shard1/
outputs/lc_motif_pol2_tss_shard2/
outputs/lc_motif_pol2_tss_shard3/
outputs/lc_motif_pol2_population_analysis/
outputs/lc_motif_pol2_effect_analysis/
agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/
```

The following similarly named directories are earlier smoke tests or exploratory
runs and are not part of the formal evidence set:

```text
data/DICs/lc_motif_batch_inputs/
data/DICs/lc_motif_batch_prepared/
outputs/lc_motif_1d_shard*/
outputs/lc_motif_tss_shard*/
outputs/lc_motif_analysis/
outputs/lc_motif_effect_analysis_local_only/
outputs/lc_motif_pol2_effect_analysis_local_only/
outputs/lc_motif_smoke/
outputs/lc_tss_4kb_*/
```

Contact maps were not run in this experiment.

## 2. Environment

Run every Python command from the repository root:

```bash
cd /home/fqijun/python/gReLU
source activate.sh
```

Key local resources:

```text
FASTA:
  /home/fqijun/.local/share/genomes/hg38/hg38.fa

GTF:
  /home/fqijun/.local/share/genomes/hg38/hg38.annotation.gtf

AlphaGenome checkpoint:
  /home/fqijun/.cache/alphagenome/model_fold_0.safetensors

AlphaGenome human track metadata:
  /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet

H13CORE motif panel:
  /home/fqijun/python/gReLU/src/grelu/resources/meme/H13CORE_meme_format.meme

CTCF/JASPAR motif file passed to generic preparation:
  /home/fqijun/python/gReLU/src/grelu/resources/meme/jaspar_2024_consensus.meme
```

`prepare_batch_ism.py` requires a `--motif` argument for the generic batch
context. For this LC-DIC motif run, the actual target PWM for each site is read
from the per-row `motif_path` and `motif_name` fields in `sites.tsv`; the
generic `--motif` file is still supplied for compatibility with shared
preparation code.

## 3. Command provenance

The AlphaGenome inference commands below are confirmed by each shard's
`run_metadata.json`. The motif scan and manifest-adapter commands are
reconstructed from the generated artifact paths, table columns, and script
arguments; the outputs match the recorded formal counts.

## 4. Step-by-step reproduction commands

### 4.1 Strict Pol2-anchored motif scan

This scans only LC-DICs from the 548-site prepared DIC batch, requires a real
Pol2 peak and summit from `dic_site_audit.tsv`, and keeps motif hits within the
same Pol2 peak and summit +/-100 bp.

```bash
python scripts/ism/scan_lc_dic_motifs.py \
  --sites data/DICs/batch_ism_prepared/ready_sites.tsv \
  --audit data/DICs/dic_site_audit.tsv \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --motifs /home/fqijun/python/gReLU/src/grelu/resources/meme/H13CORE_meme_format.meme \
  --cluster-assignments outputs/dic_batch_analysis_lc/representations/real_local_1_4kb/cluster_assignments.tsv \
  --flank-bp 100 \
  --min-relative-score 0.90 \
  --output-dir outputs/lc_dic_motif_scan_pol2_0.90
```

Expected key outputs:

```text
outputs/lc_dic_motif_scan_pol2_0.90/skipped_no_pol2_peak.tsv    246 rows
outputs/lc_dic_motif_scan_pol2_0.90/site_motif_summary.tsv      168 rows
outputs/lc_dic_motif_scan_pol2_0.90/all_motif_hits.tsv          185 rows
outputs/lc_dic_motif_scan_pol2_0.90/best_motif_hits.tsv         154 rows
outputs/lc_dic_motif_scan_pol2_0.90/motif_mutation_candidates.tsv 144 rows
```

The formal candidate table has 144 site-by-motif-family candidates.

### 4.2 Host-TSS readout construction for source LC-DICs

This maps source LC-DICs to a containing-gene proxy TSS and creates a 4 kb TSS
readout when it fits inside the 1,048,576 bp AlphaGenome context.

```bash
python scripts/ism/make_dic_tss_readouts.py \
  --sites data/DICs/batch_ism_prepared/ready_sites.tsv \
  --gtf /home/fqijun/.local/share/genomes/hg38/hg38.annotation.gtf \
  --site-class lc_dic \
  --window-bp 4096 \
  --output-dir data/DICs/lc_tss_readouts
```

Key output:

```text
data/DICs/lc_tss_readouts/tss_readouts.tsv
```

### 4.3 Build LC-DIC motif batch input manifests

This converts strict motif candidates into generic batch inputs and maps each
site-by-family candidate back to the source LC-DIC TSS readout.

```bash
python scripts/ism/make_lc_motif_batch_inputs.py \
  --candidates outputs/lc_dic_motif_scan_pol2_0.90/motif_mutation_candidates.tsv \
  --source-sites data/DICs/batch_ism_prepared/ready_sites.tsv \
  --audit data/DICs/dic_site_audit.tsv \
  --tracks data/DICs/batch_ism_inputs/tracks.tsv \
  --motifs /home/fqijun/python/gReLU/src/grelu/resources/meme/H13CORE_meme_format.meme \
  --tss-readouts data/DICs/lc_tss_readouts/tss_readouts.tsv \
  --min-relative-score 0.90 \
  --output-dir data/DICs/lc_motif_pol2_batch_inputs
```

Expected outputs:

```text
data/DICs/lc_motif_pol2_batch_inputs/sites.tsv         144 rows
data/DICs/lc_motif_pol2_batch_inputs/tracks.tsv        15 rows
data/DICs/lc_motif_pol2_batch_inputs/tss_readouts.tsv  138 rows
```

The input candidate family counts are:

```text
AP1      37
Forkhead 32
GATA3    60
TEAD4    15
```

### 4.4 Prepare target and matched-control mutations

This validates the batch manifests and designs one motif-disruption target plus
one same-Pol2-peak matched control per ready candidate.

```bash
python scripts/ism/prepare_batch_ism.py \
  --sites data/DICs/lc_motif_pol2_batch_inputs/sites.tsv \
  --tracks data/DICs/lc_motif_pol2_batch_inputs/tracks.tsv \
  --readouts data/DICs/lc_motif_pol2_batch_inputs/tss_readouts.tsv \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --motif /home/fqijun/python/gReLU/src/grelu/resources/meme/jaspar_2024_consensus.meme \
  --min-relative-motif-score 0.90 \
  --motif-controls 3 \
  --sequence-controls 1 \
  --max-control-distance-bp 5000 \
  --output-dir data/DICs/lc_motif_pol2_batch_prepared
```

Expected outputs:

```text
data/DICs/lc_motif_pol2_batch_prepared/site_manifest.tsv      144 rows
data/DICs/lc_motif_pol2_batch_prepared/ready_sites.tsv        143 rows
data/DICs/lc_motif_pol2_batch_prepared/mutation_manifest.tsv  286 rows
data/DICs/lc_motif_pol2_batch_prepared/preparation_qc.tsv     144 rows
data/DICs/lc_motif_pol2_batch_prepared/track_registry.tsv     15 rows
data/DICs/lc_motif_pol2_batch_prepared/readout_manifest.tsv   146 rows
```

One GATA3 candidate, `lc_dic_108__gata3`, is skipped because a valid three-base
motif disruption could not reduce the PWM score below threshold.

Ready candidate family counts:

```text
AP1      37
Forkhead 32
GATA3    59
TEAD4    15
```

Mutation rows:

```text
experimental target: 143
matched control:     143
```

### 4.5 Local 1D AlphaGenome inference

The formal local run uses four deterministic site shards. All commands use:

```text
--prepared-dir data/DICs/lc_motif_pol2_batch_prepared
--fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa
--weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors
--track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet
--windows-bp 1024,4096,20000,100000
--summary-strategy multiscale_log2fc
--site-shard-count 4
```

Shard commands:

```bash
python scripts/ism/run_batch_1d_features.py \
  --prepared-dir data/DICs/lc_motif_pol2_batch_prepared \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet \
  --device 0 \
  --site-shard-index 0 \
  --site-shard-count 4 \
  --windows-bp 1024,4096,20000,100000 \
  --summary-strategy multiscale_log2fc \
  --output-dir outputs/lc_motif_pol2_1d_shard0

python scripts/ism/run_batch_1d_features.py \
  --prepared-dir data/DICs/lc_motif_pol2_batch_prepared \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet \
  --device 1 \
  --site-shard-index 1 \
  --site-shard-count 4 \
  --windows-bp 1024,4096,20000,100000 \
  --summary-strategy multiscale_log2fc \
  --output-dir outputs/lc_motif_pol2_1d_shard1

python scripts/ism/run_batch_1d_features.py \
  --prepared-dir data/DICs/lc_motif_pol2_batch_prepared \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet \
  --device 2 \
  --site-shard-index 2 \
  --site-shard-count 4 \
  --windows-bp 1024,4096,20000,100000 \
  --summary-strategy multiscale_log2fc \
  --output-dir outputs/lc_motif_pol2_1d_shard2

python scripts/ism/run_batch_1d_features.py \
  --prepared-dir data/DICs/lc_motif_pol2_batch_prepared \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet \
  --device 3 \
  --site-shard-index 3 \
  --site-shard-count 4 \
  --windows-bp 1024,4096,20000,100000 \
  --summary-strategy multiscale_log2fc \
  --output-dir outputs/lc_motif_pol2_1d_shard3
```

Recorded shard sizes from `run_metadata.json`:

```text
shard0 device0: 36 sites, 72 variants, 15 tracks
shard1 device1: 36 sites, 72 variants, 15 tracks
shard2 device2: 36 sites, 72 variants, 15 tracks
shard3 device3: 35 sites, 70 variants, 15 tracks
```

### 4.6 Host-TSS 1D AlphaGenome inference

This pass reuses the same prepared mutations but supplies the TSS readout table.

Shard commands:

```bash
python scripts/ism/run_batch_1d_features.py \
  --prepared-dir data/DICs/lc_motif_pol2_batch_prepared \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet \
  --readout-table data/DICs/lc_motif_pol2_batch_inputs/tss_readouts.tsv \
  --device 0 \
  --site-shard-index 0 \
  --site-shard-count 4 \
  --windows-bp 1024,4096,20000,100000 \
  --summary-strategy multiscale_log2fc \
  --output-dir outputs/lc_motif_pol2_tss_shard0

python scripts/ism/run_batch_1d_features.py \
  --prepared-dir data/DICs/lc_motif_pol2_batch_prepared \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet \
  --readout-table data/DICs/lc_motif_pol2_batch_inputs/tss_readouts.tsv \
  --device 0 \
  --site-shard-index 1 \
  --site-shard-count 4 \
  --windows-bp 1024,4096,20000,100000 \
  --summary-strategy multiscale_log2fc \
  --output-dir outputs/lc_motif_pol2_tss_shard1

python scripts/ism/run_batch_1d_features.py \
  --prepared-dir data/DICs/lc_motif_pol2_batch_prepared \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet \
  --readout-table data/DICs/lc_motif_pol2_batch_inputs/tss_readouts.tsv \
  --device 2 \
  --site-shard-index 2 \
  --site-shard-count 4 \
  --windows-bp 1024,4096,20000,100000 \
  --summary-strategy multiscale_log2fc \
  --output-dir outputs/lc_motif_pol2_tss_shard2

python scripts/ism/run_batch_1d_features.py \
  --prepared-dir data/DICs/lc_motif_pol2_batch_prepared \
  --fasta /home/fqijun/.local/share/genomes/hg38/hg38.fa \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet \
  --readout-table data/DICs/lc_motif_pol2_batch_inputs/tss_readouts.tsv \
  --device 3 \
  --site-shard-index 3 \
  --site-shard-count 4 \
  --windows-bp 1024,4096,20000,100000 \
  --summary-strategy multiscale_log2fc \
  --output-dir outputs/lc_motif_pol2_tss_shard3
```

Recorded TSS shard sizes:

```text
shard0 device0: 35 sites, 70 variants
shard1 device0: 34 sites, 68 variants
shard2 device2: 34 sites, 68 variants
shard3 device3: 34 sites, 68 variants
```

TSS coverage is 137 site-by-motif candidates with in-context proxy TSS readouts.

### 4.7 Population clustering

```bash
python scripts/ism/analyze_batch_population.py \
  --feature-dir outputs/lc_motif_pol2_1d_shard0 \
  --feature-dir outputs/lc_motif_pol2_1d_shard1 \
  --feature-dir outputs/lc_motif_pol2_1d_shard2 \
  --feature-dir outputs/lc_motif_pol2_1d_shard3 \
  --source-sites data/DICs/lc_motif_pol2_batch_prepared/ready_sites.tsv \
  --stability-iterations 30 \
  --random-seed 20260610 \
  --output-dir outputs/lc_motif_pol2_population_analysis
```

Key outputs:

```text
outputs/lc_motif_pol2_population_analysis/representation_comparison.tsv
outputs/lc_motif_pol2_population_analysis/inference_shard_confound.tsv
outputs/lc_motif_pol2_population_analysis/representations/real_signed_depletion/cluster_assignments.tsv
outputs/lc_motif_pol2_population_analysis/figures/pca_clusters.png
```

The formal response-cluster interpretation uses:

```text
outputs/lc_motif_pol2_population_analysis/representations/real_signed_depletion/cluster_assignments.tsv
```

### 4.8 LC motif local and host-TSS statistical analysis

```bash
python scripts/ism/analyze_lc_motif_effects.py \
  --local-feature-dir outputs/lc_motif_pol2_1d_shard0 \
  --local-feature-dir outputs/lc_motif_pol2_1d_shard1 \
  --local-feature-dir outputs/lc_motif_pol2_1d_shard2 \
  --local-feature-dir outputs/lc_motif_pol2_1d_shard3 \
  --tss-feature-dir outputs/lc_motif_pol2_tss_shard0 \
  --tss-feature-dir outputs/lc_motif_pol2_tss_shard1 \
  --tss-feature-dir outputs/lc_motif_pol2_tss_shard2 \
  --tss-feature-dir outputs/lc_motif_pol2_tss_shard3 \
  --sites data/DICs/lc_motif_pol2_batch_prepared/ready_sites.tsv \
  --tss-readouts data/DICs/lc_motif_pol2_batch_inputs/tss_readouts.tsv \
  --cluster-assignments outputs/lc_motif_pol2_population_analysis/representations/real_signed_depletion/cluster_assignments.tsv \
  --output-dir outputs/lc_motif_pol2_effect_analysis
```

Key outputs:

```text
outputs/lc_motif_pol2_effect_analysis/local_track_summary.tsv
outputs/lc_motif_pol2_effect_analysis/tss_track_summary.tsv
outputs/lc_motif_pol2_effect_analysis/tss_distance_summary.tsv
outputs/lc_motif_pol2_effect_analysis/local_tss_correlations.tsv
outputs/lc_motif_pol2_effect_analysis/cluster_motif_enrichment.tsv
outputs/lc_motif_pol2_effect_analysis/candidate_ranking.tsv
outputs/lc_motif_pol2_effect_analysis/site_effects.tsv
outputs/lc_motif_pol2_effect_analysis/result_summary.md
```

### 4.9 PDF report generation

First build static chart assets and HTML:

```bash
MPLCONFIGDIR=/tmp/gReLU-mplconfig \
python scripts/ism/build_lc_motif_pdf_report.py \
  --output-dir agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report
```

Then print the HTML to PDF:

```bash
google-chrome \
  --headless=new \
  --disable-gpu \
  --no-sandbox \
  --no-first-run \
  --no-default-browser-check \
  --allow-file-access-from-files \
  --print-to-pdf-no-header \
  --print-to-pdf=/home/fqijun/python/gReLU/agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/LC_DIC_motif_ISM_experiment_report_zh.pdf \
  file:///home/fqijun/python/gReLU/agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/report.html
```

The generated report metadata records:

```text
n_ready_candidates: 143
n_source_lc_dics: 108
n_mutations: 286
contact_maps_included: false
```

### 4.10 PDF/report QA commands

The final PDF was checked with:

```bash
pdfinfo agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/LC_DIC_motif_ISM_experiment_report_zh.pdf

pdftotext \
  agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/LC_DIC_motif_ISM_experiment_report_zh.pdf \
  agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/report_extracted.txt

pdftoppm -png -r 100 \
  agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/LC_DIC_motif_ISM_experiment_report_zh.pdf \
  agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/page_previews/page

montage \
  agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/page_previews/page-*.png \
  -thumbnail 260x \
  -tile 4x2 \
  -geometry +10+10 \
  agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report/page_contact_sheet.png
```

Observed PDF properties:

```text
Pages: 8
Page size: A4
Text extraction: passed
Visual page preview: passed
```

## 5. Validation commands

Focused tests for the committed code:

```bash
python -m pytest -q tests/test_batch_ism.py tests/test_mreg_three_region_pipeline.py
```

Observed result before the current cleanup discussion:

```text
14 passed
```

If the MREG examples/tests are intentionally removed in a later cleanup, replace
the validation command with:

```bash
python -m pytest -q tests/test_batch_ism.py
```

## 6. Minimal rerun order

For a clean rerun of the formal experiment:

1. `scan_lc_dic_motifs.py`
2. `make_dic_tss_readouts.py`
3. `make_lc_motif_batch_inputs.py`
4. `prepare_batch_ism.py`
5. `run_batch_1d_features.py` for local shards 0-3
6. `run_batch_1d_features.py` for TSS shards 0-3 with `--readout-table`
7. `analyze_batch_population.py`
8. `analyze_lc_motif_effects.py`
9. `build_lc_motif_pdf_report.py`
10. Chrome PDF printing and QA commands

Do not use `outputs/lc_motif_*` without the `pol2` infix as formal evidence;
those are earlier exploratory/smoke artifacts.
