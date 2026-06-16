# HC-DIC ISM experiment index and rerun notes

Last updated: 2026-06-16

This document records the existing HC-DIC evidence and how it maps onto the
current batch-ISM code. It is an index and rerun guide, not a new model run.

For a more detailed raw-result summary and interpretation, see:

```text
experiment/hc_dic_population_result_summary.md
```

## 1. Existing reports

### Single-locus MREG report

The validated MREG experiment includes one HC-DIC edit together with the MREG
TSS and one LC-DIC edit:

```text
agent-doc/ism_context/20260611_1145_mreg_tss_hc_lc_dic_corrected/
  analysis/experimental_report.md
  analysis/experimental_report.html
  analysis/experimental_report.pdf
  analysis/key_effects.tsv
```

Use this corrected directory as the single-locus evidence. The earlier pilot
directory is invalidated and should not be cited as evidence:

```text
agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/
```

Single-locus HC-DIC conclusion:

- perturbation: strand-aware CTCF motif disruption at the HC-DIC;
- local result: strong CTCF/RAD21 depletion;
- distal result: essentially no response at the MREG TSS about 63 kb away.

### Multi-site HC-DIC population report

The existing population-scale HC-DIC report is:

```text
outputs/dic_batch_analysis_hc/report.html
```

Supporting tables and figures:

```text
outputs/dic_batch_analysis_hc/provenance.json
outputs/dic_batch_analysis_hc/site_metadata.tsv
outputs/dic_batch_analysis_hc/representation_comparison.tsv
outputs/dic_batch_analysis_hc/inference_shard_confound.tsv
outputs/dic_batch_analysis_hc/figures/
outputs/dic_batch_analysis_hc/representations/
```

This is an HC-only population clustering report. It does not include a named
host-TSS readout, and it should not be interpreted as a distal promoter test.

## 2. Current multi-site HC-DIC run state

Input and prepared manifests:

```text
data/DICs/batch_ism_inputs/
data/DICs/batch_ism_prepared/
```

Inference outputs:

```text
outputs/dic_batch_1d_shard0/
outputs/dic_batch_1d_shard1/
outputs/dic_batch_1d_shard2/
outputs/dic_batch_1d_shard3/
```

Population analysis:

```text
outputs/dic_batch_analysis_hc/
```

Current counts:

| Item | Count |
|---|---:|
| Input HC-DIC sites | 142 |
| Ready HC-DIC sites | 134 |
| HC-DIC experimental edits | 134 |
| HC-DIC matched controls | 402 |
| Total HC-DIC mutation rows | 536 |
| Tracks | 15 |
| Inference shards | 4 |

The broader prepared manifest also includes LC-DIC sites. The HC report was
run with `--site-class hc_dic`, so the population analysis is HC-only.

## 3. Mutation design

HC-DIC uses the generic `ctcf_pwm_disruption` mutation strategy in:

```text
src/grelu/interpret/ism/batch.py
```

Behavior:

- scan the HC-DIC interval for a CTCF PWM hit above the motif threshold;
- disrupt the selected CTCF motif with the configured motif-disruption policy;
- generate three local matched controls outside the motif;
- controls are `local_non_motif_control` edits, not genome-wide random edits;
- every ready target and control has `fasta_ref_match=True`.

The current prepared HC-DIC table confirms:

- every ready HC-DIC uses `ctcf_pwm_disruption`;
- every ready HC-DIC has three controls;
- 8 HC-DICs were skipped: 4 because CTCF disruption design failed and 4 because
  no CTCF motif passed the threshold.

## 4. Main local result

The existing 4 kb local peak summaries support the expected structural-control
behavior. Median matched-control-adjusted peak log2 fold changes across the
134 ready HC-DICs are:

| Track | Median 4 kb peak log2FC |
|---|---:|
| CTCF | -3.4147 |
| RAD21 | -3.1734 |
| SMC3 | -2.5159 |
| POLR2A | -0.0081 |
| H3K27ac | -0.1822 |
| H3K4me1 | -0.3132 |
| H3K4me2 | -0.5755 |
| H3K4me3 | -0.3154 |

Interpretation:

- HC-DIC CTCF motif editing strongly disrupts local CTCF/cohesin predictions.
- POLR2A is near zero at the local 4 kb peak level.
- This supports HC-DIC as a structural/cohesin control class.
- This does not prove absence of all distal effects, because this batch did not
  run named TSS readouts.

## 5. Rerun commands from current artifacts

Activate the project environment first:

```bash
cd /home/fqijun/python/gReLU
source activate.sh
```

Prepare manifests from the existing site and track tables:

```bash
python scripts/ism/prepare_batch_ism.py \
  --sites data/DICs/batch_ism_inputs/sites.tsv \
  --tracks data/DICs/batch_ism_inputs/tracks.tsv \
  --fasta data/genomes/hg38.fa \
  --motif src/grelu/resources/meme/jaspar_2024_consensus.meme \
  --output-dir data/DICs/batch_ism_prepared \
  --min-relative-motif-score 0.80 \
  --motif-controls 3 \
  --sequence-controls 1 \
  --max-control-distance-bp 5000
```

Run HC-DIC 1D inference. The completed run used four deterministic site shards,
one GPU per process:

```bash
python scripts/ism/run_batch_1d_features.py \
  --prepared-dir data/DICs/batch_ism_prepared \
  --fasta data/genomes/hg38.fa \
  --weights-path /home/fqijun/.cache/alphagenome/model_fold_0.safetensors \
  --track-metadata /home/fqijun/python/gReLU/src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet \
  --output-dir outputs/dic_batch_1d_shard0 \
  --device 0 \
  --site-class hc_dic \
  --site-shard-index 0 \
  --site-shard-count 4 \
  --windows-bp 1024,4096,20000,100000 \
  --summary-strategy multiscale_log2fc
```

Repeat the same command for shard/device pairs `1/1`, `2/2`, and `3/3`, with
matching output directories `outputs/dic_batch_1d_shard1` through
`outputs/dic_batch_1d_shard3`.

Run the HC-only population analysis:

```bash
python scripts/ism/analyze_batch_population.py \
  --feature-dir outputs/dic_batch_1d_shard0 \
  --feature-dir outputs/dic_batch_1d_shard1 \
  --feature-dir outputs/dic_batch_1d_shard2 \
  --feature-dir outputs/dic_batch_1d_shard3 \
  --source-sites data/DICs/batch_ism_prepared/ready_sites.tsv \
  --output-dir outputs/dic_batch_analysis_hc \
  --site-class hc_dic \
  --stability-iterations 30 \
  --random-seed 13
```

## 6. Code boundaries

Reusable code:

```text
src/grelu/interpret/ism/batch.py
scripts/ism/prepare_batch_ism.py
scripts/ism/run_batch_1d_features.py
scripts/ism/analyze_batch_population.py
scripts/ism/analyze_real_ctcf_group_features.py
scripts/ism/analyze_ctcf_population_clustering.py
```

MREG single-locus helpers:

```text
scripts/ism/examples/mreg/
```

Important limitation:

- The current committed active pipeline does not include a fresh raw
  `HC-DIC.modified.bed` to `sites.tsv` adapter. The existing HC rerun starts
  from `data/DICs/batch_ism_inputs/sites.tsv` or
  `data/DICs/batch_ism_prepared/ready_sites.tsv`.
- If the HC-DIC cohort is regenerated from raw paper tables, add a small
  adapter script rather than changing the generic batch runner.

## 7. Interpretation rules

- Keep HC and LC population analyses separate, because they use different
  mutation strategies.
- Treat the multi-site HC result as a local structural perturbation analysis.
- Use the corrected MREG report for the only current HC-DIC distal-TSS example.
- Do not use the old combined HC/LC clustering as biological evidence unless
  it is explicitly framed as QC.
