# gReLU ISM Project Guide

This file is the entry point for agents working on the local AlphaGenome/ISM
research in this repository. Read it before changing the pipeline or
interpreting existing results.

Last updated: 2026-06-13.

Current batch-ISM code architecture:

```text
experiment/code_design.md
```

Current formal LC-DIC motif run commands and parameters:

```text
experiment/lc_dic_motif_runbook.md
```

Existing HC-DIC report index and rerun notes:

```text
experiment/hc_dic_runbook.md
experiment/hc_dic_population_result_summary.md
```

Read that document before modifying the formal LC-DIC motif batch pipeline. It
describes stage ownership, manifest contracts, mutation and summary plugin
boundaries, GPU sharding, local/TSS analysis, known limitations, and the
planned contact-map extension point. Read the runbook before rerunning the
experiment or interpreting similarly named output directories.

## Project Scope

The upstream repository is gReLU. The local research extension uses a local
AlphaGenome checkpoint to study sequence perturbations at CTCF/cohesin sites,
with the current biological focus on HC-DIC and LC-DIC sites from Wang et al.
2022.

The working pattern is:

```text
paper/site tables
  -> site and track manifests
  -> mutation and matched-control design
  -> AlphaGenome REF/ALT inference
  -> bounded local or named distal readouts
  -> population statistics and biological interpretation
```

This is not saturation mutagenesis. Expensive inference is kept separate from
cheap downstream analysis so that tracks, summaries and statistical methods
can be changed without rerunning the model.

## Environment

Repository:

```text
/home/fqijun/python/gReLU
```

Always activate the project environment before Python commands:

```bash
cd /home/fqijun/python/gReLU
source activate.sh
```

Important local resources:

```text
AlphaGenome weights:
  /home/fqijun/.cache/alphagenome/model_fold_0.safetensors

hg38 FASTA:
  /home/fqijun/.local/share/genomes/hg38/hg38.fa

CTCF/JASPAR motifs:
  src/grelu/resources/meme/jaspar_2024_consensus.meme

HOCOMOCO H13 TF motifs:
  src/grelu/resources/meme/H13CORE_meme_format.meme
```

The plain system Python may lack dependencies such as `pyarrow`. Use
`source activate.sh`.

## Core Architecture

The reusable batch implementation is intentionally small and manifest-driven.

Core module:

```text
src/grelu/interpret/ism/batch.py
```

Stable stage interfaces:

| Stage | Main input | Main output |
|---|---|---|
| Site adapter | paper BED/audit tables | `sites.tsv`, `tracks.tsv` |
| Mutation preparation | site and track tables | prepared manifests |
| Inference | prepared manifests | mutation chunks and `group_features.parquet` |
| Population analysis | feature parquet files | clusters, tests and reports |
| Named distal analysis | readout table | site-specific TSS/other summaries |

Mutation and feature-summary behavior is extensible through registries:

```python
register_mutation_strategy(...)
register_summary_strategy(...)
```

Use a new strategy only when the biological perturbation changes. Do not add a
workflow framework or broad abstraction for a single experiment.

Main documentation for the current formal LC-DIC motif batch experiment:

```text
experiment/code_design.md
experiment/lc_dic_motif_runbook.md
experiment/hc_dic_runbook.md
```

Main current batch scripts:

```text
scripts/ism/scan_lc_dic_motifs.py
scripts/ism/make_lc_motif_batch_inputs.py
scripts/ism/prepare_batch_ism.py
scripts/ism/run_batch_1d_features.py
scripts/ism/analyze_batch_population.py
scripts/ism/make_dic_tss_readouts.py
scripts/ism/analyze_lc_motif_effects.py
scripts/ism/build_lc_motif_pdf_report.py
```

The primary model setup is a 1,048,576 bp input context with 128 bp ChIP-seq
outputs. Population runs persist bounded summaries rather than full output
arrays.

## Single-Locus MREG Experiment

The validated single-locus experiment compares the MREG TSS, one HC-DIC and one
LC-DIC. `MREG` is the correct gene name.

Authoritative experiment:

```text
agent-doc/ism_context/20260611_1145_mreg_tss_hc_lc_dic_corrected/
```

Do not use the earlier cross-region pilot as evidence:

```text
agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/
```

The corrected hg38 anchors are:

| Region | Coordinate |
|---|---:|
| MREG TSS | `chr2:216013551` |
| HC-DIC Rad21 summit | `chr2:215950160` |
| LC-DIC Rad21 summit | `chr2:215979931` |
| LC-DIC Pol2 edit anchor | `chr2:215979780` |

Perturbations:

- TSS: strand-aware CTCF motif disruption.
- HC-DIC: strand-aware CTCF motif disruption.
- LC-DIC: exploratory deterministic 3 bp substitution at the Pol2 peak center.

Main result:

- TSS editing produces a promoter/CTCF/cohesin response.
- HC-DIC editing strongly reduces local CTCF/RAD21 but has essentially no
  effect at the MREG TSS about 63 kb away.
- LC-DIC editing reduces local POLR2A and active histone marks and produces a
  weaker, directionally consistent negative response at the MREG TSS about
  34 kb away.
- This is a single-locus result, not proof that all LC-DICs regulate a TSS.

Read first:

```text
agent-doc/ism_context/20260611_1145_mreg_tss_hc_lc_dic_corrected/analysis/experimental_report.md
agent-doc/ism_context/20260611_1145_mreg_tss_hc_lc_dic_corrected/analysis/key_effects.tsv
agent-doc/ism_context/20260611_1145_mreg_tss_hc_lc_dic_corrected/figures/
```

Reusable MREG code:

```text
scripts/ism/examples/mreg/
```

## Multi-Site DIC Experiment

Input material:

```text
data/DICs/HC-DIC.modified.bed
data/DICs/LC-DIC.modified.bed
data/DICs/dic_site_audit.tsv
```

Prepared batch:

```text
data/DICs/batch_ism_inputs/
data/DICs/batch_ism_prepared/
```

Current ready sites:

- 548 total.
- 134 HC-DIC.
- 414 LC-DIC.
- 1,364 mutations including matched controls.

Current perturbations:

- HC-DIC: CTCF PWM disruption with three local non-motif controls.
- LC-DIC: deterministic 3 bp Pol2-summit substitution with one local sequence
  control.

The LC edit is not motif-guided. It changes the three bases centered at the
Pol2 summit with `A->G`, `C->T`, `G->A`, `T->C`. Do not describe it as a Pol2
motif knockout.

Four-GPU 1D inference outputs:

```text
outputs/dic_batch_1d_shard0/
outputs/dic_batch_1d_shard1/
outputs/dic_batch_1d_shard2/
outputs/dic_batch_1d_shard3/
```

Population reports:

```text
outputs/dic_batch_analysis/
outputs/dic_batch_analysis_hc/
outputs/dic_batch_analysis_lc/
```

Run HC and LC analyses separately because they use different mutation
strategies. Combined HC/LC clustering is useful QC but is confounded by
perturbation type.

### Current HC Local EDA

The existing multi-site HC-DIC report is indexed in:

```text
experiment/hc_dic_runbook.md
outputs/dic_batch_analysis_hc/report.html
```

This is an HC-only local population analysis, not a named distal-TSS analysis.
It contains 134 ready HC-DICs using CTCF PWM disruption and three local
non-motif controls per site. The current local 4 kb median peak effects show
strong CTCF/RAD21/SMC3 depletion, while POLR2A is near zero. Use the corrected
MREG single-locus report for the current HC-DIC distal-TSS example.

Detailed raw-result summary:

```text
experiment/hc_dic_population_result_summary.md
```

### Current LC Local EDA

The exploratory Pol2-center edit divided the 414 LC-DICs into:

- 353 local nonresponders.
- 61 local responders.

The 61-site group shows local decreases in CTCF, RAD21 and active chromatin,
but it is not a paper-defined biological subtype and is not enriched for the
paper-guided TF motifs.

Key files:

```text
outputs/dic_batch_analysis_lc/report.html
outputs/dic_batch_analysis_lc/responsive_61_mutations.tsv
outputs/dic_batch_analysis_lc/representations/real_local_1_4kb/
```

### Current LC Distal-TSS EDA

All 414 LC-DICs entered TSS mapping:

- 407 have a selected TSS inside the model context and were analyzed.
- 7 have a TSS outside the approximately 1 Mb model context.
- 64 of the 407 mappings are ambiguous because multiple genes overlap.

Across all 407 sites, and also within the 60/61 local responders that have a
valid TSS readout, there is no robust group-level distal TSS response.

There are six exploratory sites with concordant negative TSS POLR2A and
H3K27ac effects, of which three meet the stricter exploratory threshold. Treat
these as candidates, not a discovered class.

Key files:

```text
data/DICs/lc_tss_readouts/
outputs/lc_tss_4kb_analysis/result_summary.md
outputs/lc_tss_4kb_analysis/site_tss_effects.tsv
outputs/lc_tss_4kb_analysis/responder_tss_effects_with_mutations.tsv
```

The host TSS assignment is inferred from local gene annotation. It is not an
author-provided DIC-to-gene mapping.

## Literature-Guided LC Mutation Direction

Primary paper:

```text
agent-doc/paper/nc-1.pdf
agent-doc/paper/nc-1.txt
agent-doc/paper/nakato2022_dic_identification_algorithm.md
```

Supplementary material and source data:

```text
agent-doc/ism_context/20260610_1543_mreg_tss_hc_lc_dic_pilot/paper_data/
```

Important paper facts:

- The authors did not perform LC-DIC sequence editing.
- Their causal perturbations were E2 stimulation and global Rad21/NIPBL
  knockdown.
- LC-DICs are enhancer-like, low-CTCF sites enriched for paused Pol2,
  H3K27ac/H3K4me1, P300/CBP and Forkhead motifs.
- Machine-learning features point to FOXA1, ER, GATA3, Pol2 and LARP7.
- Only 19.2% of LC-DIC loops contacted the host-gene promoter.
- Only about 10% of LC-DIC host genes changed expression after siRad21.

The next sequence experiment should therefore test TF motif dependence rather
than repeat arbitrary peak-center substitutions.

Completed motif scan:

```text
outputs/lc_dic_motif_scan_0.90/
```

The first scan used the legacy LC anchor table and is exploratory only:

- 266/414 LC-DICs contain at least one panel motif.
- 364 site-by-family mutation candidates.
- GATA3: 157 sites.
- AP-1: 91 sites.
- FOXA1/Forkhead: 67 sites.
- TEAD4: 48 sites.
- ESR1: 1 site.
- 148 sites have no high-confidence panel motif.

Important anchor correction:

- The legacy adapter selected Pol2 Ctrl summit, then Pol2 E2-30 summit, then
  Rad21 summit as a fallback, but labelled every result `Pol2_peak_center`.
- Only 168/414 ready LC-DICs have an audited Pol2 peak and summit in Ctrl,
  E2-30 or E2-45.
- Do not use the 414-site motif scan as the formal Pol2-anchored cohort.

Candidate table:

```text
outputs/lc_dic_motif_scan_0.90/motif_mutation_candidates.tsv
```

Design plan:

```text
agent-doc/plan&reports/ctcf_alphagenome/20260612_lc_dic_literature_guided_mutation_plan.md
```

Recommended next implementation:

1. Add a general PWM motif-disruption strategy that accepts motif name/PWM
   from the site manifest.
2. For each site-by-family candidate, disrupt the three most informative PWM
   positions.
3. Reject edits that create another high-scoring panel motif.
4. Add a same-peak, non-motif matched control with matched edit count and GC
   delta.
5. Reuse the existing local 1/4 kb and named distal readouts.
6. Analyze motif families separately and compare loop-supported promoters when
   available.

Do not force motif edits onto no-hit sites. In the formal Pol2-anchored cohort,
60/168 sites have no qualifying panel motif and are useful comparison sites.

### Formal Pol2-Anchored Motif Experiment

The strict scan requires each motif to be within the selected real Pol2 summit
+/-100 bp and fully inside the same called Pol2 peak.

Formal cohort:

- 168 LC-DICs scanned after requiring a real Pol2 peak and summit.
- 108 LC-DICs contain a qualifying panel motif.
- 144 site-by-family candidates before mutation preparation.
- 143 ready motif edits with one matched same-peak control each.
- Families: 59 GATA3, 37 AP-1, 32 Forkhead and 15 TEAD4.
- One candidate, `lc_dic_108__gata3`, was skipped because no valid three-base
  edit reduced the motif below threshold.

Mutation QC:

- Exactly three changed bases in every target and control.
- Target relative PWM score is below 0.90 after editing.
- Target and control have identical edited-base count and GC delta.
- Controls avoid all high-scoring panel motifs and remain in the same Pol2
  peak.

Formal paths:

```text
outputs/lc_dic_motif_scan_pol2_0.90/
data/DICs/lc_motif_pol2_batch_inputs/
data/DICs/lc_motif_pol2_batch_prepared/
outputs/lc_motif_pol2_1d_shard0/
outputs/lc_motif_pol2_1d_shard1/
outputs/lc_motif_pol2_1d_shard2/
outputs/lc_motif_pol2_1d_shard3/
outputs/lc_motif_pol2_effect_analysis/
outputs/lc_motif_pol2_population_analysis/
```

Current local 4 kb result:

- Overall effects are modest but shifted negative for POLR2A, cohesin and
  active chromatin after matched-control adjustment.
- Forkhead edits form the clearest responsive family, with coordinated FOXA1,
  POLR2A, H3K27ac and H3K4me1 depletion.
- GATA3 motif edits are near zero as a group despite being the largest family.
- The most stable population representation selects 27/143 high-depletion
  candidates. Forkhead is enriched in this cluster (11/32, FDR 0.038), while
  GATA3 is depleted (3/59, FDR 0.0015).
- Therefore the result supports motif-family-specific LC-DIC dependence, not
  a universal LC-DIC response.

Current host-TSS proxy result:

- 137/143 candidates have a TSS readout in model context.
- Across all candidates the TSS effects are directionally negative but tiny:
  median POLR2A is about -0.0004 after matched-control adjustment.
- At distances greater than 50 kb, Forkhead candidates retain a weak
  directionally consistent POLR2A effect (median about -0.0009, 10/13
  negative), whereas GATA3 and TEAD4 remain near zero.
- Local and TSS effects are correlated even beyond 50 kb, but the distal
  magnitudes are much smaller. Treat this as candidate propagation, not proof
  of enhancer-promoter regulation.

Formal summary:

```text
outputs/lc_motif_pol2_effect_analysis/result_summary.md
outputs/lc_motif_pol2_effect_analysis/local_track_summary.tsv
outputs/lc_motif_pol2_effect_analysis/tss_distance_summary.tsv
outputs/lc_motif_pol2_effect_analysis/candidate_ranking.tsv
```

## Directory Map

```text
src/grelu/interpret/ism/
  Reusable ISM implementation.

scripts/ism/
  CLI scripts, batch adapters, analyses and documentation.

scripts/ism/examples/mreg/
  Single-locus MREG implementation.

experiment/code_design.md
  Current LC-DIC motif batch-ISM architecture and extension boundaries.

experiment/lc_dic_motif_runbook.md
  Formal LC-DIC motif run commands, parameters and artifact boundary.

experiment/hc_dic_runbook.md
  Existing HC-DIC report inventory, rerun commands and code boundary.

experiment/hc_dic_population_result_summary.md
  Detailed HC-DIC population raw-result summary and interpretation.

data/DICs/
  HC/LC BED files, audited sites and prepared batch manifests.

outputs/
  Current population inference, EDA, TSS and motif-scan artifacts.

agent-doc/paper/
  Primary papers and extracted text.

agent-doc/ism_context/
  Immutable experiment-specific inputs, runs, figures and reports.

agent-doc/plan&reports/ctcf_alphagenome/
  Research plans, reviews and interpretation notes.

agent-doc/ref/
  External reference code/data retained for provenance.
```

## Validation

Focused tests:

```bash
source activate.sh
python -m pytest -q \
  tests/test_batch_ism.py \
  tests/test_mreg_three_region_pipeline.py
```

Current baseline: 14 tests pass.

Before trusting a new run, verify:

- FASTA REF matches.
- Experimental and matched-control sequences share the intended model frame.
- The real checkpoint path is recorded in `run_metadata.json` or provenance.
- Shards use identical track selection.
- Site IDs are unique after merging.
- HC and LC are not interpreted from a combined strategy-confounded cluster.
- Distal readouts are inside model context and mapping ambiguity is reported.

## Interpretation Rules

- AlphaGenome outputs are predictions, not wet-lab validation.
- A local effect does not imply a distal promoter effect.
- A nearest or containing-gene TSS is a proxy unless supported by loop data.
- The old LC 3 bp edit is a peak-center perturbation, not a motif knockout.
- The 61 LC local responders are exploratory, not author-defined.
- The six distal LC candidates are exploratory, not statistically established.
- Prefer raw REF/ALT plus delta and matched-control-adjusted effects when
  interpreting individual sites.
- Keep RNA-seq as a later-stage readout unless explicitly requested; current
  primary work uses 128 bp ChIP tracks.

## Working Conventions

- Preserve existing user changes; the worktree may be dirty.
- Use `apply_patch` for manual edits.
- Put reusable code in `src/` or `scripts/ism/`.
- Put experiment-specific evidence under a dated `agent-doc/ism_context/`
  directory.
- Put plans and interpretive reports under
  `agent-doc/plan&reports/ctcf_alphagenome/`.
- Record exact coordinates, mutation strategy, controls, checkpoint, tracks and
  model context for every experiment.
