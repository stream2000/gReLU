# Saturation ISM Framework Design

Date: 2026-07-09

Scope: migrate the validated MREG saturation-mutagenesis ideas from the old
`/home/fqijun/python/gReLU` worktree into this clean `alphagenome` checkout,
while making saturation mutagenesis a reusable framework for fine-tuned
AlphaGenome and Borzoi models.

This document assumes "MERG" in discussion means the existing MREG locus work.

## Goal

Saturation mutagenesis should become a first-class, reusable ISM subsystem. A
specific biological experiment, such as MREG TSS/HC-DIC/LC-DIC or Saijou HSC
expression around Mdk/Col1a1/Acta2, should be a plugin that supplies regions,
readouts, tracks, and plotting choices. It should not own core sequence editing,
model prediction, batching, scoring, or provenance logic.

The framework must support:

- MREG saturation screens using original AlphaGenome and Borzoi output tracks.
- AlphaGenome fine-tuned Saijou four-track models, especially the HSC output.
- Borzoi fine-tuned Saijou four-track models.
- Model-specific input lengths, output resolutions, crop behavior, and
  checkpoint loading.
- Re-ranking existing raw mutation outputs by new tracks/readouts without
  re-running model inference when possible.
- Compact outputs that preserve enough raw REF/ALT/delta signal for later
  biology review.

## Existing Code Reviewed

Current `alphagenome` checkout:

- `scripts/run_ism.py`
- `src/grelu/interpret/score.py`
- `src/ft-scripts/plot_borzoi_positive_controls.py`
- `src/ft-scripts/plot_alphagenome_positive_controls.py`
- `src/ft-scripts/train_alphagenome.py`
- `src/ft-scripts/train_borzoi.py`

Old full ISM/MREG worktree:

- `/home/fqijun/python/gReLU/src/grelu/interpret/ism/saturation.py`
- `/home/fqijun/python/gReLU/src/grelu/interpret/ism/batch.py`
- `/home/fqijun/python/gReLU/src/grelu/interpret/ism/inference.py`
- `/home/fqijun/python/gReLU/src/grelu/interpret/ism/config.py`
- `/home/fqijun/python/gReLU/src/grelu/interpret/ism/workflow.py`
- `/home/fqijun/python/gReLU/scripts/ism/examples/mreg/README.md`
- `/home/fqijun/python/gReLU/scripts/ism/examples/mreg/prepare_mreg_pol2_saturation.py`
- `/home/fqijun/python/gReLU/scripts/ism/examples/mreg/mreg_profile_runner.py`
- `/home/fqijun/python/gReLU/scripts/ism/examples/mreg/run_mreg_alphagenome_profiles.py`
- `/home/fqijun/python/gReLU/scripts/ism/examples/mreg/run_mreg_borzoi_profiles.py`
- `/home/fqijun/python/gReLU/scripts/ism/examples/mreg/rank_mreg_saturation.py`

## What Works In The Old Code

The following pieces are worth preserving, either by cherry-pick if commits are
clean or by direct file-level port:

1. Reference-backed SNV enumeration

   `saturation.py` cleanly expands a `SaturationWindow` into all non-reference
   single-base substitutions. It records `variant_position`, `ref_base`,
   `alt_base`, `variant_id`, original region bounds, and source labels. This is
   exactly the core primitive needed for TSS windows, DIC windows, promoter
   windows, and gene-specific scans.

2. Explicit edit manifest

   `batch.py` already has an `explicit_edit` mutation strategy with FASTA REF
   validation. This is the right contract for saturation screens and should
   remain model-agnostic.

3. Strategy-based MREG profile runner

   The old MREG README says AlphaGenome and Borzoi use separate prediction
   strategies but share manifest loading, edit application, readout-bin
   extraction, long-form profile rows, timing, and provenance. This is the
   correct direction.

4. Long-form profile output

   The old MREG workflow writes long-form rows rather than full dense model
   outputs. This keeps memory and disk bounded while still preserving per-bin
   REF, ALT, delta, and log2FC for selected readout windows.

5. Re-ranking model

   `rank_mreg_saturation.py` and `RankingSpec` make ranking a downstream
   operation over stored features. That should be kept so biological questions
   can change without recomputing all mutations.

## Why The Old Framework Cannot Be Merged As-Is

The old implementation is useful but not yet a clean general framework. The
problems are architectural, not just naming problems.

1. Config is experiment-specific

   `config.py` is named as a general `RunConfig`, but its schema is
   `ctcf-ism-v1.2` and hard-codes concepts such as CTCF, MCF-7, DIC labels,
   canonical label reconstruction, preferred breast biosamples, and contact-map
   defaults. That is appropriate for the CTCF/MREG paper workflow, but not for
   Saijou HSC expression or fine-tuned four-track models.

2. Model loading assumes original AlphaGenome output heads

   `inference.py` builds `AlphaGenomeModel` with `output_key` and selects tracks
   by AlphaGenome metadata indices. Fine-tuned Saijou models instead output
   four task channels (`hsc`, `mac`, `lsec`, `chol`) from a custom head. Borzoi
   fine-tuned checkpoints similarly have a different load path, LoRA wrapping,
   label crop, and output bin size. Track selection cannot depend only on
   original AlphaGenome metadata.

3. Readout construction is partly tied to mutation-centered windows

   `_summary_masks` and default readouts center on mutation or on inferred gene
   contexts. For Saijou 10X data, the biologically useful readout may be a
   TSS window, TES/3-prime observed-signal window, gene body window, or a custom
   interval. This must be a readout plugin, not a hidden default.

4. One config object owns too many stages

   The old `workflow.py` orders labels, sampling, motifs, inference, features,
   analysis, and artifacts. For reusable saturation ISM, the core should not
   require label reconstruction or MREG sampling. Those are experiment plugins.

5. Output-resolution semantics are not explicit enough

   Single-base mutations can be enumerated at bp resolution, but model outputs
   may be 32 bp, 128 bp, or 1 bp depending on model/head. The old code often
   knows this inside AlphaGenome-specific paths. The new framework should write
   both mutation resolution and output-bin resolution into manifests and output
   tables.

6. Reverse-complement behavior is model/track-specific

   Original model stranded tracks may need reverse-complement partner mapping.
   Fine-tuned four-track HSC/mac/LSEC/chol outputs are not stranded output pairs
   in the same sense. This must live in the model adapter or track selector.

7. Experiment scripts are too close to reusable logic

   Old MREG scripts contain valuable reusable code, but some files still mix
   MREG constants, track registry, ranking presets, and profile execution. This
   makes it hard to add a clean HSC expression experiment without copying and
   editing MREG code.

## Proposed Architecture

Create a general package under:

```text
src/grelu/interpret/ism/
```

with experiment entry points under:

```text
scripts/ism/experiments/
```

The core package should not import MREG, Saijou, AlphaGenome-specific metadata,
or Borzoi-specific checkpoint logic at import time. Specific adapters and
experiments can import model-specific code.

### Layer 1: Core Data Contracts

Core dataclasses and plain TSV/Parquet schemas:

- `SequenceWindow`
  - `window_id`
  - `chrom`
  - `start`
  - `end`
  - `anchor`
  - `genome`
  - `source`
  - `label`

- `MutationSpec`
  - `mutation_id`
  - `window_id`
  - `chrom`
  - `edit_start`
  - `edit_end`
  - `ref_sequence`
  - `alt_sequence`
  - `mutation_kind`
  - `variant_position`
  - `ref_base`
  - `alt_base`
  - `control_type`
  - `matched_target_id`

- `ReadoutSpec`
  - `readout_id`
  - `window_id` or `*`
  - `chrom`
  - `start`
  - `end`
  - `anchor`
  - `relative_to`: `genomic`, `sequence_center`, `mutation`, `tss`, `tes`
  - `aggregation`: `sum`, `mean`, `max`, `peak`, `profile`
  - `description`

- `TrackSpec`
  - `track_id`
  - `model_output_key`
  - `channel_index`
  - `task_name`
  - `cell_type`
  - `modality`
  - `strand`
  - `resolution_bp`
  - `source`: `alphagenome_original`, `borzoi_original`,
    `alphagenome_finetuned`, `borzoi_finetuned`

- `PredictionProfile`
  - `mutation_id`
  - `allele`: `ref` or `alt`
  - `model_id`
  - `track_id`
  - `readout_id`
  - `chrom`
  - `bin_start`
  - `bin_end`
  - `value`

- `MutationFeature`
  - `mutation_id`
  - `model_id`
  - `track_id`
  - `readout_id`
  - `ref_mean`
  - `alt_mean`
  - `signed_delta_mean`
  - `absolute_delta_mean`
  - `log2fc_mean`
  - `delta_peak`
  - `peak_shift_bp`

These schemas should be stable and versioned. TSV is preferred for manifests;
Parquet is preferred for long profile/features.

### Layer 2: Mutation Generators

Core interface:

```python
class MutationGenerator(Protocol):
    def generate(self, windows: pd.DataFrame, fasta: FastaReference) -> pd.DataFrame:
        ...
```

Initial implementations:

- `SaturationSnvGenerator`
  - port from old `saturation.py`.
  - enumerates every non-reference SNV in each window.
  - mandatory FASTA REF validation.

- `ExplicitEditGenerator`
  - normalizes an existing mutation table.
  - useful for MREG curated motif perturbations and ranked re-profiling.

- `MotifDisruptionGenerator`
  - optional later port from old `batch.py`.
  - should live as a plugin because it depends on PWM inputs and motif policy.

The core framework should treat saturation SNV enumeration as the default
primitive. Motif disruption is a special case of mutation generation, not the
center of the framework.

### Layer 3: Model Adapters

Core interface:

```python
class SequenceToProfileModel(Protocol):
    model_id: str
    input_length_bp: int
    output_resolution_bp: int
    output_length_bins: int
    track_specs: list[TrackSpec]

    def predict_profiles(
        self,
        sequences: Sequence[str],
        tracks: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return shape [n_sequence, n_track, n_bin]."""
```

Initial adapters:

- `AlphaGenomeOriginalAdapter`
  - original AG weights.
  - supports original output heads and metadata track selection.
  - output resolution can be 1 or 128 where supported.
  - used by MREG original-model experiments.

- `BorzoiOriginalAdapter`
  - original Borzoi output tracks.
  - supports Borzoi task metadata and crop semantics.
  - used by MREG original-model comparisons.

- `AlphaGenomeFinetunedAdapter`
  - loads `grelu.lightning.LightningModel` from the Saijou LoRA checkpoint.
  - applies AG LoRA wrappers before loading state dict.
  - exposes four tracks: `hsc`, `mac`, `lsec`, `chol`.
  - default current checkpoint:
    `runs/alphagenome_split_chr10_chr11_seq524288_label196608_bin128_res128_poisson_multinomial_lora_active_h512x1/checkpoints/epochepoch=19.ckpt`.
  - output resolution is 128 bp in the current fine-tuned run.

- `BorzoiFinetunedAdapter`
  - loads Borzoi LoRA checkpoint.
  - applies Borzoi LoRA wrappers before loading state dict.
  - exposes four tracks: `hsc`, `mac`, `lsec`, `chol`.
  - output resolution is 32 bp for the existing Borzoi fine-tune unless a new
    cache/checkpoint changes it.

Adapters own:

- checkpoint loading.
- LoRA wrapper application.
- sequence tensor conversion.
- device/batch/precision settings.
- output crop alignment.
- reverse-complement behavior.
- conversion to a standard `[sequence, track, bin]` profile array.

Adapters must not own:

- mutation generation.
- readout definitions.
- ranking policy.
- experiment-specific plotting.

### Layer 4: Readout Plugins

Core interface:

```python
class ReadoutResolver(Protocol):
    def resolve(
        self,
        readouts: pd.DataFrame,
        sequence_context: SequenceContext,
        output_bin_index: OutputBinIndex,
    ) -> list[ResolvedReadout]:
        ...
```

Readouts should be explicit and should support:

- mutation-centered windows, such as 1 kb or 4 kb around each SNV.
- named genomic intervals, such as MREG TSS, HC-DIC, LC-DIC, or gene body.
- TSS windows from a GTF.
- TES/3-prime windows from a GTF.
- observed bigWig peak windows, useful for 10X 3-prime scRNA-seq.
- full profile extraction for plotting.

For the Saijou HSC experiment, the first readout set should include:

- `tss_center_1kb`
- `tss_center_4kb`
- `tes_or_3prime_1kb`
- `observed_hsc_peak_1kb`
- `gene_body_short_gene`

The 10X caveat is important: boss text previously said reads are mainly near
the TES/3-prime end. If any message says 5-prime, treat that as a contradiction
to resolve in the experiment note. The framework should support both TSS and
TES readouts so the biology question can be tested directly.

### Layer 5: Execution Engine

Core runner:

```text
prepare -> predict -> summarize -> rank -> plot/report
```

`prepare`:

- loads windows.
- extracts reference sequence contexts.
- generates mutations.
- validates edit REF against FASTA.
- writes immutable manifests.

`predict`:

- chunks mutations into bounded batches.
- builds REF and ALT sequence contexts.
- calls one model adapter.
- writes long-form profile chunks or compact summary chunks.
- records timing, model metadata, checkpoint path, git commit, and config hash.

`summarize`:

- applies readout masks to profile chunks.
- computes REF/ALT/delta/log2FC/peak metrics.
- writes `mutation_features.parquet`.

`rank`:

- applies `RankingSpec`.
- produces top variants per model/track/readout/metric.
- can be rerun without inference.

`plot/report`:

- experiment-specific.
- may use common plotting helpers, but no core dependency on a particular
  biological story.

### Layer 6: Experiment Plugins

Experiment plugins should live under:

```text
scripts/ism/experiments/mreg/
scripts/ism/experiments/saijou_hsc/
```

Each plugin should provide only:

- region/window preparation.
- readout table construction.
- track/model selection defaults.
- ranking presets.
- plots/report layout.

#### MREG Plugin

Initial files:

- `prepare_mreg_saturation.py`
- `rank_mreg_saturation.py`
- `plot_mreg_saturation.py`

Responsibilities:

- define MREG TSS/HC-DIC/LC-DIC windows.
- generate saturation SNVs or explicit motif edits.
- define MREG readouts: local mutation window, TSS, HC-DIC, LC-DIC, gene body.
- define original-model track registries for AG/Borzoi.
- keep MREG transcript/TSS coordinate choices explicit.

This plugin should not own model prediction logic.

#### Saijou HSC Plugin

Initial files:

- `prepare_saijou_hsc_tss_saturation.py`
- `run_saijou_hsc_saturation.py`
- `rank_saijou_hsc_saturation.py`
- `plot_saijou_hsc_saturation.py`

Responsibilities:

- parse candidate genes or a fallback gene list.
- resolve mm10 gene coordinates and transcript choices.
- prefer short genes for first pass.
- build windows around TSS, TES, and observed HSC signal peak.
- select the `hsc` track from fine-tuned AG or Borzoi adapters.
- produce motif-recovery plots for Mdk, Col1a1, Acta2.

This plugin should support model choices:

```text
--model alphagenome_finetuned
--model borzoi_finetuned
```

and later:

```text
--model alphagenome_original
--model borzoi_original
```

if original head proxies are intentionally used.

## Output Directory Layout

Use one run directory per biological run:

```text
experiments/ism/<YYYYMMDD_HHMM>_<slug>/
├── inputs/
│   ├── windows.tsv
│   ├── readouts.tsv
│   ├── tracks.tsv
│   └── ranking_specs.tsv
├── prepared/
│   ├── mutation_manifest.tsv
│   ├── sequence_contexts.tsv
│   ├── preparation_qc.tsv
│   └── manifest.json
├── predictions/
│   ├── profile_chunks/
│   ├── summary_chunks/
│   ├── prediction_index.tsv
│   └── model_metadata.json
├── features/
│   ├── mutation_features.parquet
│   └── mutation_features.tsv
├── ranking/
│   ├── top_variants.tsv
│   └── ranking_summary.tsv
└── figures/
    ├── saturation_logo_*.png
    └── profile_ref_alt_delta_*.png
```

Large ignored outputs stay under `experiments/ism/`. Reusable scripts and docs
stay tracked under `src/` and `scripts/`.

## Migration Plan

### Phase 0: Commit Hygiene Before Migration

Before moving code, preserve the current fine-tuning state separately from ISM.
The current branch already contains uncommitted AlphaGenome fine-tuning files.
Do not mix those edits with ISM migration.

Expected split:

1. Existing AG/Borzoi fine-tuning infrastructure commit.
2. ISM skeleton/framework commit.
3. MREG experiment plugin commit.
4. Saijou HSC experiment plugin commit.

If the first commit is already done in another session, verify `git status`
before staging anything.

### Phase 1: Port Core Skeleton

Create:

```text
src/grelu/interpret/ism/
├── __init__.py
├── schemas.py
├── fasta.py
├── mutations.py
├── adapters.py
├── readouts.py
├── engine.py
├── features.py
├── ranking.py
└── provenance.py
```

Port or rewrite:

- old `saturation.py` -> `mutations.py`.
- old explicit edit validation -> `mutations.py`.
- old long-form metric calculations -> `features.py`.
- old `RankingSpec` and ranking helpers -> `ranking.py`.
- minimal provenance helpers -> `provenance.py`.

Do not port:

- old `ctcf-ism-v1.2` config as the central config.
- MREG constants into core.
- AlphaGenome metadata track grouping into core.

Validation:

- unit test SNV enumeration on a tiny FASTA.
- unit test explicit edit REF mismatch.
- unit test readout interval-to-bin mapping for 32 bp and 128 bp outputs.
- unit test feature metrics from synthetic REF/ALT arrays.

### Phase 2: Add Model Adapters

Create:

```text
src/grelu/interpret/ism/model_adapters/
├── __init__.py
├── base.py
├── alphagenome_original.py
├── alphagenome_finetuned.py
├── borzoi_original.py
└── borzoi_finetuned.py
```

For first Saijou AG run, only `alphagenome_finetuned.py` must be fully
implemented. The others can be added behind tests or stubs, but API shape
should be fixed now.

Reuse checkpoint-loading logic from:

- `src/ft-scripts/plot_alphagenome_positive_controls.py`
- `src/ft-scripts/plot_borzoi_positive_controls.py`

Do not copy plotting code into adapters.

Validation:

- load AG e19 checkpoint.
- predict one mm10 sequence.
- assert output shape `[1, 4, 1536]` for current
  `label_len=196608`, `bin_size=128`.
- assert track names are exactly `hsc,mac,lsec,chol`.

### Phase 3: MREG Plugin

Port old MREG preparation and ranking scripts into:

```text
scripts/ism/experiments/mreg/
```

Keep MREG-specific constants and transcript choices there. MREG should call
the core engine and adapters, not duplicate profile runner logic.

Validation:

- reproduce one small known MREG saturation or explicit-edit smoke run.
- compare top-level row counts to the old validation run when using the same
  input manifests.

### Phase 4: Saijou HSC Plugin

Add the HSC experiment plugin. First technical run:

- model: `alphagenome_finetuned`.
- track: `hsc`.
- genes: `Mdk`, `Col1a1`, `Acta2` if candidate file is not yet available.
- genome: mm10.
- readouts: TSS and TES/observed-peak.
- saturation width: start small, for example 512 bp or 1 kb around TSS/TES.
- output: ranked variants plus logo-like bp-axis heatmaps.

Then compare:

- AG fine-tuned HSC versus Borzoi fine-tuned HSC.
- TSS readout versus TES/3-prime readout.
- short genes versus long genes.

Validation:

- confirm gene coordinates and transcript choices in output tables.
- confirm each SNV REF matches mm10 FASTA.
- confirm no full dense 524 kb profiles are persisted unless explicitly
  requested.

## Commit Plan

The migration should be committed in clean boundaries:

### Commit A: Existing Fine-Tuning Infrastructure

Purpose: preserve current AG/Borzoi fine-tuning and validation scripts.

Likely files:

- `src/ft-scripts/borzoi_mmap_dataset.py`
- `src/ft-scripts/train_alphagenome.py`
- `src/grelu/lightning/__init__.py`
- `src/grelu/model/heads.py`
- `src/grelu/model/models.py`
- `src/ft-scripts/compare_positive_control_models.py`
- `src/ft-scripts/plot_alphagenome_positive_controls.py`

Do this only if these changes are confirmed to belong together and are not
already committed by another session.

### Commit B: Saturation ISM Core Skeleton

Purpose: reusable framework, no MREG or Saijou-specific biology.

Files:

- `src/grelu/interpret/ism/*`
- core tests under `tests/test_ism_*.py`
- this design document can be included here if not committed earlier.

### Commit C: Model Adapters

Purpose: bridge core ISM to AG/Borzoi original/fine-tuned models.

Files:

- `src/grelu/interpret/ism/model_adapters/*`
- tests for model metadata/shape where possible.

### Commit D: MREG Plugin

Purpose: port previous complete MREG saturation workflow onto the new core.

Files:

- `scripts/ism/experiments/mreg/*`
- optional MREG-specific README.

### Commit E: Saijou HSC Plugin

Purpose: new AlphaGenome/Borzoi fine-tuned HSC saturation experiment.

Files:

- `scripts/ism/experiments/saijou_hsc/*`
- runbook or README.

This split lets framework review happen independently from biological
experiments. It also makes it possible to revert or iterate on the Saijou HSC
experiment without touching MREG or core mutation correctness.

## Immediate Next Implementation Choice

Do not rebase the old `ism` branch into this checkout. The safer path is:

1. Inspect old commits for clean cherry-pick candidates.
2. If clean, cherry-pick only generic files.
3. If commits are mixed, copy the old generic logic into the new module layout
   and rewrite imports/config boundaries manually.
4. Add tests before running large saturation jobs.

The first runnable target should be a tiny AlphaGenome-fine-tuned HSC
saturation smoke test on one gene and one readout. After that, scale to Mdk,
Col1a1, Acta2 and add Borzoi comparison.
