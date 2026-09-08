# Saijou analysis harness

This document is the entry contract for new analyses in this directory. Read it
together with `WORKFLOW_INDEX.md` before adding an entry point.

## Maintained baseline

The current maintained biological baseline is:

1. the canonical nine-gene TSS-proximal 10-bp ISM workflow; and
2. the focused, audited `Mdk` analysis built from the same run artifacts.

Liver-fibrosis hypothesis scripts are experimental progress, not part of this
baseline. Experimental code belongs under `experimental/<topic>/` and must not
be imported by canonical or retained workflows until it passes the promotion
gates below.

## Layer contract

```text
model inference artifacts (parquet + manifests)
  -> reusable analysis functions in tools/
  -> versioned analysis tables and validation_summary.json
  -> report-only renderer and external templates
  -> ignored HTML/PDF/figure deliverables
```

- Inference entry points may write model outputs but must not contain report prose.
- Analysis entry points may read raw model artifacts. Their durable interface is
  TSV/JSON plus an explicit validation summary.
- Report entry points must read the analysis interface, not raw parquet. Narrative,
  table layout, HTML and CSS belong to the report layer.
- Shared scientific definitions belong in `tools/`, currently
  `tools/effect_summary.py`; path, schema and validation helpers belong in
  `tools/harness.py`. Sequence-editing primitives a second experiment would call
  unchanged belong in `grelu.interpret.ism`, not in `tools/`: `tools/` is shared
  within this experiment, core is shared across experiments.
- Generated experiment outputs stay under `experiments/ism/` and are ignored by
  Git. Commit code, tests, templates and concise contracts, not regenerated data.

## Starting a new analysis

1. Confirm the question is not already covered by `WORKFLOW_INDEX.md`.
2. Reuse the canonical run artifacts and shared effect definitions where possible.
3. Put an unvalidated prototype under `experimental/<topic>/`.
4. Define the output table keys, units, direction and readout-selection rule before
   writing a report.
5. Add schema, uniqueness, finiteness and expected-replicate checks.
6. Add synthetic unit tests for shared calculations and one end-to-end validation
   against saved artifacts.
7. Only then promote the reusable functions or entry point into `tools/` and add it
   to `WORKFLOW_INDEX.md`.

Promotion does not require a biological result to match an expectation. Structural
invariants should fail hard; changed biological values should be surfaced in the
validation summary and interpreted conservatively rather than encoded as universal
runtime assertions.

## Current commands

Canonical nine-gene report data and artifact:

```bash
source activate.sh
python scripts/ism/experiments/saijou_hsc/build_saijou_all_genes_10bp_report.py
```

Four cell-specific nine-gene Browser PDFs:

```bash
source activate.sh
python scripts/ism/experiments/saijou_hsc/tools/build_nine_gene_browser_report.py \
  --root experiments/ism/<run> \
  --half-window-bp 1500
```

Focused `Mdk` analysis and report:

```bash
source activate.sh
python scripts/ism/experiments/saijou_hsc/tools/mdk/analyze_mdk_audited_effects.py
python scripts/ism/experiments/saijou_hsc/tools/mdk/build_mdk_audited_report.py
```

Convert the resulting HTML to PDF only after the analysis validation is `ok`.
Keep the canonical PDF with the experiment and copy one convenience copy to
`~/report`, as specified in the repository `AGENTS.md`.
