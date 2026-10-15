# Cell-specificity output audit (experimental)

This analysis asks whether the four Saijou output tracks preserve the
cell-to-cell contrasts present in held-out labels. It is an output audit only:
it cannot by itself establish catastrophic forgetting or collapse of internal
embeddings. High pairwise correlation is not sufficient evidence of collapse;
it is interpreted alongside the observed baseline, train-defined signal masks,
specificity gaps, calibration, and top-cell identity.

The CLI records only `window_idx` and `bin_idx`. It does not create, transform,
annotate, or plot genomic coordinates.

## Metric contract

For a non-negative four-track vector `x`, with `p = x / sum(x)`:

- normalized entropy specificity is `1 - H(p) / log(4)`;
- centered L2 contrast/total is
  `sqrt(sum_i (x_i - mean(x))^2) / sum_i x_i`;
- top-vs-second margin is
  `log2((top(x) + eps) / (second(x) + eps))`, where `eps` is the fixed
  1% quantile of positive **train-label** values.

Zero vectors have entropy specificity and centered-L2 contrast equal to zero.
Ties have zero top-vs-second margin. Observed top ties are excluded from
top-cell accuracy/recall and counted; predicted ties use stable `argmax` and
are also counted.

Masks are defined only from train labels, then applied unchanged to validation:
all finite bins (background-sensitive diagnostic), observed-active bins,
train total P90/P95/P99, and the union of per-track train P95 masks. The primary
interpretation belongs to active/high-signal masks because the label matrix is
sparse. Mean specificity gaps use whole validation windows as bootstrap
clusters.

The compact stratified audit additionally defines a train-only P90 peak
candidate mask. Candidates with all four tracks above their train P95 are
`shared_peak`; candidates with one or two tracks above their train P95 are
`cell_specific_peak`; the remaining three-of-four candidates are retained as
an explicit `other_peak_0or3of4` boundary stratum rather than silently
discarded. For a negative control, the
validation split globally samples the same number of all-zero bins as the two
primary positive strata when available, using a fixed seed and without looking
at predictions. These selected rows are saved in
`stratified_bins.tsv.gz`; the full bin table remains in `bin_metrics.tsv.gz`.

## Commands

```bash
source activate.sh
CUDA_VISIBLE_DEVICES=3 python \
  scripts/ism/experiments/saijou_hsc/experimental/cell_specificity_audit/run_cell_specificity_audit.py \
  --model borzoi_lora_e39 --max-windows 8 --device cuda:0

# 128-bp analysis from an existing 32-bp Borzoi prediction cache:
python scripts/ism/experiments/saijou_hsc/experimental/cell_specificity_audit/run_cell_specificity_audit.py \
  --model borzoi_lora_e39 --analysis-bin-size 128 \
  --prediction-cache-dir experiments/validation/cell_specificity_audit/borzoi_lora_e39/full_val

python scripts/ism/experiments/saijou_hsc/experimental/cell_specificity_audit/summarize_existing_controls.py
```

The saved-control command is a nine-locus full-label-window mean sensitivity
analysis. It is not a bin-level or full-validation substitute.

The Borzoi e39 checkpoint emits 32-bp bins. The 128-bp command sums each
contiguous group of four model/label bins and derives all thresholds from the
matching 128-bp training cache; it does not introduce genomic coordinates.
