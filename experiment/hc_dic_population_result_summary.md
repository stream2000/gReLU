# HC-DIC population ISM result summary

Last updated: 2026-06-16

This note summarizes the existing multi-site HC-DIC population result from the
raw output tables. It is more result-focused than `experiment/hc_dic_runbook.md`.

## 1. Source artifacts

Primary report:

```text
outputs/dic_batch_analysis_hc/report.html
outputs/dic_batch_analysis_hc/report.pdf
```

Main tables used here:

```text
outputs/dic_batch_analysis_hc/provenance.json
outputs/dic_batch_analysis_hc/representation_comparison.tsv
outputs/dic_batch_analysis_hc/inference_shard_confound.tsv
outputs/dic_batch_analysis_hc/cross_representation_ari.tsv
outputs/dic_batch_analysis_hc/representations/real_local_1_4kb/feature_matrix.parquet
outputs/dic_batch_analysis_hc/representations/real_local_1_4kb/cluster_assignments.tsv
outputs/dic_batch_analysis_hc/representations/real_local_1_4kb/cluster_summary.tsv
outputs/dic_batch_analysis_hc/representations/real_local_1_4kb/cluster_top_features.tsv
outputs/dic_batch_analysis_hc/representations/real_local_1_4kb/k_selection.tsv
```

The population report was run as an HC-only analysis:

```text
site_classes: hc_dic
feature_dirs:
  outputs/dic_batch_1d_shard0
  outputs/dic_batch_1d_shard1
  outputs/dic_batch_1d_shard2
  outputs/dic_batch_1d_shard3
```

## 2. Experiment shape

Prepared HC-DIC mutation design:

| Item | Count |
|---|---:|
| Input HC-DIC sites | 142 |
| Ready HC-DIC sites | 134 |
| Skipped HC-DIC sites | 8 |
| Experimental CTCF motif edits | 134 |
| Local non-motif matched controls | 402 |
| Total HC-DIC mutation rows | 536 |
| Tracks | 15 |
| Summary windows | 1 kb, 4 kb, 20 kb, 100 kb |
| Inference shards | 4 |

Skipped HC-DIC sites:

- 4 sites: no CTCF motif above threshold.
- 4 sites: CTCF disruption design failed.

Inference provenance:

| Shard | Device | Sites in full prepared shard | Variants | Tracks | Runtime seconds |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 137 | 342 | 15 | 897.4 |
| 1 | 1 | 137 | 342 | 15 | 897.7 |
| 2 | 2 | 137 | 340 | 15 | 891.1 |
| 3 | 3 | 137 | 340 | 15 | 892.4 |

The four inference shards were filtered to `hc_dic` for the HC population
analysis. The broader prepared shard counts include LC-DIC rows because the
shared prepared manifest contains both HC and LC sites.

## 3. Clustering result

The most stable representation was:

```text
real_local_1_4kb
```

This representation uses local 1 kb and 4 kb features after CTCF motif
perturbation and matched-control adjustment.

Cluster selection:

| k | Silhouette | Stability ARI mean | Stability ARI SD | Minimum cluster size | Selected |
|---:|---:|---:|---:|---:|---|
| 2 | 0.347 | 0.936 | 0.047 | 50 | yes |
| 3 | 0.372 | 0.659 | 0.341 | 4 | no |
| 4 | 0.292 | 0.897 | 0.095 | 4 | no |

Final cluster sizes:

| Cluster | Sites | Short label |
|---:|---:|---|
| 0 | 84 | strong local response |
| 1 | 50 | weaker structural response |

The cluster is not explained by inference shard:

| Representation | Cluster vs shard ARI | Chi-square p | Max shard fraction within cluster |
|---|---:|---:|---:|
| real_local_1_4kb | 0.001 | 0.336 | 0.300 |

The same broad structure is also visible across other representations:

| Compared representations | Adjusted Rand index |
|---|---:|
| real_local_1_4kb vs real_strength_residual | 0.773 |
| real_signed_depletion vs real_local_1_4kb | 0.721 |
| real_biological_all_scales vs real_local_1_4kb | 0.696 |

## 4. Overall 4 kb local peak effects

Across all 134 ready HC-DICs, the direct local result is dominated by
CTCF/cohesin depletion. Values below are 4 kb peak log2FC after matched-control
adjustment.

| Track | Median | Q25 | Q75 | Sites < -1 | Sites < -0.5 | Sites < -0.1 |
|---|---:|---:|---:|---:|---:|---:|
| CTCF | -3.415 | -4.051 | -1.844 | 109 | 110 | 118 |
| RAD21 | -3.173 | -4.014 | -1.678 | 106 | 111 | 114 |
| SMC3 | -2.516 | -3.245 | -1.536 | 108 | 114 | 119 |
| POLR2A | -0.008 | -0.053 | 0.006 | 0 | 2 | 22 |
| AFF4 | -0.270 | -0.541 | -0.068 | 2 | 44 | 96 |
| BRD4 | -0.018 | -0.069 | 0.002 | 0 | 0 | 21 |
| MED1 | -0.399 | -0.824 | -0.033 | 20 | 59 | 92 |
| EP300 | -0.011 | -0.105 | 0.025 | 0 | 1 | 35 |
| FOXA1 | -0.104 | -0.482 | 0.004 | 10 | 32 | 67 |
| ESR1 | -0.260 | -0.552 | -0.000 | 1 | 41 | 82 |
| GATA3 | -0.051 | -0.258 | 0.020 | 0 | 13 | 52 |
| H3K27ac | -0.182 | -0.387 | -0.052 | 5 | 23 | 85 |
| H3K4me1 | -0.313 | -0.645 | -0.043 | 13 | 49 | 93 |
| H3K4me2 | -0.576 | -1.023 | -0.073 | 35 | 71 | 98 |
| H3K4me3 | -0.315 | -0.692 | -0.058 | 19 | 50 | 92 |

Main point:

- CTCF, RAD21 and SMC3 show large, widespread local depletion.
- POLR2A is effectively near zero as a population-level local peak effect.
- Active chromatin marks show modest depletion overall, especially H3K4me1/2/3,
  but the effect is much weaker than CTCF/cohesin.

## 5. Cluster-specific 4 kb local peak effects

Values below are median 4 kb peak log2FC by cluster.

| Track | Cluster 0 | Cluster 1 | Difference: cluster 0 - cluster 1 |
|---|---:|---:|---:|
| CTCF | -3.717 | -2.125 | -1.592 |
| RAD21 | -3.826 | -2.039 | -1.787 |
| SMC3 | -3.073 | -1.562 | -1.511 |
| POLR2A | -0.019 | 0.000 | -0.019 |
| AFF4 | -0.475 | -0.134 | -0.341 |
| BRD4 | -0.029 | -0.006 | -0.023 |
| MED1 | -0.689 | -0.054 | -0.635 |
| EP300 | -0.025 | 0.000 | -0.025 |
| FOXA1 | -0.178 | -0.005 | -0.174 |
| ESR1 | -0.451 | -0.016 | -0.435 |
| GATA3 | -0.075 | -0.013 | -0.063 |
| H3K27ac | -0.291 | -0.059 | -0.231 |
| H3K4me1 | -0.503 | -0.077 | -0.426 |
| H3K4me2 | -0.839 | -0.156 | -0.683 |
| H3K4me3 | -0.501 | -0.081 | -0.420 |

This shows that the two clusters should not be described as simply
`changed` versus `unchanged`.

More accurate description:

- Cluster 0: stronger local CTCF/cohesin depletion and broader active-chromatin
  co-depletion.
- Cluster 1: weaker but still visible CTCF/cohesin depletion, with active
  chromatin and POLR2A close to zero.

## 6. Threshold view of selected tracks

Number of sites below selected 4 kb peak log2FC thresholds:

| Track | Cluster | N | < -1 | < -0.5 | < -0.1 | Median |
|---|---:|---:|---:|---:|---:|---:|
| CTCF | 0 | 84 | 74 | 74 | 79 | -3.717 |
| CTCF | 1 | 50 | 35 | 36 | 39 | -2.125 |
| RAD21 | 0 | 84 | 74 | 77 | 79 | -3.826 |
| RAD21 | 1 | 50 | 32 | 34 | 35 | -2.039 |
| SMC3 | 0 | 84 | 76 | 77 | 80 | -3.073 |
| SMC3 | 1 | 50 | 32 | 37 | 39 | -1.562 |
| POLR2A | 0 | 84 | 0 | 2 | 20 | -0.019 |
| POLR2A | 1 | 50 | 0 | 0 | 2 | 0.000 |
| H3K27ac | 0 | 84 | 5 | 23 | 64 | -0.291 |
| H3K27ac | 1 | 50 | 0 | 0 | 21 | -0.059 |
| H3K4me1 | 0 | 84 | 13 | 42 | 69 | -0.503 |
| H3K4me1 | 1 | 50 | 0 | 7 | 24 | -0.077 |
| H3K4me2 | 0 | 84 | 32 | 61 | 72 | -0.839 |
| H3K4me2 | 1 | 50 | 3 | 10 | 26 | -0.156 |
| H3K4me3 | 0 | 84 | 18 | 42 | 68 | -0.501 |
| H3K4me3 | 1 | 50 | 1 | 8 | 24 | -0.081 |

This threshold view is useful because it shows the qualitative difference:

- For CTCF/RAD21/SMC3, both clusters contain many depleted sites.
- For POLR2A, almost no site reaches strong depletion in either cluster.
- For active histone marks, cluster 0 has many mild-to-moderate depletion
  events, while cluster 1 is mostly near zero.

## 7. Baseline signal covariates

The cluster separation is not strongly explained by simple input peak strength.

From `cluster_covariate_tests.tsv`:

| Covariate | Kruskal p | Eta squared |
|---|---:|---:|
| RAD21 Ctrl signal | 0.154 | 0.014 |
| CTCF Ctrl signal | 0.065 | 0.024 |
| POLR2A Ctrl signal | 0.147 | 0.167 |

Cluster baseline medians:

| Cluster | RAD21 Ctrl median | CTCF Ctrl median | POL2 Ctrl median |
|---:|---:|---:|---:|
| 0 | 14.50 | 11.61 | 3.91 |
| 1 | 12.47 | 10.78 | 7.03 |

The p values do not support a simple claim that clusters are just high-input
versus low-input CTCF/RAD21 sites. POL2 baseline is numerically higher in
cluster 1, but that does not translate into POLR2A depletion after CTCF motif
editing.

## 8. Biological interpretation

Current best interpretation:

1. HC-DIC CTCF motif disruption does what it should locally: it strongly
   reduces predicted CTCF and cohesin signals.
2. The population splits into two response modes, but both are structural
   response modes:
   - cluster 0: strong CTCF/cohesin depletion plus active-chromatin
     co-depletion;
   - cluster 1: weaker CTCF/cohesin depletion and little active-chromatin
     response.
3. POLR2A is not a major response axis in HC-DICs. The median local 4 kb
   POLR2A effect is close to zero overall and within both clusters.
4. This supports the role of HC-DICs as structural/cohesin controls in the
   current ISM framework.
5. The multi-site HC-DIC analysis does not test distal TSS effects. The current
   distal HC-DIC evidence comes from the corrected MREG single-locus experiment,
   where HC-DIC editing had essentially no MREG TSS response.

What should not be claimed:

- Do not say one HC-DIC cluster is completely unchanged after CTCF mutation.
  Cluster 1 still shows CTCF/RAD21/SMC3 depletion.
- Do not interpret the HC population result as a distal promoter result.
- Do not mix HC and LC population clusters as one biological subtype analysis,
  because HC and LC use different perturbation strategies.

## 9. One-sentence summary

After CTCF motif disruption at 134 HC-DICs, AlphaGenome predicts widespread
local loss of CTCF/cohesin signal; the main population split separates a strong
structural-plus-active-chromatin response group from a weaker structural-only
response group, while POLR2A remains close to zero in both groups.
