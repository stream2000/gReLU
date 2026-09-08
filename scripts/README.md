# Borzoi vs AlphaGenome Benchmarks

This directory contains a suite of benchmarking scripts used to compare **Borzoi** and **AlphaGenome** models across various genomic tasks, including expression profile inference, variant effect prediction, and genome-wide accuracy.

## Experiments

### 1. Expression Inference Comparison
Runs expression-profile inference for a given gene and compares the two models. It produces CAGE track plots and a brain-vs-liver specificity bar chart.

*   **Script:** `run_inference.py`
*   **Inputs:**
    *   `--gene`: Target gene name (e.g., `SRSF11`).
    *   Reference genome (HG38) and model weights (auto-loaded).
*   **Outputs:**
    *   `comparison_cage.png`: Multi-panel plot comparing CAGE signal tracks.
    *   `comparison_specificity.png`: Bar chart of brain-vs-liver expression specificity ratios.

### 2. In Silico Mutagenesis (ISM)
Computes per-position log2FC specificity scores for a window around the first exon of a target gene. It identifies critical regulatory motifs and can optionally render sequence-logo plots.

*   **Script:** `run_ism.py`
*   **Inputs:**
    *   `--compute_ism`: Model selection (`borzoi`, `alphagenome`, or `both`).
    *   `--gene`: Target gene name for coordinates.
*   **Outputs:**
    *   `ism_results/{model}_ism.csv`: Raw ISM score matrix (4 bases × positions).
    *   `ism_results/{model}_ism_report.txt`: Summary report with top-scoring positions and base-wise statistics.
    *   `{model}_ism_logo.png`: Sequence logo visualization of the ISM scores.

### 3. Genome-wide Evaluation
Performs a sliding-window evaluation across held-out chromosomes. It calculates the Pearson R correlation between model predictions and experimental BigWig signals.

*   **Script:** `run_eval.py`
*   **Inputs:**
    *   `--eval_bigwig`: Path to the experimental BigWig signal file.
    *   `--eval_track_idx`: The specific model output track index to evaluate.
    *   `--eval_chroms`: List of chromosomes (e.g., `chr1 chr8 chr21`).
*   **Outputs:**
    *   `eval_results.json`: Summary JSON containing mean/median Pearson R for all windows and signal-only windows, stratified by chromosome.
    *   `--eval_save_per_window`: (Optional) Detailed CSV table with Pearson R scores for every sliding window.

### 4. eQTL Fine-mapping Evaluation
Ranks variants in GTEx credible sets by their predicted absolute log2FC scores and computes the Area Under the Precision-Recall Curve (AUPRC) using PIP (Posterior Inclusion Probability) as the causal label.

*   **Script:** `run_eqtl.py`
*   **Inputs:**
    *   `--eqtl_cs_files`: Path to eQTL Catalogue/GTEx credible set TSV files.
    *   `--eqtl_tissue`: Tissue keyword for track matching (e.g., `brain`, `liver`).
*   **Outputs:**
    *   `ism_results/eqtl_auprc_variants.csv`: Scores and PIPs for all evaluated variants.
    *   `ism_results/eqtl_auprc_per_locus.csv`: AUPRC metrics calculated for each independent locus.
    *   `ism_results/eqtl_auprc_report.txt`: Final summary report with mean/median AUPRC per model against a random baseline.

---

## Infrastructure and Configuration

The benchmark suite is designed for modularity and isolation:

*   **`smoke_tests/setup.py`**: Contains the `GreluTutorialApp` class, which centralizes model loading, sequence formatting (handling the different receptive fields of Borzoi vs AlphaGenome), and GPU memory management.
*   **`smoke_tests/config.py`**: The single source of truth for model architectures (input lengths, bin sizes), cache paths, and experiment constants.
*   **Subprocess Isolation**: Scripts like `run_eqtl.py` use subprocesses to ensure that heavy models are fully cleared from GPU memory between experimental runs.

## Prerequisites

- **GPU Acceleration**: Most experiments require GPUs. Use the `--devices` flag to specify indices (e.g., `0,1,2,3`) or `cpu`.
- **Data Access**: 
    - Genome-wide evaluation requires BigWig files.
    - eQTL evaluation requires GTEx credible set TSVs (by default searched in `~/.cache/eqtl_finemapping/`).
    - Model weights for Borzoi (HuggingFace) and AlphaGenome must be accessible.
