"""Experiment: Inference comparison (Borzoi vs AlphaGenome).

Runs expression-profile inference for a given gene, produces CAGE track
plots and a brain-vs-liver specificity bar chart.

Usage:
    python -m scripts.run_inference --gene SRSF11 --devices 0,1,2,3
"""
import argparse
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
from matplotlib import pyplot as plt

from scripts.smoke_tests.config import AG_META_PATH
from scripts.smoke_tests.setup import GreluTutorialApp


# ── Experiment logic ──────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    """Configuration for the inference comparison experiment."""
    gene: str
    devices: Any
    inference_device: Any
    num_workers: int


class InferenceExperiment:
    """Orchestrates comparison inference between Borzoi and AlphaGenome.

    Encapsulates the experimental lifecycle into clear preprocessing,
    model-specific inference, and postprocessing (plotting) stages.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config

    def preprocess(self, app: GreluTutorialApp):
        """Step 1: Preprocessing — Setup sequences for the target gene."""
        print(f"[Preprocessing] Setting up sequences for gene {self.config.gene}...")
        app.setup()
        app._cleanup()

    def run_borzoi_inference(self, app: GreluTutorialApp) -> Dict[str, Any]:
        """Run Borzoi specific inference and data collection."""
        app.setup_borzoi()
        print(f"\n[Inference] Running Borzoi inference on {self.config.inference_device}...")

        borzoi_preds = app.borzoi.predict_on_seqs(app.input_seqs, device=self.config.inference_device)
        b_trans = app.get_borzoi_transform()
        borzoi_specificity = float(b_trans.compute(borzoi_preds).ravel()[0])

        tasks_b = pd.DataFrame(app.borzoi.data_params["tasks"])
        # Cap analysis gene expression
        borzoi_cage_idx = tasks_b[
            (tasks_b.assay == "CAGE") & tasks_b["sample"].str.contains("brain", case=False, na=False)].head(2).index.tolist()
        borzoi_rna_brain_idx = tasks_b[
            (tasks_b.assay == "RNA") & tasks_b["sample"].str.contains("brain", case=False, na=False)].index.tolist()

        borzoi_cage_preds = borzoi_preds[0, borzoi_cage_idx, :]
        borzoi_rna_brain_preds = borzoi_preds[0, borzoi_rna_brain_idx, :]

        data = {
            "cage": borzoi_cage_preds,
            "rna": borzoi_rna_brain_preds,
            "spec": borzoi_specificity,
            "cage_names": (tasks_b.name[borzoi_cage_idx] + " " + tasks_b.description[borzoi_cage_idx]).tolist(),
            "rna_names": (tasks_b.name[borzoi_rna_brain_idx] + " " + tasks_b.description[borzoi_rna_brain_idx]).tolist()
        }
        app._cleanup()
        return data

    def run_ag_rna_inference(self, app: GreluTutorialApp) -> Dict[str, Any]:
        """Run AlphaGenome RNA specific inference and data collection."""
        app.setup_ag_rna()
        print(f"\n[Inference] Running AlphaGenome RNA inference on {self.config.inference_device}...")

        ag_rna_preds = app.ag_rna.predict_on_seqs(app.ag_seqs, device=self.config.inference_device)
        ag_trans = app.get_ag_transform(app.ag_rna)
        ag_spec = float(ag_trans.compute(ag_rna_preds).ravel()[0])

        ag_meta = pd.read_parquet(AG_META_PATH)
        rna_meta_h = ag_meta[ag_meta.output_type == "rna_seq"]
        brain_rna_idx = rna_meta_h[
            rna_meta_h.biosample_name.str.contains("brain", case=False, na=False)].track_index.tolist()
        ag_rna_brain_preds_vals = ag_rna_preds[0, brain_rna_idx, :]

        data = {
            "rna": ag_rna_brain_preds_vals,
            "spec": ag_spec,
            "rna_names": rna_meta_h[
                rna_meta_h.biosample_name.str.contains("brain", case=False, na=False)].track_name.tolist()
        }
        app._cleanup()
        return data

    def run_ag_cage_inference(self, app: GreluTutorialApp) -> Dict[str, Any]:
        """Run AlphaGenome CAGE specific inference and data collection."""
        app.setup_ag_cage()
        print(f"\n[Inference] Running AlphaGenome CAGE inference on {self.config.inference_device}...")

        ag_cage_preds_full = app.ag_cage.predict_on_seqs(app.ag_seqs, device=self.config.inference_device)
        ag_meta = pd.read_parquet(AG_META_PATH)
        cage_meta_h = ag_meta[ag_meta.output_type == "cage"]

        brain_cage_idx = cage_meta_h[cage_meta_h.biosample_name.str.contains("brain", case=False,
                                                                             na=False) & ~cage_meta_h.biosample_name.str.contains(
            "vasculature", case=False, na=False)].head(2).track_index.tolist()
        ag_cage_preds_vals = ag_cage_preds_full[0, brain_cage_idx, :]

        data = {
            "cage": ag_cage_preds_vals,
            "cage_names": cage_meta_h[cage_meta_h.biosample_name.str.contains("brain", case=False,
                                                                              na=False) & ~cage_meta_h.biosample_name.str.contains(
                "vasculature", case=False, na=False)].head(2).track_name.tolist()
        }
        app._cleanup()
        return data

    def postprocess(
        self,
        borzoi_data: Dict,
        ag_rna_data: Dict,
        ag_cage_data: Dict,
    ) -> None:
        """Step 3: Postprocessing — Render comparison plots."""
        print("\n[Postprocessing] Generating Comparison Plots...")

        # 1. CAGE Comparison Plot
        fig, axes = plt.subplots(2, 2, figsize=(18, 6), constrained_layout=True)
        for row in range(2):
            # Borzoi
            axes[row, 0].fill_between(range(borzoi_data["cage"].shape[1]), borzoi_data["cage"][row],
                                      color="steelblue", alpha=0.7)
            axes[row, 0].set_title(f"Borzoi — {borzoi_data['cage_names'][row]}", fontsize=8)

            # AlphaGenome
            axes[row, 1].fill_between(range(ag_cage_data["cage"].shape[1]),
                                      ag_cage_data["cage"][row], color="darkorange", alpha=0.7)
            axes[row, 1].set_title(f"AlphaGenome — {ag_cage_data['cage_names'][row]}", fontsize=8)

        plt.savefig("comparison_cage.png", dpi=150)
        plt.close()

        # 2. Specificity Plot
        scores = {"Borzoi": borzoi_data["spec"], "AlphaGenome": ag_rna_data["spec"]}
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(list(scores.keys()), list(scores.values()), color=["steelblue", "darkorange"], edgecolor="black",
               width=0.5)
        ax.axhline(y=1.0, color="gray", linestyle="--")
        ax.set_title(f"{self.config.gene} Brain-vs-Liver Specificity Ratio")
        plt.savefig("comparison_specificity.png", dpi=150)
        plt.close()

        print("Done. Inference results saved \u2192 comparison_cage.png and comparison_specificity.png")


def run_inference(app: GreluTutorialApp) -> None:
    """Main entry point for inference comparison."""
    config = InferenceConfig(
        gene=app.gene,
        devices=app.devices,
        inference_device=app.inference_device,
        num_workers=app.num_workers
    )

    experiment = InferenceExperiment(config)

    # 1. Preprocessing
    experiment.preprocess(app)

    # 2. Inference
    borzoi_data = experiment.run_borzoi_inference(app)
    ag_rna_data = experiment.run_ag_rna_inference(app)
    ag_cage_data = experiment.run_ag_cage_inference(app)

    # 3. Postprocessing
    experiment.postprocess(borzoi_data, ag_rna_data, ag_cage_data)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Borzoi / AlphaGenome inference and generate comparison plots."
    )
    parser.add_argument("--gene", default="SRSF11",
                        help="Gene name to centre analysis on (default: SRSF11).")
    parser.add_argument("--devices", default="0,1,2,3",
                        help="Comma-separated GPU indices, or 'cpu' (default: 0,1,2,3).")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="DataLoader worker count (default: 1).")
    args = parser.parse_args()

    app = GreluTutorialApp(
        gene=args.gene,
        devices=args.devices,
        num_workers=args.num_workers,
    )
    run_inference(app)
