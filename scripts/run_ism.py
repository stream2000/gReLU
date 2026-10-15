"""Experiment: In Silico Mutagenesis (ISM).

Computes per-position log2FC specificity scores for a ±100 bp window around
the first exon of the target gene, then optionally renders sequence-logo plots.

Because this runs as its own process it always starts with a clean CUDA
context — no subprocess isolation is needed.

Usage:
    # Compute ISM for both models
    python -m scripts.run_ism --compute_ism both --gene SRSF11 --devices 0,1,2,3

    # Plot previously computed results
    python -m scripts.run_ism --plot_ism
"""
import argparse
import os
from dataclasses import dataclass
from typing import Any, Tuple

import pandas as pd
from matplotlib import pyplot as plt

import grelu.interpret
import grelu.visualize
from scripts.smoke_tests.config import ISM_RESULTS_DIR
from scripts.smoke_tests.setup import GreluTutorialApp
from scripts.smoke_tests.utils import _write_ism_report


# ── Experiment logic ──────────────────────────────────────────────────────────

@dataclass
class ISMConfig:
    """Configuration for the In Silico Mutagenesis (ISM) experiment."""
    gene: str
    devices: Any
    num_workers: int
    batch_size: int = 1
    compare_func: str = "log2FC"


class ISMExperiment:
    """Orchestrates In Silico Mutagenesis (ISM) evaluation.

    Encapsulates the experimental lifecycle into clear preprocessing,
    inference, and postprocessing stages.
    """

    def __init__(self, config: ISMConfig):
        self.config = config

    def preprocess(
        self,
        app: GreluTutorialApp,
        model_name: str
    ) -> Tuple[Any, Any, Any, int, int]:
        """Step 1: Preprocessing — Setup model, sequences, transform, and positions."""
        app.setup()
        app._cleanup()  # Ensure absolutely clean before starting

        if model_name == "borzoi":
            app.setup_borzoi()
            trans = app.get_borzoi_transform()
            model_obj = app.borzoi
            seqs_obj = app.input_seqs
            s_pos = app.ism_region['start'] - app.borzoi_start_coord
            e_pos = app.ism_region['end'] - app.borzoi_start_coord
        else:
            app.setup_ag_rna()
            trans = app.get_ag_transform(app.ag_rna)
            model_obj = app.ag_rna
            seqs_obj = app.ag_seqs
            s_pos = app.ism_region['start'] - app.ag_start_coord
            e_pos = app.ism_region['end'] - app.ag_start_coord

        return model_obj, seqs_obj, trans, s_pos, e_pos

    def inference(
        self,
        model_obj: Any,
        seqs_obj: Any,
        trans: Any,
        s_pos: int,
        e_pos: int
    ) -> pd.DataFrame:
        """Step 2: Inference — Run ISM prediction."""
        print(f"\n[Inference] Running {self.config.compare_func} ISM on {self.config.devices}...")
        res = grelu.interpret.score.ISM_predict(
            seqs=seqs_obj,
            model=model_obj,
            prediction_transform=trans,
            devices=self.config.devices,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            start_pos=s_pos,
            end_pos=e_pos,
            compare_func=self.config.compare_func,
            return_df=True
        )
        return res

    def postprocess(
        self,
        res: pd.DataFrame,
        model_name: str,
        output_dir: str = ISM_RESULTS_DIR
    ) -> None:
        """Step 3: Postprocessing — Save results and generate report."""
        print(f"[Postprocessing] Saving ISM results for {model_name}...")
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f"{model_name}_ism.csv")
        res.to_csv(csv_path)

        txt_path = os.path.join(output_dir, f"{model_name}_ism_report.txt")
        _write_ism_report(res, model_name, self.config.gene, str(self.config.devices), txt_path)

        print(f"  Results saved \u2192 {csv_path}")
        print(f"  Text report \u2192 {txt_path}")


def run_ism(app: GreluTutorialApp, model_name: str) -> None:
    """Main entry point for running ISM for one model."""
    config = ISMConfig(
        gene=app.gene,
        devices=app.devices,
        num_workers=app.num_workers
    )

    experiment = ISMExperiment(config)

    # 1. Preprocessing
    model_obj, seqs_obj, trans, s_pos, e_pos = experiment.preprocess(app, model_name)

    # 2. Inference
    res = experiment.inference(model_obj, seqs_obj, trans, s_pos, e_pos)

    # 3. Postprocessing
    experiment.postprocess(res, model_name)

    app._cleanup()


def plot_ism(gene: str) -> None:
    """Plot sequence logos from previously saved ISM CSVs."""
    for m in ["borzoi", "alphagenome"]:
        p = os.path.join(ISM_RESULTS_DIR, f"{m}_ism.csv")
        if os.path.exists(p):
            import pandas as pd
            df = pd.read_csv(p, index_col=0)
            df.columns = [str(c).split('.')[0] for c in df.columns]
            grelu.visualize.plot_ISM(df, method="logo")
            plt.title(f"{m} Brain/Liver Ratio ISM Logo")
            plt.savefig(f"{m}_ism_logo.png", dpi=150)
            plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ISM (In Silico Mutagenesis) for Borzoi and/or AlphaGenome."
    )
    parser.add_argument("--compute_ism", choices=["borzoi", "alphagenome", "both"],
                        help="Which model(s) to run ISM for.")
    parser.add_argument("--plot_ism", action="store_true",
                        help="Plot sequence logos from previously saved ISM CSVs.")
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

    if args.compute_ism:
        if args.compute_ism in ["borzoi", "both"]:
            run_ism(app, "borzoi")
        if args.compute_ism in ["alphagenome", "both"]:
            run_ism(app, "alphagenome")

    if args.plot_ism:
        plot_ism(args.gene)
