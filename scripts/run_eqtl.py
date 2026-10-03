"""Experiment: eQTL fine-mapping AUPRC evaluation.

Ranks variants in GTEx credible sets by model |log2FC| scores and computes
AUPRC using PIP >= threshold as the causal label.  Each model is scored in a
separate subprocess so CUDA memory is fully released between models.

This script doubles as the subprocess entry point (--_eqtl_score_one) called
by run_eqtl_auprc for per-model scoring.

Normal usage:
    python -m scripts.run_eqtl \
        --eqtl_auprc \
        --eqtl_tissue brain \
        --eqtl_models borzoi alphagenome \
        --eqtl_output_prefix ism_results/eqtl_auprc

The credible-set files default to all *.tsv.gz files found in
~/.cache/eqtl_finemapping/gtex_v8_susie_ge/.
"""
import argparse
import datetime
import glob as _glob
import os
import sys
import tempfile
import subprocess as _sp
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import grelu.io
import grelu.resources
from grelu.lightning import LightningModel
from alphagenome_pytorch.config import DtypePolicy

from scripts.smoke_tests.config import (
    WEIGHTS_PATH, AG_META_PATH, BORZOI_INPUT_LEN, AG_INPUT_LEN, AG_BIN_SIZE, ISM_RESULTS_DIR,
)
from scripts.smoke_tests.setup import GreluTutorialApp
from scripts.smoke_tests.utils import _parse_variant


# ── Experiment logic ──────────────────────────────────────────────────────────

@dataclass
class EQTLConfig:
    """Configuration for the eQTL fine-mapping AUPRC evaluation."""
    tissue: str
    min_pip_causal: float
    min_cs_size: int
    max_loci: int
    output_prefix: str
    devices: Any
    num_workers: int


class EQTLExperiment:
    """Orchestrates eQTL fine-mapping AUPRC evaluation.

    Encapsulates the experimental lifecycle into clear preprocessing,
    inference (via isolated subprocesses), and postprocessing stages.
    """

    def __init__(self, config: EQTLConfig):
        self.config = config

    def preprocess(self, cs_files: List[str]) -> pd.DataFrame:
        """Step 1: Preprocessing — Load, filter, and parse credible sets."""
        print("\n[Preprocessing] Loading and filtering credible sets ...")

        dfs = [pd.read_csv(f, sep="\t", compression="gzip") for f in cs_files]
        cs = pd.concat(dfs, ignore_index=True)
        print(f"  Total variants loaded : {len(cs)}")

        # Filter credible sets by max PIP and size
        cs_max = cs.groupby(["gene_id", "cs_id"])["pip"].max().reset_index()
        cs_max.columns = ["gene_id", "cs_id", "max_pip"]
        cs_size = cs.groupby(["gene_id", "cs_id"])["pip"].count().reset_index()
        cs_size.columns = ["gene_id", "cs_id", "cs_size"]
        cs_info = cs_max.merge(cs_size)

        good = cs_info[
            (cs_info.max_pip >= self.config.min_pip_causal) &
            (cs_info.cs_size >= self.config.min_cs_size)
        ]
        print(f"  Loci after filtering  : {len(good)} "
              f"(max_pip>={self.config.min_pip_causal}, cs_size>={self.config.min_cs_size})")

        if len(good) > self.config.max_loci:
            good = good.sample(self.config.max_loci, random_state=42)
            print(f"  Sub-sampled to        : {self.config.max_loci} loci")

        cs_filtered = cs.merge(good[["gene_id", "cs_id"]], on=["gene_id", "cs_id"])
        print(f"  Variants to score     : {len(cs_filtered)}")

        # Parse variant format chr1_70355119_G_A \u2192 chrom/pos/ref/alt
        parsed = cs_filtered["variant"].apply(_parse_variant)
        variants_df = pd.DataFrame(
            list(parsed), columns=["chrom", "pos", "ref", "alt"]
        )
        variants_df["gene_id"] = cs_filtered["gene_id"].values
        variants_df["cs_id"] = cs_filtered["cs_id"].values
        variants_df["pip"] = cs_filtered["pip"].values
        variants_df["causal"] = (variants_df["pip"] >= self.config.min_pip_causal).astype(int)

        # Only keep SNPs (both ref/alt are single bases)
        snp_mask = variants_df["ref"].str.len().eq(1) & variants_df["alt"].str.len().eq(1)
        variants_df = variants_df[snp_mask].reset_index(drop=True)
        print(f"  SNPs only             : {len(variants_df)} "
              f"({snp_mask.mean() * 100:.1f}% of variants)")

        # Filter variants near chromosome ends
        chrom_sizes = grelu.io.genome.read_sizes("hg38").set_index("chrom")["size"].to_dict()
        half = BORZOI_INPUT_LEN // 2
        edge_mask = variants_df.apply(
            lambda r: (r.pos - half >= 0) and
                      (r.pos + half <= chrom_sizes.get(r.chrom, 0)),
            axis=1,
        )
        n_before = len(variants_df)
        variants_df = variants_df[edge_mask].reset_index(drop=True)
        print(f"  After edge filter     : {len(variants_df)} "
              f"(removed {n_before - len(variants_df)} near chrom ends)")

        return variants_df

    def run_inference(
        self,
        variants_df: pd.DataFrame,
        model_names: List[str],
        subprocess_script: str
    ) -> Dict[str, np.ndarray]:
        """Step 2: Inference — Score model by model via isolated subprocesses."""
        results = {}
        for mname in model_names:
            print(f"\n[Inference] Scoring {mname.upper()} in subprocess ({len(variants_df)} SNPs) \u2026")
            tmp_variants = tempfile.mktemp(suffix=".csv")
            tmp_scores = tempfile.mktemp(suffix=".npy")
            variants_df.to_csv(tmp_variants, index=False)

            devices_str = (
                ",".join(str(d) for d in self.config.devices)
                if isinstance(self.config.devices, list) else str(self.config.devices)
            )

            cmd = [
                sys.executable, subprocess_script,
                "--_eqtl_score_one",
                "--_eqtl_score_model", mname,
                "--_eqtl_score_variants", tmp_variants,
                "--_eqtl_score_output", tmp_scores,
                "--_eqtl_score_tissue", self.config.tissue,
                "--devices", devices_str,
                "--num_workers", str(self.config.num_workers),
            ]
            _sp.run(cmd, check=True)
            scores = np.load(tmp_scores)
            results[mname] = scores[: len(variants_df)]  # trim padding if any
            os.unlink(tmp_variants)
            os.unlink(tmp_scores)
        return results

    def postprocess(
        self,
        variants_df: pd.DataFrame,
        results: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Step 3: Postprocessing — Compute metrics and generate reports."""
        from sklearn.metrics import average_precision_score
        print("\n[Postprocessing] Computing per-locus AUPRC \u2026")

        for mname, scores in results.items():
            variants_df[f"score_{mname}"] = scores

        auprc_rows = []
        for (gene_id, cs_id), grp in variants_df.groupby(["gene_id", "cs_id"]):
            row = {"gene_id": gene_id, "cs_id": cs_id,
                   "cs_size": len(grp), "n_causal": grp["causal"].sum()}
            if row["n_causal"] == 0 or row["n_causal"] == row["cs_size"]:
                continue  # Cannot calculate AUPRC
            for mname in results:
                auprc = average_precision_score(grp["causal"], grp[f"score_{mname}"])
                row[f"auprc_{mname}"] = auprc
            auprc_rows.append(row)

        auprc_df = pd.DataFrame(auprc_rows)

        # Save and Report
        self._save_results(variants_df, auprc_df, results)
        return auprc_df

    def _save_results(self, variants_df: pd.DataFrame, auprc_df: pd.DataFrame, results: Dict):
        os.makedirs(ISM_RESULTS_DIR, exist_ok=True)
        csv_variants = os.path.join(ISM_RESULTS_DIR, f"{self.config.output_prefix}_variants.csv")
        csv_auprc = os.path.join(ISM_RESULTS_DIR, f"{self.config.output_prefix}_per_locus.csv")
        txt_report = os.path.join(ISM_RESULTS_DIR, f"{self.config.output_prefix}_report.txt")

        variants_df.to_csv(csv_variants, index=False)
        auprc_df.to_csv(csv_auprc, index=False)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(txt_report, "w") as f:
            f.write(f"eQTL AUPRC Evaluation Report\n")
            f.write("=" * 68 + "\n")
            f.write(f"Run Time    : {ts}\n")
            f.write(f"Tissue      : {self.config.tissue}\n")
            f.write(f"Causal Threshold : PIP >= {self.config.min_pip_causal}\n")
            f.write(f"Total Loci  : {len(auprc_df)}\n")
            f.write(f"Total Variants : {len(variants_df)}\n\n")
            f.write("── Model AUPRC Summary ──────────────────────────────────────────\n")

            baseline = auprc_df["n_causal"].sum() / auprc_df["cs_size"].sum()
            f.write(f"  Random Baseline AUPRC          : {baseline:.4f}\n")

            for mname in results:
                col = f"auprc_{mname}"
                if col in auprc_df.columns:
                    mean_a = auprc_df[col].mean()
                    med_a = auprc_df[col].median()
                    f.write(f"\n  {mname.upper():15s}\n")
                    f.write(f"    mean AUPRC  : {mean_a:.4f}\n")
                    f.write(f"    median AUPRC: {med_a:.4f}\n")
                    print(f"  {mname.upper():15s}  mean_AUPRC={mean_a:.4f}  median={med_a:.4f}")

        print(f"\n  Report \u2192 {txt_report}")


def run_eqtl_auprc(
        app: GreluTutorialApp,
        cs_files: list,
        tissue: str = "brain",
        min_pip_causal: float = 0.5,
        min_cs_size: int = 5,
        max_loci: int = 200,
        model_names: list = None,
        output_prefix: str = "eqtl_auprc",
        _subprocess_script: str | None = None,
) -> pd.DataFrame:
    """Evaluate AUPRC of variant effect scores against eQTL fine-mapping PIPs."""
     
    if model_names is None:
        model_names = ["borzoi", "alphagenome"]
    if _subprocess_script is None:
        raise ValueError("run_eqtl_auprc requires _subprocess_script= (path to run_eqtl.py)")

    config = EQTLConfig(
        tissue=tissue,
        min_pip_causal=min_pip_causal,
        min_cs_size=min_cs_size,
        max_loci=max_loci,
        output_prefix=output_prefix,
        devices=app.devices,
        num_workers=app.num_workers
    )

    experiment = EQTLExperiment(config)

    # 1. Preprocessing
    variants_df = experiment.preprocess(cs_files)

    # 2. Inference (Subprocess isolation)
    results = experiment.run_inference(variants_df, model_names, _subprocess_script)

    # 3. Postprocessing
    return experiment.postprocess(variants_df, results)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="eQTL fine-mapping AUPRC evaluation (Borzoi vs AlphaGenome)."
    )

    # ── Normal mode ──────────────────────────────────────────────────────────
    parser.add_argument("--eqtl_auprc", action="store_true",
                        help="Run eQTL fine-mapping AUPRC evaluation.")
    parser.add_argument("--eqtl_cs_files", nargs="+", metavar="PATH",
                        help="eQTL Catalogue credible_sets.tsv.gz file(s).")
    parser.add_argument("--eqtl_tissue", default="brain",
                        help="Tissue keyword for track selection (default: brain).")
    parser.add_argument("--eqtl_min_pip", type=float, default=0.5,
                        help="PIP threshold for causal label (default: 0.5).")
    parser.add_argument("--eqtl_max_loci", type=int, default=200,
                        help="Max loci to evaluate (default: 200).")
    parser.add_argument("--eqtl_models", nargs="+", default=["borzoi", "alphagenome"],
                        choices=["borzoi", "alphagenome"],
                        help="Models to evaluate (default: both).")
    parser.add_argument("--eqtl_output_prefix", default="eqtl_auprc",
                        help="Output file prefix (default: eqtl_auprc).")

    # ── Subprocess mode (called internally, not for direct use) ──────────────
    parser.add_argument("--_eqtl_score_one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_eqtl_score_model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_eqtl_score_variants", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_eqtl_score_output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_eqtl_score_tissue", default="brain", help=argparse.SUPPRESS)

    # ── Shared ────────────────────────────────────────────────────────────────
    parser.add_argument("--devices", default="0,1,2,3",
                        help="Comma-separated GPU indices, or 'cpu' (default: 0,1,2,3).")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="DataLoader worker count (default: 1).")
    args = parser.parse_args()

    # ── Subprocess: score one model, save scores, exit ───────────────────────
    if args._eqtl_score_one:
        from grelu.variant import predict_variant_effects
        from grelu.transforms.prediction_transforms import Aggregate

        mname = args._eqtl_score_model
        tissue = args._eqtl_score_tissue
        vdf = pd.read_csv(args._eqtl_score_variants)
        devices = [int(x) for x in args.devices.split(",")]

        if mname == "borzoi":
            print("[subprocess] Loading Borzoi \u2026")
            model_obj = grelu.resources.load_model(
                repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt"
            )
            tasks_df = pd.DataFrame(model_obj.data_params["tasks"])
            brain_idx = tasks_df[
                (tasks_df.assay == "RNA") &
                tasks_df["sample"].str.contains(tissue, case=False, na=False)
            ].index.tolist()
            bs = 2
        else:
            print("[subprocess] Loading AlphaGenome RNA \u2026")
            ag_params = dict(
                weights_path=WEIGHTS_PATH,
                dtype_policy=DtypePolicy.mixed_precision(),
                resolution=128,
            )
            model_obj = LightningModel(
                model_params={"model_type": "AlphaGenomeModel",
                              "output_key": "rna_seq", **ag_params},
                train_params={"task": "regression", "loss": "mse"},
            )
            model_obj.data_params["train"] = {"seq_len": AG_INPUT_LEN, "bin_size": AG_BIN_SIZE}
            model_obj.model_params["crop_len"] = 0
            ag_meta = pd.read_parquet(AG_META_PATH)
            rna_meta = ag_meta[ag_meta.output_type == "rna_seq"]
            brain_idx = rna_meta[
                rna_meta.biosample_name.str.contains(tissue, case=False, na=False)
            ].track_index.tolist()
            bs = 4

        print(f"[subprocess] {mname} {tissue} tracks: {len(brain_idx)}, "
              f"variants: {len(vdf)}, devices: {devices}, batch_size: {bs}")
        transform = Aggregate(tasks=brain_idx, length_aggfunc="mean", task_aggfunc="mean")
        odds = predict_variant_effects(
            variants=vdf,
            model=model_obj,
            devices=devices,
            num_workers=args.num_workers,
            batch_size=bs,
            genome="hg38",
            compare_func="log2FC",
            return_ad=False,
            prediction_transform=transform,
        )
        scores = np.abs(odds.squeeze())
        np.save(args._eqtl_score_output, scores)
        print(f"[subprocess] Scores saved \u2192 {args._eqtl_score_output}  shape={scores.shape}")
        sys.exit(0)

    # ── Normal mode ───────────────────────────────────────────────────────────
    if args.eqtl_auprc:
        cs_files = args.eqtl_cs_files or sorted(
            _glob.glob(os.path.expanduser(
                "~/.cache/eqtl_finemapping/gtex_v8_susie_ge/*.tsv.gz"
            ))
        )
        if not cs_files:
            parser.error(
                "--eqtl_auprc: no credible set files found. "
                "Use --eqtl_cs_files or download data first."
            )
        print(f"[eQTL-AUPRC] Using {len(cs_files)} credible set file(s)")

        app = GreluTutorialApp(devices=args.devices, num_workers=args.num_workers)
        run_eqtl_auprc(
            app=app,
            cs_files=cs_files,
            tissue=args.eqtl_tissue,
            min_pip_causal=args.eqtl_min_pip,
            max_loci=args.eqtl_max_loci,
            model_names=args.eqtl_models,
            output_prefix=args.eqtl_output_prefix,
            _subprocess_script=__file__,
        )
