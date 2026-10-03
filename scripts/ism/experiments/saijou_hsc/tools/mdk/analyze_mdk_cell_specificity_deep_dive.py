#!/usr/bin/env python
"""Build audit tables for the focused Mdk cell-specificity deep dive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_targeted_original_comparison"
CELLS = ["hsc", "mac", "lsec", "chol"]
FINAL_RUNS = {
    "AlphaGenome e19": "runs/alphagenome_finetuned",
    "Borzoi e39": "runs/borzoi_finetuned",
}
CHECKPOINT_RUNS = {
    "AlphaGenome e12": "checkpoint_stability/alphagenome_e12",
    "AlphaGenome e16": "checkpoint_stability/alphagenome_e16",
    "AlphaGenome e19": "runs/alphagenome_finetuned",
    "Borzoi e00": "checkpoint_stability/borzoi_e00",
    "Borzoi e37": "checkpoint_stability/borzoi_e37",
    "Borzoi e39": "runs/borzoi_finetuned",
}
BACKGROUND_RUNS = {
    "AlphaGenome e19": "specificity_background/alphagenome_e19",
    "Borzoi e39": "specificity_background/borzoi_e39",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def read_features(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path / "features/combined_mutation_features.tsv", sep="\t")
    table["cell_type"] = table["track_id"].str.lower()
    return table.loc[
        table["gene"].eq("Mdk")
        & table["cell_type"].isin(CELLS)
        & table["readout_role"].eq("gene_body")
    ].copy()


def center_cell_table(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(
            ["variant_offset_from_tss_transcription_bp", "cell_type"], sort=True
        )
        .agg(
            median_abs_effect=("log2fc_ratio_of_sums", lambda x: float(np.median(np.abs(x)))),
            median_effect=("log2fc_ratio_of_sums", "median"),
            median_abs_delta=("signed_delta_sum", lambda x: float(np.median(np.abs(x)))),
            median_ref_sum=("ref_sum", "median"),
            shuffles=("mutation_id", "nunique"),
        )
        .reset_index()
    )


def add_specificity(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    out["max_other"] = out[[c for c in CELLS if c != "hsc"]].max(axis=1)
    out["hsc_margin"] = out["hsc"] - out["max_other"]
    out["hsc_vs_max_other_ratio"] = out["hsc"] / out["max_other"].clip(lower=1e-12)
    out["top_cell"] = out[CELLS].idxmax(axis=1)
    return out


def center_specificity(model: str, data: pd.DataFrame) -> pd.DataFrame:
    center = center_cell_table(data)
    wide = center.pivot(
        index="variant_offset_from_tss_transcription_bp",
        columns="cell_type",
        values="median_abs_effect",
    ).reset_index()
    wide = add_specificity(wide)
    wide.insert(0, "model", model)
    return wide


def aggregate_specificity(model: str, data: pd.DataFrame) -> dict:
    agg = (
        data.groupby("cell_type")["log2fc_ratio_of_sums"]
        .apply(lambda x: float(np.median(np.abs(x))))
        .reindex(CELLS)
    )
    raw = (
        data.groupby("cell_type")["signed_delta_sum"]
        .apply(lambda x: float(np.median(np.abs(x))))
        .reindex(CELLS)
    )
    ref = data.groupby("cell_type")["ref_sum"].median().reindex(CELLS)
    other = [c for c in CELLS if c != "hsc"]
    return {
        "model": model,
        "hsc_relative_effect": agg["hsc"],
        "max_other_relative_effect": agg[other].max(),
        "relative_hsc_ratio": agg["hsc"] / max(agg[other].max(), 1e-12),
        "relative_top_cell": agg.idxmax(),
        "hsc_raw_delta": raw["hsc"],
        "max_other_raw_delta": raw[other].max(),
        "raw_delta_hsc_ratio": raw["hsc"] / max(raw[other].max(), 1e-12),
        "raw_delta_top_cell": raw.idxmax(),
        "hsc_ref_sum": ref["hsc"],
        "max_other_ref_sum": ref[other].max(),
        "baseline_hsc_ratio": ref["hsc"] / max(ref[other].max(), 1e-12),
        "baseline_top_cell": ref.idxmax(),
    }


def bootstrap_specificity(
    model: str, center: pd.DataFrame, n_bootstrap: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    values = center.set_index("variant_offset_from_tss_transcription_bp")[CELLS].to_numpy()
    n = len(values)
    ratios = np.empty(n_bootstrap)
    margins = np.empty(n_bootstrap)
    is_top = np.empty(n_bootstrap, dtype=bool)
    for i in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        cell_values = np.median(sample, axis=0)
        max_other = cell_values[1:].max()
        ratios[i] = cell_values[0] / max(max_other, 1e-12)
        margins[i] = cell_values[0] - max_other
        is_top[i] = cell_values.argmax() == 0
    return {
        "model": model,
        "centers": n,
        "bootstrap_replicates": n_bootstrap,
        "probability_hsc_top": is_top.mean(),
        "ratio_median": np.median(ratios),
        "ratio_ci_low": np.quantile(ratios, 0.025),
        "ratio_ci_high": np.quantile(ratios, 0.975),
        "margin_median": np.median(margins),
        "margin_ci_low": np.quantile(margins, 0.025),
        "margin_ci_high": np.quantile(margins, 0.975),
    }


def checkpoint_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    profiles = {}
    for model, relpath in CHECKPOINT_RUNS.items():
        data = read_features(root / relpath)
        data = data.loc[data["locus_id"].eq("mdk_intron_hub_441_507")]
        spec = center_specificity(model, data)
        profiles[model] = spec.set_index("variant_offset_from_tss_transcription_bp")[
            "hsc_margin"
        ]
        agg = aggregate_specificity(model, data)
        agg.update(
            {
                "centers": len(spec),
                "hsc_top_center_fraction": float(spec["top_cell"].eq("hsc").mean()),
                "median_center_hsc_margin": float(spec["hsc_margin"].median()),
            }
        )
        rows.append(agg)
    correlations = []
    for family, labels in {
        "AlphaGenome": ["AlphaGenome e12", "AlphaGenome e16", "AlphaGenome e19"],
        "Borzoi": ["Borzoi e00", "Borzoi e37", "Borzoi e39"],
    }.items():
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                joined = pd.concat([profiles[left], profiles[right]], axis=1, join="inner").dropna()
                joined.columns = ["left", "right"]
                correlations.append(
                    {
                        "family": family,
                        "left": left,
                        "right": right,
                        "centers": len(joined),
                        "spearman_hsc_margin": spearmanr(joined["left"], joined["right"]).statistic,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(correlations)


def observed_signal(root: Path) -> pd.DataFrame:
    import pyBigWig

    readouts = pd.read_csv(root / "prepared/readouts.tsv", sep="\t")
    readouts = readouts.loc[readouts["gene"].eq("Mdk")]
    manifest = json.loads(
        (REPO_ROOT / "dataset_store/mmap_caches/split_chr10_chr11__poisson_multinomial__seq524288__label196608__bin128/manifest.json").read_text()
    )
    rows = []
    for cell, path in zip(manifest["task_names"], manifest["bw_files"]):
        with pyBigWig.open(path) as bw:
            for r in readouts.itertuples(index=False):
                value = bw.stats(r.chrom, int(r.start), int(r.end), type="sum", exact=True)[0]
                rows.append(
                    {
                        "gene": "Mdk",
                        "cell_type": cell,
                        "readout_id": r.readout_id,
                        "readout_role": r.role,
                        "start": int(r.start),
                        "end": int(r.end),
                        "length_bp": int(r.end) - int(r.start),
                        "observed_cpm_sum": 0.0 if value is None else float(value),
                    }
                )
    return pd.DataFrame(rows)


def reference_profile_fit(root: Path) -> pd.DataFrame:
    import pyBigWig

    manifest = json.loads(
        (REPO_ROOT / "dataset_store/mmap_caches/split_chr10_chr11__poisson_multinomial__seq524288__label196608__bin128/manifest.json").read_text()
    )
    rows = []
    for model, relpath in BACKGROUND_RUNS.items():
        run = root / relpath
        context = pd.read_csv(run / "sequence_contexts.tsv", sep="\t").iloc[0]
        metadata = json.loads((run / "model_metadata.json").read_text())
        profiles = np.load(run / "profiles/Mdk.ref_profiles.npy")
        track_order = json.loads((run / "profiles/Mdk.track_order.json").read_text())
        if profiles.shape[0] != len(track_order):
            profiles = profiles.T
        resolution = int(metadata["output_resolution_bp"])
        output_start = int(context.output_start)
        n_bins = int(profiles.shape[1])
        output_end = output_start + n_bins * resolution
        for cell, bw_path in zip(manifest["task_names"], manifest["bw_files"]):
            idx = track_order.index(cell)
            with pyBigWig.open(bw_path) as bw:
                obs = bw.stats(
                    "chr2", output_start, output_end, nBins=n_bins, type="sum", exact=True
                )
            obs = np.nan_to_num(np.asarray(obs, dtype=float), nan=0.0)
            pred = np.asarray(profiles[idx], dtype=float)
            gene_start, gene_end = 91929804, 91932298
            mask = np.arange(n_bins) * resolution + output_start
            mask = (mask + resolution > gene_start) & (mask < gene_end)
            for scope, use in [("full_output", np.ones(n_bins, dtype=bool)), ("gene_body", mask)]:
                o, p = obs[use], pred[use]
                rows.append(
                    {
                        "model": model,
                        "cell_type": cell,
                        "scope": scope,
                        "resolution_bp": resolution,
                        "bins": int(use.sum()),
                        "pearson": float(pearsonr(o, p).statistic) if np.std(o) and np.std(p) else np.nan,
                        "spearman": float(spearmanr(o, p).statistic) if np.std(o) and np.std(p) else np.nan,
                        "observed_sum": float(o.sum()),
                        "predicted_sum": float(p.sum()),
                        "predicted_to_observed_ratio": float(p.sum() / max(o.sum(), 1e-12)),
                    }
                )
    return pd.DataFrame(rows)


def training_overlap(root: Path) -> pd.DataFrame:
    genes = pd.read_csv(root / "prepared/genes.tsv", sep="\t")
    split = json.loads((REPO_ROOT / "dataset_store/splits/split_chr10_chr11/manifest.json").read_text())
    crop = (int(split["seq_len"]) - int(split["label_len"])) // 2
    beds = {}
    for split_name in ["train", "val", "test"]:
        beds[split_name] = pd.read_csv(
            REPO_ROOT / f"dataset_store/splits/split_chr10_chr11/{split_name}_intervals.bed",
            sep="\t",
            header=None,
            names=["chrom", "start", "end"],
        )
    rows = []
    for gene in genes.itertuples(index=False):
        gene_start, gene_end = sorted([int(gene.analysis_tss), int(gene.tes)])
        gene_end += 1
        split_name = "ood"
        input_overlaps = label_overlaps = 0
        for candidate, bed in beds.items():
            chrom = bed.loc[bed.chrom.eq(gene.chrom)].copy()
            inp = (chrom.start < gene_end) & (chrom.end > gene_start)
            lab_start, lab_end = chrom.start + crop, chrom.start + crop + int(split["label_len"])
            lab = (lab_start < gene_end) & (lab_end > gene_start)
            if inp.any() or lab.any():
                split_name = candidate
                input_overlaps = int(inp.sum())
                label_overlaps = int(lab.sum())
                break
        rows.append(
            {
                "gene": gene.gene,
                "chrom": gene.chrom,
                "split": split_name,
                "input_window_overlaps": input_overlaps,
                "supervised_label_window_overlaps": label_overlaps,
            }
        )
    return pd.DataFrame(rows)


def head_geometry() -> pd.DataFrame:
    import torch

    specs = {
        "AlphaGenome e19": (
            REPO_ROOT / "runs/alphagenome_split_chr10_chr11_seq524288_label196608_bin128_res128_poisson_multinomial_lora_active_h512x1/checkpoints/epochepoch=19.ckpt",
            "model.head.net.3.weight",
        ),
        "Borzoi e39": (
            REPO_ROOT / "runs/borzoi_split_chr10_chr11_poisson_multinomial_lora/checkpoints/epochepoch=39.ckpt",
            "model.head.channel_transform.conv.layer.weight",
        ),
    }
    rows = []
    for model, (path, key) in specs.items():
        state = torch.load(path, map_location="cpu")["state_dict"]
        weight = state[key].float().flatten(1)
        norms = torch.linalg.vector_norm(weight, dim=1)
        cos = torch.nn.functional.normalize(weight, dim=1) @ torch.nn.functional.normalize(weight, dim=1).T
        for i, cell in enumerate(CELLS):
            rows.append(
                {
                    "model": model,
                    "cell_type": cell,
                    "head_weight_norm": float(norms[i]),
                    "mean_cosine_to_other_heads": float((cos[i].sum() - 1) / (len(CELLS) - 1)),
                    "largest_head_norm": bool(i == int(norms.argmax())),
                }
            )
    return pd.DataFrame(rows)


def background_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    centers_all, windows_all, hub_rows = [], [], []
    for model, relpath in BACKGROUND_RUNS.items():
        data = read_features(root / relpath)
        center = center_specificity(model, data).sort_values(
            "variant_offset_from_tss_transcription_bp"
        )
        center["in_target_hub"] = center["variant_offset_from_tss_transcription_bp"].between(441, 507)
        centers_all.append(center)
        offsets = center["variant_offset_from_tss_transcription_bp"].to_numpy()
        for start_idx in range(len(center) - 33):
            subset = center.iloc[start_idx : start_idx + 34]
            start, end = int(offsets[start_idx]), int(offsets[start_idx + 33])
            if end - start != 66:
                continue
            cell_values = subset[CELLS].median()
            max_other = cell_values[[c for c in CELLS if c != "hsc"]].max()
            windows_all.append(
                {
                    "model": model,
                    "start_offset": start,
                    "end_offset": end,
                    "centers": 34,
                    "hsc_effect": cell_values["hsc"],
                    "max_other_effect": max_other,
                    "hsc_margin": cell_values["hsc"] - max_other,
                    "hsc_vs_max_other_ratio": cell_values["hsc"] / max(max_other, 1e-12),
                    "hsc_top_center_fraction": subset["top_cell"].eq("hsc").mean(),
                    "is_target_hub": start == 441 and end == 507,
                }
            )
    centers = pd.concat(centers_all, ignore_index=True)
    windows = pd.DataFrame(windows_all)
    for model, group in windows.groupby("model"):
        hub = group.loc[group["is_target_hub"]].iloc[0]
        hub_rows.append(
            {
                "model": model,
                "eligible_windows": len(group),
                "hub_start": int(hub.start_offset),
                "hub_end": int(hub.end_offset),
                "hub_hsc_ratio": hub.hsc_vs_max_other_ratio,
                "hub_hsc_margin": hub.hsc_margin,
                "hub_hsc_top_center_fraction": hub.hsc_top_center_fraction,
                "ratio_percentile": float((group.hsc_vs_max_other_ratio <= hub.hsc_vs_max_other_ratio).mean()),
                "margin_percentile": float((group.hsc_margin <= hub.hsc_margin).mean()),
                "top_fraction_percentile": float((group.hsc_top_center_fraction <= hub.hsc_top_center_fraction).mean()),
            }
        )
    return centers, windows, pd.DataFrame(hub_rows)


def shuffle_seed_replication(root: Path, background_centers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, relpath in FINAL_RUNS.items():
        targeted = read_features(root / relpath)
        targeted = targeted.loc[targeted["locus_id"].eq("mdk_intron_hub_441_507")]
        old = center_specificity(model, targeted).set_index("variant_offset_from_tss_transcription_bp")
        new = background_centers.loc[
            background_centers["model"].eq(model) & background_centers["in_target_hub"]
        ].set_index("variant_offset_from_tss_transcription_bp")
        joined = old[["hsc_margin", "hsc_vs_max_other_ratio"]].join(
            new[["hsc_margin", "hsc_vs_max_other_ratio"]], lsuffix="_seed17", rsuffix="_seed23"
        )
        rows.append(
            {
                "model": model,
                "centers": len(joined),
                "seed17_hsc_ratio": float(
                    np.median(old.hsc)
                    / max(old[["mac", "lsec", "chol"]].median().max(), 1e-12)
                ),
                "seed23_hsc_ratio": float(
                    np.median(new.hsc)
                    / max(new[["mac", "lsec", "chol"]].median().max(), 1e-12)
                ),
                "hsc_margin_spearman": float(spearmanr(joined.hsc_margin_seed17, joined.hsc_margin_seed23).statistic),
                "ratio_spearman": float(spearmanr(joined.hsc_vs_max_other_ratio_seed17, joined.hsc_vs_max_other_ratio_seed23).statistic),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / "analysis/cell_specificity_deep_dive"
    out.mkdir(parents=True, exist_ok=True)

    final_data = {}
    aggregate_rows, center_tables, bootstrap_rows = [], [], []
    for i, (model, relpath) in enumerate(FINAL_RUNS.items()):
        data = read_features(root / relpath)
        data = data.loc[data["locus_id"].eq("mdk_intron_hub_441_507")]
        final_data[model] = data
        aggregate_rows.append(aggregate_specificity(model, data))
        center = center_specificity(model, data)
        center_tables.append(center)
        bootstrap_rows.append(bootstrap_specificity(model, center, args.bootstrap, args.seed + i))

    aggregate = pd.DataFrame(aggregate_rows)
    centers = pd.concat(center_tables, ignore_index=True)
    bootstrap = pd.DataFrame(bootstrap_rows)
    checkpoint, checkpoint_corr = checkpoint_tables(root)
    observed = observed_signal(root)
    training = training_overlap(root)
    heads = head_geometry()
    background_centers, background_windows, background_hub = background_tables(root)
    seed_replication = shuffle_seed_replication(root, background_centers)
    reference_fit = reference_profile_fit(root)

    cross = centers.pivot(
        index="variant_offset_from_tss_transcription_bp", columns="model", values="hsc_margin"
    ).dropna()
    cross_model = pd.DataFrame(
        [{
            "left": "AlphaGenome e19",
            "right": "Borzoi e39",
            "centers": len(cross),
            "spearman_hsc_margin": float(spearmanr(cross.iloc[:, 0], cross.iloc[:, 1]).statistic),
        }]
    )

    outputs = {
        "cell_metric_decomposition.tsv": aggregate,
        "mdk_hub_center_specificity.tsv": centers,
        "bootstrap_specificity.tsv": bootstrap,
        "checkpoint_stability.tsv": checkpoint,
        "checkpoint_profile_concordance.tsv": checkpoint_corr,
        "cross_model_selectivity_concordance.tsv": cross_model,
        "observed_signal_by_readout.tsv": observed,
        "training_split_overlap.tsv": training,
        "head_geometry.tsv": heads,
        "background_center_specificity.tsv": background_centers,
        "background_window_scan.tsv": background_windows,
        "background_hub_enrichment.tsv": background_hub,
        "shuffle_seed_replication.tsv": seed_replication,
        "reference_profile_fit.tsv": reference_fit,
    }
    for name, table in outputs.items():
        table.to_csv(out / name, sep="\t", index=False)

    validation = {
        "status": "ok",
        "tables": {name: int(len(table)) for name, table in outputs.items()},
        "background_models": sorted(background_centers.model.unique()),
        "background_centers_per_model": background_centers.groupby("model").size().to_dict(),
        "target_hub_present_per_model": background_hub.set_index("model")["eligible_windows"].gt(0).to_dict(),
        "bootstrap_replicates": args.bootstrap,
    }
    (out / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
