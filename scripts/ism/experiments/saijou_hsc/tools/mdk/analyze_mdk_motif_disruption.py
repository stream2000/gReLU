#!/usr/bin/env python
"""Associate Mdk motif disruption with HSC-selective ISM effects."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from grelu.interpret.motifs import scan_sequences
from grelu.io.motifs import get_jaspar

if __package__:
    from ..genomics import build_edit_context_sequences
else:
    SAIJOU_DIR = Path(__file__).resolve().parents[2]
    if str(SAIJOU_DIR) not in sys.path:
        sys.path.insert(0, str(SAIJOU_DIR))
    from tools.genomics import build_edit_context_sequences


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_targeted_original_comparison"
DEFAULT_FASTA = Path("/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa")
RUNS = {
    "AlphaGenome e19": "runs/alphagenome_finetuned",
    "Borzoi e39": "runs/borzoi_finetuned",
}
CELLS = ["hsc", "mac", "lsec", "chol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--context-bp", type=int, default=80)
    parser.add_argument("--pthresh", type=float, default=1e-3)
    return parser.parse_args()


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    values = pvalues.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return pd.Series(out, index=pvalues.index)


def mdk_motif_family(name: str) -> str:
    upper = name.upper()
    if "CTCF" in upper:
        return "CTCF"
    if "NFY" in upper:
        return "NFY"
    if "RREB1" in upper:
        return "RREB1"
    if "EGR" in upper or "ZBTB" in upper:
        return "EGR_ZBTB"
    if "KLF" in upper or re.match(r"^SP\d", upper.split("_")[-1]):
        return "KLF_SP_GC_BOX"
    return "OTHER"


def hsc_margin(root: Path, model: str, relpath: str) -> pd.Series:
    data = pd.read_csv(root / relpath / "features/combined_mutation_features.tsv", sep="\t")
    data = data.loc[
        data["gene"].eq("Mdk")
        & data["locus_id"].eq("mdk_intron_hub_441_507")
        & data["readout_role"].eq("gene_body")
        & data["track_id"].isin(CELLS)
    ].copy()
    data["abs_effect"] = data["log2fc_ratio_of_sums"].abs()
    mutation = data.pivot_table(
        index=["mutation_id", "variant_offset_from_tss_transcription_bp"],
        columns="track_id",
        values="abs_effect",
        aggfunc="median",
    ).reset_index()
    mutation["hsc_margin"] = mutation.hsc - mutation[["mac", "lsec", "chol"]].max(axis=1)
    return mutation.groupby("variant_offset_from_tss_transcription_bp").hsc_margin.median().rename(model)


def _scan_mdk_motifs(
    manifest: pd.DataFrame, args: argparse.Namespace
) -> tuple[list[str], object, pd.DataFrame]:
    sequences, sequence_ids, sequence_records = build_edit_context_sequences(
        manifest, args.fasta, args.context_bp
    )
    # Use the vertebrate CORE collection: mouse-only JASPAR omits canonical
    # SP/KLF matrices needed to test the GC-box hypothesis.
    motifs = get_jaspar(release="JASPAR2024", tax_group="vertebrates")
    hits = scan_sequences(
        sequences,
        motifs,
        seq_ids=sequence_ids,
        pthresh=args.pthresh,
        rc=True,
    )
    hits = hits.merge(sequence_records, on="sequence", how="left", validate="many_to_one")
    hits = hits.loc[(hits.start < hits.edit_rel_end) & (hits.end > hits.edit_rel_start)].copy()
    hits["family"] = hits.motif.map(mdk_motif_family)
    hits = hits.sort_values(
        ["sequence", "motif", "start", "end", "strand"], kind="stable"
    ).reset_index(drop=True)
    return sequences, motifs, hits


def _motif_disruption_scores(hits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score = hits.pivot_table(
        index=["mutation_id", "center", "motif", "family"],
        columns="state",
        values="fimo_score",
        aggfunc="max",
        fill_value=0.0,
    ).reset_index()
    for state in ["ref", "alt"]:
        if state not in score:
            score[state] = 0.0
    score["signed_score_change"] = score["ref"] - score["alt"]
    score["absolute_score_change"] = score.signed_score_change.abs()
    center_motif = (
        score.groupby(["center", "motif", "family"], sort=False)
        .agg(
            median_abs_score_change=("absolute_score_change", "median"),
            median_signed_score_change=("signed_score_change", "median"),
            shuffle_replicates=("mutation_id", "nunique"),
        )
        .reset_index()
    )
    return score, center_motif


def _motif_associations(
    center_motif: pd.DataFrame, margins: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for (motif, family), group in center_motif.groupby(["motif", "family"]):
        disruption = group.set_index("center").median_abs_score_change
        for model in RUNS:
            # Every motif is evaluated on the same 34 centers. A missing FIMO
            # hit is a zero disruption, avoiding selection on motif presence.
            joined = pd.concat([disruption, margins[model]], axis=1, join="outer").fillna(0)
            if len(joined) < 5 or joined.iloc[:, 0].nunique() < 2:
                continue
            stat = spearmanr(joined.iloc[:, 0], joined.iloc[:, 1])
            rows.append(
                {
                    "level": "motif",
                    "feature": motif,
                    "family": family,
                    "model": model,
                    "centers": len(joined),
                    "spearman": float(stat.statistic),
                    "pvalue": float(stat.pvalue),
                }
            )
    associations = pd.DataFrame(rows)
    associations["qvalue"] = associations.groupby("model", group_keys=False).pvalue.apply(bh_adjust)
    return associations


def _family_associations(score: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    family_center = (
        score.loc[score.family.ne("OTHER")]
        .groupby(["center", "family"], sort=False)
        .absolute_score_change.max()
        .reset_index()
    )
    family_rows = []
    for family, group in family_center.groupby("family"):
        disruption = group.set_index("center").absolute_score_change
        for model in RUNS:
            joined = pd.concat([disruption, margins[model]], axis=1, join="outer").fillna(0)
            stat = spearmanr(joined.iloc[:, 0], joined.iloc[:, 1])
            family_rows.append(
                {
                    "level": "family",
                    "feature": family,
                    "family": family,
                    "model": model,
                    "centers": len(joined),
                    "spearman": float(stat.statistic),
                    "pvalue": float(stat.pvalue),
                }
            )
    families = pd.DataFrame(family_rows)
    families["qvalue"] = families.groupby("model", group_keys=False).pvalue.apply(bh_adjust)
    return families


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / "analysis/cell_specificity_deep_dive"
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(root / "prepared/mutation_manifest.tsv", sep="\t")
    manifest = manifest.loc[manifest.locus_id.eq("mdk_intron_hub_441_507")].copy()
    sequences, motifs, hits = _scan_mdk_motifs(manifest, args)
    score, center_motif = _motif_disruption_scores(hits)
    margins = pd.concat(
        [hsc_margin(root, model, relpath) for model, relpath in RUNS.items()],
        axis=1,
    )
    associations = _motif_associations(center_motif, margins)
    families = _family_associations(score, margins)
    associations = pd.concat([associations, families], ignore_index=True)
    hits.to_csv(out / "motif_overlapping_hits.tsv", sep="\t", index=False)
    score.to_csv(out / "motif_disruption_by_mutation.tsv", sep="\t", index=False)
    associations.sort_values(["level", "model", "qvalue", "pvalue"]).to_csv(
        out / "motif_disruption_associations.tsv", sep="\t", index=False
    )
    print(
        {
            "sequences": len(sequences),
            "motifs": len(motifs),
            "overlapping_hits": len(hits),
            "motifs_with_associations": int((associations.level == "motif").sum()),
            "significant_fdr_005": int((associations.qvalue < 0.05).sum()),
        }
    )


if __name__ == "__main__":
    main()
