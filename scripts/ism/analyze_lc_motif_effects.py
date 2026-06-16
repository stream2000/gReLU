#!/usr/bin/env python
"""Summarize matched-control-adjusted LC-DIC motif knockout effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, spearmanr, wilcoxon


LOCAL_TRACKS = (
    "ctcf",
    "rad21",
    "smc3",
    "polr2a",
    "aff4",
    "brd4",
    "med1",
    "ep300",
    "foxa1",
    "esr1",
    "gata3",
    "h3k27ac",
    "h3k4me1",
    "h3k4me2",
    "h3k4me3",
)
PRIMARY_TRACKS = (
    "polr2a",
    "rad21",
    "smc3",
    "foxa1",
    "esr1",
    "gata3",
    "ep300",
    "brd4",
    "h3k27ac",
    "h3k4me1",
    "h3k4me2",
    "h3k4me3",
)
COGNATE_TRACK = {
    "Forkhead": "foxa1",
    "GATA3": "gata3",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-feature-dir", action="append", required=True)
    parser.add_argument("--tss-feature-dir", action="append", default=[])
    parser.add_argument("--sites", required=True)
    parser.add_argument("--tss-readouts")
    parser.add_argument("--cluster-assignments")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_features(directories: list[str]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(Path(directory) / "group_features.parquet")
        for directory in directories
    ]
    table = pd.concat(frames, ignore_index=True)
    if table["site_id"].duplicated().any():
        duplicates = table.loc[
            table["site_id"].duplicated(keep=False), "site_id"
        ].head(10)
        raise ValueError(f"duplicate site IDs across shards: {duplicates.tolist()}")
    return table


def _bh_adjust(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    result = np.full(len(array), np.nan)
    valid = np.isfinite(array)
    if not valid.any():
        return pd.Series(result, index=values.index)
    order = np.argsort(array[valid])
    ranked = array[valid][order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[np.flatnonzero(valid)[order]] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def _wilcoxon(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(wilcoxon(nonzero, alternative="two-sided").pvalue)


def _summary_rows(
    table: pd.DataFrame,
    *,
    column_template: str,
    tracks: tuple[str, ...],
    windows: tuple[int, ...] | tuple[None, ...],
) -> pd.DataFrame:
    records: list[dict] = []
    groups = [("all", table)]
    groups.extend(
        (str(family), group)
        for family, group in table.groupby("motif_family", sort=True)
    )
    for group_name, group in groups:
        for window in windows:
            for track in tracks:
                column = column_template.format(track=track, window=window)
                if column not in group:
                    continue
                values = group[column].to_numpy(dtype=float)
                records.append(
                    {
                        "group": group_name,
                        "track": track,
                        "window_bp": window,
                        "n": len(values),
                        "median": float(np.median(values)),
                        "mean": float(np.mean(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "fraction_negative": float(np.mean(values < 0)),
                        "fraction_le_minus_0.01": float(
                            np.mean(values <= -0.01)
                        ),
                        "fraction_le_minus_0.05": float(
                            np.mean(values <= -0.05)
                        ),
                        "wilcoxon_p": _wilcoxon(values),
                    }
                )
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result["wilcoxon_q"] = result.groupby(
            ["group", "window_bp"], dropna=False
        )["wilcoxon_p"].transform(_bh_adjust)
    return result


def _family_comparisons(
    table: pd.DataFrame,
    *,
    column_template: str,
    tracks: tuple[str, ...],
    windows: tuple[int, ...] | tuple[None, ...],
    analysis: str,
) -> pd.DataFrame:
    records = []
    for window in windows:
        for track in tracks:
            column = column_template.format(track=track, window=window)
            if column not in table:
                continue
            groups = [
                group[column].to_numpy(dtype=float)
                for _, group in table.groupby("motif_family", sort=True)
            ]
            statistic, pvalue = kruskal(*groups)
            records.append(
                {
                    "analysis": analysis,
                    "track": track,
                    "window_bp": window,
                    "kruskal_statistic": statistic,
                    "kruskal_p": pvalue,
                }
            )
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result["kruskal_q"] = _bh_adjust(result["kruskal_p"])
    return result


def _local_tss_correlations(table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for minimum_distance in (0, 10_000, 50_000, 100_000):
        subset = table[
            table["distance_tss_from_edit"].abs() > minimum_distance
        ]
        for track in PRIMARY_TRACKS:
            local_column = f"{track}__w4096__log2fc_signed_mean"
            tss_column = f"{track}__host_tss_4kb__log2fc_signed_mean"
            if local_column not in subset or tss_column not in subset:
                continue
            valid = subset[[local_column, tss_column]].dropna()
            rho, pvalue = spearmanr(valid[local_column], valid[tss_column])
            records.append(
                {
                    "minimum_distance_bp": minimum_distance,
                    "track": track,
                    "n": len(valid),
                    "spearman_rho": rho,
                    "spearman_p": pvalue,
                }
            )
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result["spearman_q"] = result.groupby(
            "minimum_distance_bp"
        )["spearman_p"].transform(_bh_adjust)
    return result


def _tss_distance_summary(table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for minimum_distance in (0, 10_000, 50_000, 100_000):
        distant = table[
            table["distance_tss_from_edit"].abs() > minimum_distance
        ]
        groups = [("all", distant)]
        groups.extend(
            (str(family), group)
            for family, group in distant.groupby("motif_family", sort=True)
        )
        for group_name, group in groups:
            for track in PRIMARY_TRACKS:
                column = f"{track}__host_tss_4kb__log2fc_signed_mean"
                values = group[column].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                records.append(
                    {
                        "minimum_distance_bp": minimum_distance,
                        "group": group_name,
                        "track": track,
                        "n": len(values),
                        "median": float(np.median(values)),
                        "fraction_negative": float(np.mean(values < 0)),
                        "wilcoxon_p": _wilcoxon(values),
                    }
                )
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result["wilcoxon_q"] = result.groupby(
            ["minimum_distance_bp", "group"]
        )["wilcoxon_p"].transform(_bh_adjust)
    return result


def _cluster_motif_enrichment(table: pd.DataFrame) -> pd.DataFrame:
    cluster_medians = table.groupby("cluster")[
        "polr2a__w4096__log2fc_signed_mean"
    ].median()
    response_cluster = cluster_medians.idxmin()
    table["response_cluster"] = table["cluster"].eq(response_cluster)
    records = []
    for family in sorted(table["motif_family"].unique()):
        in_family = table["motif_family"].eq(family)
        response = table["response_cluster"]
        contingency = [
            [
                int((in_family & response).sum()),
                int((in_family & ~response).sum()),
            ],
            [
                int((~in_family & response).sum()),
                int((~in_family & ~response).sum()),
            ],
        ]
        odds_ratio, pvalue = fisher_exact(contingency)
        records.append(
            {
                "motif_family": family,
                "response_cluster": response_cluster,
                "family_n": int(in_family.sum()),
                "family_response_n": int((in_family & response).sum()),
                "family_response_fraction": float(
                    table.loc[in_family, "response_cluster"].mean()
                ),
                "other_response_fraction": float(
                    table.loc[~in_family, "response_cluster"].mean()
                ),
                "odds_ratio": odds_ratio,
                "fisher_p": pvalue,
            }
        )
    result = pd.DataFrame.from_records(records)
    result["fisher_q"] = _bh_adjust(result["fisher_p"])
    return result


def _write_report(
    output_dir: Path,
    sites: pd.DataFrame,
    local_summary: pd.DataFrame,
    tss_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    cluster_enrichment: pd.DataFrame,
) -> None:
    family_counts = sites["motif_family"].value_counts().sort_index()
    lines = [
        "# LC-DIC motif knockout analysis",
        "",
        "All effects are target motif edits minus same-Pol2-peak matched controls.",
        "",
        "## Cohort",
        "",
        f"- {len(sites)} site-by-motif-family candidates.",
        f"- {sites['source_site_id'].nunique()} distinct source LC-DICs.",
    ]
    lines.extend(
        f"- {family}: {count} candidates."
        for family, count in family_counts.items()
    )
    lines.extend(["", "## Primary local 4 kb effects", ""])
    primary_local = local_summary[
        local_summary["group"].eq("all")
        & local_summary["window_bp"].eq(4096)
        & local_summary["track"].isin(PRIMARY_TRACKS)
    ].copy()
    for row in primary_local.sort_values("median").itertuples(index=False):
        lines.append(
            f"- {row.track}: median {row.median:+.4f}, "
            f"{row.fraction_negative:.1%} negative, q={row.wilcoxon_q:.3g}."
        )
    if not tss_summary.empty:
        lines.extend(["", "## Host-TSS proxy effects", ""])
        primary_tss = tss_summary[
            tss_summary["group"].eq("all")
            & tss_summary["track"].isin(PRIMARY_TRACKS)
        ].copy()
        for row in primary_tss.sort_values("median").itertuples(index=False):
            lines.append(
                f"- {row.track}: median {row.median:+.4f}, "
                f"{row.fraction_negative:.1%} negative, q={row.wilcoxon_q:.3g}."
            )
    if not correlations.empty:
        lines.extend(["", "## Distal local-to-TSS association (>50 kb)", ""])
        distant_correlations = correlations[
            correlations["minimum_distance_bp"].eq(50_000)
        ]
        for row in distant_correlations.sort_values(
            "spearman_q"
        ).itertuples(index=False):
            lines.append(
                f"- {row.track}: rho={row.spearman_rho:+.3f}, "
                f"q={row.spearman_q:.3g}, n={row.n}."
            )
    if not distance_summary.empty:
        lines.extend(["", "## Distal POLR2A by motif family (>50 kb)", ""])
        distant_pol2 = distance_summary[
            distance_summary["minimum_distance_bp"].eq(50_000)
            & distance_summary["track"].eq("polr2a")
        ]
        for row in distant_pol2.sort_values("group").itertuples(index=False):
            lines.append(
                f"- {row.group}: median {row.median:+.4f}, "
                f"{row.fraction_negative:.1%} negative, "
                f"q={row.wilcoxon_q:.3g}, n={row.n}."
            )
    if not cluster_enrichment.empty:
        response_cluster = cluster_enrichment["response_cluster"].iloc[0]
        response_n = int(
            cluster_enrichment["family_response_n"].sum()
        )
        lines.extend(
            [
                "",
                "## Population response cluster",
                "",
                f"- High-depletion cluster: {response_cluster}; "
                f"{response_n}/{len(sites)} candidates.",
            ]
        )
        for row in cluster_enrichment.sort_values(
            "fisher_q"
        ).itertuples(index=False):
            lines.append(
                f"- {row.motif_family}: "
                f"{row.family_response_n}/{row.family_n} responders, "
                f"odds ratio {row.odds_ratio:.2f}, q={row.fisher_q:.3g}."
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- These are AlphaGenome predictions, not experimental validation.",
            "- The distal readout is an inferred host TSS unless loop evidence is added.",
            "- AP1 and TEAD4 cognate TF tracks were not present in this track panel.",
        ]
    )
    (output_dir / "result_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(args.sites, sep="\t", low_memory=False)
    local = _load_features(args.local_feature_dir)
    table = sites.merge(
        local, on=["site_id", "site_class"], validate="one_to_one"
    )
    if len(table) != len(sites):
        raise ValueError("local feature shards do not exactly cover ready sites")

    local_summary = _summary_rows(
        table,
        column_template="{track}__w{window}__log2fc_signed_mean",
        tracks=LOCAL_TRACKS,
        windows=(1024, 4096),
    )
    local_summary.to_csv(
        output_dir / "local_track_summary.tsv", sep="\t", index=False
    )
    comparisons = [
        _family_comparisons(
            table,
            column_template="{track}__w{window}__log2fc_signed_mean",
            tracks=LOCAL_TRACKS,
            windows=(1024, 4096),
            analysis="local",
        )
    ]
    cluster_enrichment = pd.DataFrame()
    if args.cluster_assignments:
        assignments = pd.read_csv(
            args.cluster_assignments, sep="\t", low_memory=False
        )
        table = table.merge(
            assignments[["site_id", "cluster"]],
            on="site_id",
            validate="one_to_one",
        )
        cluster_enrichment = _cluster_motif_enrichment(table)
        cluster_enrichment.to_csv(
            output_dir / "cluster_motif_enrichment.tsv",
            sep="\t",
            index=False,
        )

    tss_summary = pd.DataFrame()
    distance_summary = pd.DataFrame()
    correlations = pd.DataFrame()
    if args.tss_feature_dir:
        tss = _load_features(args.tss_feature_dir)
        tss_columns = [
            column
            for column in tss.columns
            if "__host_tss_4kb__" in column
        ]
        table = table.merge(
            tss[["site_id", *tss_columns]],
            on="site_id",
            how="left",
            validate="one_to_one",
        )
        if args.tss_readouts:
            readouts = pd.read_csv(args.tss_readouts, sep="\t", low_memory=False)
            readout_columns = [
                column for column in readouts.columns if column != "source_site_id"
            ]
            table = table.merge(
                readouts[readout_columns],
                on="site_id",
                how="left",
                validate="one_to_one",
            )
        tss_ready = table[
            table["polr2a__host_tss_4kb__log2fc_signed_mean"].notna()
        ].copy()
        tss_summary = _summary_rows(
            tss_ready,
            column_template="{track}__host_tss_4kb__log2fc_signed_mean",
            tracks=LOCAL_TRACKS,
            windows=(None,),
        )
        tss_summary.to_csv(
            output_dir / "tss_track_summary.tsv", sep="\t", index=False
        )
        comparisons.append(
            _family_comparisons(
                tss_ready,
                column_template="{track}__host_tss_4kb__log2fc_signed_mean",
                tracks=LOCAL_TRACKS,
                windows=(None,),
                analysis="host_tss",
            )
        )
        correlations = _local_tss_correlations(tss_ready)
        correlations.to_csv(
            output_dir / "local_tss_correlations.tsv", sep="\t", index=False
        )
        distance_summary = _tss_distance_summary(tss_ready)
        distance_summary.to_csv(
            output_dir / "tss_distance_summary.tsv", sep="\t", index=False
        )

    family_comparison = pd.concat(comparisons, ignore_index=True)
    family_comparison.to_csv(
        output_dir / "motif_family_comparisons.tsv", sep="\t", index=False
    )

    table["cognate_track"] = table["motif_family"].map(COGNATE_TRACK)
    table["cognate_local_4kb_effect"] = [
        row.get(
            f"{row['cognate_track']}__w4096__log2fc_signed_mean",
            np.nan,
        )
        if pd.notna(row["cognate_track"])
        else np.nan
        for _, row in table.iterrows()
    ]
    ranking_columns = [
        "site_id",
        "source_site_id",
        "motif_family",
        "motif_name",
        "motif_relative_score",
        "cognate_track",
        "cognate_local_4kb_effect",
        "polr2a__w4096__log2fc_signed_mean",
        "h3k27ac__w4096__log2fc_signed_mean",
        "h3k4me1__w4096__log2fc_signed_mean",
        "rad21__w4096__log2fc_signed_mean",
    ]
    ranking_columns.extend(
        column
        for column in (
            "gene_name",
            "distance_tss_from_edit",
            "mapping_ambiguous",
            "polr2a__host_tss_4kb__log2fc_signed_mean",
            "h3k27ac__host_tss_4kb__log2fc_signed_mean",
        )
        if column in table
    )
    table[ranking_columns].sort_values(
        [
            "polr2a__w4096__log2fc_signed_mean",
            "h3k27ac__w4096__log2fc_signed_mean",
        ]
    ).to_csv(output_dir / "candidate_ranking.tsv", sep="\t", index=False)
    table.to_csv(output_dir / "site_effects.tsv", sep="\t", index=False)
    _write_report(
        output_dir,
        sites,
        local_summary,
        tss_summary,
        distance_summary,
        correlations,
        cluster_enrichment,
    )


if __name__ == "__main__":
    main()
