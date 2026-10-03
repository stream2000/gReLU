"""Candidate-region selection policy for the Saijou nine-gene scan."""

from __future__ import annotations

import pandas as pd


# Requested motif controls occupy one of the two per-gene candidate slots.
KNOWN_CONTROL_CENTERS = {
    "Acta2": [(-61, "promoter_CArG_SRF_positive_control")],
    "Col1a1": [(-71, "inverted_CCAAT_NFY_positive_control")],
}

# These Mdk submodules were preregistered by the independent final audit.
FIXED_REGULATORY_RANGES = {
    "Mdk": [
        (463, 475, "Mdk_intron1_SP_KLF_submodule"),
        (499, 507, "Mdk_intron1_secondary_peak"),
    ]
}


def candidate_segment_record(
    *,
    gene: str,
    tx_start: int,
    tx_end: int,
    centers: pd.DataFrame,
    biological_prior: str | None = None,
) -> dict[str, object]:
    """Summarize one candidate segment from its retained center rows."""

    peak = centers.nlargest(1, "selection_priority").iloc[0]
    return {
        "gene": gene,
        "tx_start": int(tx_start),
        "tx_end": int(tx_end),
        "centers": int(centers.variant_offset_from_tss_transcription_bp.nunique()),
        "peak_tx_offset": int(peak.variant_offset_from_tss_transcription_bp),
        "peak_score": float(peak.raw_selection_priority),
        "peak_selection_score": float(peak.selection_priority),
        "peak_discovery_score": float(peak.discovery_score),
        "peak_readout_role": peak.readout_role,
        "peak_discovery_class": peak.discovery_class,
        "peak_region_type": peak.region_type,
        "peak_transcript_feature": str(peak.get("transcript_feature", "unannotated")),
        "peak_ag_hsc_ratio": float(peak.ag_hsc_ratio),
        "peak_bz_hsc_ratio": float(peak.bz_hsc_ratio),
        "peak_both_hsc_top": bool(peak.both_hsc_top),
        "peak_biological_prior": (
            str(biological_prior)
            if biological_prior is not None
            else str(peak.biological_prior)
        ),
        "minimum_splice_distance_bp": float(
            centers.nearest_splice_distance_bp.min()
        ),
    }


def _collapse_centers(annotated: pd.DataFrame) -> pd.DataFrame:
    collapsed = (
        annotated.loc[~annotated.overlaps_splice_site]
        .sort_values("selection_priority", ascending=False)
        .drop_duplicates(["gene", "variant_offset_from_tss_transcription_bp"])
        .copy()
    )
    collapsed["raw_selection_priority"] = collapsed.selection_priority
    collapsed["biological_prior"] = ""
    return collapsed


def _prioritize_positive_controls(collapsed: pd.DataFrame) -> None:
    for gene, controls in KNOWN_CONTROL_CENTERS.items():
        gene_rows = collapsed.loc[collapsed.gene.eq(gene)]
        if gene_rows.empty:
            continue
        gene_max = float(gene_rows.selection_priority.max())
        for target, label in controls:
            distance = (
                gene_rows.variant_offset_from_tss_transcription_bp.astype(int)
                - int(target)
            ).abs()
            nearest_index = distance.idxmin()
            if int(distance.loc[nearest_index]) > 2:
                raise RuntimeError(
                    f"No scanned center within 2 bp of {gene} control at {target:+d}"
                )
            collapsed.loc[nearest_index, "biological_prior"] = label
            collapsed.loc[nearest_index, "selection_priority"] = gene_max + 1.0


def _discovered_segments(selected: pd.DataFrame) -> list[dict[str, object]]:
    segments = []
    for gene, group in selected.groupby("gene"):
        if gene in FIXED_REGULATORY_RANGES:
            continue
        group = group.sort_values("variant_offset_from_tss_transcription_bp")
        cluster_ids = (
            group.variant_offset_from_tss_transcription_bp.diff().gt(4).cumsum()
        )
        for _, cluster in group.groupby(cluster_ids):
            if len(cluster) < 2:
                continue
            segments.append(
                candidate_segment_record(
                    gene=gene,
                    tx_start=int(
                        cluster.variant_offset_from_tss_transcription_bp.min()
                    ),
                    tx_end=int(
                        cluster.variant_offset_from_tss_transcription_bp.max()
                    ),
                    centers=cluster,
                )
            )
    return segments


def _fixed_segments(collapsed: pd.DataFrame) -> list[dict[str, object]]:
    segments = []
    for gene, ranges in FIXED_REGULATORY_RANGES.items():
        gene_rows = collapsed.loc[collapsed.gene.eq(gene)]
        if gene_rows.empty:
            continue
        for start, end, label in ranges:
            region = gene_rows.loc[
                gene_rows.variant_offset_from_tss_transcription_bp.between(start, end)
            ]
            if region.empty:
                raise RuntimeError(
                    f"No non-splice scanned centers in fixed {gene} range {start}:{end}"
                )
            segments.append(
                candidate_segment_record(
                    gene=gene,
                    tx_start=start,
                    tx_end=end,
                    centers=region,
                    biological_prior=label,
                )
            )
    return segments


def _fill_missing_segments(
    collapsed: pd.DataFrame,
    segments: list[dict[str, object]],
    segments_per_gene: int,
) -> None:
    for gene, group in collapsed.groupby("gene"):
        existing = [row for row in segments if row["gene"] == gene]
        needed = max(0, segments_per_gene - len(existing))
        if not needed:
            continue
        peaks = [row["peak_tx_offset"] for row in existing]
        for _, peak in group.sort_values("selection_priority", ascending=False).iterrows():
            offset = int(peak.variant_offset_from_tss_transcription_bp)
            if any(abs(offset - used) <= 10 for used in peaks):
                continue
            segments.append(
                candidate_segment_record(
                    gene=gene,
                    tx_start=offset,
                    tx_end=offset,
                    centers=pd.DataFrame([peak.to_dict()]),
                )
            )
            peaks.append(offset)
            needed -= 1
            if needed == 0:
                break


def _rank_segments(
    segments: list[dict[str, object]], segments_per_gene: int
) -> pd.DataFrame:
    table = pd.DataFrame.from_records(segments)
    if table.empty:
        raise RuntimeError("No non-splice candidate segments were found")
    table["segment_rank"] = (
        table.groupby("gene")["peak_selection_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    table = table.loc[table.segment_rank.le(segments_per_gene)].sort_values(
        ["gene", "segment_rank"]
    )
    table["candidate_segment_id"] = table.apply(
        lambda row: (
            f"{row.gene.lower()}_candidate_{int(row.segment_rank):02d}_"
            f"tx{int(row.tx_start):+d}_{int(row.tx_end):+d}"
        ),
        axis=1,
    )
    return table


def _select_segment_centers(
    annotated: pd.DataFrame,
    segments: pd.DataFrame,
    max_centers_per_segment: int,
) -> pd.DataFrame:
    candidates = []
    for segment in segments.itertuples(index=False):
        pool = annotated.loc[
            annotated.gene.eq(segment.gene)
            & annotated.variant_offset_from_tss_transcription_bp.between(
                segment.tx_start, segment.tx_end
            )
            & ~annotated.overlaps_splice_site
        ].copy()
        pool = (
            pool.sort_values("selection_priority", ascending=False)
            .drop_duplicates(["variant_offset_from_tss_transcription_bp"])
            .head(max_centers_per_segment)
        )
        pool["candidate_segment_id"] = segment.candidate_segment_id
        pool["segment_rank"] = segment.segment_rank
        candidates.append(pool)
    return pd.concat(candidates, ignore_index=True)


def cluster_candidates(
    annotated: pd.DataFrame,
    *,
    segments_per_gene: int,
    max_centers_per_segment: int,
    candidate_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select splice-protected candidates under the preregistered policy."""

    collapsed = _collapse_centers(annotated)
    _prioritize_positive_controls(collapsed)
    collapsed["gene_score_percentile"] = collapsed.groupby("gene")[
        "selection_priority"
    ].rank(method="average", pct=True)
    selected = collapsed.loc[
        collapsed.gene_score_percentile.ge(candidate_quantile)
    ].copy()
    segments = _discovered_segments(selected) + _fixed_segments(collapsed)
    _fill_missing_segments(collapsed, segments, segments_per_gene)
    segment_table = _rank_segments(segments, segments_per_gene)
    candidate_centers = _select_segment_centers(
        annotated, segment_table, max_centers_per_segment
    )
    return segment_table, candidate_centers
