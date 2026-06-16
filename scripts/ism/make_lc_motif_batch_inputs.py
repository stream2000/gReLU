#!/usr/bin/env python
"""Build batch inputs for literature-guided LC-DIC motif perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PANEL_MOTIF_NAMES = (
    "FOXA1.H13CORE.0.P.B",
    "ESR1.H13CORE.0.P.B",
    "GATA3.H13CORE.0.PS.A",
    "FOS.H13CORE.0.P.B",
    "JUN.H13CORE.0.P.B",
    "TEAD4.H13CORE.0.PS.A",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--source-sites", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--motifs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tss-readouts")
    parser.add_argument("--min-relative-score", type=float, default=0.90)
    return parser.parse_args()


def _parse_peak_interval(value: object) -> tuple[int, int] | None:
    text = "" if pd.isna(value) else str(value).strip()
    if not text or ":" not in text or "-" not in text:
        return None
    coordinates = text.split(":", 1)[1]
    start, end = coordinates.split("-", 1)
    return int(start), int(end)


def _pol2_peak(
    row: pd.Series, anchor: int
) -> tuple[int, int, int, str] | None:
    candidate_start = pd.to_numeric(
        row.get("pol2_peak_start"), errors="coerce"
    )
    candidate_end = pd.to_numeric(
        row.get("pol2_peak_end"), errors="coerce"
    )
    if pd.notna(candidate_start) and pd.notna(candidate_end):
        return (
            int(candidate_start),
            int(candidate_end),
            anchor,
            str(row.get("pol2_condition", "")),
        )
    choices = []
    for prefix in ("pol2_ctrl", "pol2_e2_30", "pol2_e2_45"):
        interval = _parse_peak_interval(row.get(f"{prefix}_best_peak"))
        summit = pd.to_numeric(
            row.get(f"{prefix}_best_summit"), errors="coerce"
        )
        if interval is not None:
            choices.append(
                (
                    0 if pd.notna(summit) and int(summit) == anchor else 1,
                    abs(int(summit) - anchor) if pd.notna(summit) else 10**12,
                    interval,
                    int(summit) if pd.notna(summit) else anchor,
                    prefix,
                )
            )
    if not choices:
        return None
    selected = min(choices, key=lambda value: (value[0], value[1]))
    return selected[2][0], selected[2][1], selected[3], selected[4]


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates, sep="\t", low_memory=False)
    source_sites = pd.read_csv(args.source_sites, sep="\t", low_memory=False)
    source_sites = source_sites[source_sites["site_class"].eq("lc_dic")].copy()
    source_columns = [
        "site_id",
        "chrom",
        "start",
        "end",
        "rad21_ctrl_signal",
        "ctcf_ctrl_signal",
        "pol2_ctrl_signal",
    ]
    source_sites = source_sites[source_columns]
    audit = pd.read_csv(args.audit, sep="\t", low_memory=False)
    audit = audit[audit["paper_class"].eq("lc_dic")].copy()
    motifs_path = str(Path(args.motifs).resolve())

    merged = (
        candidates.merge(
            source_sites,
            on=["site_id", "chrom"],
            suffixes=("_candidate", "_source"),
            validate="many_to_one",
        )
        .merge(
            audit,
            on=["site_id", "chrom"],
            suffixes=("", "_audit"),
            validate="many_to_one",
        )
        .reset_index(drop=True)
    )

    rows = []
    skipped = []
    for _, row in merged.iterrows():
        pol2_anchor = int(row["anchor"])
        peak_info = _pol2_peak(row, pol2_anchor)
        if peak_info is None:
            skipped.append(
                {
                    "candidate_id": row["candidate_id"],
                    "site_id": row["site_id"],
                    "reason": "no Pol2 peak interval in audit table",
                }
            )
            continue
        control_start, control_end, true_pol2_anchor, pol2_condition = peak_info
        motif_center = (
            int(row["motif_start"]) + int(row["motif_end"])
        ) // 2
        if not (
            int(row["motif_start"]) >= control_start
            and int(row["motif_end"]) <= control_end
            and abs(motif_center - true_pol2_anchor) <= 100
        ):
            skipped.append(
                {
                    "candidate_id": row["candidate_id"],
                    "site_id": row["site_id"],
                    "reason": (
                        "motif is not inside the selected Pol2 peak and "
                        "summit +/-100 bp"
                    ),
                }
            )
            continue
        rows.append(
            {
                "site_id": row["candidate_id"],
                "source_site_id": row["site_id"],
                "site_class": "lc_dic_motif",
                "motif_family": row["motif_family"],
                "motif_label": row["motif_label"],
                "chrom": row["chrom"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "anchor": motif_center,
                "pol2_anchor": true_pol2_anchor,
                "pol2_condition": pol2_condition,
                "mutation_strategy": "motif_pwm_disruption",
                "target_feature": f"{row['motif_family']}_motif",
                "motif_path": motifs_path,
                "motif_name": row["motif_name"],
                "motif_start": int(row["motif_start"]),
                "motif_end": int(row["motif_end"]),
                "motif_strand": row["motif_strand"],
                "motif_sequence": row["motif_sequence"],
                "motif_relative_score_scanned": row["relative_score"],
                "min_relative_motif_score": args.min_relative_score,
                "panel_motif_path": motifs_path,
                "panel_motif_names": ";".join(PANEL_MOTIF_NAMES),
                "control_region_start": control_start,
                "control_region_end": control_end,
                "source": args.candidates,
                "label": row["candidate_id"],
                "rad21_ctrl_signal": row.get("rad21_ctrl_signal"),
                "ctcf_ctrl_signal": row.get("ctcf_ctrl_signal"),
                "pol2_ctrl_signal": row.get("pol2_ctrl_signal"),
            }
        )

    sites = pd.DataFrame.from_records(rows)
    if sites.empty:
        raise ValueError("no LC-DIC motif candidates have a Pol2 peak interval")
    if sites["site_id"].duplicated().any():
        raise ValueError("candidate IDs must be unique")
    sites.to_csv(output_dir / "sites.tsv", sep="\t", index=False)
    pd.read_csv(args.tracks, sep="\t", low_memory=False).to_csv(
        output_dir / "tracks.tsv", sep="\t", index=False
    )
    pd.DataFrame.from_records(skipped).to_csv(
        output_dir / "skipped_candidates.tsv", sep="\t", index=False
    )

    if args.tss_readouts:
        tss = pd.read_csv(args.tss_readouts, sep="\t", low_memory=False)
        readouts = sites[["site_id", "source_site_id"]].merge(
            tss.drop(columns=["site_class"], errors="ignore"),
            left_on="source_site_id",
            right_on="site_id",
            suffixes=("", "_source"),
            validate="many_to_one",
        )
        readouts = readouts.drop(columns=["site_id_source"])
        readouts.to_csv(
            output_dir / "tss_readouts.tsv", sep="\t", index=False
        )

    print(
        f"Wrote {len(sites)} motif candidates across "
        f"{sites['source_site_id'].nunique()} LC-DICs; "
        f"skipped={len(skipped)}"
    )


if __name__ == "__main__":
    main()
