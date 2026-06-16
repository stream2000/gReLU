#!/usr/bin/env python
"""Scan literature-guided TF motifs around LC-DIC Pol2 summits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.motifs import (  # noqa: E402
    FastaReference,
    load_meme_pwm,
    scan_motif_hits,
)

DEFAULT_MOTIFS = {
    "FOXA1": "FOXA1.H13CORE.0.P.B",
    "ESR1": "ESR1.H13CORE.0.P.B",
    "GATA3": "GATA3.H13CORE.0.PS.A",
    "FOS": "FOS.H13CORE.0.P.B",
    "JUN": "JUN.H13CORE.0.P.B",
    "TEAD4": "TEAD4.H13CORE.0.PS.A",
}
MOTIF_FAMILIES = {
    "FOXA1": "Forkhead",
    "ESR1": "ESR1",
    "GATA3": "GATA3",
    "FOS": "AP1",
    "JUN": "AP1",
    "TEAD4": "TEAD4",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--motifs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--audit",
        help=(
            "Optional DIC audit table. When provided, LC-DICs without a real "
            "Pol2 peak are excluded and anchors are replaced with Pol2 summits."
        ),
    )
    parser.add_argument("--cluster-assignments")
    parser.add_argument("--flank-bp", type=int, default=100)
    parser.add_argument("--min-relative-score", type=float, default=0.80)
    return parser.parse_args()


def _parse_peak_interval(value: object) -> tuple[int, int] | None:
    text = "" if pd.isna(value) else str(value).strip()
    if not text or ":" not in text or "-" not in text:
        return None
    coordinates = text.split(":", 1)[1]
    start, end = coordinates.split("-", 1)
    return int(start), int(end)


def _attach_pol2_peaks(
    sites: pd.DataFrame, audit_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = pd.read_csv(audit_path, sep="\t", low_memory=False)
    audit = audit[audit["paper_class"].eq("lc_dic")].copy()
    merged = sites.drop(columns=["anchor"], errors="ignore").merge(
        audit,
        on=["site_id", "chrom"],
        suffixes=("", "_audit"),
        validate="one_to_one",
    )
    ready = []
    skipped = []
    for _, row in merged.iterrows():
        selected = None
        for condition in ("pol2_ctrl", "pol2_e2_30", "pol2_e2_45"):
            peak = _parse_peak_interval(row.get(f"{condition}_best_peak"))
            summit = pd.to_numeric(
                row.get(f"{condition}_best_summit"), errors="coerce"
            )
            if peak is not None and pd.notna(summit):
                selected = (condition, int(summit), peak)
                break
        if selected is None:
            skipped.append(
                {
                    "site_id": row["site_id"],
                    "reason": "no real Pol2 peak and summit in audit table",
                }
            )
            continue
        condition, summit, peak = selected
        record = row.to_dict()
        record.update(
            {
                "anchor": summit,
                "pol2_condition": condition,
                "pol2_peak_start": peak[0],
                "pol2_peak_end": peak[1],
            }
        )
        ready.append(record)
    return pd.DataFrame.from_records(ready), pd.DataFrame.from_records(skipped)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(args.sites, sep="\t", low_memory=False)
    sites = sites[sites["site_class"].eq("lc_dic")].copy()
    skipped = pd.DataFrame()
    if args.audit:
        sites, skipped = _attach_pol2_peaks(sites, args.audit)
        skipped.to_csv(
            output_dir / "skipped_no_pol2_peak.tsv", sep="\t", index=False
        )
    if sites.empty:
        raise ValueError("no LC-DIC sites remain for motif scanning")
    fasta = FastaReference(args.fasta)
    pwms = {
        label: load_meme_pwm(args.motifs, motif_name=name)
        for label, name in DEFAULT_MOTIFS.items()
    }

    records = []
    for row in sites.itertuples(index=False):
        anchor = int(row.anchor)
        start = max(0, anchor - args.flank_bp)
        end = min(fasta.chrom_length(str(row.chrom)), anchor + args.flank_bp)
        sequence = fasta.extract(str(row.chrom), start, end)
        for label, pwm in pwms.items():
            hits = scan_motif_hits(
                sequence,
                start,
                pwm,
                min_relative_score=args.min_relative_score,
            )
            for rank, hit in enumerate(hits, start=1):
                if (
                    "pol2_peak_start" in sites
                    and not (
                        hit.start >= int(row.pol2_peak_start)
                        and hit.end <= int(row.pol2_peak_end)
                    )
                ):
                    continue
                records.append(
                    {
                        "site_id": row.site_id,
                        "chrom": row.chrom,
                        "anchor": anchor,
                        "motif_label": label,
                        "motif_name": pwm.name,
                        "hit_rank": rank,
                        "motif_start": hit.start,
                        "motif_end": hit.end,
                        "motif_strand": hit.strand,
                        "motif_sequence": hit.genomic_sequence,
                        "pwm_score": hit.score,
                        "relative_score": hit.relative_score,
                        "distance_to_pol2_summit": hit.center - anchor,
                        "pol2_condition": getattr(
                            row, "pol2_condition", ""
                        ),
                        "pol2_peak_start": getattr(
                            row, "pol2_peak_start", pd.NA
                        ),
                        "pol2_peak_end": getattr(
                            row, "pol2_peak_end", pd.NA
                        ),
                    }
                )

    hits = pd.DataFrame.from_records(records)
    if hits.empty:
        raise ValueError("no motif hits passed the requested score threshold")
    hits.to_csv(output_dir / "all_motif_hits.tsv", sep="\t", index=False)
    best = (
        hits.sort_values(
            ["site_id", "motif_label", "hit_rank"], kind="stable"
        )
        .drop_duplicates(["site_id", "motif_label"])
        .reset_index(drop=True)
    )
    best.to_csv(output_dir / "best_motif_hits.tsv", sep="\t", index=False)

    presence = pd.crosstab(best["site_id"], best["motif_label"]).gt(0).reset_index()
    summary_columns = ["site_id", "chrom", "start", "end", "anchor"]
    summary_columns.extend(
        column
        for column in ("pol2_condition", "pol2_peak_start", "pol2_peak_end")
        if column in sites
    )
    site_summary = sites[summary_columns].merge(
        presence, on="site_id", how="left", validate="one_to_one"
    )
    for label in DEFAULT_MOTIFS:
        if label not in site_summary:
            site_summary[label] = False
        site_summary[label] = site_summary[label].eq(True)
    site_summary["n_panel_motifs"] = site_summary[
        list(DEFAULT_MOTIFS)
    ].sum(axis=1)

    family_candidates = best.assign(
        motif_family=best["motif_label"].map(MOTIF_FAMILIES)
    )
    family_candidates = (
        family_candidates.sort_values(
            ["site_id", "motif_family", "relative_score", "motif_label"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .drop_duplicates(["site_id", "motif_family"])
        .reset_index(drop=True)
    )
    family_candidates.insert(
        0,
        "candidate_id",
        family_candidates["site_id"]
        + "__"
        + family_candidates["motif_family"].str.lower(),
    )
    family_candidates["recommended_strategy"] = "pwm_max_disruption_3bp"
    family_candidates.to_csv(
        output_dir / "motif_mutation_candidates.tsv",
        sep="\t",
        index=False,
    )

    enrichment = []
    if args.cluster_assignments:
        clusters = pd.read_csv(
            args.cluster_assignments, sep="\t", low_memory=False
        )
        site_summary = site_summary.merge(
            clusters, on="site_id", validate="one_to_one"
        )
        for label in DEFAULT_MOTIFS:
            table = pd.crosstab(site_summary["cluster"], site_summary[label])
            table = table.reindex(index=[0, 1], columns=[False, True], fill_value=0)
            odds_ratio, pvalue = fisher_exact(
                [
                    [table.loc[1, True], table.loc[1, False]],
                    [table.loc[0, True], table.loc[0, False]],
                ]
            )
            enrichment.append(
                {
                    "motif_label": label,
                    "responder_with_motif": int(table.loc[1, True]),
                    "responder_total": int(table.loc[1].sum()),
                    "nonresponder_with_motif": int(table.loc[0, True]),
                    "nonresponder_total": int(table.loc[0].sum()),
                    "responder_fraction": (
                        table.loc[1, True] / table.loc[1].sum()
                    ),
                    "nonresponder_fraction": (
                        table.loc[0, True] / table.loc[0].sum()
                    ),
                    "odds_ratio": odds_ratio,
                    "fisher_p": pvalue,
                }
            )
    site_summary.to_csv(
        output_dir / "site_motif_summary.tsv", sep="\t", index=False
    )
    if enrichment:
        pd.DataFrame.from_records(enrichment).to_csv(
            output_dir / "responder_motif_enrichment.tsv",
            sep="\t",
            index=False,
        )

    coverage = {
        label: int(site_summary[label].sum()) for label in DEFAULT_MOTIFS
    }
    print(
        f"Scanned {len(site_summary)} LC-DICs; "
        f"sites with >=1 panel motif={int((site_summary['n_panel_motifs'] > 0).sum())}; "
        f"coverage={coverage}; skipped_no_pol2={len(skipped)}"
    )


if __name__ == "__main__":
    main()
