#!/usr/bin/env python
"""Build a matched Nakato-boundary CTCF perturbation input table.

This experiment creates an input table for ``run_tf_context.py`` with explicit
single-base variants. Each selected CTCF motif gets:

- one PWM max-information-content motif-disrupting SNV;
- one nearby non-motif control SNV inside the same CTCF peak when possible.

The output is intentionally small and cache-friendly: default 4 classes x
6 motif sites x 2 variants = 48 variants, which is divisible by 6 for the
current 4-GPU / batch_size=3 paired prediction path.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
from pyfaidx import Fasta

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.ism.prepare_ctcf_sites import (  # noqa: E402
    CANONICAL_HG38,
    DEFAULT_MEME,
    DEFAULT_UCSC_API,
    BASES,
    best_motif_hit,
    load_meme_pwm,
    pwm_to_pssm,
)

AG_INPUT_LEN = 1_048_576
NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_idx(ref: str) -> int:
    n = 0
    for ch in "".join(x for x in ref if x.isalpha()):
        n = n * 26 + ord(ch.upper()) - 64
    return n - 1


def read_xlsx_first_sheet(path: str | Path) -> pd.DataFrame:
    """Read a simple XLSX first sheet without requiring openpyxl."""

    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in ss_root.findall("a:si", NS):
            shared_strings.append("".join(t.text or "" for t in si.findall(".//a:t", NS)))

        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in root.findall(".//a:sheetData/a:row", NS):
            tmp: dict[int, str] = {}
            maxc = -1
            for cell in row.findall("a:c", NS):
                idx = _col_idx(cell.attrib.get("r", "A1"))
                maxc = max(maxc, idx)
                cell_type = cell.attrib.get("t")
                value = cell.find("a:v", NS)
                if cell_type == "s" and value is not None:
                    cell_value = shared_strings[int(value.text)]
                elif cell_type == "inlineStr":
                    cell_value = "".join(x.text or "" for x in cell.findall(".//a:t", NS))
                elif value is not None:
                    cell_value = value.text or ""
                else:
                    cell_value = ""
                tmp[idx] = cell_value
            rows.append([tmp.get(i, "") for i in range(maxc + 1)])

    return pd.DataFrame(rows[1:], columns=rows[0])


def rank_boundaries(boundaries: pd.DataFrame, boundary_class: str) -> pd.DataFrame:
    """Rank boundaries using the depletion logic described by Nakato 2023."""

    data = boundaries[boundaries["class"] == boundary_class].copy()
    if boundary_class == "CTCF-dependent":
        data["rank_score"] = data["CTCFKD"] - data["Control"]
    elif boundary_class == "CTCF-separated":
        data["rank_score"] = data["Control"] - data["CTCFKD"]
    elif boundary_class == "Cohesin-dependent":
        data["rank_score"] = pd.concat(
            [data["Rad21KD"] - data["Control"], data["NIPBLKD"] - data["Control"]],
            axis=1,
        ).max(axis=1)
    elif boundary_class == "Cohesin-separated":
        data["rank_score"] = pd.concat(
            [data["Control"] - data["Rad21KD"], data["Control"] - data["NIPBLKD"]],
            axis=1,
        ).max(axis=1)
    elif boundary_class == "All-dependent":
        data["rank_score"] = (
            (data["CTCFKD"] - data["Control"]).abs()
            + (data["Rad21KD"] - data["Control"]).abs()
            + (data["NIPBLKD"] - data["Control"]).abs()
        )
    else:
        effect_cols = [
            "Rad21KD",
            "NIPBLKD",
            "CTCFKD",
            "ESCO1KD",
            "Mau2KD",
            "NIPBL_Rad21_KD",
            "WAPLKD",
            "PDS5AKD_72h",
            "PDS5BKD",
            "PDS5ABKD",
        ]
        data["rank_score"] = -data[effect_cols].sub(data["Control"], axis=0).abs().sum(axis=1)
    return data.sort_values("rank_score", ascending=False)


def get_ctcf_peaks(
    session: requests.Session,
    chrom: str,
    start: int,
    end: int,
    api_url: str = DEFAULT_UCSC_API,
) -> list[dict]:
    response = session.get(
        api_url,
        params={
            "genome": "hg38",
            "track": "TFrPeakClusters",
            "chrom": chrom,
            "start": int(start),
            "end": int(end),
            "maxItemsOutput": 100_000,
        },
        timeout=60,
    )
    response.raise_for_status()
    peaks = []
    for item in response.json().get("TFrPeakClusters", []):
        if item.get("factor") != "CTCF":
            continue
        peaks.append(
            {
                "chrom": item["chrom"],
                "start": int(item["chromStart"]),
                "end": int(item["chromEnd"]),
                "name": item["name"],
                "peak_score": int(item["score"]),
                "rpeak_strand": item.get("strand", "."),
                "rpeak_ubiquity": item.get("ubiquity", ""),
                "cCRE": item.get("cCRE", ""),
            }
        )
    return peaks


def information_content(pwm: list[list[float]]) -> list[float]:
    """Information content per PWM row against uniform background."""

    values: list[float] = []
    for row in pwm:
        ic = 2.0
        for prob in row:
            if prob > 0:
                ic += prob * math.log2(prob)
        values.append(ic)
    return values


def choose_max_ic_variant(
    matched_seq: str,
    pwm: list[list[float]],
    motif_start: int,
    strand: str,
) -> tuple[int, str, str, int, float, float, float, int, str, str]:
    """Choose max-IC concrete base and mutate it to a low-probability base.

    ``best_motif_hit`` scores reverse-strand hits using the reverse complement
    of the genomic window. Therefore, PWM probabilities must be interpreted on
    the motif-oriented sequence, then mapped back to genomic coordinates.
    """

    def _rc_base(base: str) -> str:
        return {"A": "T", "C": "G", "G": "C", "T": "A"}[base]

    matched_seq = matched_seq.upper()
    oriented_seq = matched_seq
    if strand == "-":
        oriented_seq = "".join(_rc_base(base) for base in reversed(matched_seq))

    width = len(matched_seq)
    ic = information_content(pwm)
    order = sorted(range(len(pwm)), key=lambda i: (-ic[i], i))
    for pwm_offset in order:
        oriented_ref = oriented_seq[pwm_offset].upper()
        if oriented_ref not in BASES:
            continue
        row = pwm[pwm_offset]
        ref_idx = BASES.index(oriented_ref)
        alt_idx = min((i for i in range(4) if i != ref_idx), key=lambda i: row[i])
        oriented_alt = BASES[alt_idx]
        if strand == "-":
            genomic_offset = width - 1 - pwm_offset
            genomic_ref = _rc_base(oriented_ref)
            genomic_alt = _rc_base(oriented_alt)
        else:
            genomic_offset = pwm_offset
            genomic_ref = oriented_ref
            genomic_alt = oriented_alt
        if matched_seq[genomic_offset] != genomic_ref:
            raise ValueError(
                "Motif orientation mapping failed: "
                f"strand={strand} pwm_offset={pwm_offset} genomic_offset={genomic_offset} "
                f"matched_base={matched_seq[genomic_offset]} expected={genomic_ref}"
            )
        return (
            motif_start + genomic_offset + 1,
            genomic_ref,
            genomic_alt,
            genomic_offset,
            float(row[ref_idx]),
            float(row[alt_idx]),
            float(row[alt_idx] - row[ref_idx]),
            pwm_offset,
            oriented_ref,
            oriented_alt,
        )
    raise ValueError(f"No concrete A/C/G/T base found in motif sequence: {matched_seq}")


def choose_nearby_control(
    fasta: Fasta,
    chrom: str,
    peak_start: int,
    peak_end: int,
    motif_start: int,
    motif_end: int,
) -> tuple[int, str, str]:
    """Choose a deterministic nearby non-motif base in the same peak."""

    center = (motif_start + motif_end) // 2
    candidates: list[int] = []
    for distance in (25, 50, 75, 100, 150, 200):
        candidates.extend([center - distance, center + distance])
    candidates.extend([peak_start + 10, peak_end - 11])

    for pos0 in candidates:
        if pos0 < peak_start or pos0 >= peak_end:
            continue
        if motif_start <= pos0 < motif_end:
            continue
        ref = str(fasta[chrom][pos0 : pos0 + 1]).upper()
        if ref in BASES:
            return pos0 + 1, ref, {"A": "C", "C": "A", "G": "T", "T": "G"}[ref]
    raise ValueError(f"Could not choose nearby control for {chrom}:{peak_start}-{peak_end}")


def random_motif_replacement(matched_seq: str, rng: random.Random) -> str:
    """Return a deterministic random sequence with no base kept unchanged."""

    out: list[str] = []
    for base in matched_seq.upper():
        choices = [b for b in BASES if b != base]
        out.append(rng.choice(choices))
    return "".join(out)


def load_boundaries(path: str | Path) -> pd.DataFrame:
    boundaries = read_xlsx_first_sheet(path)
    for col in boundaries.columns:
        if col not in {"chromosome", "class"}:
            boundaries[col] = pd.to_numeric(boundaries[col], errors="coerce")
    return boundaries


def collect_candidates(
    boundaries: pd.DataFrame,
    fasta: Fasta,
    pwm: list[list[float]],
    pssm: list[list[float]],
    classes: list[str],
    candidates_per_class: int,
    max_boundaries_per_class: int,
    max_motif_to_boundary_center_bp: int | None = None,
    max_boundary_width_bp: int | None = None,
) -> pd.DataFrame:
    session = requests.Session()
    records: list[dict] = []
    seen_peaks: set[tuple[str, int, int]] = set()
    half = AG_INPUT_LEN // 2

    for boundary_class in classes:
        picked = 0
        ranked = rank_boundaries(boundaries, boundary_class).head(max_boundaries_per_class)
        print(f"[collect] {boundary_class}: scanning up to {len(ranked)} boundaries")
        for _, boundary in ranked.iterrows():
            if picked >= candidates_per_class:
                break
            chrom = str(boundary["chromosome"])
            if chrom not in CANONICAL_HG38 or chrom not in fasta:
                continue
            boundary_start = max(0, int(boundary["start"]))
            boundary_end = min(int(boundary["end"]), CANONICAL_HG38[chrom])
            boundary_width = boundary_end - boundary_start
            if max_boundary_width_bp is not None and boundary_width > max_boundary_width_bp:
                continue
            boundary_center = (boundary_start + boundary_end) // 2
            try:
                peaks = get_ctcf_peaks(session, chrom, boundary_start, boundary_end)
            except Exception as exc:
                print(f"  skip API error {chrom}:{boundary_start}-{boundary_end}: {exc}")
                continue

            motif_records: list[dict] = []
            for peak in peaks:
                peak_key = (peak["chrom"], peak["start"], peak["end"])
                if peak_key in seen_peaks:
                    continue
                seq = str(fasta[peak["chrom"]][peak["start"] : peak["end"]]).upper()
                if len(seq) != peak["end"] - peak["start"]:
                    continue
                rel_start, rel_end, strand, motif_score = best_motif_hit(seq, pssm)
                motif_start = peak["start"] + rel_start
                motif_end = peak["start"] + rel_end
                motif_center = (motif_start + motif_end) // 2
                motif_to_boundary_center_bp = abs(motif_center - boundary_center)
                if (
                    max_motif_to_boundary_center_bp is not None
                    and motif_to_boundary_center_bp > max_motif_to_boundary_center_bp
                ):
                    continue
                if motif_center - half < 0 or motif_center + half > len(fasta[peak["chrom"]]):
                    continue
                matched_seq = seq[rel_start:rel_end]
                try:
                    variant = choose_max_ic_variant(matched_seq, pwm, motif_start, strand)
                    control = choose_nearby_control(
                        fasta, peak["chrom"], peak["start"], peak["end"], motif_start, motif_end
                    )
                except ValueError:
                    continue
                motif_records.append(
                    {
                        **peak,
                        "strand": strand,
                        "motif_start": motif_start,
                        "motif_end": motif_end,
                        "matched_seq": matched_seq,
                        "motif_score": float(motif_score),
                        "motif_center": motif_center,
                        "boundary_center": boundary_center,
                        "boundary_width_bp": boundary_width,
                        "motif_to_boundary_center_bp": motif_to_boundary_center_bp,
                        "variant_position": variant[0],
                        "variant_ref": variant[1],
                        "variant_alt": variant[2],
                        "variant_offset": variant[3],
                        "variant_ref_pwm_prob": variant[4],
                        "variant_alt_pwm_prob": variant[5],
                        "variant_delta_pwm_prob": variant[6],
                        "variant_pwm_offset": variant[7],
                        "variant_oriented_ref": variant[8],
                        "variant_oriented_alt": variant[9],
                        "control_position": control[0],
                        "control_ref": control[1],
                        "control_alt": control[2],
                    }
                )
            if not motif_records:
                continue

            # Pick the strongest site per boundary. Later we match distributions
            # across classes before taking the final subset.
            motif_records = sorted(
                motif_records, key=lambda row: (row["peak_score"], row["motif_score"]), reverse=True
            )
            rec = motif_records[0]
            seen_peaks.add((rec["chrom"], rec["start"], rec["end"]))
            picked += 1
            rec.update(
                {
                    "boundary_class": boundary_class,
                    "dic_class": boundary_class,
                    "boundary_chrom": chrom,
                    "boundary_start": boundary_start,
                    "boundary_end": boundary_end,
                    "boundary_center": boundary_center,
                    "boundary_width_bp": boundary_width,
                    "motif_to_boundary_center_bp": rec["motif_to_boundary_center_bp"],
                    "boundary_rank_score": float(boundary["rank_score"]),
                    "boundary_Control": boundary["Control"],
                    "boundary_Rad21KD": boundary["Rad21KD"],
                    "boundary_NIPBLKD": boundary["NIPBLKD"],
                    "boundary_CTCFKD": boundary["CTCFKD"],
                }
            )
            records.append(rec)
            print(
                "  candidate",
                boundary_class,
                rec["chrom"],
                rec["motif_center"],
                rec["peak_score"],
                f"motif={rec['motif_score']:.3f}",
                f"dist={rec['motif_to_boundary_center_bp']}",
            )
            time.sleep(0.03)
        if picked < candidates_per_class:
            print(f"WARNING: only collected {picked}/{candidates_per_class} for {boundary_class}")

    return pd.DataFrame.from_records(records)


def select_matched_sites(
    candidates: pd.DataFrame,
    classes: list[str],
    sites_per_class: int,
    match_geometry: bool = True,
) -> pd.DataFrame:
    """Greedy class-balanced matching on score and boundary geometry."""

    candidates = candidates[candidates["dic_class"].isin(classes)].copy()
    match_specs = [
        ("peak_score", "peak_norm"),
        ("motif_score", "motif_norm"),
    ]
    if match_geometry:
        for col, norm_col in [
            ("motif_to_boundary_center_bp", "distance_norm"),
            ("boundary_width_bp", "boundary_width_norm"),
        ]:
            if col in candidates.columns:
                match_specs.append((col, norm_col))

    for col, norm_col in match_specs:
        values = candidates[col].astype(float)
        span = float(values.max() - values.min())
        if span == 0:
            candidates[norm_col] = 0.0
        else:
            candidates[norm_col] = (values - float(values.min())) / span

    quantiles = (0.35, 0.5, 0.65, 0.8)
    centers: list[tuple[float, ...]] = [()]
    for _, norm_col in match_specs:
        col_centers = [float(candidates[norm_col].quantile(q)) for q in quantiles]
        centers = [prev + (center,) for prev in centers for center in col_centers]
    norm_cols = [norm_col for _, norm_col in match_specs]

    best: pd.DataFrame | None = None
    best_score = float("inf")
    for center in centers:
        parts = []
        for boundary_class in classes:
            sub = candidates[candidates["dic_class"] == boundary_class].copy()
            sub["match_distance"] = 0.0
            for norm_col, target in zip(norm_cols, center):
                sub["match_distance"] += (sub[norm_col] - target) ** 2
            parts.append(sub.sort_values("match_distance").head(sites_per_class))
        selected = pd.concat(parts, ignore_index=True)
        if selected.groupby("dic_class").size().min() < sites_per_class:
            continue
        # Lower cross-class mean/variance difference is better.
        stats = selected.groupby("dic_class")[norm_cols].mean()
        score = float(stats.var().sum()) + float(selected["match_distance"].mean()) * 0.01
        if score < best_score:
            best_score = score
            best = selected
    if best is None:
        raise ValueError("Could not build a balanced matched set")
    return best.drop(columns=norm_cols, errors="ignore").reset_index(drop=True)


def expand_with_controls(
    selected: pd.DataFrame,
    include_random_motif_replacement: bool = False,
    random_seed: int = 20260519,
) -> pd.DataFrame:
    rows: list[dict] = []
    rng = random.Random(random_seed)
    for class_idx, (_, rec) in enumerate(selected.iterrows(), start=1):
        base_id = f"matched_{rec['dic_class'].replace('-', '_')}_{class_idx:02d}"
        common = rec.to_dict()
        common.update(
            {
                "gene": "",
                "paper_context": "Nakato 2023 matched CTCF boundary-class pilot",
                "paper_note": (
                    f"Supplementary Data 5 boundary {rec['boundary_chrom']}:"
                    f"{int(rec['boundary_start'])}-{int(rec['boundary_end'])}; "
                    f"class={rec['dic_class']}; rank_score={rec['boundary_rank_score']:.6g}"
                ),
            }
        )
        motif_row = dict(common)
        motif_row.update(
            {
                "site_id": base_id + "_motif",
                "paired_site_id": base_id,
                "control_type": "ctcf_motif_max_ic_disruption",
                "variant_position": int(rec["variant_position"]),
                "variant_ref": rec["variant_ref"],
                "variant_alt": rec["variant_alt"],
                "variant_start": int(rec["variant_position"]) - 1,
                "variant_end": int(rec["variant_position"]),
                "variant_ref_seq": rec["variant_ref"],
                "variant_alt_seq": rec["variant_alt"],
                "mutation_strategy": "ctcf_pwm_max_ic_to_min_prob",
            }
        )
        control_row = dict(common)
        control_row.update(
            {
                "site_id": base_id + "_nearby_control",
                "paired_site_id": base_id,
                "control_type": "same_peak_non_motif_nearby_base",
                "variant_position": int(rec["control_position"]),
                "variant_ref": rec["control_ref"],
                "variant_alt": rec["control_alt"],
                "variant_start": int(rec["control_position"]) - 1,
                "variant_end": int(rec["control_position"]),
                "variant_ref_seq": rec["control_ref"],
                "variant_alt_seq": rec["control_alt"],
                "mutation_strategy": "same_peak_non_motif_control",
            }
        )
        rows.extend([motif_row, control_row])
        if include_random_motif_replacement:
            motif_start = int(rec["motif_start"])
            motif_end = int(rec["motif_end"])
            ref_seq = str(rec["matched_seq"]).upper()
            alt_seq = random_motif_replacement(ref_seq, rng)
            center_position = motif_start + (motif_end - motif_start) // 2 + 1
            random_row = dict(common)
            random_row.update(
                {
                    "site_id": base_id + "_whole_motif_random",
                    "paired_site_id": base_id,
                    "control_type": "whole_motif_random_replacement",
                    "variant_position": center_position,
                    "variant_ref": ref_seq,
                    "variant_alt": alt_seq,
                    "variant_start": motif_start,
                    "variant_end": motif_end,
                    "variant_ref_seq": ref_seq,
                    "variant_alt_seq": alt_seq,
                    "mutation_strategy": "whole_motif_random_replacement",
                    "random_seed": random_seed,
                }
            )
            rows.append(random_row)
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary_xlsx",
        default="agent-doc/ref/nakato_2023_supplementary/supplementary_data_5.xlsx",
    )
    parser.add_argument("--fasta", default="/home/fqijun/.local/share/genomes/hg38/hg38.fa")
    parser.add_argument("--meme", default=str(DEFAULT_MEME))
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["CTCF-dependent", "Cohesin-dependent", "Robust", "Cohesin-separated"],
    )
    parser.add_argument("--sites_per_class", type=int, default=6)
    parser.add_argument("--candidates_per_class", type=int, default=30)
    parser.add_argument("--max_boundaries_per_class", type=int, default=800)
    parser.add_argument(
        "--max_motif_to_boundary_center_bp",
        type=int,
        default=None,
        help="Drop candidate motifs farther than this distance from the Nakato boundary center.",
    )
    parser.add_argument(
        "--max_boundary_width_bp",
        type=int,
        default=None,
        help="Drop Nakato boundary intervals wider than this threshold before motif selection.",
    )
    parser.add_argument(
        "--no_match_geometry",
        action="store_true",
        help="Match only peak_score and motif_score, not distance-to-boundary-center or boundary width.",
    )
    parser.add_argument(
        "--include_random_motif_replacement",
        action="store_true",
        help="Add a third perturbation per site replacing the full CTCF motif with a deterministic random sequence.",
    )
    parser.add_argument("--random_seed", type=int, default=20260519)
    parser.add_argument(
        "--output",
        default="agent-doc/ism_context/literature_ctcf_sites/nakato2023_matched_4class_6sites_48variants.tsv",
    )
    parser.add_argument(
        "--candidate_output",
        default="agent-doc/ism_context/literature_ctcf_sites/nakato2023_matched_4class_candidates.tsv",
    )
    args = parser.parse_args()

    boundaries = load_boundaries(args.boundary_xlsx)
    fasta = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    pwm = load_meme_pwm(args.meme, motif_name="CTCF")
    pssm = pwm_to_pssm(pwm)

    candidates = collect_candidates(
        boundaries=boundaries,
        fasta=fasta,
        pwm=pwm,
        pssm=pssm,
        classes=args.classes,
        candidates_per_class=args.candidates_per_class,
        max_boundaries_per_class=args.max_boundaries_per_class,
        max_motif_to_boundary_center_bp=args.max_motif_to_boundary_center_bp,
        max_boundary_width_bp=args.max_boundary_width_bp,
    )
    candidate_output = Path(args.candidate_output)
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidate_output, sep="\t", index=False)

    selected = select_matched_sites(
        candidates,
        classes=args.classes,
        sites_per_class=args.sites_per_class,
        match_geometry=not args.no_match_geometry,
    )
    output = expand_with_controls(
        selected,
        include_random_motif_replacement=args.include_random_motif_replacement,
        random_seed=args.random_seed,
    )

    keep_cols = [
        "site_id",
        "paired_site_id",
        "chrom",
        "start",
        "end",
        "gene",
        "paper_context",
        "dic_class",
        "motif_start",
        "motif_end",
        "matched_seq",
        "paper_note",
        "control_type",
        "mutation_strategy",
        "variant_position",
        "variant_ref",
        "variant_alt",
        "variant_start",
        "variant_end",
        "variant_ref_seq",
        "variant_alt_seq",
        "variant_offset",
        "variant_ref_pwm_prob",
        "variant_alt_pwm_prob",
        "variant_delta_pwm_prob",
        "variant_pwm_offset",
        "variant_oriented_ref",
        "variant_oriented_alt",
        "random_seed",
        "peak_score",
        "motif_score",
        "motif_center",
        "boundary_center",
        "motif_to_boundary_center_bp",
        "boundary_width_bp",
        "strand",
        "rpeak_ubiquity",
        "cCRE",
        "boundary_chrom",
        "boundary_start",
        "boundary_end",
        "boundary_class",
        "boundary_rank_score",
        "boundary_Control",
        "boundary_Rad21KD",
        "boundary_NIPBLKD",
        "boundary_CTCFKD",
    ]
    output = output[[col for col in keep_cols if col in output.columns]]

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, sep="\t", index=False)

    print(f"Wrote candidates: {candidate_output} ({len(candidates)} rows)")
    print(f"Wrote matched variants: {path} ({len(output)} rows)")
    print(output.groupby(["dic_class", "control_type"]).size().to_string())
    selected_stats = selected.groupby("dic_class")[
        [
            col
            for col in [
                "peak_score",
                "motif_score",
                "motif_to_boundary_center_bp",
                "boundary_width_bp",
            ]
            if col in selected.columns
        ]
    ].agg(["mean", "min", "max"])
    print("\nSelected motif-site score distribution:")
    print(selected_stats.round(3).to_string())


if __name__ == "__main__":
    main()
