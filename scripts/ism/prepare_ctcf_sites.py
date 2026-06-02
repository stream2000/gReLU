#!/usr/bin/env python
"""Prepare hg38 CTCF motif sites from UCSC ENCODE4 rPeak clusters.

The output schema is compatible with ``scripts/ism/run_tf_context.py``:

chrom, start, end, name, peak_score, strand, motif_start, motif_end, matched_seq
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests
from pyfaidx import Fasta


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RPEAK_BB = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/ENCODE4/TFrPeakClusters.bb"
DEFAULT_UCSC_API = "https://api.genome.ucsc.edu/getData/track"
DEFAULT_MEME = REPO_ROOT / "src/grelu/resources/meme/jaspar_2024_consensus.meme"
DEFAULT_FASTA = Path("/home/fqijun/.local/share/genomes/hg38/hg38.fa")
DEFAULT_BIGBEDTOBED = "/work/git/UCSC_utils/bigBedToBed"
CANONICAL_HG38 = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
}

BASES = "ACGT"
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1].upper()


def load_meme_pwm(path: str | Path, motif_name: str = "CTCF") -> list[list[float]]:
    """Load the first MEME text-format motif whose header contains motif_name."""

    pwm: list[list[float]] = []
    in_motif = False
    in_matrix = False
    with Path(path).open() as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("MOTIF "):
                in_motif = motif_name.upper() in line.upper()
                in_matrix = False
                pwm = []
                continue
            if not in_motif:
                continue
            if line.startswith("letter-probability matrix"):
                in_matrix = True
                continue
            if in_matrix:
                if not line or line.startswith("URL") or line.startswith("MOTIF "):
                    break
                values = [float(x) for x in line.split()]
                if len(values) == 4:
                    pwm.append(values)
    if not pwm:
        raise ValueError(f"Could not find motif containing {motif_name!r} in {path}")
    return pwm


def pwm_to_pssm(pwm: list[list[float]], pseudocount: float = 1e-4) -> list[list[float]]:
    """Convert A/C/G/T probabilities to log2 odds against uniform background."""

    pssm: list[list[float]] = []
    for row in pwm:
        total = sum(row) + 4 * pseudocount
        pssm.append([math.log2(((x + pseudocount) / total) / 0.25) for x in row])
    return pssm


def score_window(seq: str, pssm: list[list[float]]) -> float:
    score = 0.0
    for base, row in zip(seq, pssm):
        idx = BASES.find(base)
        if idx < 0:
            return float("-inf")
        score += row[idx]
    return score


def best_motif_hit(seq: str, pssm: list[list[float]]) -> tuple[int, int, str, float]:
    """Return start, end, strand, score for the best PWM match in seq."""

    seq = seq.upper()
    width = len(pssm)
    best = (0, width, "+", float("-inf"))
    if len(seq) < width:
        return best

    for start in range(0, len(seq) - width + 1):
        end = start + width
        window = seq[start:end]
        plus = score_window(window, pssm)
        if plus > best[3]:
            best = (start, end, "+", plus)
        minus = score_window(reverse_complement(window), pssm)
        if minus > best[3]:
            best = (start, end, "-", minus)
    return best


def iter_ucsc_ctcf_rpeaks(
    bigbedtobed: str | Path,
    rpeak_bb: str,
    max_peaks: int | None = None,
):
    """Yield CTCF rows from UCSC ENCODE4 TFrPeakClusters bigBed."""

    cmd = [str(bigbedtobed), rpeak_bb, "stdout"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    seen = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 12:
            continue
        factor = fields[12]
        if factor != "CTCF":
            continue
        seen += 1
        yield {
            "chrom": fields[0],
            "start": int(fields[1]),
            "end": int(fields[2]),
            "name": fields[3],
            "peak_score": int(fields[4]),
            "rpeak_strand": fields[5],
            "factor": factor,
            "ubiquity": fields[13] if len(fields) > 13 else "",
            "cCRE": fields[14] if len(fields) > 14 else "",
        }
        if max_peaks is not None and seen >= max_peaks:
            proc.terminate()
            break
    _, stderr = proc.communicate()
    if proc.returncode not in (0, -15):
        raise RuntimeError(f"bigBedToBed failed with code {proc.returncode}: {stderr}")


def iter_ucsc_api_ctcf_rpeaks(
    api_url: str = DEFAULT_UCSC_API,
    chunk_bp: int = 5_000_000,
    max_peaks: int | None = None,
    chromosomes: list[str] | None = None,
):
    """Yield CTCF rows from UCSC ENCODE4 TFrPeakClusters using the REST API."""

    seen = 0
    chroms = chromosomes or list(CANONICAL_HG38)
    session = requests.Session()
    for chrom in chroms:
        chrom_size = CANONICAL_HG38[chrom]
        for start in range(0, chrom_size, chunk_bp):
            end = min(start + chunk_bp, chrom_size)
            response = session.get(
                api_url,
                params={
                    "genome": "hg38",
                    "track": "TFrPeakClusters",
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "maxItemsOutput": 1_000_000,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            for item in data.get("TFrPeakClusters", []):
                if item.get("factor") != "CTCF":
                    continue
                seen += 1
                yield {
                    "chrom": item["chrom"],
                    "start": int(item["chromStart"]),
                    "end": int(item["chromEnd"]),
                    "name": item["name"],
                    "peak_score": int(item["score"]),
                    "rpeak_strand": item.get("strand", "."),
                    "factor": item["factor"],
                    "ubiquity": item.get("ubiquity", ""),
                    "cCRE": item.get("cCRE", ""),
                }
                if max_peaks is not None and seen >= max_peaks:
                    return


def prepare_sites(
    fasta_path: str | Path,
    output: str | Path,
    rpeak_bb: str = DEFAULT_RPEAK_BB,
    source: str = "api",
    api_url: str = DEFAULT_UCSC_API,
    chunk_bp: int = 5_000_000,
    meme_path: str | Path = DEFAULT_MEME,
    bigbedtobed: str | Path = DEFAULT_BIGBEDTOBED,
    motif_name: str = "CTCF",
    max_peaks: int | None = None,
    min_motif_score: float | None = None,
) -> pd.DataFrame:
    pwm = load_meme_pwm(meme_path, motif_name=motif_name)
    pssm = pwm_to_pssm(pwm)
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    records: list[dict] = []
    if source == "api":
        peak_iter = iter_ucsc_api_ctcf_rpeaks(
            api_url=api_url,
            chunk_bp=chunk_bp,
            max_peaks=max_peaks,
        )
    elif source == "bigbed":
        peak_iter = iter_ucsc_ctcf_rpeaks(bigbedtobed, rpeak_bb, max_peaks=max_peaks)
    else:
        raise ValueError(f"Unknown source: {source}")

    for peak in peak_iter:
        if peak["chrom"] not in fasta:
            continue
        seq = str(fasta[peak["chrom"]][peak["start"] : peak["end"]]).upper()
        if len(seq) != peak["end"] - peak["start"]:
            continue
        motif_rel_start, motif_rel_end, strand, motif_score = best_motif_hit(seq, pssm)
        if min_motif_score is not None and motif_score < min_motif_score:
            continue
        motif_start = peak["start"] + motif_rel_start
        motif_end = peak["start"] + motif_rel_end
        matched_seq = seq[motif_rel_start:motif_rel_end]
        records.append(
            {
                "chrom": peak["chrom"],
                "start": peak["start"],
                "end": peak["end"],
                "name": peak["name"],
                "peak_score": peak["peak_score"],
                "strand": strand,
                "motif_start": motif_start,
                "motif_end": motif_end,
                "matched_seq": matched_seq,
                "motif_score": motif_score,
                "rpeak_ubiquity": peak["ubiquity"],
                "cCRE": peak["cCRE"],
            }
        )

    out = pd.DataFrame.from_records(records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, sep="\t", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default=str(DEFAULT_FASTA))
    parser.add_argument("--output", default="agent-doc/ism_context/ctcf_sites/encode4_ctcf_rpeak_jaspar_sites.tsv")
    parser.add_argument("--source", choices=["api", "bigbed"], default="api")
    parser.add_argument("--api_url", default=DEFAULT_UCSC_API)
    parser.add_argument("--rpeak_bb", default=DEFAULT_RPEAK_BB)
    parser.add_argument("--chunk_bp", type=int, default=5_000_000)
    parser.add_argument("--meme", default=str(DEFAULT_MEME))
    parser.add_argument("--bigBedToBed", default=DEFAULT_BIGBEDTOBED)
    parser.add_argument("--motif_name", default="CTCF")
    parser.add_argument("--max_peaks", type=int, default=None)
    parser.add_argument("--min_motif_score", type=float, default=None)
    args = parser.parse_args()

    out = prepare_sites(
        fasta_path=args.fasta,
        output=args.output,
        rpeak_bb=args.rpeak_bb,
        source=args.source,
        api_url=args.api_url,
        chunk_bp=args.chunk_bp,
        meme_path=args.meme,
        bigbedtobed=args.bigBedToBed,
        motif_name=args.motif_name,
        max_peaks=args.max_peaks,
        min_motif_score=args.min_motif_score,
    )
    print(f"Wrote {len(out)} CTCF sites to {args.output}")
    if len(out):
        print(out.head().to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
