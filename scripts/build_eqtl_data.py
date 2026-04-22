"""Build eQTL benchmark data: positive VCFs (PIP >= 0.9) + TSS-matched negative VCFs.

Output layout (per tissue):
  ~/.cache/eqtl_finemapping/benchmark/
      {tissue}_pos.vcf   — fine-mapped causal SNPs (PIP >= 0.9)
      {tissue}_neg.vcf   — TSS-distance-matched non-eQTL SNPs
      manifest.tsv       — tissue → dataset_id mapping

Negative SNP modes:
  --neg_mode random    (default) Random genomic positions anchored to real TSS.
                       Fast, but not real genotyped variants.
  --neg_mode sumstat   Download the tissue's sumstat file, extract all tested SNPs
                       not in the cross-tissue blacklist, then TSS-distance match.
                       Closer to paper standard (Linder 2025 Methods).

Usage:
    source activate.sh
    python scripts/build_eqtl_data.py [--tissues all|brain|blood] [--pip 0.9] [--seed 42]
    python scripts/build_eqtl_data.py --tissues brain_cortex --neg_mode sumstat
"""

import argparse
import gzip
import os
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
CS_DIR = Path("~/.cache/eqtl_finemapping/gtex_v8_susie_ge").expanduser()
GENCODE_GTF = Path("~/.cache/eqtl_finemapping/gencode/gencode41_basic_nort_protein.gtf").expanduser()
OUT_DIR = Path("~/.cache/eqtl_finemapping/benchmark").expanduser()
TABIX_PATHS = Path("~/.cache/eqtl_finemapping/tabix_ftp_paths.tsv").expanduser()

BORZOI_INPUT_LEN = 524_288
AG_INPUT_LEN = 1_048_576
EDGE_HALF = AG_INPUT_LEN // 2  # use largest model's receptive field

# hg38 chromosome sizes (canonical autosomes + X)
HG38_CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895,
}

VCF_HEADER = """\
##fileformat=VCFv4.2
##reference=hg38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""


# ── Step 1: Load GENCODE TSS ──────────────────────────────────────────────────

def load_tss(gtf_path: Path) -> pd.DataFrame:
    """Extract one canonical TSS per protein-coding gene from GENCODE GTF."""
    print(f"Loading TSS from {gtf_path} ...")
    rows = []
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if parts[2] != "transcript":
                continue
            attrs = parts[8]
            if 'transcript_type "protein_coding"' not in attrs:
                continue
            chrom = parts[0]
            if chrom not in HG38_CHROM_SIZES:
                continue
            strand = parts[6]
            start = int(parts[3])
            end = int(parts[4])
            tss = start if strand == "+" else end
            # extract gene_id
            gene_id = ""
            for tok in attrs.split(";"):
                tok = tok.strip()
                if tok.startswith("gene_id"):
                    gene_id = tok.split('"')[1].split(".")[0]  # strip version suffix
                    break
            rows.append({"chrom": chrom, "tss": tss, "gene_id": gene_id, "strand": strand})

    df = pd.DataFrame(rows)
    # keep one TSS per gene (first encountered, already canonical transcript first in GENCODE)
    df = df.drop_duplicates(subset="gene_id", keep="first")
    print(f"  {len(df)} protein-coding genes with TSS")
    return df.set_index("gene_id")


# ── Step 2: Load credible sets ────────────────────────────────────────────────

def load_cs(cs_file: Path) -> pd.DataFrame:
    """Load a credible set TSV.gz and parse variant fields."""
    df = pd.read_csv(cs_file, sep="\t", compression="gzip",
                     usecols=["gene_id", "cs_id", "variant", "pip", "beta", "se"])
    # parse variant: chr1_12345_A_G → chrom, pos, ref, alt
    split = df["variant"].str.split("_", expand=True)
    df["chrom"] = split[0]
    df["pos"] = split[1].astype(int)
    df["ref"] = split[2]
    df["alt"] = split[3]
    return df


# ── Step 3: Filter positives ──────────────────────────────────────────────────

def filter_positives(df: pd.DataFrame, pip_threshold: float, tss_df: pd.DataFrame) -> pd.DataFrame:
    """Return SNPs meeting PIP >= threshold with valid chromosome edge margin."""
    # SNP only (single nucleotide)
    snp = df["ref"].str.len().eq(1) & df["alt"].str.len().eq(1)
    df = df[snp].copy()

    # PIP threshold
    df = df[df["pip"] >= pip_threshold].copy()

    # valid chroms only
    df = df[df["chrom"].isin(HG38_CHROM_SIZES)].copy()

    # chromosome edge filter (use AG's larger window)
    chrom_sz = df["chrom"].map(HG38_CHROM_SIZES)
    edge_ok = (df["pos"] - EDGE_HALF >= 0) & (df["pos"] + EDGE_HALF <= chrom_sz)
    df = df[edge_ok].copy()

    # keep only genes with known TSS
    df = df[df["gene_id"].isin(tss_df.index)].copy()

    # compute TSS distance
    df["tss"] = df["gene_id"].map(tss_df["tss"])
    df["tss_dist"] = (df["pos"] - df["tss"]).abs()

    return df.reset_index(drop=True)


# ── Step 4: Build all-tissue SNP blacklist ────────────────────────────────────

def build_eqtl_blacklist(all_cs_files: list[Path]) -> set[tuple]:
    """Any variant appearing in any credible set is blacklisted from negatives."""
    print("Building cross-tissue eQTL blacklist ...")
    blacklist: set[tuple] = set()
    for f in all_cs_files:
        try:
            df = pd.read_csv(f, sep="\t", compression="gzip", usecols=["variant"])
            for v in df["variant"]:
                parts = v.split("_")
                if len(parts) >= 4:
                    blacklist.add((parts[0], int(parts[1])))
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")
    print(f"  Blacklist size: {len(blacklist):,} variants")
    return blacklist


# ── Step 5a: Build negatives — random mode (original) ────────────────────────

def build_negatives_random(positives: pd.DataFrame, tss_df: pd.DataFrame,
                           blacklist: set, rng: random.Random) -> pd.DataFrame:
    """For each positive SNP, sample a random TSS-distance-matched negative.

    Negatives are random genomic positions anchored to real TSS offsets.
    Fast but not real genotyped variants (see build_negatives_sumstat for
    paper-standard real SNPs).
    """
    tss_by_chrom: dict[str, np.ndarray] = {}
    for chrom, grp in tss_df.reset_index().groupby("chrom"):
        tss_by_chrom[chrom] = np.sort(grp["tss"].values)

    neg_rows = []
    for _, pos_row in positives.iterrows():
        target_dist = int(pos_row["tss_dist"])
        found = False
        for _ in range(300):
            chroms = list(HG38_CHROM_SIZES.keys())
            weights = [HG38_CHROM_SIZES[c] for c in chroms]
            chrom = rng.choices(chroms, weights=weights, k=1)[0]

            if chrom not in tss_by_chrom or len(tss_by_chrom[chrom]) == 0:
                continue
            anchor_tss = int(rng.choice(tss_by_chrom[chrom]))

            direction = rng.choice([-1, 1])
            neg_pos = anchor_tss + direction * target_dist

            chrom_len = HG38_CHROM_SIZES[chrom]
            if neg_pos - EDGE_HALF < 0 or neg_pos + EDGE_HALF > chrom_len:
                continue
            if neg_pos < 0 or neg_pos > chrom_len:
                continue

            if (chrom, neg_pos) in blacklist:
                continue

            ref_base = rng.choice(["A", "C", "G", "T"])
            alt_base = rng.choice([b for b in ["A", "C", "G", "T"] if b != ref_base])

            neg_rows.append({
                "chrom": chrom, "pos": neg_pos, "ref": ref_base, "alt": alt_base,
                "gene_id": pos_row["gene_id"], "tss_dist": target_dist,
            })
            found = True
            break

        if not found:
            neg_rows.append(None)

    valid = [r for r in neg_rows if r is not None]
    print(f"  Matched negatives (random): {len(valid)}/{len(positives)}")
    return pd.DataFrame(valid) if valid else pd.DataFrame()


# Alias for backward compatibility
build_negatives = build_negatives_random


# ── Step 5b: Build negatives — sumstat mode (paper standard) ─────────────────

SUMSTAT_CACHE_DIR = Path("~/.cache/eqtl_finemapping/sumstats").expanduser()


def _download_sumstat(ftp_url: str, tissue: str) -> Path:
    """Download and cache a sumstat file. Returns local path."""
    SUMSTAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = ftp_url.split("/")[-1]
    local = SUMSTAT_CACHE_DIR / fname
    if local.exists():
        print(f"  Sumstat cached: {local}")
        return local
    # Convert ftp:// to https:// (eQTL Catalogue supports both)
    https_url = ftp_url.replace("ftp://ftp.ebi.ac.uk", "https://ftp.ebi.ac.uk", 1)
    print(f"  Downloading sumstat: {https_url}")
    print(f"  → {local}  (may take several minutes for large files)")
    urllib.request.urlretrieve(https_url, local)
    print(f"  Download complete: {local.stat().st_size / 1e6:.0f} MB")
    return local


def _load_sumstat_snps(sumstat_path: Path, blacklist: set) -> pd.DataFrame:
    """Load all SNPs from a sumstat .all.tsv.gz that are not in the blacklist.

    Sumstat columns: molecular_trait_id, chromosome, position, ref, alt,
                     variant, ..., type, ...
    We keep unique (chrom, pos, ref, alt) SNP rows not in the blacklist.
    """
    print(f"  Parsing sumstat: {sumstat_path.name} ...")
    rows = []
    seen: set[tuple] = set()
    with gzip.open(sumstat_path, "rt") as fh:
        header = fh.readline().rstrip().split("\t")
        col_idx = {c: i for i, c in enumerate(header)}
        chrom_i = col_idx["chromosome"]
        pos_i = col_idx["position"]
        ref_i = col_idx["ref"]
        alt_i = col_idx["alt"]
        type_i = col_idx.get("type", -1)

        for line in fh:
            parts = line.rstrip().split("\t")
            if type_i >= 0 and parts[type_i] != "SNP":
                continue
            chrom = parts[chrom_i]
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            if chrom not in HG38_CHROM_SIZES:
                continue
            try:
                pos = int(parts[pos_i])
            except ValueError:
                continue
            ref = parts[ref_i]
            alt = parts[alt_i]
            # Keep only SNPs (single base change)
            if len(ref) != 1 or len(alt) != 1:
                continue
            key = (chrom, pos)
            if key in seen or key in blacklist:
                continue
            # Chromosome edge filter
            chrom_len = HG38_CHROM_SIZES[chrom]
            if pos - EDGE_HALF < 0 or pos + EDGE_HALF > chrom_len:
                continue
            seen.add(key)
            rows.append({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt})

    df = pd.DataFrame(rows)
    print(f"  Candidate negatives from sumstat: {len(df):,} unique SNPs")
    return df


def _compute_tss_dist(candidates: pd.DataFrame, tss_df: pd.DataFrame) -> pd.DataFrame:
    """Add tss_dist column = distance to nearest protein-coding TSS."""
    tss_by_chrom: dict[str, np.ndarray] = {}
    tss_pos_by_chrom: dict[str, np.ndarray] = {}
    for chrom, grp in tss_df.reset_index().groupby("chrom"):
        tss_by_chrom[chrom] = np.sort(grp["tss"].values)

    dists = []
    for _, row in candidates.iterrows():
        chrom = row["chrom"]
        if chrom not in tss_by_chrom:
            dists.append(np.nan)
            continue
        arr = tss_by_chrom[chrom]
        idx = np.searchsorted(arr, row["pos"])
        candidates_near = arr[max(0, idx-1): idx+2]
        dist = int(np.min(np.abs(candidates_near - row["pos"])))
        dists.append(dist)
    candidates = candidates.copy()
    candidates["tss_dist"] = dists
    return candidates.dropna(subset=["tss_dist"])


def build_negatives_sumstat(positives: pd.DataFrame, tss_df: pd.DataFrame,
                            blacklist: set, ftp_url: str, tissue: str,
                            rng: random.Random) -> pd.DataFrame:
    """TSS-distance-matched negatives from real sumstat SNPs (paper standard).

    1. Download sumstat file (cached).
    2. Keep unique SNPs not in the cross-tissue blacklist.
    3. Compute distance to nearest TSS for each candidate.
    4. For each positive SNP, find a candidate with matching TSS distance ±5%.
    """
    local = _download_sumstat(ftp_url, tissue)
    candidates = _load_sumstat_snps(local, blacklist)
    if candidates.empty:
        print("  WARNING: no candidates from sumstat, falling back to random mode")
        return build_negatives_random(positives, tss_df, blacklist, rng)

    print("  Computing TSS distances for sumstat candidates ...")
    candidates = _compute_tss_dist(candidates, tss_df)

    # Sort by tss_dist for fast range lookup
    candidates = candidates.sort_values("tss_dist").reset_index(drop=True)
    dist_arr = candidates["tss_dist"].values

    neg_rows = []
    used_positions: set[tuple] = set()
    for _, pos_row in positives.iterrows():
        target_dist = int(pos_row["tss_dist"])
        # Allow ±10% window, min ±1000 bp
        delta = max(1000, int(target_dist * 0.10))
        lo, hi = target_dist - delta, target_dist + delta

        lo_idx = int(np.searchsorted(dist_arr, lo, side="left"))
        hi_idx = int(np.searchsorted(dist_arr, hi, side="right"))
        pool_idx = list(range(lo_idx, hi_idx))

        found = False
        rng.shuffle(pool_idx)
        for idx in pool_idx:
            row = candidates.iloc[idx]
            key = (row["chrom"], row["pos"])
            if key in used_positions:
                continue
            used_positions.add(key)
            neg_rows.append({
                "chrom": row["chrom"], "pos": row["pos"],
                "ref": row["ref"], "alt": row["alt"],
                "gene_id": pos_row["gene_id"], "tss_dist": target_dist,
            })
            found = True
            break

        if not found:
            neg_rows.append(None)

    valid = [r for r in neg_rows if r is not None]
    print(f"  Matched negatives (sumstat): {len(valid)}/{len(positives)}")
    return pd.DataFrame(valid) if valid else pd.DataFrame()


# ── Step 6: Write VCF ─────────────────────────────────────────────────────────

def write_vcf(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(VCF_HEADER)
        for _, row in df.sort_values(["chrom", "pos"]).iterrows():
            vid = f"{row['chrom']}_{row['pos']}_{row['ref']}_{row['alt']}"
            fh.write(f"{row['chrom']}\t{int(row['pos'])}\t{vid}\t{row['ref']}\t{row['alt']}\t.\tPASS\t.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build eQTL benchmark VCF pairs.")
    parser.add_argument("--tissues", default="all",
                        help="Comma-sep tissue keywords or 'all' (default: all)")
    parser.add_argument("--pip", type=float, default=0.9,
                        help="PIP threshold for positive label (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--neg_mode", default="random", choices=["random", "sumstat"],
                        help="Negative SNP source: 'random' (default, fast) or "
                             "'sumstat' (paper-standard: real genotyped SNPs from sumstat file).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Load tissue manifest
    tab = pd.read_csv(TABIX_PATHS, sep="\t")
    gtex_ge = tab[(tab["study_label"] == "GTEx") & (tab["quant_method"] == "ge")].copy()

    # Filter by tissue keyword
    if args.tissues != "all":
        keywords = [t.strip().lower() for t in args.tissues.split(",")]
        mask = gtex_ge["sample_group"].apply(
            lambda sg: any(kw in sg.lower() for kw in keywords)
        )
        gtex_ge = gtex_ge[mask]
    print(f"Processing {len(gtex_ge)} tissues")

    # Check which CS files are available; also record sumstat URL per tissue
    available = {}
    sumstat_urls = {}
    for _, row in gtex_ge.iterrows():
        fname = CS_DIR / f"{row['dataset_id']}_{row['sample_group']}.credible_sets.tsv.gz"
        if fname.exists():
            available[row["sample_group"]] = fname
            sumstat_urls[row["sample_group"]] = row["ftp_path"]  # .all.tsv.gz URL

    missing = set(gtex_ge["sample_group"]) - set(available.keys())
    if missing:
        print(f"WARNING: {len(missing)} tissues missing CS files: {sorted(missing)[:5]} ...")

    all_cs_files = list(available.values())

    # Load TSS database
    if not GENCODE_GTF.exists():
        sys.exit(f"ERROR: GENCODE GTF not found at {GENCODE_GTF}")
    tss_df = load_tss(GENCODE_GTF)

    # Build cross-tissue blacklist
    blacklist = build_eqtl_blacklist(all_cs_files)

    print(f"Negative mode: {args.neg_mode}")

    # Save manifest
    manifest_rows = []

    for tissue, cs_file in sorted(available.items()):
        print(f"\n── {tissue} ({cs_file.name}) ──")

        df = load_cs(cs_file)
        positives = filter_positives(df, args.pip, tss_df)
        print(f"  Positives (PIP >= {args.pip}): {len(positives)}")

        if len(positives) == 0:
            print("  SKIP: no qualifying positives")
            continue

        # Remove indel-contaminated credible sets (any indel PIP > 0.1 in same CS)
        indels = df[df["ref"].str.len().ne(1) | df["alt"].str.len().ne(1)]
        bad_cs = set(zip(indels[indels["pip"] > 0.1]["gene_id"],
                         indels[indels["pip"] > 0.1]["cs_id"]))
        if bad_cs:
            pos_key = list(zip(positives["gene_id"], positives["cs_id"]))
            keep = [k not in bad_cs for k in pos_key]
            positives = positives[keep].reset_index(drop=True)
            print(f"  After indel-CS filter: {len(positives)}")

        if args.neg_mode == "sumstat":
            ftp_url = sumstat_urls.get(tissue, "")
            if not ftp_url:
                print(f"  WARNING: no sumstat URL for {tissue}, falling back to random")
                negatives = build_negatives_random(positives, tss_df, blacklist, rng)
            else:
                negatives = build_negatives_sumstat(
                    positives, tss_df, blacklist, ftp_url, tissue, rng
                )
        else:
            negatives = build_negatives_random(positives, tss_df, blacklist, rng)

        pos_vcf = out_dir / f"{tissue}_pos.vcf"
        neg_vcf = out_dir / f"{tissue}_neg.vcf"
        write_vcf(positives, pos_vcf)
        if not negatives.empty:
            write_vcf(negatives, neg_vcf)
        print(f"  Written: {pos_vcf.name} ({len(positives)} SNPs), {neg_vcf.name} ({len(negatives)} SNPs)")

        manifest_rows.append({
            "tissue": tissue,
            "dataset_id": cs_file.stem.split("_")[0],
            "neg_mode": args.neg_mode,
            "n_pos": len(positives),
            "n_neg": len(negatives),
            "pos_vcf": str(pos_vcf),
            "neg_vcf": str(neg_vcf),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.tsv", sep="\t", index=False)
    print(f"\nDone. Manifest: {out_dir / 'manifest.tsv'}")
    print(manifest[["tissue", "n_pos", "n_neg"]].to_string())


if __name__ == "__main__":
    main()
