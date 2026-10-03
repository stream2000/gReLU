"""AlphaGenome single-track inference demo.

What was run
------------
Region  : chr2:215,930,000–216,020,000 (hg38)
Window  : chr2:215,450,712–216,499,288 (1,048,576 bp, centered on the region above)
Sequence: fetched from the UCSC REST API (https://api.genome.ucsc.edu)
Weights : /work/qijun/model_weights/model_fold_0.safetensors
Output  : chip_tf, resolution = 128 bp

Tracks predicted (703 bins × 128 bp covering the ROI):
  track 776 — CTCF  / MCF-7
  track 812 — RAD21 / MCF-7  (replicate 1)
  track 813 — RAD21 / MCF-7  (replicate 2)

Tracks NOT available in the model:
  CTCF  / RPE1  — RPE1 was not a training target; no output head exists
  RAD21 / RPE1  — same reason

Results saved to /work/qijun/result/:
  alphagenome_CTCF_MCF7.bw / .bedgraph
  alphagenome_RAD21_MCF7_rep1.bw / .bedgraph
  alphagenome_RAD21_MCF7_rep2.bw / .bedgraph



Usage
-----
# Test with a random sequence (verify the environment is working):
python demo_inference.py --random

# Provide a DNA string (must be exactly 1,048,576 bp):
python demo_inference.py --seq ACGT...

# Use a FASTA file with genomic coordinates:
python demo_inference.py --fasta hg38.fa --region chr1:1000000-2048576

# Specify which track to extract (default: print first 5 CAGE tracks):
python demo_inference.py --random --output_type cage --track_index 0

# List all available track metadata for a given output type:
python demo_inference.py --list_tracks --output_type cage | head -20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 1_048_576  # 2^20 bp — AlphaGenome's required input length
MODEL_REPO = "gtca/alphagenome_pytorch"
MODEL_FILENAME = "model_fold_0.safetensors"
VALID_OUTPUT_TYPES = ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_dna(seq: str) -> torch.Tensor:
    """Convert a DNA string to a one-hot tensor of shape (1, L, 4)."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        idx = mapping.get(base)
        if idx is not None:
            arr[i, idx] = 1.0
        # N or unknown base → all-zero row (no encoding)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, L, 4)


def fetch_ucsc_sequence(region: str, genome: str = "hg38") -> str:
    """Fetch DNA sequence from the UCSC REST API.

    Args:
        region: e.g. "chr2:215450712-216499288"
        genome: UCSC genome assembly (default: hg38)
    """
    import urllib.request, json
    chrom, coords = region.rsplit(":", 1)
    start, end = (int(x) for x in coords.split("-"))
    if end - start != SEQUENCE_LENGTH:
        sys.exit(
            f"Region length {end-start:,} bp ≠ required {SEQUENCE_LENGTH:,} bp. "
            f"Adjust coordinates so end - start == {SEQUENCE_LENGTH}."
        )
    url = (
        f"https://api.genome.ucsc.edu/getData/sequence"
        f"?genome={genome};chrom={chrom};start={start};end={end}"
    )
    print(f"[→] Fetching {chrom}:{start}-{end} from UCSC ({genome}) …")
    with urllib.request.urlopen(url, timeout=120) as r:
        seq = json.loads(r.read())["dna"]
    print(f"[✓] Got {len(seq):,} bp")
    return seq


def load_fasta_region(fasta_path: str, region: str) -> str:
    """Extract sequence from a FASTA file using pyfaidx."""
    try:
        from pyfaidx import Fasta
    except ImportError:
        sys.exit("pyfaidx is required for --fasta mode. Run: pip install pyfaidx")

    chrom, coords = region.rsplit(":", 1)
    start, end = coords.split("-")
    start, end = int(start), int(end)
    length = end - start
    if length != SEQUENCE_LENGTH:
        sys.exit(
            f"Region length {length:,} bp ≠ required {SEQUENCE_LENGTH:,} bp. "
            f"Adjust coordinates so end - start == {SEQUENCE_LENGTH}."
        )
    fa = Fasta(fasta_path)
    seq = fa[chrom][start:end].seq
    return seq


def download_model(cache_dir: Path) -> Path:
    """Download model weights from HuggingFace Hub if not already cached."""
    weights_path = cache_dir / MODEL_FILENAME
    if weights_path.exists():
        print(f"[✓] Using cached weights: {weights_path}")
        return weights_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required to auto-download weights.\n"
            "Run: pip install huggingface_hub\n"
            f"Or manually place {MODEL_FILENAME} at {weights_path}"
        )

    print(f"[↓] Downloading {MODEL_FILENAME} from HuggingFace ({MODEL_REPO}) …")
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(cache_dir),
    )
    print(f"[✓] Saved to {weights_path}")
    return weights_path


def load_track_metadata(output_type: str):
    """Load track metadata parquet for a given output_type."""
    try:
        import pandas as pd
    except ImportError:
        return None

    data_dir = Path(__file__).parent / "src" / "alphagenome_pytorch" / "src" / "alphagenome_pytorch" / "data"
    parquet = data_dir / "track_metadata_human.parquet"
    if not parquet.exists():
        return None

    df = pd.read_parquet(parquet)
    return df[df["output_type"] == output_type].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AlphaGenome inference demo — input DNA, get one track's predictions."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--random", action="store_true", help="Use random DNA (for testing)")
    input_group.add_argument("--seq", type=str, help=f"DNA string, exactly {SEQUENCE_LENGTH:,} bp")
    input_group.add_argument("--fasta", type=str, help="Path to FASTA file")
    input_group.add_argument("--ucsc", type=str, metavar="REGION",
                             help=f"Fetch from UCSC REST API, e.g. chr2:215450712-216499288 (must be {SEQUENCE_LENGTH:,} bp)")

    parser.add_argument("--region", type=str,
                        help="Genomic region e.g. chr1:1000000-2048576 (required with --fasta)")
    parser.add_argument("--genome", type=str, default="hg38",
                        help="UCSC genome assembly for --ucsc mode (default: hg38)")
    parser.add_argument("--output_type", type=str, default="cage",
                        choices=VALID_OUTPUT_TYPES,
                        help="Which assay type to extract (default: cage)")
    parser.add_argument("--track_index", type=int, default=None,
                        help="Track index within the output type (default: print first 5)")
    parser.add_argument("--resolution", type=int, default=128, choices=[1, 128],
                        help="Output resolution in bp (default: 128)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (.safetensors or .pth). "
                             "If omitted, downloads from HuggingFace.")
    parser.add_argument("--cache_dir", type=str, default=str(Path.home() / ".cache" / "alphagenome"),
                        help="Directory to cache downloaded weights")
    parser.add_argument("--list_tracks", action="store_true",
                        help="Print track metadata for --output_type and exit (no inference)")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch device (default: cuda if available, else cpu)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (.npy and .bedgraph). "
                             "Requires --ucsc or --fasta to derive genomic coordinates.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # --list_tracks: just print metadata and exit
    # ------------------------------------------------------------------
    if args.list_tracks:
        meta = load_track_metadata(args.output_type)
        if meta is None:
            print("Track metadata not found (pandas or parquet file missing).")
        else:
            print(meta[["track_index", "track_name", "biosample_name", "assay_title"]].to_string(index=False))
        return

    # ------------------------------------------------------------------
    # Prepare input sequence
    # ------------------------------------------------------------------
    if args.ucsc:
        dna_str = fetch_ucsc_sequence(args.ucsc, genome=args.genome)
        dna_input = encode_dna(dna_str)
    elif args.fasta:
        if not args.region:
            parser.error("--fasta requires --region")
        print(f"[→] Loading sequence from {args.fasta} ({args.region})")
        dna_str = load_fasta_region(args.fasta, args.region)
        dna_input = encode_dna(dna_str)
    elif args.seq:
        if len(args.seq) != SEQUENCE_LENGTH:
            parser.error(f"--seq must be exactly {SEQUENCE_LENGTH:,} bp, got {len(args.seq):,}")
        dna_input = encode_dna(args.seq)
    else:
        # Default: use a random sequence
        print(f"[→] Using random DNA sequence ({SEQUENCE_LENGTH:,} bp)")
        np.random.seed(0)
        rand_idx = np.random.randint(0, 4, size=(1, SEQUENCE_LENGTH))
        dna_input = torch.from_numpy(np.eye(4, dtype=np.float32)[rand_idx])  # (1, L, 4)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[→] Device: {device}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    weights_path = Path(args.weights) if args.weights else download_model(Path(args.cache_dir))

    from alphagenome_pytorch import AlphaGenome

    print("[→] Loading model …")
    model = AlphaGenome.from_pretrained(str(weights_path), device=device)
    model.eval()
    print("[✓] Model ready")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    dna_input = dna_input.to(device)
    organism_index = torch.tensor([0], dtype=torch.long, device=device)  # 0 = human (hg38)

    print("[→] Running inference …")
    with torch.no_grad():
        outputs = model(dna_input, organism_index)
    print("[✓] Inference done")

    # ------------------------------------------------------------------
    # Extract requested track
    # ------------------------------------------------------------------
    out_type = args.output_type
    res = args.resolution

    raw = outputs.get(out_type)
    if raw is None:
        sys.exit(f"Output type '{out_type}' not found. Available: {list(outputs.keys())}")

    if isinstance(raw, dict):
        if res not in raw:
            available = list(raw.keys())
            sys.exit(f"Resolution {res}bp not available for '{out_type}'. Available: {available}")
        tensor = raw[res]  # shape: (1, L_bins, N_tracks)
    else:
        tensor = raw  # some output heads return a flat tensor

    arr = tensor[0].float().cpu().numpy()  # shape: (L_bins, N_tracks)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    meta = load_track_metadata(out_type)

    if args.track_index is not None:
        indices = [args.track_index]
    else:
        indices = list(range(min(5, arr.shape[1])))

    print(f"\n{'='*60}")
    print(f"Output type : {out_type.upper()}")
    print(f"Resolution  : {res} bp")
    print(f"Shape       : {arr.shape}  (positions × tracks)")
    print(f"{'='*60}")

    # Resolve genomic coordinates for output (needed for bedgraph)
    win_chrom, win_start = None, None
    if args.ucsc:
        win_chrom, coords = args.ucsc.rsplit(":", 1)
        win_start = int(coords.split("-")[0])
    elif args.fasta and args.region:
        win_chrom, coords = args.region.rsplit(":", 1)
        win_start = int(coords.split("-")[0])

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        track_arr = arr[:, idx]
        name, safe_name = "unknown", f"track{idx}"
        if meta is not None and idx < len(meta):
            row = meta.iloc[idx]
            name = f"{row['biosample_name']} ({row['assay_title']})"
            tf   = str(row.get("transcription_factor") or "").strip() or out_type
            bs   = str(row.get("biosample_name") or "").strip().replace(" ", "_").replace("/", "-")
            safe_name = f"{out_type}_{tf}_{bs}_track{idx}"

        print(f"\n  Track {idx:4d}: {name}")
        print(f"    min={track_arr.min():.4f}  max={track_arr.max():.4f}  "
              f"mean={track_arr.mean():.4f}  shape={track_arr.shape}")

        if out_dir:
            # .npy — raw float32 array
            npy_path = out_dir / f"{safe_name}.npy"
            np.save(npy_path, track_arr)
            print(f"    saved → {npy_path}")

            # .bedgraph — only if genomic coordinates are known
            if win_chrom and win_start is not None:
                bg_path = out_dir / f"{safe_name}.bedgraph"
                with open(bg_path, "w") as f:
                    f.write(f'track type=bedGraph name="{safe_name}" '
                            f'description="AlphaGenome {out_type} {res}bp"\n')
                    for i, val in enumerate(track_arr):
                        start = win_start + i * res
                        f.write(f"{win_chrom}\t{start}\t{start + res}\t{val:.4f}\n")
                print(f"    saved → {bg_path}")

    print(f"\n[✓] Done.")


if __name__ == "__main__":
    main()
