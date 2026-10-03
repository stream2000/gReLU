#!/usr/bin/env python
"""Plot observed-vs-LoRA Borzoi predictions on positive-control loci."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
import torch
from PIL import Image
from pyfaidx import Fasta

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "ft-scripts"))

import grelu.lightning  # noqa: E402
from borzoi_mmap_dataset import transform_binned_signal  # noqa: E402
from train_borzoi import (  # noqa: E402
    BORZOI_LORA_TARGET_MODULES,
    DEFAULT_BIGWIG_DIR,
    DEFAULT_GENOME,
    TASK_NAMES,
    apply_lora_to_borzoi_embedding,
)
from grelu.sequence.format import BASE_TO_INDEX_HASH, indices_to_one_hot  # noqa: E402


DEFAULT_CONTROLS = [
    ("Cd68", "Cd68-201", "-", "chr11", 69659212, 69671153, "mac", "test"),
    ("Col1a1", "Col1a1-201", "+", "chr11", 94931223, 94958042, "hsc", "test"),
    ("Epcam", "Epcam-201", "+", "chr17", 87630978, 87656106, "chol", "train_chrom"),
    ("Fabp4", "Fabp4-201", "-", "chr3", 10199087, 10213576, "lsec", "train_chrom"),
    ("Mmp2", "Mmp2-201", "+", "chr8", 92822290, 92858420, "hsc", "train_chrom"),
    ("Myc", "Myc-205", "+", "chr15", 61980390, 61995350, "", "train_chrom"),
    ("Nipbl", "Nipbl-201", "-", "chr15", 8285616, 8449463, "", "train_chrom"),
    ("Pcdh17", "Pcdh17-201", "+", "chr14", 84438562, 84544002, "lsec", "train_chrom"),
    ("Trem2", "Trem2-203", "+", "chr17", 48341439, 48359147, "mac", "train_chrom"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/borzoi_split_chr10_chr11_poisson_multinomial_lora/checkpoints/epochepoch=19.ckpt",
    )
    parser.add_argument(
        "--out_dir",
        default="experiments/validation/borzoi_lora_poisson_multinomial_epoch19_controls_raw_cpm",
    )
    parser.add_argument("--genome", default=DEFAULT_GENOME)
    parser.add_argument("--bigwig_dir", default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--seq_len", type=int, default=524_288)
    parser.add_argument("--label_len", type=int, default=196_608)
    parser.add_argument("--bin_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--original_pdf_dir", default="referrence/validation")
    parser.add_argument("--skip_original_left", action="store_true")
    parser.add_argument(
        "--share_right_y",
        action="store_true",
        help="Use one y-axis scale across the four right-side overlay tracks for each locus.",
    )
    return parser.parse_args()


def centered_interval(chrom: str, start: int, end: int, length: int) -> tuple[int, int]:
    center = (start + end) / 2
    out_start = int(math.ceil(center - (length / 2)))
    return out_start, out_start + length


def read_sequence_tensor(fasta: Fasta, chrom: str, start: int, end: int) -> torch.Tensor:
    record = fasta.get_seq(chrom, start + 1, end)
    seq = getattr(record, "seq", str(record)).upper()
    if len(seq) != end - start:
        raise ValueError((chrom, start, end, len(seq)))
    n_index = BASE_TO_INDEX_HASH["N"]
    indices = np.fromiter(
        (BASE_TO_INDEX_HASH.get(base, n_index) for base in seq),
        dtype=np.int8,
        count=len(seq),
    )
    return indices_to_one_hot(indices).unsqueeze(0)


def read_observed(
    bw_files: list[Path],
    chrom: str,
    label_start: int,
    label_end: int,
    bin_size: int,
) -> np.ndarray:
    tracks = []
    for bw_file in bw_files:
        with pyBigWig.open(str(bw_file), "r") as bw:
            raw = bw.values(chrom, label_start, label_end, numpy=True)
        tracks.append(transform_binned_signal(raw, "poisson_multinomial", bin_size))
    return np.stack(tracks, axis=0)


def build_lora_model(checkpoint: Path, device: torch.device) -> grelu.lightning.LightningModel:
    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = ckpt["hyper_parameters"]
    model = grelu.lightning.LightningModel(
        model_params=hparams["model_params"],
        train_params=hparams["train_params"],
    )
    apply_lora_to_borzoi_embedding(
        model,
        target_modules=BORZOI_LORA_TARGET_MODULES,
        rank=8,
        alpha=16,
        dropout=0.0,
    )
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    model.to(device)
    return model


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def plot_gene(
    gene: str,
    chrom: str,
    gene_start: int,
    gene_end: int,
    expected_track: str,
    split_role: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    label_start: int,
    bin_size: int,
    out_path: Path,
) -> list[dict[str, object]]:
    # Training uses 32 bp bin sums for poisson_multinomial, but the wet-track
    # PDFs and reader-facing overlays are in CPM height. Convert sums back to
    # per-base CPM for plotting and metrics.
    observed = observed / bin_size
    predicted = predicted / bin_size
    x = label_start + (np.arange(observed.shape[1]) * bin_size) + (bin_size / 2)
    x_kb = (x - ((gene_start + gene_end) / 2)) / 1000
    gene_left = (gene_start - ((gene_start + gene_end) / 2)) / 1000
    gene_right = (gene_end - ((gene_start + gene_end) / 2)) / 1000

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, dpi=160)
    rows = []
    for task_idx, (ax, task) in enumerate(zip(axes, TASK_NAMES)):
        obs = observed[task_idx]
        pred = predicted[task_idx]
        ax.plot(x_kb, obs, color="#2f5597", lw=1.0, label="observed raw CPM")
        ax.plot(x_kb, pred, color="#c43c39", lw=1.0, alpha=0.9, label="LoRA pred")
        ax.axvspan(gene_left, gene_right, color="#999999", alpha=0.12, lw=0)
        ax.set_ylabel(task)
        ax.grid(True, alpha=0.2)
        if task == expected_track:
            ax.text(
                0.01,
                0.82,
                "expected",
                transform=ax.transAxes,
                fontsize=8,
                color="#176f3d",
            )
        rows.append(
            {
                "gene": gene,
                "chrom": chrom,
                "start": gene_start,
                "end": gene_end,
                "split_role": split_role,
                "expected_track": expected_track,
                "track": task,
                "pearson_full_label_window": pearson(obs, pred),
                "mse_full_label_window": float(np.mean((obs - pred) ** 2)),
                "observed_mean_full_label_window": float(np.mean(obs)),
                "predicted_mean_full_label_window": float(np.mean(pred)),
                "observed_max_full_label_window": float(np.max(obs)),
                "predicted_max_full_label_window": float(np.max(pred)),
                "observed_peak_pos": int(x[int(np.argmax(obs))]),
                "predicted_peak_pos": int(x[int(np.argmax(pred))]),
            }
        )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("kb relative to gene center")
    fig.suptitle(f"{gene} {chrom}:{gene_start:,}-{gene_end:,} ({split_role})", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    return rows


def plot_prediction_overlay(
    gene: str,
    transcript: str,
    strand: str,
    chrom: str,
    gene_start: int,
    gene_end: int,
    split_role: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    label_start: int,
    bin_size: int,
    out_path: Path,
    share_y: bool = False,
) -> None:
    observed = observed / bin_size
    predicted = predicted / bin_size
    x = label_start + (np.arange(observed.shape[1]) * bin_size) + (bin_size / 2)
    view_mask = (x >= gene_start) & (x <= gene_end)
    if np.any(view_mask):
        x = x[view_mask]
        observed = observed[:, view_mask]
        predicted = predicted[:, view_mask]
    x_mb = x / 1e6
    y_max = None
    if share_y:
        finite = np.concatenate([observed[np.isfinite(observed)], predicted[np.isfinite(predicted)]])
        if finite.size:
            data_max = float(np.max(finite))
            if data_max > 0:
                y_max = data_max * 1.08
    fig, axes = plt.subplots(4, 1, figsize=(9.2, 5.0), sharex=True, dpi=180)
    for task_idx, (ax, task) in enumerate(zip(axes, TASK_NAMES)):
        ax.plot(x_mb, observed[task_idx], color="#4b5563", lw=1.1, label="observed")
        ax.plot(x_mb, predicted[task_idx], color="#f97316", lw=1.1, label="LoRA predicted")
        if y_max is not None:
            ax.set_ylim(0, y_max)
        ax.set_ylabel(f"{task}\nCPM", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        if task_idx == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    title = (
        f"{gene} ({transcript}, {strand}) "
        f"{chrom}:{gene_start:,}-{gene_end:,} [{split_role}; CPM]"
    )
    axes[0].set_title(title, fontsize=10)
    axes[-1].set_xlabel(f"Genomic coordinate on {chrom} (Mb)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def original_pdf_path(pdf_dir: Path, gene: str) -> Path | None:
    candidates = [
        pdf_dir / f"{gene}.pdf",
        pdf_dir / f"{gene.upper()}.pdf",
        pdf_dir / f"{gene.capitalize()}.pdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_pdf_title(pdf_path: Path) -> dict[str, object]:
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf_path), "-"], text=True)
    pattern = re.compile(
        r"(?P<gene>\S+) \((?P<transcript>[^,]+), (?P<strand>[+-])\) "
        r"(?P<chrom>chr[^:]+):(?P<start>[\d,]+)-(?P<end>[\d,]+)"
    )
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            row = match.groupdict()
            row["start"] = int(row["start"].replace(",", ""))
            row["end"] = int(row["end"].replace(",", ""))
            return row
    raise ValueError(f"Could not parse PDF title coordinates: {pdf_path}")


def render_pdf_first_page(pdf_path: Path, dpi: int = 170) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "1",
                "-l",
                "1",
                "-singlefile",
                "-png",
                "-r",
                str(dpi),
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return Image.open(prefix.with_suffix(".png")).convert("RGB")


def compose_original_left_overlay_right(
    original: Image.Image,
    overlay: Image.Image,
    out_path: Path,
    target_height: int = 980,
) -> None:
    def resize_to_height(img: Image.Image, height: int) -> Image.Image:
        width = int(img.width * (height / img.height))
        return img.resize((width, height), Image.Resampling.LANCZOS)

    left = resize_to_height(original, target_height)
    right = resize_to_height(overlay, target_height)
    pad = 36
    sheet = Image.new("RGB", (left.width + right.width + pad, target_height), "white")
    sheet.paste(left, (0, 0))
    sheet.paste(right, (left.width + pad, 0))
    sheet.save(out_path)


def make_contact_sheet(page_paths: list[Path], out_path: Path) -> None:
    thumbs = []
    for path in page_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((900, 720))
        thumbs.append(img.copy())
    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    cell_w = max(img.width for img in thumbs)
    cell_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, img in enumerate(thumbs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        sheet.paste(img, (x, y))
    sheet.save(out_path)


def make_vertical_contact_sheet(page_paths: list[Path], out_path: Path, width: int = 1600) -> None:
    images = []
    for path in page_paths:
        img = Image.open(path).convert("RGB")
        height = int(img.height * (width / img.width))
        images.append(img.resize((width, height), Image.Resampling.LANCZOS))
    total_height = sum(img.height for img in images)
    sheet = Image.new("RGB", (width, total_height), "white")
    y = 0
    for img in images:
        sheet.paste(img, (0, y))
        y += img.height
    sheet.save(out_path)


def make_multipage_pdf(page_paths: list[Path], out_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in page_paths]
    if not images:
        return
    first, rest = images[0], images[1:]
    first.save(out_path, save_all=True, append_images=rest)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    page_dir = out_dir / "pages"
    summary_dir = out_dir / "summary"
    side_by_side_dir = summary_dir / "original_left_overlay_right_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = build_lora_model(Path(args.checkpoint), device)
    fasta = Fasta(args.genome, rebuild=False)
    bw_files = [Path(args.bigwig_dir) / f"{task}.CPM.mapq10.bw" for task in TASK_NAMES]

    all_rows = []
    qc_rows = []
    page_paths = []
    side_by_side_paths = []
    with torch.no_grad():
        for gene, transcript, strand, chrom, start, end, expected_track, split_role in DEFAULT_CONTROLS:
            seq_start, seq_end = centered_interval(chrom, start, end, args.seq_len)
            label_start, label_end = centered_interval(chrom, start, end, args.label_len)
            observed = read_observed(bw_files, chrom, label_start, label_end, args.bin_size)
            x = read_sequence_tensor(fasta, chrom, seq_start, seq_end).to(device)
            predicted = model.forward(x).detach().cpu().numpy()[0]
            page_path = page_dir / f"{gene}.observed_vs_lora_poisson_multinomial_raw_cpm.png"
            all_rows.extend(
                plot_gene(
                    gene,
                    chrom,
                    start,
                    end,
                    expected_track,
                    split_role,
                    observed,
                    predicted,
                    label_start,
                    args.bin_size,
                    page_path,
                )
            )
            page_paths.append(page_path)
            overlay_path = side_by_side_dir / f"{gene}.overlay_right.png"
            plot_prediction_overlay(
                gene,
                transcript,
                strand,
                chrom,
                start,
                end,
                split_role,
                observed,
                predicted,
                label_start,
                args.bin_size,
                overlay_path,
                share_y=args.share_right_y,
            )
            if not args.skip_original_left:
                pdf_path = original_pdf_path(Path(args.original_pdf_dir), gene)
                if pdf_path is not None:
                    pdf_row = parse_pdf_title(pdf_path)
                    qc_rows.append(
                        {
                            "gene": gene,
                            "pdf_path": str(pdf_path),
                            "pdf_gene": pdf_row["gene"],
                            "pdf_transcript": pdf_row["transcript"],
                            "pdf_strand": pdf_row["strand"],
                            "pdf_chrom": pdf_row["chrom"],
                            "pdf_start": pdf_row["start"],
                            "pdf_end": pdf_row["end"],
                            "overlay_transcript": transcript,
                            "overlay_strand": strand,
                            "overlay_chrom": chrom,
                            "overlay_start": start,
                            "overlay_end": end,
                            "coordinate_match": (
                                pdf_row["gene"] == gene
                                and pdf_row["transcript"] == transcript
                                and pdf_row["strand"] == strand
                                and pdf_row["chrom"] == chrom
                                and pdf_row["start"] == start
                                and pdf_row["end"] == end
                            ),
                            "y_scale": "observed and predicted are 32bp bin sums divided by bin_size to CPM",
                        }
                    )
                    original_img = render_pdf_first_page(pdf_path)
                    overlay_img = Image.open(overlay_path).convert("RGB")
                    combined_path = (
                        side_by_side_dir
                        / f"{gene}.original_left_lora_poisson_multinomial_overlay_right.png"
                    )
                    compose_original_left_overlay_right(original_img, overlay_img, combined_path)
                    side_by_side_paths.append(combined_path)
            print(f"wrote {page_path}")

    metrics_path = out_dir / "per_track_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    if qc_rows:
        qc_path = out_dir / "coordinate_qc.csv"
        with qc_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(qc_rows[0]))
            writer.writeheader()
            writer.writerows(qc_rows)
    make_contact_sheet(page_paths, out_dir / "positive_controls_observed_vs_lora_poisson_multinomial_contact_sheet.png")
    if side_by_side_paths:
        make_vertical_contact_sheet(
            side_by_side_paths,
            summary_dir
            / "positive_controls_original_left_lora_poisson_multinomial_overlay_right_contact_sheet.png",
        )
        make_multipage_pdf(
            side_by_side_paths,
            summary_dir / "positive_controls_original_left_lora_poisson_multinomial_overlay_right.pdf",
        )
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
