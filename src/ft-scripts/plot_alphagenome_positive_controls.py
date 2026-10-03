#!/usr/bin/env python
"""Plot observed-vs-AlphaGenome LoRA predictions on positive-control loci."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from pyfaidx import Fasta

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "ft-scripts"))

import grelu.lightning  # noqa: E402
from plot_borzoi_positive_controls import (  # noqa: E402
    DEFAULT_CONTROLS,
    centered_interval,
    compose_original_left_overlay_right,
    make_contact_sheet,
    make_multipage_pdf,
    make_vertical_contact_sheet,
    original_pdf_path,
    parse_pdf_title,
    plot_gene,
    plot_prediction_overlay,
    read_observed,
    read_sequence_tensor,
    render_pdf_first_page,
)
from train_alphagenome import (  # noqa: E402
    ALPHAGENOME_LORA_TARGETS,
    DEFAULT_BIGWIG_DIR,
    DEFAULT_GENOME,
    TASK_NAMES,
    apply_lora_to_alphagenome_embedding,
    choose_lora_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/alphagenome_split_chr10_chr11_seq131072_bin128_"
            "poisson_multinomial_lora/checkpoints/last.ckpt"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=(
            "experiments/validation/"
            "alphagenome_lora_poisson_multinomial_last_controls_raw_cpm"
        ),
    )
    parser.add_argument("--genome", default=DEFAULT_GENOME)
    parser.add_argument("--bigwig_dir", default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--seq_len", type=int, default=131_072)
    parser.add_argument("--label_len", type=int, default=131_072)
    parser.add_argument("--bin_size", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=128, choices=[1, 128])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_conv_rank", type=int, default=8)
    parser.add_argument("--lora_conv_alpha", type=int, default=16)
    parser.add_argument(
        "--lora_preset",
        default="tower",
        choices=["tower", "active", "all"],
        help=(
            "LoRA target preset used by the checkpoint. Use active for the "
            "higher-coverage Linear+Conv1d AlphaGenome runs."
        ),
    )
    parser.add_argument("--lora_targets", default=",".join(ALPHAGENOME_LORA_TARGETS))
    parser.add_argument(
        "--lora_conv_targets",
        default=None,
        help="Comma-separated Conv1d target substrings; defaults follow --lora_preset.",
    )
    parser.add_argument("--original_pdf_dir", default="referrence/validation")
    parser.add_argument("--skip_original_left", action="store_true")
    parser.add_argument(
        "--share_right_y",
        action="store_true",
        help="Use one y-axis scale across the four right-side overlay tracks for each locus.",
    )
    return parser.parse_args()


def build_lora_model(
    checkpoint: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[grelu.lightning.LightningModel, dict[str, object]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = ckpt["hyper_parameters"]
    model = grelu.lightning.LightningModel(
        model_params=hparams["model_params"],
        train_params=hparams["train_params"],
    )
    linear_targets, conv_targets = choose_lora_targets(args)
    lora_result = apply_lora_to_alphagenome_embedding(
        model,
        linear_targets=linear_targets,
        conv_targets=conv_targets,
        linear_rank=args.lora_rank,
        linear_alpha=args.lora_alpha,
        conv_rank=args.lora_conv_rank,
        conv_alpha=args.lora_conv_alpha,
    )
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    model.to(device)
    return model, {
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_global_step": ckpt.get("global_step"),
        "lora_preset": args.lora_preset,
        "lora_linear_targets": ",".join(linear_targets),
        "lora_conv_targets": ",".join(conv_targets),
        "lora_linear_target_count": lora_result["linear_targets"],
        "lora_conv_target_count": lora_result["conv_targets"],
        "lora_wrappers": lora_result["lora_wrappers"],
        "locon_wrappers": lora_result["locon_wrappers"],
    }


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
    model, metadata = build_lora_model(
        Path(args.checkpoint),
        device,
        args=args,
    )
    print("checkpoint:", args.checkpoint)
    for key, value in metadata.items():
        print(f"{key}: {value}")

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
            page_path = page_dir / f"{gene}.observed_vs_alphagenome_lora_poisson_multinomial_raw_cpm.png"
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
            overlay_path = side_by_side_dir / f"{gene}.alphagenome_overlay_right.png"
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
                            "y_scale": "observed and predicted are 128bp bin sums divided by bin_size to CPM",
                        }
                    )
                    original_img = render_pdf_first_page(pdf_path)
                    overlay_img = Image.open(overlay_path).convert("RGB")
                    combined_path = (
                        side_by_side_dir
                        / f"{gene}.original_left_alphagenome_lora_poisson_multinomial_overlay_right.png"
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
    make_contact_sheet(
        page_paths,
        out_dir / "positive_controls_observed_vs_alphagenome_lora_poisson_multinomial_contact_sheet.png",
    )
    if side_by_side_paths:
        make_vertical_contact_sheet(
            side_by_side_paths,
            summary_dir
            / "positive_controls_original_left_alphagenome_lora_poisson_multinomial_overlay_right_contact_sheet.png",
        )
        make_multipage_pdf(
            side_by_side_paths,
            summary_dir / "positive_controls_original_left_alphagenome_lora_poisson_multinomial_overlay_right.pdf",
        )
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
