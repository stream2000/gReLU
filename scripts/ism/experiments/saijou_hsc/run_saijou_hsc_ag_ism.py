#!/usr/bin/env python
"""Minimal Saijou HSC saturation ISM with fine-tuned AlphaGenome."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.fasta import FastaReference  # noqa: E402
from grelu.interpret.ism.model_adapters import AlphaGenomeFinetunedAdapter  # noqa: E402
from grelu.interpret.ism.mutations import (  # noqa: E402
    SaturationWindow,
    apply_equal_length_edit,
    enumerate_snv_site_table,
)
from grelu.interpret.ism.readouts import map_readouts_to_bins, summarize_profile_pair  # noqa: E402

DEFAULT_FASTA = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
DEFAULT_GTF = "/work/Database/Database_fromDocker/Referencedata_mm10/gtf_chrUCSC/chr.gtf"
DEFAULT_HSC_BIGWIG = (
    "/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/"
    "bigwig/hsc.CPM.mapq10.bw"
)
DEFAULT_CHECKPOINT = (
    "runs/alphagenome_split_chr10_chr11_seq524288_label196608_bin128_res128_"
    "poisson_multinomial_lora_active_h512x1/checkpoints/epochepoch=19.ckpt"
)
DEFAULT_GENES = "Mdk,Col1a1,Acta2"


@dataclass(frozen=True)
class GeneRecord:
    gene: str
    gene_id: str
    chrom: str
    start: int
    end: int
    strand: str

    @property
    def tss(self) -> int:
        return self.start if self.strand == "+" else self.end

    @property
    def tes(self) -> int:
        return self.end if self.strand == "+" else self.start

    @property
    def length(self) -> int:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--hsc_bigwig", default=DEFAULT_HSC_BIGWIG)
    parser.add_argument("--genes", default=DEFAULT_GENES)
    parser.add_argument("--window_bp", type=int, default=8)
    parser.add_argument("--readout_bp", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=524_288)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--ranking_top_n", type=int, default=25)
    parser.add_argument("--motif_top_n", type=int, default=10)
    parser.add_argument("--motif_flank_bp", type=int, default=30)
    parser.add_argument("--motif_file", default="consensus")
    parser.add_argument("--motif_pthresh", type=float, default=1e-3)
    parser.add_argument("--progress_every_batches", type=int, default=50)
    parser.add_argument(
        "--out_dir",
        default="experiments/ism/saijou_hsc_ag_smoke",
    )
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--determinism_check", action="store_true", default=True)
    return parser.parse_args()


def _gtf_attr(attributes: str, key: str) -> str:
    match = re.search(rf'(?:^|;\s*){re.escape(key)} "([^"]+)"', attributes)
    return match.group(1).strip() if match else ""


def load_gene_records(gtf_path: str | Path, genes: list[str]) -> list[GeneRecord]:
    wanted = {gene.casefold(): gene for gene in genes}
    rows: list[GeneRecord] = []
    columns = [
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attributes",
    ]
    table = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=columns,
        usecols=range(9),
        low_memory=False,
    )
    gene_rows = table.loc[table["feature"].eq("gene")].copy()
    for row in gene_rows.itertuples(index=False):
        gene_name = _gtf_attr(str(row.attributes), "gene_name")
        if gene_name.casefold() not in wanted:
            continue
        rows.append(
            GeneRecord(
                gene=gene_name,
                gene_id=_gtf_attr(str(row.attributes), "gene_id"),
                chrom=str(row.chrom),
                start=int(row.start) - 1,
                end=int(row.end),
                strand=str(row.strand),
            )
        )
    by_gene = {record.gene.casefold(): record for record in rows}
    missing = [gene for gene in genes if gene.casefold() not in by_gene]
    if missing:
        raise ValueError(f"Genes not found in GTF: {missing}")
    return [by_gene[gene.casefold()] for gene in genes]


def centered_interval(center: int, width: int) -> tuple[int, int]:
    start = int(center) - int(width) // 2
    return start, start + int(width)


def build_windows(genes: list[GeneRecord], window_bp: int) -> list[SaturationWindow]:
    windows = []
    for gene in genes:
        start, end = centered_interval(gene.tss, window_bp)
        windows.append(
            SaturationWindow(
                window_id=f"{gene.gene}__tss_saturation_{window_bp}bp",
                gene=gene.gene,
                chrom=gene.chrom,
                start=start,
                end=end,
                anchor=gene.tss,
                source="mm10_gtf_tss",
                label=f"{gene.gene} TSS saturation",
            )
        )
    return windows


def observed_peak_center(gene: GeneRecord, bigwig_path: str | Path) -> int | None:
    try:
        import pyBigWig
    except ImportError:
        return None
    path = Path(bigwig_path)
    if not path.exists():
        return None
    with pyBigWig.open(str(path), "r") as bw:
        values = bw.values(gene.chrom, gene.start, gene.end, numpy=True)
    finite = np.asarray(values, dtype=float)
    finite[~np.isfinite(finite)] = 0.0
    if finite.size == 0 or float(finite.max()) <= 0:
        return None
    return int(gene.start + int(np.argmax(finite)))


def build_readouts(
    genes: list[GeneRecord],
    *,
    readout_bp: int,
    hsc_bigwig: str | Path,
) -> pd.DataFrame:
    rows = []
    for gene in genes:
        for role, center in (("tss", gene.tss), ("tes_3prime", gene.tes)):
            start, end = centered_interval(center, readout_bp)
            rows.append(
                {
                    "readout_id": f"{gene.gene}__{role}_{readout_bp}bp",
                    "gene": gene.gene,
                    "chrom": gene.chrom,
                    "start": start,
                    "end": end,
                    "anchor": center,
                    "role": role,
                }
            )
        peak = observed_peak_center(gene, hsc_bigwig)
        if peak is not None:
            start, end = centered_interval(peak, readout_bp)
            rows.append(
                {
                    "readout_id": f"{gene.gene}__hsc_observed_peak_{readout_bp}bp",
                    "gene": gene.gene,
                    "chrom": gene.chrom,
                    "start": start,
                    "end": end,
                    "anchor": peak,
                    "role": "hsc_observed_peak",
                }
            )
        rows.append(
            {
                "readout_id": f"{gene.gene}__gene_body",
                "gene": gene.gene,
                "chrom": gene.chrom,
                "start": gene.start,
                "end": gene.end,
                "anchor": gene.tss,
                "role": "gene_body",
            }
        )
    return pd.DataFrame.from_records(rows)


def write_table(path: Path, table: pd.DataFrame, *, sep: str = "\t") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep=sep, index=False)


def plot_gene_heatmaps(features: pd.DataFrame, output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    bases = ["A", "C", "G", "T"]
    for (gene, readout_id), group in features.groupby(["gene", "readout_id"], sort=False):
        positions = sorted(group["variant_position"].astype(int).unique())
        if not positions:
            continue
        matrix = np.full((len(bases), len(positions)), np.nan, dtype=float)
        pos_index = {pos: idx for idx, pos in enumerate(positions)}
        base_index = {base: idx for idx, base in enumerate(bases)}
        for row in group.itertuples(index=False):
            matrix[base_index[str(row.alt_base)], pos_index[int(row.variant_position)]] = float(
                row.log2fc_mean
            )
        fig_width = min(24.0, max(5.0, len(positions) * 0.055))
        fig, ax = plt.subplots(figsize=(fig_width, 2.2), dpi=180)
        vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
        vmax = max(float(vmax), 1e-6)
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_yticks(np.arange(len(bases)), bases)
        tick_stride = max(1, math.ceil(len(positions) / 32))
        tick_idx = np.arange(0, len(positions), tick_stride)
        ax.set_xticks(tick_idx, [str(positions[idx]) for idx in tick_idx], rotation=90)
        ax.set_title(f"{gene} {readout_id} HSC log2FC")
        ax.set_xlabel("0-based variant position")
        ax.set_ylabel("ALT base")
        fig.colorbar(image, ax=ax, shrink=0.75, label="mean log2FC")
        fig.tight_layout()
        out_path = output_dir / f"{gene}.{readout_id}.hsc_log2fc_heatmap.png"
        fig.savefig(out_path)
        plt.close(fig)
        paths.append(str(out_path))
    return paths


def rank_features(features: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Rank variants by absolute mean log2FC within each gene/readout."""

    ranked = features.copy()
    ranked["abs_log2fc_mean"] = ranked["log2fc_mean"].abs()
    ranked = ranked.sort_values(
        [
            "gene",
            "readout_role",
            "abs_log2fc_mean",
            "absolute_delta_mean",
            "mutation_id",
        ],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)
    ranked["rank_within_gene_readout"] = (
        ranked.groupby(["gene", "readout_role"]).cumcount() + 1
    )
    return ranked.loc[ranked["rank_within_gene_readout"] <= int(top_n)].copy()


def _safe_scan_motifs(
    *,
    sequences: list[str],
    seq_ids: list[str],
    motif_file: str,
    pthresh: float,
) -> pd.DataFrame:
    from grelu.interpret.motifs import scan_sequences

    hits = scan_sequences(
        sequences,
        motif_file,
        seq_ids=seq_ids,
        pthresh=pthresh,
        rc=True,
    )
    if hits is None or hits.empty:
        return pd.DataFrame()
    return hits


def annotate_top_variant_motifs(
    *,
    ranked: pd.DataFrame,
    mutation_manifest: pd.DataFrame,
    fasta_path: str | Path,
    output_dir: Path,
    motif_file: str,
    top_n: int,
    flank_bp: int,
    pthresh: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan reference sequence around top variants and mark motif overlaps."""

    selected = (
        ranked.loc[ranked["rank_within_gene_readout"] <= int(top_n)]
        .drop_duplicates(["gene", "readout_role", "mutation_id"])
        .copy()
    )
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    manifest = mutation_manifest.set_index("mutation_id", drop=False)
    sequence_records = []
    sequences: list[str] = []
    seq_ids: list[str] = []
    with FastaReference(fasta_path) as fasta:
        for row in selected.itertuples(index=False):
            mut = manifest.loc[str(row.mutation_id)]
            center = int(mut.variant_position)
            start = max(0, center - int(flank_bp))
            end = center + int(flank_bp) + 1
            seq = fasta.extract(str(mut.chrom), start, end)
            seq_id = (
                f"{row.gene}|{row.readout_role}|{row.rank_within_gene_readout}|"
                f"{row.mutation_id}"
            )
            seq_ids.append(seq_id)
            sequences.append(seq)
            sequence_records.append(
                {
                    "sequence_name": seq_id,
                    "gene": row.gene,
                    "readout_role": row.readout_role,
                    "rank_within_gene_readout": int(row.rank_within_gene_readout),
                    "mutation_id": row.mutation_id,
                    "chrom": mut.chrom,
                    "variant_position": int(mut.variant_position),
                    "ref_base": mut.ref_base,
                    "alt_base": mut.alt_base,
                    "sequence_start": start,
                    "sequence_end": end,
                    "reference_sequence": seq,
                    "abs_log2fc_mean": float(row.abs_log2fc_mean),
                    "log2fc_mean": float(row.log2fc_mean),
                    "signed_delta_mean": float(row.signed_delta_mean),
                }
            )
    sequence_table = pd.DataFrame.from_records(sequence_records)
    hits = _safe_scan_motifs(
        sequences=sequences,
        seq_ids=seq_ids,
        motif_file=motif_file,
        pthresh=pthresh,
    )
    if hits.empty:
        return sequence_table, hits
    if "sequence_name" not in hits.columns and "sequence" in hits.columns:
        hits = hits.rename(columns={"sequence": "sequence_name"})
    hits = hits.merge(sequence_table, on="sequence_name", how="left")
    hits["motif_genomic_start"] = hits["sequence_start"] + hits["start"].astype(int)
    hits["motif_genomic_end"] = hits["sequence_start"] + hits["end"].astype(int)
    hits["overlaps_variant"] = (
        (hits["motif_genomic_start"] <= hits["variant_position"])
        & (hits["variant_position"] < hits["motif_genomic_end"])
    )
    motif_center = (hits["motif_genomic_start"] + hits["motif_genomic_end"]) / 2
    hits["distance_to_variant_bp"] = motif_center - hits["variant_position"]
    hits = hits.sort_values(
        [
            "gene",
            "readout_role",
            "rank_within_gene_readout",
            "overlaps_variant",
            "fimo_p-value",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return sequence_table, hits


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    input_dir = out_dir / "inputs"
    prepared_dir = out_dir / "prepared"
    feature_dir = out_dir / "features"
    ranking_dir = out_dir / "ranking"
    motif_dir = out_dir / "motifs"
    figure_dir = out_dir / "figures"
    for path in (input_dir, prepared_dir, feature_dir, ranking_dir, motif_dir, figure_dir):
        path.mkdir(parents=True, exist_ok=True)

    genes = [gene.strip() for gene in args.genes.split(",") if gene.strip()]
    gene_records = load_gene_records(args.gtf, genes)
    windows = build_windows(gene_records, args.window_bp)
    readouts = build_readouts(
        gene_records,
        readout_bp=args.readout_bp,
        hsc_bigwig=args.hsc_bigwig,
    )
    write_table(input_dir / "genes.tsv", pd.DataFrame.from_records([asdict(g) for g in gene_records]))
    write_table(input_dir / "windows.tsv", pd.DataFrame.from_records([asdict(w) for w in windows]))
    write_table(input_dir / "readouts.tsv", readouts)

    with FastaReference(args.fasta) as fasta:
        mutation_manifest = enumerate_snv_site_table(fasta=fasta, windows=windows)
        expected_mutations = 0
        for window in windows:
            seq = fasta.extract(window.chrom, window.start, window.end)
            expected_mutations += sum(base in "ACGT" for base in seq) * 3
        ref_mismatches = []
        for row in mutation_manifest.itertuples(index=False):
            observed = fasta.extract(row.chrom, int(row.edit_start), int(row.edit_end))
            if observed.upper() != str(row.ref_sequence).upper():
                ref_mismatches.append(row.mutation_id)
        if ref_mismatches:
            raise ValueError(f"REF mismatches in mutation manifest: {ref_mismatches[:5]}")
        write_table(prepared_dir / "mutation_manifest.tsv", mutation_manifest)

        adapter = AlphaGenomeFinetunedAdapter(
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
            input_length_bp=args.seq_len,
        )
        adapter.setup()
        model_meta = adapter.metadata
        (out_dir / "model_metadata.json").write_text(
            json.dumps(model_meta, indent=2, default=str) + "\n"
        )
        label_len = adapter.output_length_bins * adapter.output_resolution_bp
        crop_bp = (adapter.input_length_bp - label_len) // 2

        feature_rows = []
        sequence_context_rows = []
        deterministic_max_abs_diff = None
        for gene_idx, gene in enumerate(gene_records):
            seq_start = int(gene.tss - adapter.input_length_bp // 2)
            seq_end = seq_start + adapter.input_length_bp
            if seq_start < 0 or seq_end > fasta.chrom_length(gene.chrom):
                raise ValueError(f"{gene.gene} model sequence is out of chromosome bounds")
            output_start = seq_start + crop_bp
            ref_seq = fasta.extract(gene.chrom, seq_start, seq_end)
            gene_mutations = mutation_manifest.loc[mutation_manifest["gene"].eq(gene.gene)].copy()
            gene_readouts = readouts.loc[readouts["gene"].eq(gene.gene)].copy()
            print(
                f"[saijou-hsc-ism] gene={gene.gene} "
                f"mutations={len(gene_mutations)} readouts={len(gene_readouts)}",
                flush=True,
            )
            readout_bins = map_readouts_to_bins(
                gene_readouts,
                output_start=output_start,
                n_bins=adapter.output_length_bins,
                bin_size=adapter.output_resolution_bp,
                chrom=gene.chrom,
            )
            ref_profile = adapter.predict_profiles([ref_seq], tracks=["hsc"])[0, 0]
            if args.determinism_check and gene_idx == 0:
                ref_profile_repeat = adapter.predict_profiles([ref_seq], tracks=["hsc"])[0, 0]
                deterministic_max_abs_diff = float(np.max(np.abs(ref_profile - ref_profile_repeat)))
            sequence_context_rows.append(
                {
                    "gene": gene.gene,
                    "chrom": gene.chrom,
                    "seq_start": seq_start,
                    "seq_end": seq_end,
                    "output_start": output_start,
                    "output_end": output_start + label_len,
                    "tss": gene.tss,
                    "tes": gene.tes,
                    "n_readouts": len(readout_bins),
                    "n_mutations": len(gene_mutations),
                }
            )

            n_batches = math.ceil(len(gene_mutations) / args.batch_size)
            for batch_index, batch_start in enumerate(
                range(0, len(gene_mutations), args.batch_size), start=1
            ):
                batch = gene_mutations.iloc[batch_start : batch_start + args.batch_size]
                alt_sequences = [
                    apply_equal_length_edit(ref_seq, seq_start, row)
                    for _, row in batch.iterrows()
                ]
                alt_profiles = adapter.predict_profiles(alt_sequences, tracks=["hsc"])[:, 0, :]
                for alt_idx, (_, row) in enumerate(batch.iterrows()):
                    alt_profile = alt_profiles[alt_idx]
                    for _, readout in gene_readouts.iterrows():
                        left, right = readout_bins[str(readout["readout_id"])]
                        summary = summarize_profile_pair(
                            ref_profile[left:right],
                            alt_profile[left:right],
                        )
                        summary["peak_shift_bp"] = (
                            summary["alt_peak_bin"] - summary["ref_peak_bin"]
                        ) * adapter.output_resolution_bp
                        feature_rows.append(
                            {
                                "gene": gene.gene,
                                "mutation_id": row["mutation_id"],
                                "window_id": row["window_id"],
                                "variant_position": int(row["variant_position"]),
                                "ref_base": row["ref_base"],
                                "alt_base": row["alt_base"],
                                "readout_id": str(readout["readout_id"]),
                                "readout_role": str(readout["role"]),
                                "track_id": "hsc",
                                "output_start": output_start,
                                "output_resolution_bp": adapter.output_resolution_bp,
                                "readout_bin_start": left,
                                "readout_bin_end": right,
                                **summary,
                            }
                        )
                if (
                    args.progress_every_batches > 0
                    and (
                        batch_index % args.progress_every_batches == 0
                        or batch_index == n_batches
                    )
                ):
                    print(
                        f"[saijou-hsc-ism] gene={gene.gene} "
                        f"batch={batch_index}/{n_batches}",
                        flush=True,
                    )

        features = pd.DataFrame.from_records(feature_rows)
        write_table(prepared_dir / "sequence_contexts.tsv", pd.DataFrame.from_records(sequence_context_rows))
        write_table(feature_dir / "mutation_features.tsv", features)
        try:
            features.to_parquet(feature_dir / "mutation_features.parquet", index=False)
        except Exception as exc:
            (feature_dir / "parquet_write_error.txt").write_text(str(exc) + "\n")

    figure_paths: list[str] = []
    if not args.skip_plots and not features.empty:
        figure_paths = plot_gene_heatmaps(features, figure_dir)

    ranked = rank_features(features, top_n=args.ranking_top_n)
    write_table(ranking_dir / "top_variants_by_gene_readout.tsv", ranked)
    motif_sequence_table, motif_hits = annotate_top_variant_motifs(
        ranked=ranked,
        mutation_manifest=mutation_manifest,
        fasta_path=args.fasta,
        output_dir=motif_dir,
        motif_file=args.motif_file,
        top_n=args.motif_top_n,
        flank_bp=args.motif_flank_bp,
        pthresh=args.motif_pthresh,
    )
    write_table(motif_dir / "top_variant_sequences.tsv", motif_sequence_table)
    if motif_hits.empty:
        (motif_dir / "top_variant_motif_hits.tsv").write_text("")
    else:
        write_table(motif_dir / "top_variant_motif_hits.tsv", motif_hits)

    validation = {
        "status": "ok",
        "genes": genes,
        "window_bp": args.window_bp,
        "readout_bp": args.readout_bp,
        "expected_mutations": int(expected_mutations),
        "observed_mutations": int(len(mutation_manifest)),
        "ref_mismatch_count": 0,
        "feature_rows": int(len(features)),
        "finite_log2fc_rows": int(np.isfinite(features["log2fc_mean"].to_numpy(dtype=float)).sum()),
        "ranking_rows": int(len(ranked)),
        "motif_sequences": int(len(motif_sequence_table)),
        "motif_hits": int(len(motif_hits)),
        "motif_hits_overlapping_variant": int(motif_hits["overlaps_variant"].sum())
        if not motif_hits.empty
        else 0,
        "deterministic_ref_max_abs_diff": deterministic_max_abs_diff,
        "model_output_length_bins": int(model_meta["output_length_bins"]),
        "model_output_resolution_bp": int(model_meta["output_resolution_bp"]),
        "figures": figure_paths,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    if validation["expected_mutations"] != validation["observed_mutations"]:
        raise RuntimeError(validation)
    (out_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, default=str) + "\n"
    )
    print(json.dumps(validation, indent=2, default=str))


if __name__ == "__main__":
    main()
