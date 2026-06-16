#!/usr/bin/env python
"""Assign intragenic DICs to a containing-gene TSS and build 4 kb readouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.batch import AG_INPUT_LEN  # noqa: E402
from grelu.interpret.ism.inference import load_gene_annotation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--site-class", default="lc_dic")
    parser.add_argument("--window-bp", type=int, default=4096)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(args.sites, sep="\t", low_memory=False)
    sites = sites[sites["site_class"].eq(args.site_class)].copy()
    annotation = load_gene_annotation(args.gtf)
    half_window = args.window_bp // 2
    records = []
    for row in sites.itertuples(index=False):
        center = (int(row.start) + int(row.end)) // 2
        genes = [
            gene
            for gene in annotation.by_chrom.get(str(row.chrom), [])
            if gene.start < int(row.end) and gene.end > int(row.start)
        ]
        containing = [
            gene for gene in genes if gene.start <= center < gene.end
        ]
        candidates = containing or genes
        chosen = (
            min(
                candidates,
                key=lambda gene: (
                    abs(gene.tss - center),
                    -(gene.end - gene.start),
                    gene.start,
                ),
            )
            if candidates
            else None
        )
        context_center = int(row.anchor)
        context_start = context_center - AG_INPUT_LEN // 2
        context_end = context_start + AG_INPUT_LEN
        readout_start = chosen.tss - half_window if chosen else -1
        readout_end = readout_start + args.window_bp if chosen else -1
        in_context = bool(
            chosen
            and readout_start >= context_start
            and readout_end <= context_end
        )
        records.append(
            {
                "site_id": row.site_id,
                "site_class": row.site_class,
                "chrom": row.chrom,
                "readout_id": "host_tss_4kb",
                "start": readout_start,
                "end": readout_end,
                "anchor": chosen.tss if chosen else -1,
                "gene_id": chosen.gene_id if chosen else "",
                "gene_name": chosen.gene_name if chosen else "",
                "gene_strand": chosen.strand if chosen else ".",
                "distance_tss_from_edit": (
                    chosen.tss - context_center if chosen else pd.NA
                ),
                "n_overlapping_genes": len(genes),
                "n_center_containing_genes": len(containing),
                "mapping_ambiguous": len(containing) > 1,
                "readout_in_model_context": in_context,
                "mapping_status": (
                    "ready"
                    if in_context
                    else "outside_model_context"
                    if chosen
                    else "no_overlapping_gene"
                ),
            }
        )
    audit = pd.DataFrame.from_records(records)
    audit.to_csv(output / "tss_mapping_audit.tsv", sep="\t", index=False)
    ready = audit[audit["mapping_status"].eq("ready")].copy()
    ready.to_csv(output / "tss_readouts.tsv", sep="\t", index=False)
    print(
        f"Mapped {len(audit)} {args.site_class} sites: "
        f"{audit['mapping_status'].value_counts().to_dict()}, "
        f"ambiguous={int(audit['mapping_ambiguous'].sum())}"
    )


if __name__ == "__main__":
    main()
