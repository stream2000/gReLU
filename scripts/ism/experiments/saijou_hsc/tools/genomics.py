"""Genomic, sequence-context, and motif helpers for Saijou ISM."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


TRANSCRIPT_AUTHORITY = (
    Path(__file__).resolve().parents[1]
    / "configs/provided_nine_gene_transcripts.tsv"
)


def load_transcript_overrides(
    path: Path = TRANSCRIPT_AUTHORITY,
) -> dict[str, str]:
    """Load the one versioned transcript definition shared by all stages."""

    table = pd.read_csv(path, sep="\t", usecols=["gene", "transcript_id"])
    if table.gene.duplicated().any() or table.transcript_id.duplicated().any():
        raise ValueError(f"Transcript authority keys are not unique: {path}")
    return dict(zip(table.gene.astype(str), table.transcript_id.astype(str)))


# Maintained canonical behavior. Wider or externally registered transcript
# presets must be passed explicitly through ``load_transcript_overrides``.
TRANSCRIPT_OVERRIDES = {"Acta2": "ENSMUST00000238147"}
GTF_COLUMNS = (
    "chrom",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attributes",
)


def gtf_attr(attributes: str, key: str) -> str:
    match = re.search(rf'(?:^|;\s*){re.escape(key)} "([^"]+)"', attributes)
    return match.group(1).strip() if match else ""


def _read_gtf(gtf_path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=GTF_COLUMNS,
        usecols=range(9),
        low_memory=False,
    )
    attributes = table.attributes.astype(str)
    table["gene_name"] = attributes.map(lambda value: gtf_attr(value, "gene_name"))
    table["transcript_id"] = attributes.map(
        lambda value: gtf_attr(value, "transcript_id")
    )
    return table


def _select_transcript(
    table: pd.DataFrame,
    gene_name: str,
    analysis_tss: int,
    transcript_overrides: Mapping[str, str],
) -> pd.Series:
    transcripts = table.loc[
        table.feature.eq("transcript") & table.gene_name.eq(gene_name)
    ].copy()
    override = transcript_overrides.get(gene_name)
    if override:
        selected = transcripts.loc[transcripts.transcript_id.eq(override)]
    else:
        transcripts["length"] = (
            transcripts.end.astype(int) - transcripts.start.astype(int) + 1
        )
        transcripts["tss"] = np.where(
            transcripts.strand.eq("+"),
            transcripts.start.astype(int) - 1,
            transcripts.end.astype(int),
        )
        same_tss = transcripts.loc[transcripts.tss.eq(analysis_tss)]
        selected = (same_tss if not same_tss.empty else transcripts).nlargest(
            1, "length"
        )
    if len(selected) != 1:
        raise ValueError(f"Could not select one representative transcript for {gene_name}")
    return selected.iloc[0]


def _transcript_metadata(selected: pd.Series, analysis_tss: int) -> dict[str, object]:
    attributes = str(selected.attributes)
    transcript_tss = (
        int(selected.start) - 1
        if str(selected.strand) == "+"
        else int(selected.end)
    )
    return {
        "representative_transcript_id": str(selected.transcript_id),
        "transcript_name": gtf_attr(attributes, "transcript_name"),
        "transcript_biotype": gtf_attr(attributes, "transcript_biotype"),
        "transcript_support_level": gtf_attr(
            attributes, "transcript_support_level"
        ),
        "is_basic": 'tag "basic"' in attributes,
        "has_ccds_tag": 'tag "CCDS"' in attributes,
        "transcript_start": int(selected.start) - 1,
        "transcript_end": int(selected.end),
        "transcript_tss": transcript_tss,
        "analysis_tss": analysis_tss,
        "tss_matches_analysis": transcript_tss == analysis_tss,
    }


def _feature_intervals(
    table: pd.DataFrame, transcript_id: str, feature: str
) -> list[tuple[int, int]]:
    rows = table.loc[
        table.feature.eq(feature) & table.transcript_id.eq(transcript_id)
    ]
    return sorted((int(row.start) - 1, int(row.end)) for row in rows.itertuples())


def _splice_sites(
    intervals: list[tuple[int, int]], strand: str
) -> list[tuple[int, str]]:
    sites = []
    for index, (start, end) in enumerate(intervals):
        if strand == "+":
            if index > 0:
                sites.append((start, "splice_acceptor"))
            if index < len(intervals) - 1:
                sites.append((end, "splice_donor"))
        else:
            if index > 0:
                sites.append((start, "splice_donor"))
            if index < len(intervals) - 1:
                sites.append((end, "splice_acceptor"))
    return sites


def load_gtf_annotations(
    gtf_path: str | Path,
    genes: pd.DataFrame,
    *,
    transcript_overrides: Mapping[str, str] | None = None,
) -> tuple[
    dict[str, str],
    dict[str, list[tuple[int, int]]],
    dict[str, list[tuple[int, str]]],
    dict[str, dict[str, list[tuple[int, int]]]],
    dict[str, dict[str, object]],
]:
    """Select one transcript per gene and return its interval annotations."""

    table = _read_gtf(gtf_path)
    if transcript_overrides is None:
        transcript_overrides = TRANSCRIPT_OVERRIDES
    transcript_ids = {}
    exon_intervals = {}
    splice_sites = {}
    transcript_features = {}
    transcript_metadata = {}
    for gene in genes.itertuples(index=False):
        selected = _select_transcript(
            table,
            gene.gene,
            int(gene.analysis_tss),
            transcript_overrides,
        )
        transcript_id = str(selected.transcript_id)
        exons = _feature_intervals(table, transcript_id, "exon")
        transcript_ids[gene.gene] = transcript_id
        exon_intervals[gene.gene] = exons
        splice_sites[gene.gene] = _splice_sites(exons, gene.strand)
        transcript_features[gene.gene] = {
            feature: _feature_intervals(table, transcript_id, feature)
            for feature in ("CDS", "UTR", "start_codon", "stop_codon")
        }
        transcript_metadata[gene.gene] = _transcript_metadata(
            selected, int(gene.analysis_tss)
        )
    return (
        transcript_ids,
        exon_intervals,
        splice_sites,
        transcript_features,
        transcript_metadata,
    )


def _contains(intervals: list[tuple[int, int]], position: int) -> bool:
    return any(start <= position < end for start, end in intervals)


def _transcript_feature(
    center: int,
    tx_offset: int,
    gene: dict[str, object],
    exons: list[tuple[int, int]],
    features: dict[str, list[tuple[int, int]]],
) -> str:
    if _contains(features.get("CDS", []), center):
        return "coding_exon"
    if not _contains(features.get("UTR", []), center):
        return "noncoding_or_untyped_exon" if _contains(exons, center) else "intron_or_intergenic"

    analysis_tss = int(gene["analysis_tss"])
    cds_offsets = []
    for start, end in features.get("CDS", []):
        if gene["strand"] == "+":
            cds_offsets.extend([start - analysis_tss, end - 1 - analysis_tss])
        else:
            cds_offsets.extend([analysis_tss - end, analysis_tss - start - 1])
    if not cds_offsets:
        return "UTR"
    return "five_prime_UTR" if tx_offset < min(cds_offsets) else "three_prime_UTR"


def _region_type(tx_offset: int, in_exon: bool, gene_length: int) -> str:
    if -500 <= tx_offset <= 200:
        return "promoter_tss"
    if 0 <= tx_offset and in_exon:
        return "exon"
    if 0 <= tx_offset <= gene_length:
        return "intron"
    return "upstream" if tx_offset < -500 else "downstream_intergenic"


def add_genomic_annotation(
    combined: pd.DataFrame,
    genes: pd.DataFrame,
    exon_intervals: dict[str, list[tuple[int, int]]],
    splice_sites: dict[str, list[tuple[int, str]]],
    splice_buffer_bp: int,
    transcript_features: dict[str, dict[str, list[tuple[int, int]]]] | None = None,
) -> pd.DataFrame:
    """Annotate each scanned edit center with transcript and splice context.

    This mirrors ``AnchoredScan.genomic_center`` rather than calling it: this
    module is imported by entry points that have no ``grelu`` bootstrap on
    ``sys.path``, so it must not depend on core at import or call time.
    """

    gene_map = genes.set_index("gene").to_dict("index")
    rows = []
    for row in combined.itertuples(index=False):
        gene = gene_map[row.gene]
        tx_offset = int(row.variant_offset_from_tss_transcription_bp)
        direction = 1 if gene["strand"] == "+" else -1
        center = int(gene["analysis_tss"]) + direction * tx_offset
        edit_start, edit_end = center - 5, center + 5
        sites = splice_sites[row.gene]
        nearby = [
            (site, kind)
            for site, kind in sites
            if edit_start <= site + splice_buffer_bp
            and edit_end >= site - splice_buffer_bp
        ]
        exons = exon_intervals[row.gene]
        in_exon = _contains(exons, center)
        record = row._asdict()
        record.update(
            {
                "chrom": gene["chrom"],
                "gene_strand": gene["strand"],
                "genomic_center": center,
                "edit_start": edit_start,
                "edit_end": edit_end,
                "region_type": _region_type(tx_offset, in_exon, int(gene["length"])),
                "transcript_feature": _transcript_feature(
                    center,
                    tx_offset,
                    gene,
                    exons,
                    (transcript_features or {}).get(row.gene, {}),
                ),
                "overlaps_splice_site": bool(nearby),
                "splice_site_types": ",".join(sorted({kind for _, kind in nearby})),
                "nearest_splice_distance_bp": (
                    min(abs(center - site) for site, _ in sites) if sites else np.nan
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def centered_interval(center: int, width: int) -> tuple[int, int]:
    """Return a half-open interval of ``width`` centered on one coordinate."""

    start = int(center) - int(width) // 2
    return start, start + int(width)


def observed_peak(
    chrom: str,
    start: int,
    end: int,
    bigwig_path: str | Path,
) -> int | None:
    """Return the strongest finite positive bigWig position in one gene span."""

    import pyBigWig

    with pyBigWig.open(str(bigwig_path), "r") as bigwig:
        values = np.asarray(
            bigwig.values(str(chrom), int(start), int(end), numpy=True),
            dtype=float,
        )
    values[~np.isfinite(values)] = 0.0
    if values.size == 0 or float(values.max()) <= 0:
        return None
    return int(start) + int(np.argmax(values))


def canonical_motif_family(name: str) -> str:
    """Map a JASPAR motif name to the report's canonical family labels."""

    factor = name.upper().split("_", 1)[-1]
    rules = [
        ("CTCF_CTCFL", ("CTCF", "CTCFL")),
        ("NFY_CCAAT", ("NFYA", "NFYB", "NFYC")),
        ("SRF_CARG", ("SRF",)),
        ("KLF_SP_GC", ("KLF",)),
        ("EGR_ZBTB", ("EGR", "ZBTB")),
        ("ETS", ("ETS", "ELF", "ERG", "FLI", "SPI1", "GABP")),
        ("AP1", ("JUN", "FOS", "BATF", "ATF3")),
        ("STAT", ("STAT",)),
        ("HNF", ("HNF", "FOXA")),
        ("CEBP", ("CEBP",)),
        ("SMAD", ("SMAD",)),
        ("TEAD", ("TEAD",)),
        ("HIF", ("HIF", "ARNT")),
        ("NFKB", ("NFKB", "RELA", "REL")),
        ("RREB1", ("RREB1",)),
    ]
    if re.match(r"^SP\d", factor):
        return "KLF_SP_GC"
    for family, tokens in rules:
        if any(token in factor for token in tokens):
            return family
    return "OTHER"


def build_edit_context_sequences(
    manifest: pd.DataFrame,
    fasta_path: Path,
    context_bp: int,
    *,
    metadata_columns: dict[str, str] | None = None,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Build paired reference/alternate contexts for every manifest edit."""

    from grelu.interpret.ism.fasta import FastaReference

    if context_bp % 2:
        raise ValueError("context-bp must be even")
    sequences = []
    sequence_ids = []
    records = []
    flank = context_bp // 2
    with FastaReference(fasta_path) as fasta:
        for mutation in manifest.itertuples(index=False):
            context_start = int(mutation.edit_center_position) - flank
            reference = fasta.extract(
                str(mutation.chrom), context_start, context_start + context_bp
            ).upper()
            edit_start = int(mutation.edit_start) - context_start
            edit_end = int(mutation.edit_end) - context_start
            observed = reference[edit_start:edit_end]
            expected = str(mutation.ref_sequence).upper()
            if observed != expected:
                raise ValueError(
                    f"Reference mismatch for {mutation.mutation_id}: {observed}"
                )
            alternate = (
                reference[:edit_start]
                + str(mutation.alt_sequence).upper()
                + reference[edit_end:]
            )
            common = {
                "mutation_id": mutation.mutation_id,
                "center": int(mutation.variant_offset_from_tss_transcription_bp),
                "edit_rel_start": edit_start,
                "edit_rel_end": edit_end,
            }
            for output_name, manifest_name in (metadata_columns or {}).items():
                common[output_name] = getattr(mutation, manifest_name)
            for state, sequence in (("ref", reference), ("alt", alternate)):
                sequence_id = f"{mutation.mutation_id}::{state}"
                sequence_ids.append(sequence_id)
                sequences.append(sequence)
                records.append(
                    {"sequence": sequence_id, **common, "state": state}
                )
    return sequences, sequence_ids, pd.DataFrame.from_records(records)
