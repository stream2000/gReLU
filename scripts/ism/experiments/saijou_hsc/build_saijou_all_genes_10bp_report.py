#!/usr/bin/env python
"""Build the canonical portable technical report for nine-gene 10-bp ISM."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__:
    from .tools.report_data import prepare_report_data
    from .tools.report_figures import build_all_genes_static_figures
    from .tools.report_spec import TITLE, build_report_spec
    from .tools.report_summary import build_summary_tables
else:
    from tools.report_data import prepare_report_data
    from tools.report_figures import build_all_genes_static_figures
    from tools.report_spec import TITLE, build_report_spec
    from tools.report_summary import build_summary_tables


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_all_genes_10bp_scan"
STYLE_ID = "saijou-print-layout"
PRINT_STYLE = f"""<style id="{STYLE_ID}">
@media print {{
  .portable-block-stack {{ display: block !important; }}
  .portable-block {{ margin-bottom: 20px; }}
  [data-artifact-block-type="markdown"] +
  [data-artifact-block-type="html"] {{
    break-before: page;
    page-break-before: always;
  }}
}}
</style>
"""

CHART_MAP = """# Chart map

| Section | Question | Family / type | Fields | Supported claim | Palette policy |
|---|---|---|---|---|---|
| Candidate specificity | How HSC-selective is each candidate in each model? | comparison / horizontal grouped bar | segment, ratio, model | strength and specificity differ | hard two-root |
| Four-cell matrix | Which head is actually strongest? | matrix / heatmap | segment-model, cell, median absolute effect | five candidates are not HSC-top in both | hard two-root |
| Signed HSC Browser | What is the direction and magnitude across the full 1-kb scan? | ordered-axis / nine small multiples | TSS offset, signed median log2FC, model | position and effect size can be read directly | hard two-root plus zero line; gene-specific symmetric scales |
| Mdk robustness | Does the hub replicate across shuffle manifests? | ordered-axis / multi-series line | offset, margin, model-manifest | segment-level Borzoi result is more stable than center ranking | hard two-root plus zero line |
| Original context | Were candidates already sequence-sensitive before fine-tuning? | comparison / horizontal grouped bar | segment, max original effect, model | most candidates have pre-existing context | hard two-root |
"""

DESIGN_NOTES = """# DESIGN

- Surface: single-column technical report.
- Palette: AlphaGenome blue, Borzoi orange, neutrals for references; no red/green semantics.
- Charts: full-width; horizontal bars for long segment labels; heatmap for four-cell matrix; zero-aware line chart for signed Mdk margins.
- Tables: spacious for interpretation and controls, dense for audit detail.
- Accessibility: model identity is carried by labels and ordering in addition to color.
"""

REPORT_DATASET_ORDER = (
    "summary",
    "hsc_ratio",
    "fine_cell_matrix",
    "original_top",
    "original_group_matrix",
    "overview",
    "controls",
    "motif_top",
    "gene_interpretation",
    "genes",
    "transcripts",
    "mdk_hub",
    "mdk_centers",
    "mdk_concordance",
    "validation",
    "fine_signed_browser",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--finalize-html",
        type=Path,
        help="Apply print pagination to an already packaged HTML report.",
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _records(table: pd.DataFrame) -> list[dict]:
    return json.loads(table.to_json(orient="records"))


def write_summary_tables(root: Path) -> dict[str, object]:
    """Write the stable report tables consumed by the artifact builder."""

    out = root / "analysis/report_data"
    out.mkdir(parents=True, exist_ok=True)
    summary = build_summary_tables(root)
    for name, table in summary.frames.items():
        table.to_csv(out / f"{name}.tsv", sep="\t", index=False)
    (out / "validation_summary.json").write_text(
        json.dumps(summary.validation, indent=2) + "\n"
    )
    return summary.validation


def build_report_artifact(
    frames: dict[str, pd.DataFrame],
    static_figures: dict[str, str],
    generated_at: str,
) -> dict[str, object]:
    """Combine declarative report specs with the prepared data snapshot."""

    sources, cards, charts, tables, blocks = build_report_spec(static_figures)
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Complete nine-gene Saijou HSC 10-bp ISM comparison across fine-tuned and original models.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [
                {"id": item["id"], "label": item["label"], "path": item["path"]}
                for item in sources
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {name: _records(frames[name]) for name in REPORT_DATASET_ORDER},
        },
        "sources": sources,
    }


def _source_notes(generated_at: str) -> str:
    return f"""# Report source notes

- Audience: technical; primary question is which TSS-proximal sequence regions drive HSC-head sensitivity and how original-model tracks change the interpretation.
- Delivery: canonical portable HTML followed by PDF export.
- Comparison basis: HSC versus the maximum of mac/LSEC/chol using median absolute ratio-of-sums log2 effect at the candidate-selected readout.
- Selection conditioning: 13/18 HSC-top is not an unbiased hit rate and is explicitly labeled in the report.
- Original tracks: curated liver/fibroblast/smooth-muscle/mesenchymal proxies, not matched Saijou cell types.
- Transcript caveat: Col1a2-205 and Hgf-202 are non-basic alternative/retained-intron representatives.
- Artifact generated: {generated_at}
"""


def write_report_artifacts(
    report_dir: Path,
    artifact: dict[str, object],
    gene_interpretation: pd.DataFrame,
    generated_at: str,
) -> Path:
    """Write the portable artifact and its human-readable companions."""

    artifact_path = report_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    gene_interpretation.to_csv(
        report_dir / "gene_interpretation.tsv", sep="\t", index=False
    )
    (report_dir / "chart_map.md").write_text(CHART_MAP)
    (report_dir / "source_notes.md").write_text(_source_notes(generated_at))
    (report_dir / "DESIGN.md").write_text(DESIGN_NOTES)
    return artifact_path


def apply_print_layout(path: Path) -> bool:
    """Inject print-only pagination once into a packaged HTML report."""

    html = path.read_text()
    if f'id="{STYLE_ID}"' in html:
        return False
    if "</head>" not in html:
        raise ValueError(f"Packaged report has no </head>: {path}")
    path.write_text(html.replace("</head>", PRINT_STYLE + "</head>", 1))
    return True


def main() -> None:
    args = parse_args()
    if args.finalize_html:
        path = args.finalize_html.resolve()
        changed = apply_print_layout(path)
        print(f"{path}: {'updated' if changed else 'already finalized'}")
        return

    root = args.root.resolve()
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_validation = write_summary_tables(root)
    prepared = prepare_report_data(root)
    frames = prepared.frames
    static_figures = build_all_genes_static_figures(
        report_dir,
        frames["overview"],
        frames["hsc_ratio"],
        frames["fine_cell_matrix"],
        frames["mdk_centers"],
        frames["original_top"],
        frames["fine_signed_browser"],
    )
    generated_at = utc_timestamp()
    artifact = build_report_artifact(frames, static_figures, generated_at)
    artifact_path = write_report_artifacts(
        report_dir,
        artifact,
        frames["gene_interpretation"],
        generated_at,
    )
    print(artifact_path)
    print(
        json.dumps(
            {
                "summary_validation": summary_validation,
                "candidate_validation": prepared.candidate_validation,
                "motif_validation": prepared.motif_validation,
                "report_datasets": len(artifact["snapshot"]["datasets"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
