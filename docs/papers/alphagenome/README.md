# AlphaGenome paper — local copies

Downloaded 2026-10-04.

| File | What it is |
|---|---|
| `alphagenome_nature_2026.md` | Main text of the Nature paper (Avsec et al., 2026, doi:10.1038/s41586-025-10014-0) converted to Markdown from the Europe PMC full-text XML. Includes figure and extended-data captions and the reference list; table bodies are omitted. |
| `alphagenome_nature_2026_PMC12851941.xml` | Source JATS XML (PMC12851941, CC BY). |
| `alphagenome_nature_2026_supplementary.pdf` / `.txt` | Nature Supplementary Information (76 pp.). The Nature main text has no Methods section of its own; data, architecture, losses, training and distillation are all described here. |
| `alphagenome_biorxiv_v1.pdf` / `.txt` | bioRxiv preprint v1 (doi:10.1101/2025.06.25.661532), with its Methods in the same PDF. The `.txt` is `pdftotext -layout` output. |

Where to look in `alphagenome_nature_2026_supplementary.txt`:

- Training data: around line 58
- Output heads and losses: around line 619 (track loss around line 825; RNA-seq gene-level loss around line 856)
- Training parameters and distillation: around lines 877–915
- Benchmarking: around line 917
