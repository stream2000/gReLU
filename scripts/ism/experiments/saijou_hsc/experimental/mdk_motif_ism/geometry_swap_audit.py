#!/usr/bin/env python3
"""Readout-geometry swap audit for the two Mdk WT1 candidate sites.

`run_mdk_motif_ism.original_scores` scores the original backends only under
`tss_1024bp`, while `ft_scores` scores the fine-tuned backends under
`gene_body_output_clipped`. The published original-vs-fine-tuned contrast is
therefore confounded by readout geometry. `center_effects.tsv` already stores
every backend under all five readout roles, so the confound can be removed
without new inference.

This audit recomputes, for both WT1 sites and every (backend, readout role,
track), the same center statistic and same-geometry background percentile the
frozen pipeline uses, so geometry and track composition can be separated.

Unvalidated prototype: `experimental/` only, per ANALYSIS_HARNESS.md.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[6]
OUT = REPO / 'experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5'
# mm10 Mdk-201 (ENSMUST00000028672, chr2 minus strand); coordinates propagated
# unchanged from candidate_instances.tsv / the r5 WT1 audit. 0-based half-open.
SITES = {
    'wt1_0001_promoter': (91933589, 91933598),  # TSS -1300..-1290, GAGGGGGAGG
    'wt1_0002_exon': (91930029, 91930038),      # TSS +2260..+2270, GTGGGAGAGG
}
EDIT_SPAN = 10  # scan edit window width; centers are window midpoints
TX_MODALITIES = ('rna', 'rna_seq', 'cage', '10x_scRNAseq_3prime_pseudobulk')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    return p.parse_args()


def load_centers(out: Path) -> pd.DataFrame:
    cols = ['model_backend', 'track_id', 'track_modality', 'readout_role',
            'edit_center_position', 'median_absolute_effect', 'median_signed_effect']
    x = pd.read_csv(out / 'center_effects.tsv', sep='\t', usecols=cols)
    return x.rename(columns={'model_backend': 'backend', 'track_modality': 'modality',
                             'edit_center_position': 'center',
                             'median_absolute_effect': 'abs_eff',
                             'median_signed_effect': 'signed_eff'})


def support(centers: np.ndarray, start: int, end: int) -> list[int]:
    """Centers whose edit window overlaps the motif; mirrors run_mdk_motif_ism.support."""
    h = EDIT_SPAN // 2
    return [int(c) for c in centers if c - h < end and c + h > start]


def window_index(centers: np.ndarray, sup: dict[str, list[int]]):
    """Same-geometry background windows as index arrays, excluding every selected support."""
    pos = {int(c): i for i, c in enumerate(centers)}
    excluded = set().union(*(set(v) for v in sup.values()))
    out = {}
    for site, s in sup.items():
        rel = [pos[x] - pos[s[0]] for x in s]
        rows = [[i + d for d in rel] for i in range(len(centers))
                if i + rel[-1] < len(centers)
                and not {int(centers[i + d]) for d in rel} & excluded]
        out[site] = (np.array([pos[x] for x in s]), np.array(rows))
    return out


def score(df: pd.DataFrame, centers: np.ndarray, windows) -> pd.DataFrame:
    rows = []
    for (backend, track, modality, role), g in df.groupby(
            ['backend', 'track_id', 'modality', 'readout_role'], sort=False):
        g = g.set_index('center')
        a = g.abs_eff.reindex(centers).to_numpy(float)
        s = g.signed_eff.reindex(centers).to_numpy(float)
        for site, (sup_i, bg_i) in windows.items():
            stat = float(np.median(a[sup_i]))
            bg = np.median(a[bg_i], axis=1)
            rows.append(dict(site=site, backend=backend, readout_role=role,
                             track_id=track, modality=modality, raw_statistic=stat,
                             percentile_score=100.0 * float((bg < stat).mean()),
                             median_signed_effect=float(np.median(s[sup_i])),
                             covered_centers=len(sup_i), background_n=len(bg)))
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    df = load_centers(args.out)
    centers = np.array(sorted(df.center.unique()))
    sup = {k: support(centers, *v) for k, v in SITES.items()}
    windows = window_index(centers, sup)
    table = score(df, centers, windows)

    if not table.percentile_score.between(0, 100).all():
        raise ValueError('percentile outside [0, 100]')
    if not np.isfinite(table.raw_statistic).all():
        raise ValueError('non-finite center statistic')
    dup = table.duplicated(['site', 'backend', 'readout_role', 'track_id'])
    if dup.any():
        raise ValueError(f'{int(dup.sum())} duplicate table keys')

    path = args.out / 'wt1_geometry_swap_scores.tsv'
    table.to_csv(path, sep='\t', index=False)

    tx = table[table.modality.isin(TX_MODALITIES)]
    summary = (tx.groupby(['site', 'backend', 'readout_role'])
                 .agg(tracks=('percentile_score', 'size'),
                      median_percentile=('percentile_score', 'median'),
                      max_percentile=('percentile_score', 'max'),
                      median_raw_statistic=('raw_statistic', 'median'),
                      median_signed_effect=('median_signed_effect', 'median'))
                 .reset_index())
    summary.to_csv(args.out / 'wt1_geometry_swap_summary.tsv', sep='\t', index=False)

    with open(args.out / 'wt1_geometry_swap_validation.json', 'w') as fh:
        json.dump(dict(
            version='wt1-geometry-swap-v1',
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            source_table=str((args.out / 'center_effects.tsv').relative_to(REPO)),
            sites={k: dict(genomic_start=v[0], genomic_end=v[1],
                           covered_centers=len(sup[k])) for k, v in SITES.items()},
            background_n={k: int(v[1].shape[0]) for k, v in windows.items()},
            statistic='median over support of per-center median absolute log2 effect',
            background='same relative center geometry; all selected supports excluded',
            rows=int(len(table)), status='ok'), fh, indent=1)

    print(summary[summary.site.eq('wt1_0002_exon')].to_string(index=False))
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
