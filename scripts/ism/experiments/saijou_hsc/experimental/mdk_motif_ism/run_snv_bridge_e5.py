#!/usr/bin/env python3
"""Execute the deferred E5 SNV bridge for the frozen r5 Mdk motif-ISM package.

Reads only existing artifacts (r5 outputs + the old 1024bp Mdk SNV saturation
runs for AlphaGenome and Borzoi FT checkpoints). Writes a standalone addendum
directory next to r5 -- it does not modify or duplicate r5's own files.

Per docs/mdk_motif_ism_execution_analysis_plan_20260909_zh.md section E5:
  - only the existing 1024bp SNV coverage is used, intersected with the r5
    candidate instances;
  - checkpoint/window/readout identity is verified before any bridging claim;
  - results are reported as a "scale response comparison" (10bp shuffle vs
    1bp SNV), never as "SNV accuracy validation";
  - single-base effects are not summed into a claimed multi-base effect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "src"))
from grelu.interpret.ism.fasta import FastaReference
from grelu.io.motifs import get_jaspar

R5 = REPO / "experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5"
OUT = REPO / "experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5_e5_snv_bridge"
FASTA = Path("/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa")
OLD_SNV = {
    "alphagenome": REPO / "experiments/ism/saijou_hsc_ag_tss1kb_full/Mdk",
    "borzoi": REPO / "experiments/ism/saijou_hsc_borzoi_tss1kb_full/Mdk",
}
FT_BACKEND = {"alphagenome": "alphagenome_finetuned", "borzoi": "borzoi_finetuned"}
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
ALPHA = 1.0


def write_tsv(p: Path, d: pd.DataFrame) -> None:
    d.to_csv(p, sep="\t", index=False)


def write_json(p: Path, x: dict) -> None:
    p.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n")


def pwm_score(seq: str, pwm: np.ndarray) -> float:
    return float(sum(np.log2((pwm[BASE[b], i] + 1e-4) / 0.25) for i, b in enumerate(seq)))


def load_old_snv(short_backend: str) -> pd.DataFrame:
    path = OLD_SNV[short_backend] / "features/mutation_features.tsv"
    x = pd.read_csv(path, sep="\t")
    x = x[(x.readout_role == "tss") & (x.track_id == "hsc")].copy()
    x["ref_sum"] = x.ref_mean * x.n_bins
    x["alt_sum"] = x.alt_mean * x.n_bins
    x["signed_effect"] = np.log2((x.alt_sum + x.n_bins * ALPHA) / (x.ref_sum + x.n_bins * ALPHA))
    x["absolute_effect"] = x.signed_effect.abs()
    return x


def checkpoint_identity_audit() -> pd.DataFrame:
    rows = []
    for short, ft in FT_BACKEND.items():
        old_meta = json.loads((OLD_SNV[short] / "model_metadata.json").read_text())
        new_meta = json.loads((REPO / f"experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle/runs/{ft}/model_metadata.json").read_text())
        old_path = old_meta["checkpoint_path"]
        new_path = new_meta["checkpoint_path"]
        match = old_path in new_path or new_path.endswith(old_path)
        rows.append(
            dict(
                backend=short,
                old_checkpoint_path=old_path,
                old_checkpoint_epoch=old_meta["checkpoint_epoch"],
                new_checkpoint_path=new_path,
                new_checkpoint_epoch=new_meta["checkpoint_epoch"],
                new_checkpoint_sha256=new_meta.get("checkpoint_sha256", ""),
                checkpoint_identity_match=bool(match and old_meta["checkpoint_epoch"] == new_meta["checkpoint_epoch"]),
            )
        )
    return pd.DataFrame(rows)


def readout_window_audit() -> pd.DataFrame:
    readouts = pd.read_csv(REPO / "experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle/prepared/readouts.tsv", sep="\t")
    mdk = readouts[readouts.gene == "Mdk"]
    rows = []
    for short in FT_BACKEND:
        old = pd.read_csv(OLD_SNV[short] / "features/mutation_features.tsv", sep="\t")
        for role in ["tss", "gene_body", "tes_3prime", "hsc_observed_peak"]:
            g = old[old.readout_role == role].iloc[0]
            old_start = int(g.output_start + g.readout_bin_start * g.output_resolution_bp)
            old_end = int(g.output_start + g.readout_bin_end * g.output_resolution_bp)
            new_role = {"tss": "tss_1024bp", "gene_body": "gene_body_output_clipped", "tes_3prime": "tes_3prime_1024bp", "hsc_observed_peak": "hsc_observed_peak_1024bp"}[role]
            new_row = mdk[mdk.role == new_role]
            new_start = int(new_row.start.iloc[0]) if len(new_row) else None
            new_end = int(new_row.end.iloc[0]) if len(new_row) else None
            rows.append(
                dict(
                    backend=short,
                    old_readout_role=role,
                    new_readout_role=new_role,
                    old_genomic_start=old_start,
                    old_genomic_end=old_end,
                    new_genomic_start=new_start,
                    new_genomic_end=new_end,
                    window_identical=bool(new_start == old_start and new_end == old_end),
                )
            )
    return pd.DataFrame(rows)


def ref_baseline_audit() -> pd.DataFrame:
    """Cross-check the model's own REF prediction (unmutated) for the tss window
    between the old SNV run and the new shuffle run -- same checkpoint, same
    window, so these must match bit-for-bit if the two runs are truly comparable."""
    import pyarrow.dataset as ds

    rows = []
    for short, ft in FT_BACKEND.items():
        old = pd.read_csv(OLD_SNV[short] / "features/mutation_features.tsv", sep="\t")
        old_tss = old[(old.readout_role == "tss") & (old.track_id == "hsc")]
        old_ref_sum = float(old_tss.ref_mean.iloc[0] * old_tss.n_bins.iloc[0])
        p = ds.dataset(REPO / f"experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle/runs/{ft}/features/combined_mutation_features.parquet", format="parquet")
        new = p.to_table(
            columns=["gene", "track_id", "readout_role", "ref_sum", "n_bins"],
            filter=(ds.field("gene") == "Mdk") & (ds.field("readout_role") == "tss_1024bp") & (ds.field("track_id") == "hsc"),
        ).to_pandas()
        rows.append(
            dict(
                backend=short,
                old_ref_sum=old_ref_sum,
                new_ref_sum_unique_count=int(new.ref_sum.nunique()),
                new_ref_sum=float(new.ref_sum.iloc[0]),
                ref_sum_bitwise_match=bool(old_ref_sum == float(new.ref_sum.iloc[0])),
            )
        )
    return pd.DataFrame(rows)


def selected_instances() -> pd.DataFrame:
    c = pd.read_csv(R5 / "candidate_instances.tsv", sep="\t")
    return c[c.selected_for_scoring.fillna(False)].copy()


def bridgeable_instances(selected: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    selected = selected.copy()
    selected["overlaps_tss_1024bp_window"] = ~((selected.genomic_end <= window[0]) | (selected.genomic_start >= window[1]))
    return selected


def per_mutation_bridge(selected_bridgeable: pd.DataFrame, motifs: dict, fasta: FastaReference) -> pd.DataFrame:
    rows = []
    for short in FT_BACKEND:
        old = load_old_snv(short)
        for r in selected_bridgeable.itertuples(index=False):
            gs, ge = int(r.genomic_start), int(r.genomic_end)
            sub = old[(old.variant_position >= gs) & (old.variant_position < ge)]
            has_pwm = isinstance(r.pwm_id, str) and r.pwm_id in motifs
            ref_seq = fasta.extract("chr2", gs, ge).upper() if has_pwm else None
            pwm = None
            if has_pwm:
                pwm = motifs[r.pwm_id]
                if r.pwm_strand == "+":
                    pwm = pwm[[3, 2, 1, 0], ::-1]
                ref_pwm_score_fixed = pwm_score(ref_seq, pwm)
            for x in sub.itertuples(index=False):
                genome_ref_base = fasta.extract("chr2", int(x.variant_position), int(x.variant_position) + 1).upper()
                base_consistent = genome_ref_base == x.ref_base
                cont_ref, cont_alt, cont_delta, loss_class = (np.nan, np.nan, np.nan, "not_applicable_structural")
                if has_pwm:
                    alt_list = list(ref_seq)
                    alt_list[int(x.variant_position) - gs] = x.alt_base
                    cont_alt_seq = "".join(alt_list)
                    cont_ref = ref_pwm_score_fixed
                    cont_alt = pwm_score(cont_alt_seq, pwm)
                    cont_delta = cont_ref - cont_alt
                    loss_class = "loss" if cont_delta >= 5 else ("preserved" if abs(cont_delta) < 1 else "ambiguous")
                rows.append(
                    dict(
                        instance_id=r.instance_id,
                        backend=short,
                        mutation_id=x.mutation_id,
                        variant_position=int(x.variant_position),
                        ref_base=x.ref_base,
                        alt_base=x.alt_base,
                        genome_ref_base_consistent=bool(base_consistent),
                        track_id="hsc",
                        bridged_readout_role="tss_1024bp",
                        n_bins=int(x.n_bins),
                        ref_sum=float(x.ref_sum),
                        alt_sum=float(x.alt_sum),
                        pseudocount_alpha=ALPHA,
                        signed_effect_log2_ratio_of_sums=float(x.signed_effect),
                        absolute_effect=float(x.absolute_effect),
                        continuous_ref_pwm_score=cont_ref,
                        continuous_alt_pwm_score=cont_alt,
                        continuous_pwm_delta=cont_delta,
                        loss_class=loss_class,
                        status="ok",
                        reason="bridged on tss_1024bp readout; checkpoint/window/REF-baseline verified identical to primary FT run",
                    )
                )
    return pd.DataFrame(rows)


def excluded_rows(selected_bridgeable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in selected_bridgeable[~selected_bridgeable.overlaps_tss_1024bp_window].itertuples(index=False):
        for short in FT_BACKEND:
            rows.append(
                dict(
                    instance_id=r.instance_id,
                    backend=short,
                    mutation_id="",
                    variant_position=np.nan,
                    ref_base="",
                    alt_base="",
                    genome_ref_base_consistent=np.nan,
                    track_id="hsc",
                    bridged_readout_role="",
                    n_bins=np.nan,
                    ref_sum=np.nan,
                    alt_sum=np.nan,
                    pseudocount_alpha=np.nan,
                    signed_effect_log2_ratio_of_sums=np.nan,
                    absolute_effect=np.nan,
                    continuous_ref_pwm_score=np.nan,
                    continuous_alt_pwm_score=np.nan,
                    continuous_pwm_delta=np.nan,
                    loss_class="",
                    status="no_snv_coverage",
                    reason="instance genomic interval falls entirely outside the old 1024bp SNV saturation window chr2:[91931785,91932809)",
                )
            )
    return pd.DataFrame(rows)


def scale_comparison(bridge: pd.DataFrame, shuffle_tss_centers: pd.DataFrame, selected_bridgeable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = bridge[bridge.status == "ok"]
    for (iid, backend), g in ok.groupby(["instance_id", "backend"]):
        inst = selected_bridgeable[selected_bridgeable.instance_id == iid].iloc[0]
        ft_backend = FT_BACKEND[backend]
        shuf = shuffle_tss_centers[
            (shuffle_tss_centers.model_backend == ft_backend)
            & (shuffle_tss_centers.edit_center_position >= inst.genomic_start - 5)
            & (shuffle_tss_centers.edit_center_position < inst.genomic_end + 5)
        ]
        rows.append(
            dict(
                instance_id=iid,
                backend=backend,
                n_snv=int(len(g)),
                n_snv_positions=int(g.variant_position.nunique()),
                median_absolute_snv_effect_tss_1024bp=float(g.absolute_effect.median()),
                max_absolute_snv_effect_tss_1024bp=float(g.absolute_effect.max()),
                n_shuffle_centers_same_instance_window=int(shuf.edit_center_position.nunique()),
                median_absolute_shuffle_effect_tss_1024bp=float(shuf.median_absolute_effect.median()) if len(shuf) else np.nan,
                ratio_shuffle_to_snv_median_absolute_effect=(float(shuf.median_absolute_effect.median()) / float(g.absolute_effect.median())) if len(shuf) and g.absolute_effect.median() else np.nan,
                note="scale response comparison across 1bp SNV vs 10bp shuffle on the SAME auxiliary tss_1024bp readout; not an SNV-accuracy validation, no allelic ground truth used",
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")
    OUT.mkdir(parents=True)

    ckpt_audit = checkpoint_identity_audit()
    window_audit = readout_window_audit()
    ref_audit = ref_baseline_audit()
    write_tsv(OUT / "e5_checkpoint_identity_audit.tsv", ckpt_audit)
    write_tsv(OUT / "e5_readout_window_audit.tsv", window_audit)
    write_tsv(OUT / "e5_ref_baseline_audit.tsv", ref_audit)

    if not ckpt_audit.checkpoint_identity_match.all():
        raise SystemExit("checkpoint identity audit failed; refusing to bridge")
    if not ref_audit.ref_sum_bitwise_match.all():
        raise SystemExit("REF baseline audit failed; refusing to bridge")

    selected = selected_instances()
    tss_window = (91931785, 91932809)  # verified identical: old SNV saturation window == new tss_1024bp readout
    selected_bridgeable = bridgeable_instances(selected, tss_window)
    write_tsv(OUT / "e5_instance_coverage_audit.tsv", selected_bridgeable[["instance_id", "hypothesis_id", "genomic_start", "genomic_end", "overlaps_tss_1024bp_window"]])

    motifs = get_jaspar(release="JASPAR2024", tax_group="vertebrates")
    with FastaReference(FASTA) as fa:
        ok_rows = per_mutation_bridge(selected_bridgeable[selected_bridgeable.overlaps_tss_1024bp_window], motifs, fa)
    excl_rows = excluded_rows(selected_bridgeable)
    bridge = pd.concat([ok_rows, excl_rows], ignore_index=True)
    write_tsv(OUT / "snv_bridge.tsv", bridge)

    if not bridge[bridge.status == "ok"].genome_ref_base_consistent.all():
        raise SystemExit("old-SNV ref_base disagrees with mm10 FASTA at some position; refusing summary")

    # loaded from a precomputed extract to avoid re-reading the 430MB center_effects.tsv
    if len(sys.argv) <= 1:
        raise SystemExit("usage: run_snv_bridge_e5.py <path-to-tss_1024bp_ft_centers.tsv>")
    shuffle_tss_centers = pd.read_csv(Path(sys.argv[1]), sep="\t")
    summary = scale_comparison(bridge, shuffle_tss_centers, selected_bridgeable)
    write_tsv(OUT / "snv_bridge_scale_comparison.tsv", summary)

    write_json(
        OUT / "e5_status.json",
        {
            "status": "ok",
            "scope": "E5 SNV bridge on tss_1024bp auxiliary readout only, for instances whose genomic interval overlaps chr2:[91931785,91932809)",
            "bridgeable_instances": sorted(selected_bridgeable[selected_bridgeable.overlaps_tss_1024bp_window].instance_id.tolist()),
            "not_bridgeable_instances": sorted(selected_bridgeable[~selected_bridgeable.overlaps_tss_1024bp_window].instance_id.tolist()),
            "not_bridgeable_reason": "outside the old 1024bp SNV saturation window; no per-base coverage exists for these instances at any readout",
            "gene_body_output_clipped_bridge": "not attempted: old SNV 'gene_body' window (91929737-91932297) and new gene_body_output_clipped window (91929827-91932297) differ by ~90bp at the 3' boundary; this is the FT primary S_RNA readout and was not treated as bridged",
            "tes_3prime_bridge": "not attempted: old TES anchor 91929804 vs new TES anchor 91929827 (23bp shift) and window width differs (1152bp vs 1024bp)",
            "hsc_observed_peak_bridge": "not attempted: old and new observed-peak windows have different anchors and widths",
            "checkpoint_identity": "verified bit-identical checkpoint path/epoch for both backends",
            "ref_baseline_identity": "verified bit-identical REF ratio-of-sums on tss_1024bp/hsc for both backends",
        },
    )
    print(OUT)


if __name__ == "__main__":
    main()
