# MREG Single-Locus Algorithm Review

This note documents the biology-facing algorithm used by the current MREG
single-locus AlphaGenome and Borzoi experiments. It follows the flow from
inputs to final outputs and shows the code paths that determine the biological
interpretation.

The central question is:

```text
If we perturb selected bases near MREG, how do predicted MCF-7 regulatory
tracks and contact-map summaries change at the edited locus and at the MREG TSS?
```

## 1. Inputs

The experiment starts from curated MREG regions, genome resources, motif
resources, model track metadata and model checkpoints.

The manifest builder expects curated regions with biological roles:

```python
def prepare_manifests(
    *,
    curated_regions: pd.DataFrame,
    fasta_path: Path,
    gtf_path: Path,
    meme_path: Path,
    gene_name: str = "MREG",
    output_dir: Path,
) -> dict:
    """Main entry point: produce all manifests from curated region input.

    Parameters
    ----------
    curated_regions : pd.DataFrame
        Must have columns: region_role, chrom, start, end, summit, strand, source, label.
        region_role ∈ {tss, hc_dic, lc_dic}.
```

The code loads hg38 sequence, JASPAR CTCF motif, MREG gene annotation and MREG
transcripts:

```python
fasta = FastaReference(str(fasta_path))
pwm = load_meme_pwm(str(meme_path), motif_name="CTCF", pseudocount=PSEUDOCOUNT)

# Extract MREG gene info from GTF
gene_chrom, gene_start, gene_end, gene_strand = _parse_gtf_gene_region(gtf_path, gene_name)
transcripts = _parse_gtf_transcripts(gtf_path, gene_name)
```

Biological meaning:

- `curated_regions` defines the three loci: MREG TSS, HC-DIC and LC-DIC.
- `hg38 FASTA` provides the actual reference sequence to edit.
- `GTF` defines the MREG gene body and fallback TSS.
- `JASPAR CTCF PWM` defines which local sequence is treated as a CTCF motif.
- model track metadata later determines which experimental assays/cell contexts
  are read from AlphaGenome or Borzoi.

## 2. Region QC And Biological Anchors

The primary TSS comes from the curated TSS region when present; otherwise it is
derived from GTF transcripts.

```python
tss_rows = curated_regions[curated_regions["region_role"] == "tss"]
if tss_rows.empty:
    # Fall back to GTF: use the most upstream TSS
    primary_tss = int(transcripts["tss"].min()) if not transcripts.empty else None
    if primary_tss is None:
        raise ValueError("No TSS region provided and no transcripts found in GTF")
    print(f"[prepare] Using primary TSS from GTF: {gene_chrom}:{primary_tss}")
else:
    primary_tss = int(tss_rows.iloc[0]["summit"])
    print(f"[prepare] Using primary TSS from curated regions: {gene_chrom}:{primary_tss}")
```

HC-DIC and LC-DIC candidates are checked against the MREG gene body and kept
away from gene ends. This avoids interpreting a boundary or promoter-adjacent
site as an internal DIC.

```python
if role in ("hc_dic", "lc_dic"):
    if region_start < gene_start or region_end > gene_end:
        qc_issues.append(
            f"{role}: candidate {gene_chrom}:{region_start}-{region_end} "
            f"is outside the MREG gene body {gene_start}-{gene_end}"
        )
    distance_to_gene_ends = min(
        abs(region_summit - gene_start),
        abs(region_summit - gene_end),
    )
    if distance_to_gene_ends <= 10_000:
        qc_issues.append(
            f"{role}: candidate summit {region_summit} is only "
            f"{distance_to_gene_ends} bp from a gene end; DIC requires >10 kb"
        )
```

The region manifest records the biological role, distance to MREG TSS, CTCF
motif fields and Pol2 peak fields:

```python
region_records.append({
    "region_id": _make_region_id(role),
    "region_role": role,
    "chrom": str(row["chrom"]),
    "start": region_start,
    "end": region_end,
    "summit": region_summit,
    "host_gene": gene_name,
    "tss": primary_tss,
    "distance_to_tss": int(row.get("summit", (int(row["start"]) + int(row["end"])) // 2)) - primary_tss,
    "has_ctcf_motif": False,
    "ctcf_motif_start": -1,
    "ctcf_motif_end": -1,
    "ctcf_motif_score": np.nan,
    "pol2_peak_start": _optional_int(row, "pol2_peak_start"),
    "pol2_peak_end": _optional_int(row, "pol2_peak_end"),
    "pol2_peak_summit": _optional_int(row, "pol2_peak_summit"),
    "review_status": review_status,
})
```

Output:

```text
prepared/mreg_region_manifest.tsv
```

## 3. Perturbation Design

The experiment uses different mutation strategies for the three biological
regions:

```text
MREG TSS  -> CTCF motif disruption
HC-DIC    -> CTCF motif disruption
LC-DIC    -> Pol2 peak-center perturbation
```

### 3.1 TSS And HC-DIC: CTCF Motif Disruption

Only TSS and HC-DIC enter the CTCF motif-disruption path:

```python
if role in ("tss", "hc_dic"):
    # ---- CTCF motif disruption strategy ----
    region_seq = fasta.extract(chrom, region_start, region_end)
    hits = _scan_region_for_ctcf(fasta, chrom, region_start, region_end, pwm)
    if not hits:
        qc_issues.append(f"{region_id}: no CTCF motif found in region {chrom}:{region_start}-{region_end}")
        continue

    best_hit = hits[0]  # highest-scoring motif
```

The scan checks both strands and keeps CTCF hits with relative PWM score >=
`0.80`. Hits are sorted by strongest raw PWM score:

```python
for offset in range(0, len(sequence) - pwm.width + 1):
    window = sequence[offset : offset + pwm.width]
    for strand, oriented in (("+", window), ("-", reverse_complement(window))):
        score = score_oriented_sequence(oriented, pwm)
        if not np.isfinite(score):
            continue
        relative = pwm.relative_score(score)
        if min_relative_score is not None and relative < min_relative_score:
            continue
        hits.append(
            MotifHit(
                start=genomic_start + offset,
                end=genomic_start + offset + pwm.width,
                strand=strand,
                score=score,
                relative_score=relative,
                genomic_sequence=window,
            )
        )
return sorted(
    hits,
    key=lambda hit: (-hit.score, hit.start, hit.strand),
)
```

For the selected motif, exactly three motif positions are edited. The edited
motif must drop below relative PWM score `0.80`, and a +/-100 bp rescan must
not create a new high-scoring CTCF motif:

```python
@dataclass
class _MinimalMutationConfig:
    positions_per_motif: int = 3
    max_positions_per_motif: int = 3
    min_relative_motif_score: float = 0.80
    motif_rescan_flank_bp: int = 100
    reject_new_high_scoring_motif: bool = True
    max_control_distance_bp: int = 500
    max_gc_change_difference: float = 0.0
    local_controls_per_motif: int = 3
    min_local_controls_per_motif: int = 1
    include_motif_preserving_control: bool = False
    motif_name: str = "CTCF"
```

The three edited bases are chosen by their PWM contribution. For each position,
the algorithm finds the lowest-scoring non-reference base and ranks positions
by the score drop from reference to that alternative:

```python
oriented_ref = _oriented_sequence(hit)
contributions: list[Tuple[float, int, str]] = []
for offset, ref_base in enumerate(oriented_ref):
    ref_index = BASE_INDEX[ref_base]
    alternative_scores = [
        (pwm.pssm[offset, index], base)
        for index, base in enumerate(BASES)
        if index != ref_index
    ]
    min_score, min_base = min(
        alternative_scores,
        key=lambda item: (item[0], item[1]),
    )
    contribution = float(pwm.pssm[offset, ref_index] - min_score)
    contributions.append((contribution, offset, min_base))
contributions.sort(key=lambda item: (-item[0], item[1]))
```

The selected offsets are edited, scored, converted back to genomic strand, and
rescanned for newly created motifs:

```python
selected = contributions[:edit_count]
oriented_alt = list(oriented_ref)
offsets = []
for _, offset, alt_base in selected:
    oriented_alt[offset] = alt_base
    offsets.append(offset)
genomic_alt = _genomic_alt_sequence(oriented_alt, hit.strand)
alt_score = score_oriented_sequence("".join(oriented_alt), pwm)
alt_relative = pwm.relative_score(alt_score)
if alt_relative >= config.mutation.min_relative_motif_score:
    continue
```

```python
creates_new = _creates_new_high_scoring_motif(
    ref_context,
    alt_context,
    context_start,
    pwm,
    config.mutation.min_relative_motif_score,
)
if config.mutation.reject_new_high_scoring_motif and creates_new:
    continue
```

The accepted edit becomes a mutation-manifest row:

```python
mutation_records.append({
    "mutation_id": mutation_id,
    "region_id": region_id,
    "mutation_strategy": "strand_aware_ctcf_pwm_disruption",
    "edit_start": disruption.edit_start,
    "edit_end": disruption.edit_end,
    "ref_sequence": disruption.ref_sequence,
    "alt_sequence": disruption.alt_sequence,
    "target_feature": "CTCF_motif",
    "matched_control_id": f"{mutation_id}_ctrl",
    "ref_pwm_score": disruption.pwm_ref_score,
    "alt_pwm_score": disruption.pwm_alt_score,
    "delta_pwm_score": disruption.pwm_alt_score - disruption.pwm_ref_score,
    "fasta_ref_match": True,
    "qc_status": "ready",
})
```

### 3.2 LC-DIC: Pol2 Peak-Center Perturbation

LC-DIC does not use the CTCF disruption path. It uses a deterministic 3 bp
sequence perturbation at the Pol2 peak center, falling back to the region
coordinates if explicit Pol2 peak fields are absent:

```python
elif role == "lc_dic":
    # ---- LC-DIC Pol2 peak-center perturbation ----
    pol2_peak_start = int(region.get("pol2_peak_start", -1))
    pol2_peak_end = int(region.get("pol2_peak_end", -1))
    pol2_peak_summit = int(region.get("pol2_peak_summit", -1))
    # Use region coordinates as fallback if pol2_peak_* not set
    if pol2_peak_start <= 0:
        pol2_peak_start = region_start
    if pol2_peak_end <= 0:
        pol2_peak_end = region_end
    if pol2_peak_summit <= 0:
        pol2_peak_summit = region_summit

    pert = _design_pol2_peak_center_perturbation(
        fasta=fasta,
        chrom=chrom,
        peak_start=pol2_peak_start,
        peak_end=pol2_peak_end,
        peak_summit=pol2_peak_summit,
        edit_length=3,
        gc_change_tolerance=1.0,
    )
```

The LC-DIC row is marked as Pol2 peak-center, not CTCF:

```python
mutation_records.append({
    "mutation_id": mutation_id,
    "region_id": region_id,
    "mutation_strategy": "pol2_peak_center_sequence_perturbation",
    "edit_start": pert["edit_start"],
    "edit_end": pert["edit_end"],
    "ref_sequence": pert["ref_sequence"],
    "alt_sequence": pert["alt_sequence"],
    "target_feature": "Pol2_peak_center",
    "matched_control_id": f"{mutation_id}_ctrl",
    "ref_pwm_score": np.nan,
    "alt_pwm_score": np.nan,
    "delta_pwm_score": np.nan,
    "fasta_ref_match": True,
    "qc_status": "ready",
})
```

Output:

```text
prepared/mreg_mutation_manifest.tsv
prepared/mreg_sequence_contexts.tsv
```

## 4. Matched Controls

The biological readout is not only raw ALT-REF. Each experimental edit has
local matched controls so that sequence-edit artifacts can be subtracted.

For CTCF edits, local non-motif controls are designed after the CTCF disruption:

```python
controls = _design_local_controls(
    fasta=fasta,
    chrom=chrom,
    site_id=region_id,
    hit=best_hit,
    disruption=disruption,
    pwm=pwm,
    config=config,
)
for ctrl_idx, control in enumerate(controls):
    ctrl_id = f"{mutation_id}_ctrl_{ctrl_idx:02d}"
    mutation_records.append({
        "mutation_id": ctrl_id,
        "region_id": region_id,
        "mutation_strategy": "local_non_motif_control",
        "edit_start": control.edit_start,
        "edit_end": control.edit_end,
        "ref_sequence": control.ref_sequence,
        "alt_sequence": control.alt_sequence,
        "target_feature": "non_motif_control",
        "matched_control_id": "",
```

For LC-DIC, the matched control is a non-peak local sequence control:

```python
ctrl = _design_local_sequence_control(
    fasta=fasta,
    chrom=chrom,
    target_edit_start=pert["edit_start"],
    target_edit_end=pert["edit_end"],
    target_ref=pert["ref_sequence"],
    target_alt=pert["alt_sequence"],
    target_gc_delta=pert["gc_delta_count"],
    target_edit_length=pert["edited_base_count"],
    peak_start=pol2_peak_start,
    peak_end=pol2_peak_end,
    max_distance_bp=5_000,
    pwm=pwm,
)
```

Biological meaning:

- CTCF controls test whether the observed change is specific to disrupting the
  motif instead of simply changing nearby DNA.
- LC controls test whether the Pol2-center change is stronger than similar
  local non-peak changes.

## 5. Readout Windows

The experiment defines named readouts before model inference. These include
MREG TSS, HC-DIC, LC-DIC and mutation-centered windows.

```python
records.append({
    "readout_id": "hc_dic_4kb",
    "source_region_id": "mreg_hc_dic",
    "chrom": chrom,
    "start": max(0, hc_dic_summit - radius_bp),
    "end": hc_dic_summit + radius_bp,
    "role": "hc_dic_readout",
    "anchor_coordinate": hc_dic_summit,
    "expected_bin_count": (2 * radius_bp) // 128,
})
```

```python
records.append({
    "readout_id": "mutation_local_4kb",
    "source_region_id": "_per_mutation_",
    "chrom": chrom,
    "start": -1,
    "end": -1,
    "role": "mutation_local_readout",
    "anchor_coordinate": -1,
    "expected_bin_count": 4096 // 128,
})
```

Biological meaning:

- Local readouts ask whether the edit changes the edited peak itself.
- DIC-to-TSS readouts ask whether a distal edit changes predicted signal at the
  MREG promoter.
- Mutation-centered windows make local interpretation comparable across TSS,
  HC-DIC and LC-DIC.

Output:

```text
prepared/mreg_readout_regions.tsv
```

## 6. Track Selection

Track selection is separate from mutation design. It determines which predicted
experimental signals are read from the model.

### 6.1 AlphaGenome Tracks

The AlphaGenome registry focuses on 128 bp ChIP-seq tracks. It asks for MCF-7
tracks for CTCF, RAD21, POLR2A and selected active chromatin marks.

```python
def _build_track_registry() -> pd.DataFrame:
    """Build the hardcoded track registry for the three-region pilot.

    Focused on 128 bp ChIP-seq tracks: CTCF, RAD21, POLR2A, SMC3, H3K4me3,
    H3K27ac, H3K4me2. MCF-7 preferred, with deterministic fallback.
    """
    targets = [
        # (target_name, selection_priority, is_primary, output_type)
        ("CTCF", 1, True, "chip_tf"),
        ("RAD21", 2, True, "chip_tf"),
        ("POLR2A", 3, True, "chip_tf"),
        ("POLR2B", 4, False, "chip_tf"),  # fallback if no POLR2A
        ("SMC3", 5, True, "chip_tf"),
        ("H3K4me3", 6, True, "chip_histone"),
        ("H3K27ac", 7, False, "chip_histone"),
        ("H3K4me2", 8, False, "chip_histone"),
    ]
```

At runtime, the registry is resolved against AlphaGenome metadata by matching
target name and preferring exact `MCF-7`. Substring matching is intentionally
avoided because it can confuse `MCF-7` with other MCF samples.

```python
# Try to match by target name (case-insensitive)
tx_col = "transcription_factor" if output_type == "chip_tf" else "histone_mark"
if tx_col in sub.columns:
    sub["_target"] = sub[tx_col].fillna("").astype(str).str.upper()
    sub = sub[sub["_target"] == target]
elif "target" in sub.columns:
    sub["_target"] = sub["target"].fillna("").astype(str).str.upper()
    sub = sub[sub["_target"] == target]
```

```python
# Prefer the requested biosample exactly. Substring matching is unsafe:
# "MCF" matches MCF 10A before MCF-7 in the current metadata.
biosample_col = "biosample_name" if "biosample_name" in sub.columns else "biosample"
requested_biosample = str(reg_row.get("biosample", "")).strip().upper()
exact_biosample = sub[
    sub[biosample_col].fillna("").astype(str).str.strip().str.upper()
    == requested_biosample
]
if not exact_biosample.empty:
    best = exact_biosample.iloc[0]
    fallback = ""
else:
    best = sub.iloc[0]
    best_biosample = str(best.get(biosample_col, best.get("biosample_name", "unknown")))
    fallback = (
        f"no exact {requested_biosample} track for {target}; "
        f"using {best_biosample}"
    )
```

Output:

```text
run/resolved_tracks.tsv
```

### 6.2 Borzoi Tracks

Borzoi uses its own task metadata, so the track set is specified separately.
The MCF-7-centered set includes CAGE, RNA, DNase, CTCF, POLR2A, TFs and active
histone marks when those tasks exist.

```python
TRACK_SPECS = [
    ("CAGE_MCF7_PLUS", "CAGE", "CNhs11943+", "CAGE", "breast carcinoma cell line:MCF7", "promoter"),
    ("CAGE_MCF7_MINUS", "CAGE", "CNhs11943-", "CAGE", "breast carcinoma cell line:MCF7", "promoter"),
    ("RNA_MCF7_PLUS", "RNA", "ENCFF795XIC+", "RNA", "MCF-7", "rna"),
    ("RNA_MCF7_MINUS", "RNA", "ENCFF795XIC-", "RNA", "MCF-7", "rna"),
    ("DNASE_MCF7", "DNASE", "ENCFF924FJR", "DNASE", "MCF-7", "accessibility"),
    ("CTCF_MCF7", "CTCF", None, "CHIP", "CTCF:MCF-7", "structural"),
    ("POLR2A_MCF7", "POLR2A", None, "CHIP", "POLR2A:MCF-7", "polymerase"),
    ("FOXA1_MCF7", "FOXA1", None, "CHIP", "FOXA1:MCF-7", "tf"),
    ("GATA3_MCF7", "GATA3", None, "CHIP", "GATA3:MCF-7", "tf"),
    ("H3K27ac_MCF7", "H3K27ac", None, "CHIP", "H3K27ac:MCF-7", "active_mark"),
    ("H3K4me1_MCF7", "H3K4me1", None, "CHIP", "H3K4me1:MCF-7", "active_mark"),
    ("H3K4me2_MCF7", "H3K4me2", None, "CHIP", "H3K4me2:MCF-7", "active_mark"),
    ("H3K4me3_MCF7", "H3K4me3", None, "CHIP", "H3K4me3:MCF-7", "active_mark"),
]
```

Tasks are selected by exact task name when available, otherwise by exact or
contains-style assay/sample description. Missing tasks are not synthesized.

```python
for rank, (track_id, target, name, assay, sample_token, group) in enumerate(TRACK_SPECS, start=1):
    if name is not None:
        sub = tasks[tasks["name"].astype(str) == name].copy()
    else:
        exact = f"{assay}:{sample_token}"
        sub = tasks[tasks["description"].astype(str) == exact].copy()
        if sub.empty:
            sub = tasks[
                (tasks["assay"].astype(str) == assay)
                & tasks["description"].astype(str).str.contains(sample_token, case=False, na=False)
            ].copy()
```

Output:

```text
resolved_borzoi_tracks.tsv
```

## 7. REF/ALT Model Inference

For each mutation and matched control, the runner builds one shared model
window per region. This keeps experimental and control bins aligned.

```python
def _shared_context_centers(mutations: pd.DataFrame) -> Dict[str, int]:
    """Use one model window per region so experimental and control bins align."""
    centers: Dict[str, int] = {}
    for _, group in mutations.groupby("region_id", sort=False):
        experimental = group[
            ~group["mutation_strategy"].isin(CONTROL_STRATEGIES)
        ]
        anchor_row = experimental.iloc[0] if not experimental.empty else group.iloc[0]
        context_center = (
            int(anchor_row["edit_start"]) + int(anchor_row["edit_end"])
        ) // 2
        for mutation_id in group["mutation_id"].astype(str):
            centers[mutation_id] = context_center
    return centers
```

The runner extracts the reference sequence, applies the manifest edit, and
predicts REF and ALT.

```python
alt_seq, edit_info = _apply_variant_edit(ref_seq, seq_start, var_row)
interval_records.append({
    "mutation_id": mutation_id,
    "chrom": chrom,
    "seq_start": seq_start,
    "seq_end": seq_end,
    "edit_center": edit_center,
    "context_center": context_center,
    **edit_info,
})

# One-hot encode
ref_dna = _one_hot_1mb(ref_seq).to(device_obj)
alt_dna = _one_hot_1mb(alt_seq).to(device_obj)
```

AlphaGenome predictions are run for the ChIP TF and ChIP histone heads at 128
bp resolution. Resolved track indices are then copied into compact arrays.

```python
for head_key in OUTPUT_KEYS:
    with torch.no_grad():
        ref_out = ag_model.predict(
            ref_dna, organism, heads=(head_key,), resolutions=(RESOLUTION,), channels_last=False,
        )
        alt_out = ag_model.predict(
            alt_dna, organism, heads=(head_key,), resolutions=(RESOLUTION,), channels_last=False,
        )
    ref_head_preds.append(ref_out[head_key][RESOLUTION][0].detach().cpu().numpy().astype(np.float32))
    alt_head_preds.append(alt_out[head_key][RESOLUTION][0].detach().cpu().numpy().astype(np.float32))
```

```python
for our_idx, (_, track_row) in enumerate(valid_tracks.iterrows()):
    head = str(track_row["output_type"])
    local_idx = int(track_row["track_index"])
    global_offset = head_offsets[head]
    global_idx = global_offset + local_idx
    if global_idx < ref_concat.shape[0]:
        ref_all[mut_idx, our_idx, :] = ref_concat[global_idx, :]
        alt_all[mut_idx, our_idx, :] = alt_concat[global_idx, :]
```

Output:

```text
run/scored_intervals.tsv
run/mreg_chromatin_profiles.parquet
```

## 8. Convert Model Bins To Biological Readouts

Readout regions are converted from genomic coordinates to model output bins.
The code rejects readouts that fall outside the model context.

```python
def _compute_readout_bins(
    readout_regions: pd.DataFrame,
    seq_start: int,
    n_bins: int,
) -> BinMapping:
    """Map genomic readout regions to model output bin indices.

    Each model output bin covers 128 bp. The i-th bin corresponds to genomic
    interval [seq_start + i*128, seq_start + (i+1)*128).
    """
    mapping: BinMapping = {}
    for _, row in readout_regions.iterrows():
        readout_id = str(row["readout_id"])
        if readout_id in ("mutation_center_1kb", "mutation_local_4kb"):
            continue
        r_start = int(row["start"])
        r_end = int(row["end"])
        rel_start = r_start - seq_start
        rel_end = r_end - seq_start
        if rel_start < 0 or rel_end > n_bins * RESOLUTION:
            raise ValueError(
                f"Readout {readout_id} ({r_start}-{r_end}) is not fully contained in "
                f"model context (seq_start={seq_start}, n_bins={n_bins})"
            )
        bin_start = rel_start // RESOLUTION
        bin_end = (rel_end + RESOLUTION - 1) // RESOLUTION
        mapping[readout_id] = (int(bin_start), int(bin_end))
    return mapping
```

For each mutation, readout, track and output bin, the profile table records
REF, ALT, raw delta and log2FC:

```python
rv = float(ref_vals[bin_offset])
av = float(alt_vals[bin_offset])
delta = av - rv
pc = pseudocount
log2fc = float(np.log2((av + pc) / (rv + pc))) if rv + pc > 0 else 0.0

records.append({
    "mutation_id": mutation_id,
    "control_type": control_type,
    "readout_id": readout_id,
    "track_id": str(track_row["track_id"]),
    "target_name": str(track_row["target_name"]),
    "biosample": str(track_row.get("biosample_resolved", "")),
    "genomic_start": int(bin_genomic_starts[bin_offset]),
    "genomic_end": int(bin_genomic_starts[bin_offset] + RESOLUTION),
    "offset_from_readout_anchor_bp": int(
        bin_genomic_starts[bin_offset] - readout_anchor
    ),
    "ref_value": rv,
    "alt_value": av,
    "delta": delta,
    "log2fc": log2fc,
})
```

Biological meaning:

- `ref_value` and `alt_value` are the raw predicted track signals.
- `delta = ALT - REF` is the direction and magnitude of the predicted change.
- `log2fc` gives a scale-stabilized fold-change style readout.
- Each value remains tied to a genomic bin, readout window and target track.

## 9. Matched-Control Adjustment And Report Tables

The distribution report loads raw profiles, raw metrics and control-adjusted
metrics:

```python
def _load(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = pd.read_parquet(run_dir / "run/mreg_chromatin_profiles.parquet")
    metrics = pd.read_csv(run_dir / "analysis/mreg_chromatin_metrics.tsv", sep="\t")
    adjusted = pd.read_csv(run_dir / "analysis/mreg_control_adjusted_metrics.tsv", sep="\t")
    mutations = pd.read_csv(run_dir / "prepared/mreg_mutation_manifest.tsv", sep="\t")
    return profiles, metrics, adjusted, mutations
```

The key local and distal biological comparisons are:

```python
local_cases = [
    ("TSS edit", "mreg_tss_00_ctcf_snv", "mreg_tss_4kb"),
    ("HC-DIC edit", "mreg_hc_dic_00_ctcf_snv", "hc_dic_4kb"),
    ("LC-DIC edit", "mreg_lc_dic_00_pol2_perturbation", "lc_dic_4kb"),
]
distal_cases = [
    ("HC-DIC edit -> TSS", "mreg_hc_dic_00_ctcf_snv", "mreg_tss_4kb"),
    ("LC-DIC edit -> TSS", "mreg_lc_dic_00_pol2_perturbation", "mreg_tss_4kb"),
]
```

The output table shows raw REF/ALT, target ALT-REF, control ALT-REF and
matched-control adjusted effects:

```python
{
    "comparison": label,
    "readout": READOUT_LABELS[readout_id],
    "track": track,
    "REF mean": _raw_mean(metrics, mutation_id, readout_id, track, "ref_mean"),
    "ALT mean": _raw_mean(metrics, mutation_id, readout_id, track, "alt_mean"),
    "target ALT-REF": _raw_mean(metrics, mutation_id, readout_id, track, "signed_delta_mean"),
    "control ALT-REF mean": _adj_value(adjusted, mutation_id, readout_id, track, "control_signed_delta_mean"),
    "adjusted ALT-REF": _adj_value(adjusted, mutation_id, readout_id, track, "adjusted_signed_delta_mean"),
    "adjusted mean log2FC": _adj_value(adjusted, mutation_id, readout_id, track, "adjusted_log2fc_mean"),
}
```

Outputs:

```text
analysis/mreg_chromatin_metrics.tsv
analysis/mreg_control_adjusted_metrics.tsv
local_distribution_summary.tsv
dic_to_tss_distribution_summary.tsv
```

## 10. Contact-Map Summary Branch

The contact-map branch asks whether an edit changes predicted local contacts
or edit-to-readout contacts. It uses the same mutation manifest and readout
regions, but predicts AlphaGenome `contact_maps`.

```python
def _predict_contacts(model, seq: str, device: torch.device) -> np.ndarray:
    dna = _one_hot_nlc(seq).to(device)
    organism = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.predict(dna, organism, heads=(CONTACT_HEAD,), channels_last=False)
    contacts = normalize_contact_maps(out[CONTACT_HEAD].detach().cpu().numpy())[0]
    del dna, organism, out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return contacts.astype(np.float32)
```

Contact metrics include local cross-boundary signal around the edit and
edit-anchor-to-readout signal such as DIC-to-TSS:

```python
for window_bp, min_distance_bp in [(20_000, 2_048), (100_000, 10_000), (500_000, 100_000)]:
    left, right = _local_cross_bins(edit_bin, n_bins, bin_bp, window_bp, min_distance_bp)
    mean, median, minv, maxv, n_pairs = _mean_pair_delta(delta_maps, left, right)
    records.append(
        {
            "mutation_id": mutation_id,
            "region_id": region_id,
            "control_type": control_type,
            "metric": "local_cross_boundary",
            "target_region": "edit_left_vs_right",
            "window_bp": window_bp,
            "min_distance_bp": min_distance_bp,
            "delta_contact_mean": mean,
```

```python
anchor = np.array([edit_bin], dtype=int)
for readout_id, bins in region_bin_map.items():
    if len(bins) == 0:
        continue
    mean, median, minv, maxv, n_pairs = _mean_pair_delta(delta_maps, anchor, bins)
    records.append(
        {
            "mutation_id": mutation_id,
            "region_id": region_id,
            "control_type": control_type,
            "metric": "edit_anchor_to_readout",
            "target_region": readout_id,
            "window_bp": np.nan,
            "min_distance_bp": bin_bp,
            "delta_contact_mean": mean,
```

The same matched-control adjustment is applied to contact means:

```python
control_mean = float(control.loc[mask, "delta_contact_mean"].mean()) if mask.any() else float("nan")
records.append(
    {
        "mutation_id": mutation_id,
        "region_id": row["region_id"],
        **keys,
        "target_delta_contact_mean": float(row["delta_contact_mean"]),
        "control_delta_contact_mean": control_mean,
        "adjusted_delta_contact_mean": float(row["delta_contact_mean"]) - control_mean,
        "n_controls": int(mask.sum()),
        "n_bin_pairs": int(row["n_bin_pairs"]),
    }
)
```

Output:

```text
contact_metric_raw.tsv
contact_metric_adjusted.tsv
```

## 11. Final Biological Outputs

The full experiment produces these reviewable outputs:

```text
prepared/mreg_region_manifest.tsv
  Biological loci and QC fields.

prepared/mreg_mutation_manifest.tsv
  REF/ALT edits, target feature, PWM scores and matched-control IDs.

prepared/mreg_readout_regions.tsv
  Local and distal readout windows.

run/resolved_tracks.tsv
  AlphaGenome tracks actually used after MCF-7 exact matching/fallback.

run/mreg_chromatin_profiles.parquet
  Per-bin REF, ALT, delta and log2FC for each mutation x readout x track.

analysis/mreg_chromatin_metrics.tsv
  Raw per-mutation/readout/track summaries.

analysis/mreg_control_adjusted_metrics.tsv
  Target effects after subtracting matched-control means.

resolved_borzoi_tracks.tsv
  Borzoi tasks actually used.

borzoi_mreg_profiles.parquet
  Borzoi per-bin REF/ALT profiles for selected tasks.

borzoi_mreg_effect_summary.tsv
  Borzoi mean delta/log2FC summaries.

contact_metric_adjusted.tsv
  Control-adjusted AlphaGenome contact-map summary metrics.
```

## 12. Actual MREG Edits Used

Coordinates are hg38, 0-based, half-open intervals.

| mutation_id | interval | ref motif/window | alt motif/window | changed bases | target feature | PWM ref | PWM alt | delta |
|---|---:|---|---|---|---|---:|---:|---:|
| `mreg_tss_00_ctcf_snv` | `chr2:216013628-216013640` | `GCCCTCTAAGGG` | `GCGCTATAAGCG` | `216013630 C>G`; `216013633 C>A`; `216013638 G>C` | CTCF motif | `8.851` | `-13.722` | `-22.574` |
| `mreg_hc_dic_00_ctcf_snv` | `chr2:215950149-215950161` | `CCACTAGGTGGC` | `CGACTATGTCGC` | `215950150 C>G`; `215950155 G>T`; `215950158 G>C` | CTCF motif | `17.688` | `-4.885` | `-22.574` |
| `mreg_lc_dic_00_pol2_perturbation` | `chr2:215979779-215979782` | `GAT` | `AGC` | `215979779 G>A`; `215979780 A>G`; `215979781 T>C` | Pol2 peak center | NA | NA | NA |

In short, the experiment is:

```text
curated MREG TSS/HC-DIC/LC-DIC regions
  -> region QC and MREG anchors
  -> CTCF motif disruption or Pol2 peak-center perturbation
  -> local matched controls
  -> MCF-7-focused track/task resolution
  -> REF/ALT AlphaGenome and Borzoi prediction
  -> local and DIC-to-TSS readout extraction
  -> raw delta/log2FC plus matched-control adjusted effects
  -> optional contact-map summary metrics
```
