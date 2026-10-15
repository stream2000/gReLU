#!/usr/bin/env bash
# eQTL AUROC: run Borzoi + AlphaGenome tissue-by-tissue for real-time comparison.
# Uses 3 GPUs. Survives terminal close (nohup).
#
# Approach: subprocess-per-tissue (calls run_eqtl_auroc.py for each tissue).
# This avoids the DDP-worker-respawn problem: Lightning DDP popen re-runs the
# entire script in each worker. By keeping each tissue in its own subprocess,
# workers exit cleanly after each tissue and never accumulate.
#
# Usage:
#   nohup bash scripts/eqtl/run_all_auto.sh &> results/eqtl/master.log &
#   disown
set -euo pipefail
cd "$(dirname "$0")/../.."

set +u; source /work/miniconda3/etc/profile.d/conda.sh; source activate.sh; set -u
export PYTHONPATH="$(pwd):$(pwd)/src"
export HF_HUB_OFFLINE=1

MANIFEST="$HOME/.cache/eqtl_finemapping/paper_vcfs/manifest.tsv"
OUTDIR="results/eqtl"
DEVICES="0,1,2"
SCRIPT="scripts/eqtl/run_eqtl_auroc.py"

mkdir -p "$OUTDIR/per_tissue_borzoi" "$OUTDIR/per_tissue_ag" \
         "$OUTDIR/features_borzoi" "$OUTDIR/features_ag"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=========================================="
echo "$(timestamp) START tissue-by-tissue pipeline (3 GPUs, subprocess-per-tissue)"
echo "=========================================="

# Read tissue list from manifest (skip header)
mapfile -t TISSUES < <(tail -n +2 "$MANIFEST" | cut -f1)
N=${#TISSUES[@]}
echo "Tissues: $N"
echo "GPUs: $DEVICES"
echo ""

borzoi_tsv="$OUTDIR/borzoi_sad_rf_all.tsv"
ag_tsv="$OUTDIR/ag_sad_rf_all.tsv"

# Accumulate per-tissue TSVs into master files
aggregate_tsv() {
    local model="$1"   # borzoi or ag
    local outfile="$2"
    local perdir="$OUTDIR/per_tissue_${model}"
    python - <<PYEOF
import pandas as pd, glob, sys
files = sorted(glob.glob("${perdir}/*.tsv"))
if not files:
    sys.exit(0)
df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
df.to_csv("${outfile}", sep="\t", index=False)
PYEOF
}

compare_tissue() {
    local tissue="$1"
    local b_tsv="$OUTDIR/per_tissue_borzoi/${tissue}.tsv"
    local ag_tsv_t="$OUTDIR/per_tissue_ag/${tissue}.tsv"
    python - <<PYEOF
import pandas as pd, os
b_file, ag_file = "${b_tsv}", "${ag_tsv_t}"
b = pd.read_csv(b_file, sep="\t").iloc[0] if os.path.exists(b_file) else None
ag = pd.read_csv(ag_file, sep="\t").iloc[0] if os.path.exists(ag_file) else None
b_str  = f"{b['auroc']:.4f}"  if b  is not None else "  skip  "
ag_str = f"{ag['auroc']:.4f}" if ag is not None else "  skip  "
delta  = f"  Δ={ag['auroc']-b['auroc']:+.4f}" if (b is not None and ag is not None) else ""
print(f"  [${tissue}]  Borzoi={b_str}  AG={ag_str}{delta}  |  $(timestamp)")
PYEOF
}

for i in "${!TISSUES[@]}"; do
    tissue="${TISSUES[$i]}"
    idx=$((i + 1))
    b_out="$OUTDIR/per_tissue_borzoi/${tissue}.tsv"
    ag_out="$OUTDIR/per_tissue_ag/${tissue}.tsv"

    echo "[$idx/$N] $tissue ..."

    # ── Borzoi ──────────────────────────────────────────────────────────────
    if [[ -f "$b_out" ]]; then
        echo "  [$tissue] Borzoi: cached, skipping"
    else
        python -u "$SCRIPT" \
            --model borzoi \
            --tissue "$tissue" \
            --score sum --rc --rf \
            --devices "$DEVICES" \
            --num_workers 4 \
            --output "$b_out" \
            --save_features "$OUTDIR/features_borzoi" \
            || echo "  [$tissue] Borzoi ERROR (exit $?)"
    fi

    # ── AlphaGenome ─────────────────────────────────────────────────────────
    if [[ -f "$ag_out" ]]; then
        echo "  [$tissue] AlphaGenome: cached, skipping"
    else
        python -u "$SCRIPT" \
            --model alphagenome \
            --tissue "$tissue" \
            --score sum --rc --rf \
            --devices "$DEVICES" \
            --num_workers 4 \
            --output "$ag_out" \
            --save_features "$OUTDIR/features_ag" \
            || echo "  [$tissue] AlphaGenome ERROR (exit $?)"
    fi

    compare_tissue "$tissue"
    aggregate_tsv "borzoi" "$borzoi_tsv"
    aggregate_tsv "ag"     "$ag_tsv"
done

# ── Final summary ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "$(timestamp) ALL DONE"
echo "=========================================="

python - <<PYEOF
import pandas as pd, os

b_file = "${borzoi_tsv}"
ag_file = "${ag_tsv}"

for label, path in [("Borzoi", b_file), ("AlphaGenome", ag_file)]:
    if not os.path.exists(path):
        print(f"{label}: no results")
        continue
    df = pd.read_csv(path, sep="\t")
    anom = df[df["auroc"] >= 0.95]
    print(f"\n{label} ({len(df)} tissues):")
    print(f"  mean={df['auroc'].mean():.4f}  std={df['auroc'].std():.4f}  "
          f"range=[{df['auroc'].min():.4f}, {df['auroc'].max():.4f}]")
    if len(anom):
        print(f"  ⚠  {len(anom)} ANOMALIES (AUROC>=0.95): {list(anom['tissue'])}")

if os.path.exists(b_file) and os.path.exists(ag_file):
    b = pd.read_csv(b_file, sep="\t")
    ag = pd.read_csv(ag_file, sep="\t")
    common = set(b["tissue"]) & set(ag["tissue"])
    if common:
        bm = b[b["tissue"].isin(common)]["auroc"].mean()
        am = ag[ag["tissue"].isin(common)]["auroc"].mean()
        print(f"\n  Δ(AG - Borzoi) = {am - bm:+.4f}  ({len(common)} tissues)")

print("\nPaper refs:  Borzoi single+SAD+RF+RC=0.7880  ensemble=0.7943  Enformer=0.7469")
PYEOF

echo "$(timestamp) Script exit."
