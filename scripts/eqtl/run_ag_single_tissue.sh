#!/usr/bin/env bash
# Minimal AlphaGenome single-tissue eQTL AUROC.
# Usage:
#   bash scripts/eqtl/run_ag_single_tissue.sh <tissue> [devices]
#   bash scripts/eqtl/run_ag_single_tissue.sh brain_cortex 0
set -euo pipefail
cd "$(dirname "$0")/../.."

TISSUE="${1:?Usage: $0 <tissue> [devices]}"
DEVICES="${2:-0}"

source /work/miniconda3/etc/profile.d/conda.sh
source activate.sh
export PYTHONPATH="$(pwd):$(pwd)/src"

python -u scripts/eqtl/run_eqtl_auroc.py \
    --model alphagenome \
    --tissue "$TISSUE" \
    --score sum --rc --rf \
    --devices "$DEVICES" \
    --num_workers 4 \
    --output "results/eqtl/ag_${TISSUE}.tsv"
