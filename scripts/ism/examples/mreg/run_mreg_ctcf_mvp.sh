#!/usr/bin/env bash
# MVP pipeline for the second CTCF step:
#   MREG-region motif SNV -> AlphaGenome 1D ref/alt -> center-1kb per-track effects.

set -euo pipefail

FASTA="${FASTA:-/home/fqijun/.local/share/genomes/hg38/hg38.fa}"
GTF="${GTF:-/home/fqijun/.local/share/genomes/hg38/hg38.annotation.gtf}"
OUT_DIR="${OUT_DIR:-agent-doc/ism_context/mreg_ctcf_mvp}"
DEVICES="${DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PRECISION="${PRECISION:-bf16-mixed}"
TRACK_WINDOW_BP="${TRACK_WINDOW_BP:-500}"
CENTER_BP="${CENTER_BP:-1000}"
TOP_N="${TOP_N:-50}"
N_SITES="${N_SITES:-1}"
WEIGHTS_PATH="${WEIGHTS_PATH:-/home/fqijun/.cache/alphagenome/model_fold_0.safetensors}"

mkdir -p "${OUT_DIR}"

python scripts/ism/examples/mreg/make_mreg_ctcf_example.py \
  --fasta "${FASTA}" \
  --gtf "${GTF}" \
  --n_sites "${N_SITES}" \
  --output "${OUT_DIR}/input_sites.tsv"

run_args=(
  scripts/ism/run_tf_context.py
  --sites "${OUT_DIR}/input_sites.tsv"
  --fasta "${FASTA}"
  --devices "${DEVICES}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --precision "${PRECISION}"
  --feature_mode track_window
  --track_window_bp "${TRACK_WINDOW_BP}"
  --track_predict_chunk_size "${N_SITES}"
  --output_dir "${OUT_DIR}"
)

if [[ -f "${WEIGHTS_PATH}" ]]; then
  run_args+=(--weights_path "${WEIGHTS_PATH}")
else
  echo "WARNING: WEIGHTS_PATH does not exist (${WEIGHTS_PATH}); running without an explicit checkpoint." >&2
fi

python "${run_args[@]}"

python scripts/ism/summarize_center_track_effects.py \
  --run_dir "${OUT_DIR}" \
  --center_bp "${CENTER_BP}" \
  --top_n "${TOP_N}" \
  --output_prefix center_1kb

cat <<EOF

MREG CTCF MVP completed.
Input sites: ${OUT_DIR}/input_sites.tsv
Per-track center-1kb effects: ${OUT_DIR}/center_1kb_track_effects.tsv
Top tracks per site: ${OUT_DIR}/center_1kb_top${TOP_N}_tracks_by_site.tsv
EOF
