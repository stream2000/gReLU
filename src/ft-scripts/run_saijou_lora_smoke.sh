#!/usr/bin/env bash
# Run the approved one-epoch Saijou LoRA integration smoke test.
set -eo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
data_root="${SAIJOU_DATA_ROOT:-$project_root/data}"
runtime_env="${GRELU_CONDA_ENV:-$project_root/runtime/grelu_dev}"
timeout_seconds="${SMOKE_TIMEOUT_SECONDS:-180}"

cd "$repo_root"
export GRELU_CONDA_ENV="$runtime_env"
source activate.sh

for required in \
  "$data_root/bigwig/hsc.CPM.mapq10.bw" \
  "$data_root/mm10/genome.fa" \
  "$data_root/mm10/genometable.txt" \
  "$data_root/pretrained/borzoi/human_state_dict_rep0.h5"; do
  [[ -r "$required" ]] || { echo "Missing readable dependency: $required" >&2; exit 2; }
done

profile_dir="$data_root/profiles/example_smoke_v1"
if [[ ! -f "$profile_dir/manifest.json" ]]; then
  python src/ft-scripts/make_borzoi_smoke_profile.py \
    --out_dir "$profile_dir" \
    --chrom_sizes "$data_root/mm10/genometable.txt"
fi

if [[ -z "${GPU:-}" ]]; then
  GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr | head -n1 | cut -d, -f1 | tr -d ' ')"
fi
[[ -n "$GPU" ]] || { echo "No GPU selected. Set GPU to a free GPU index." >&2; exit 2; }

echo "Using physical GPU: $GPU"
echo "Smoke-test limit: ${timeout_seconds}s"

timeout --signal=INT --kill-after=20s "${timeout_seconds}s" \
  env CUDA_VISIBLE_DEVICES="$GPU" \
  python src/ft-scripts/train_borzoi.py \
    --split_name example_smoke_v1 \
    --finetune_mode lora \
    --target_mode log1p_mse \
    --bigwig_dir "$data_root/bigwig" \
    --genome "$data_root/mm10/genome.fa" \
    --pretrained_weights "$data_root/pretrained/borzoi/human_state_dict_rep0.h5" \
    --split_dir "$data_root/profiles" \
    --cache_dir "$data_root/caches" \
    --run_root "$data_root/runs" \
    --devices 0 \
    --num_workers 0 \
    --batch_size 1 \
    --accumulate_grad_batches 1 \
    --max_epochs 1 \
    --disable_early_stopping
