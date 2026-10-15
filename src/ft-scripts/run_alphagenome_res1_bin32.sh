#!/usr/bin/env bash
# AlphaGenome LoRA finetune at 1-bp embeddings with 32-bp supervised bins.
#
# Matches the validated res=128/bin=128 run
# (alphagenome_split_chr10_chr11_seq524288_label196608_bin128_res128_...)
# on split, genomic window, loss, LoRA preset, head and effective batch size.
# Only the embedding resolution and the target bin width change, so the two
# runs are directly comparable.
#
# Reused label cache: dataset_store/mmap_caches/split_chr10_chr11__poisson_multinomial__bin32
# (already built for seq_len=524288, label_len=196608, bin_size=32).
#
# Memory: peak 36.9 GiB/GPU measured with fwd+bwd+step at batch_size=1.
# batch_size=2 OOMs on a 48 GB card, so accumulate_grad_batches doubles from
# 5 to 10 to keep the effective batch at 1 x 10 x 3 = 30, same as the reference.

set -eo pipefail

cd "$(dirname "$0")/../.."
source activate.sh
set -u

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="experiments/logs"
mkdir -p "${log_dir}"

split_name="${SPLIT_NAME:-split_chr10_chr11}"
gpus="${GPUS:-1,2,3}"
devices="${DEVICES:-0,1,2}"
num_workers="${NUM_WORKERS:-6}"
batch_size="${BATCH_SIZE:-1}"
accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES:-10}"
target_mode="${TARGET_MODE:-poisson_multinomial}"
total_weight="${TOTAL_WEIGHT:-0.2}"
max_epochs="${MAX_EPOCHS:-20}"
lr="${LR:-1e-5}"
seq_len="${SEQ_LEN:-524288}"
label_len="${LABEL_LEN:-196608}"
bin_size="${BIN_SIZE:-32}"
resolution="${RESOLUTION:-1}"
checkpoint_path="${CHECKPOINT_PATH:-}"

log="${log_dir}/ag_lora_res1_bin32_3gpu_${timestamp}.log"

echo "AlphaGenome res=1 bin=32 LoRA finetune started: $(date)"
echo "physical_gpus=${gpus}  lightning_devices=${devices}"
echo "seq_len=${seq_len} label_len=${label_len} bin_size=${bin_size} resolution=${resolution}"
echo "batch_size=${batch_size} accumulate_grad_batches=${accumulate_grad_batches}"
echo "max_epochs=${max_epochs} lr=${lr}"
echo "log=${log}"

extra_args=()
if [[ -n "${checkpoint_path}" ]]; then
  extra_args+=(--checkpoint_path "${checkpoint_path}")
fi

CUDA_VISIBLE_DEVICES="${gpus}" python src/ft-scripts/train_alphagenome.py \
  --split_name "${split_name}" \
  --target_mode "${target_mode}" \
  --finetune_mode lora \
  --lora_preset active \
  --seq_len "${seq_len}" \
  --label_len "${label_len}" \
  --bin_size "${bin_size}" \
  --resolution "${resolution}" \
  --head_hidden_channels 512 \
  --head_hidden_layers 1 \
  --gradient_checkpointing \
  --devices "${devices}" \
  --num_workers "${num_workers}" \
  --batch_size "${batch_size}" \
  --accumulate_grad_batches "${accumulate_grad_batches}" \
  --max_epochs "${max_epochs}" \
  --lr "${lr}" \
  --total_weight "${total_weight}" \
  --disable_early_stopping \
  --skip_shape_check \
  "${extra_args[@]}" \
  2>&1 | tee "${log}"

echo "AlphaGenome res=1 bin=32 LoRA finetune finished: $(date)"
