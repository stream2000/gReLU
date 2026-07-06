#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/../.."

source activate.sh

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="experiments/logs"
mkdir -p "$log_dir"

split_name="${SPLIT_NAME:-split_chr10_chr11}"
target_mode="${TARGET_MODE:-poisson_multinomial}"
total_weight="${TOTAL_WEIGHT:-0.2}"
gpus="${GPUS:-1,3}"
devices="${DEVICES:-0,1}"
num_workers="${NUM_WORKERS:-2}"
batch_size="${BATCH_SIZE:-1}"
accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES:-8}"
lora_epochs="${LORA_EPOCHS:-20}"
headonly_epochs="${HEADONLY_EPOCHS:-20}"
lora_lr="${LORA_LR:-1e-5}"
headonly_lr="${HEADONLY_LR:-3e-6}"

lora_log="${log_dir}/borzoi_${split_name}_${target_mode}_lora_2gpu_local_serial_${timestamp}.log"
headonly_log="${log_dir}/borzoi_${split_name}_${target_mode}_headonly_2gpu_local_serial_${timestamp}.log"

echo "Local serial retrain started: $(date)"
echo "host=$(hostname)"
echo "cwd=$(pwd)"
echo "conda_prefix=${CONDA_PREFIX}"
echo "python=$(command -v python)"
echo "physical_gpus=${gpus}"
echo "lightning_devices=${devices}"
echo "split_name=${split_name}"
echo "target_mode=${target_mode}"
echo "total_weight=${total_weight}"
echo "lora_log=${lora_log}"
echo "headonly_log=${headonly_log}"

echo "Starting LoRA retrain: $(date)"
CUDA_VISIBLE_DEVICES="${gpus}" python src/ft-scripts/train_borzoi.py \
  --split_name "${split_name}" \
  --target_mode "${target_mode}" \
  --finetune_mode lora \
  --devices "${devices}" \
  --num_workers "${num_workers}" \
  --batch_size "${batch_size}" \
  --accumulate_grad_batches "${accumulate_grad_batches}" \
  --max_epochs "${lora_epochs}" \
  --lr "${lora_lr}" \
  --total_weight "${total_weight}" \
  2>&1 | tee "${lora_log}"

echo "LoRA retrain finished: $(date)"
echo "Starting head-only retrain: $(date)"
CUDA_VISIBLE_DEVICES="${gpus}" python src/ft-scripts/train_borzoi.py \
  --split_name "${split_name}" \
  --target_mode "${target_mode}" \
  --finetune_mode headonly \
  --devices "${devices}" \
  --num_workers "${num_workers}" \
  --batch_size "${batch_size}" \
  --accumulate_grad_batches "${accumulate_grad_batches}" \
  --max_epochs "${headonly_epochs}" \
  --lr "${headonly_lr}" \
  --total_weight "${total_weight}" \
  2>&1 | tee "${headonly_log}"

echo "Head-only retrain finished: $(date)"
echo "Local serial retrain completed: $(date)"
