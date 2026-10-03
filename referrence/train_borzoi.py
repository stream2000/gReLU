# train_borzoi.py

import argparse
import os
import torch
from torch.utils.data import DataLoader

import grelu.lightning
from dataset_cache import build_dataset_pair


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_name", default="split_chr10_chr11")
    parser.add_argument(
        "--target_mode",
        default="log1p_mse",
        choices=["poisson", "poisson_multinomial", "log1p_mse", "borzoi_squash_mse"],
    )
    parser.add_argument("--devices", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--force_rebuild", action="store_true")
    return parser.parse_args()


args = parse_args()
split_name = args.split_name
target_mode = args.target_mode

bw_files = [
    "bigwig/hsc.CPM.mapq10.bw",
    "bigwig/mac.CPM.mapq10.bw",
    "bigwig/lsec.CPM.mapq10.bw",
    "bigwig/chol.CPM.mapq10.bw"
]

task_names = ["hsc", "mac", "lsec", "chol"]

train_dataset, val_dataset = build_dataset_pair(
    split_name=split_name,
    target_mode=target_mode,
    bw_files=bw_files,
    task_names=task_names,
    genome="mm10",
    force_rebuild=args.force_rebuild,
)

model_params = {
    "model_type": "BorzoiPretrainedModel",
    "n_tasks": len(bw_files),
    "fold": 0,
    "n_transformers": 8,
    "crop_len": 5120,
    "final_pool_func": None,
}

if target_mode == "poisson":
    loss = "poisson"
    extra_train_params = {}

elif target_mode == "poisson_multinomial":
    loss = "poisson_multinomial"
    extra_train_params = {"total_weight": 1.0}

elif target_mode in {"log1p_mse", "borzoi_squash_mse"}:
    loss = "mse"
    extra_train_params = {}

else:
    raise ValueError(target_mode)

run_name = f"borzoi_{split_name}_{target_mode}_headonly"
ckpt_dir = f"runs/{run_name}/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

train_params = {
    "task": "regression",
    "loss": loss,
    "lr": 3e-6,
    "optimizer": "adam",
    "batch_size": 1,
    "num_workers": args.num_workers,
    "devices": args.devices,
    "logger": "csv",
    "name": run_name,
    "save_dir": "runs",
    "max_epochs": args.max_epochs,
    "checkpoint": {
        "dirpath": ckpt_dir,
#        "filename": "epoch{epoch:02d}-valloss{val_loss:.4f}",
        "filename": "epoch{epoch:02d}",
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 3,
        "save_last": True,
        "every_n_epochs": 1,
    },
    "clip": 1.0,
    "accumulate_grad_batches": 8,
    "early_stopping": True,
    "patience": 3,
    "monitor": "val_loss",
    "mode": "min",
}

train_params.update(extra_train_params)

model = grelu.lightning.LightningModel(
    model_params=model_params,
    train_params=train_params,
)

# head-only fine-tune
for p in model.model.embedding.parameters():
    p.requires_grad = False

# shape check
loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
x, y = next(iter(loader))

with torch.no_grad():
    yhat = model.forward(x, logits=True)

print("x:", x.shape)
print("y:", y.shape)
print("yhat:", yhat.shape)

assert yhat.shape == y.shape, (yhat.shape, y.shape)
assert torch.isfinite(y).all()

trainer = model.train_on_dataset(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)

print("best checkpoint:", trainer.checkpoint_callback.best_model_path)
print("last checkpoint:", trainer.checkpoint_callback.last_model_path)
