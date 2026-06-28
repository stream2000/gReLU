#!/usr/bin/env python
"""Fine-tune Borzoi on Saijou pseudobulk bigWig tracks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from torch import nn
import torch
from torch.utils.data import DataLoader

import grelu.lightning
from borzoi_mmap_dataset import build_dataset_pair


DEFAULT_BIGWIG_DIR = "/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig"
DEFAULT_GENOME = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
TASK_NAMES = ["hsc", "mac", "lsec", "chol"]
BORZOI_LORA_TARGET_MODULES = (
    r"(?:conv_tower\.blocks\.\d+\.conv"
    r"|unet_tower\.blocks\.\d+\.conv\.conv"
    r"|unet_tower\.blocks\.\d+\.channel_transform\.conv\.layer"
    r"|pointwise_conv\.conv"
    r"|transformer_tower\.blocks\.\d+\.mha\.to_[qv]"
    r"|transformer_tower\.blocks\.\d+\.ffn\.dense[12]\.linear)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_name", default="split_chr10_chr11")
    parser.add_argument(
        "--finetune_mode",
        default="lora",
        choices=["headonly", "lora"],
        help="headonly freezes the Borzoi embedding; lora trains PEFT LoRA adapters plus the new head.",
    )
    parser.add_argument(
        "--target_mode",
        default="log1p_mse",
        choices=["poisson", "poisson_multinomial", "log1p_mse", "borzoi_squash_mse"],
    )
    parser.add_argument("--bigwig_dir", default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--genome", default=DEFAULT_GENOME)
    parser.add_argument("--split_dir", default="dataset_store/splits")
    parser.add_argument("--cache_dir", default="dataset_store/mmap_caches")
    parser.add_argument("--seq_len", type=int, default=524_288)
    parser.add_argument("--label_len", type=int, default=196_608)
    parser.add_argument("--bin_size", type=int, default=32)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default=BORZOI_LORA_TARGET_MODULES)
    parser.add_argument(
        "--total_weight",
        type=float,
        default=0.2,
        help="Poisson total-count weight for poisson_multinomial loss.",
    )
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument(
        "--disable_early_stopping",
        action="store_true",
        help="Run exactly to max_epochs unless interrupted.",
    )
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def parse_devices(value: str):
    if value == "cpu":
        return "cpu"
    devices = [int(item) for item in value.split(",")]
    return devices[0] if len(devices) == 1 else devices


def freeze_embedding(model: grelu.lightning.LightningModel) -> None:
    for param in model.model.embedding.parameters():
        param.requires_grad = False


def count_lora_targets(embedding: nn.Module, target_modules: str) -> int:
    pattern = re.compile(target_modules)
    return sum(
        1
        for name, module in embedding.named_modules()
        if isinstance(module, (nn.Conv1d, nn.Linear)) and pattern.fullmatch(name)
    )


def apply_lora_to_borzoi_embedding(
    model: grelu.lightning.LightningModel,
    target_modules: str,
    rank: int,
    alpha: int,
    dropout: float,
) -> int:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA mode requires the Scooby-compatible PEFT fork. Install it with: "
            "python -m pip install 'peft @ git+https://github.com/lauradmartens/peft.git@f00283c5ef35bead9bdc68113797c4619a2cc06d'"
        ) from exc

    n_targets = count_lora_targets(model.model.embedding, target_modules)
    if n_targets == 0:
        raise ValueError(f"No Borzoi LoRA target modules matched: {target_modules}")

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model.model.embedding = get_peft_model(model.model.embedding, config)
    return n_targets


def main() -> None:
    args = parse_args()
    bigwig_dir = Path(args.bigwig_dir)
    bw_files = [
        str(bigwig_dir / "hsc.CPM.mapq10.bw"),
        str(bigwig_dir / "mac.CPM.mapq10.bw"),
        str(bigwig_dir / "lsec.CPM.mapq10.bw"),
        str(bigwig_dir / "chol.CPM.mapq10.bw"),
    ]

    train_dataset, val_dataset = build_dataset_pair(
        split_name=args.split_name,
        target_mode=args.target_mode,
        bw_files=bw_files,
        task_names=TASK_NAMES,
        genome=args.genome,
        split_dir=Path(args.split_dir),
        cache_dir=Path(args.cache_dir),
        seq_len=args.seq_len,
        label_len=args.label_len,
        bin_size=args.bin_size,
        force_rebuild=args.force_rebuild,
    )

    model_params = {
        "model_type": "BorzoiPretrainedModel",
        "n_tasks": len(TASK_NAMES),
        "fold": 0,
        "n_transformers": 8,
        "crop_len": 5120,
        "final_pool_func": None,
    }

    if args.target_mode == "poisson":
        loss = "poisson"
        extra_train_params = {}
    elif args.target_mode == "poisson_multinomial":
        loss = "poisson_multinomial"
        extra_train_params = {"total_weight": args.total_weight}
    elif args.target_mode in {"log1p_mse", "borzoi_squash_mse"}:
        loss = "mse"
        extra_train_params = {}
    else:
        raise ValueError(args.target_mode)

    default_lr = 3e-6 if args.finetune_mode == "headonly" else 1e-5
    lr = args.lr if args.lr is not None else default_lr
    run_name = f"borzoi_{args.split_name}_{args.target_mode}_{args.finetune_mode}"
    ckpt_dir = f"runs/{run_name}/checkpoints"
    train_params = {
        "task": "regression",
        "loss": loss,
        "lr": lr,
        "optimizer": "adam",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "devices": parse_devices(args.devices),
        "logger": "csv",
        "name": run_name,
        "save_dir": "runs",
        "max_epochs": args.max_epochs,
        "checkpoint": {
            "dirpath": ckpt_dir,
            "filename": "epoch{epoch:02d}",
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 3,
            "save_last": True,
            "every_n_epochs": 1,
        },
        "clip": args.clip,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "early_stopping": not args.disable_early_stopping,
        "freeze_embedding_eval": args.finetune_mode == "headonly",
        "freeze_embedding_norm_stats": args.finetune_mode in {"headonly", "lora"},
        "patience": 3,
        "monitor": "val_loss",
        "mode": "min",
    }
    train_params.update(extra_train_params)

    model = grelu.lightning.LightningModel(
        model_params=model_params,
        train_params=train_params,
    )

    if args.finetune_mode == "headonly":
        freeze_embedding(model)
        lora_targets = 0
    else:
        lora_targets = apply_lora_to_borzoi_embedding(
            model,
            target_modules=args.lora_target_modules,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("finetune_mode:", args.finetune_mode)
    print("lr:", lr)
    print("early_stopping:", not args.disable_early_stopping)
    print("freeze_embedding_eval:", args.finetune_mode == "headonly")
    print("freeze_embedding_norm_stats:", args.finetune_mode in {"headonly", "lora"})
    if args.target_mode == "poisson_multinomial":
        print("total_weight:", args.total_weight)
    if args.finetune_mode == "lora":
        print("lora_targets:", lora_targets)
        print("lora_rank:", args.lora_rank)
        print("lora_alpha:", args.lora_alpha)
        print("lora_dropout:", args.lora_dropout)
    print(f"trainable_params: {trainable_params:,} / {total_params:,}")

    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    with torch.no_grad():
        yhat = model.forward(x, logits=True)

    print("x:", x.shape)
    print("y:", y.shape)
    print("yhat:", yhat.shape)
    assert yhat.shape == y.shape, (yhat.shape, y.shape)
    assert torch.isfinite(y).all()
    train_dataset.close_handles()
    val_dataset.close_handles()
    if args.dry_run:
        print("dry_run: completed setup and shape check; skipping training")
        return

    trainer = model.train_on_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_path=args.checkpoint_path,
    )
    print("best checkpoint:", trainer.checkpoint_callback.best_model_path)
    print("last checkpoint:", trainer.checkpoint_callback.last_model_path)


if __name__ == "__main__":
    main()
