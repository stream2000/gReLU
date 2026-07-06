#!/usr/bin/env python
"""Fine-tune AlphaGenome embeddings on Saijou pseudobulk bigWig tracks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import grelu.lightning
from alphagenome_pytorch.config import DtypePolicy
from borzoi_mmap_dataset import build_dataset_pair


DEFAULT_BIGWIG_DIR = "/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig"
DEFAULT_GENOME = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
DEFAULT_WEIGHTS = (
    "~/.cache/huggingface/hub/models--gtca--alphagenome_pytorch/"
    "snapshots/b01c0ffa73e07c053491f3b5ea8bcf67d93b9920/model_fold_0.safetensors"
)
TASK_NAMES = ["hsc", "mac", "lsec", "chol"]
ALPHAGENOME_LORA_TARGETS = ["mha", "mlp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_name", default="split_chr10_chr11_tiny50")
    parser.add_argument(
        "--finetune_mode",
        default="headonly",
        choices=["headonly", "lora"],
        help="headonly freezes AlphaGenome and trains a new 4-track head; lora trains AlphaGenome LoRA adapters plus the head.",
    )
    parser.add_argument(
        "--target_mode",
        default="poisson_multinomial",
        choices=["poisson", "poisson_multinomial", "log1p_mse", "borzoi_squash_mse"],
    )
    parser.add_argument("--bigwig_dir", default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--genome", default=DEFAULT_GENOME)
    parser.add_argument("--weights_path", default=DEFAULT_WEIGHTS)
    parser.add_argument("--split_dir", default="dataset_store/splits")
    parser.add_argument("--cache_dir", default="dataset_store/mmap_caches")
    parser.add_argument("--seq_len", type=int, default=131_072)
    parser.add_argument("--label_len", type=int, default=131_072)
    parser.add_argument("--bin_size", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=128, choices=[1, 128])
    parser.add_argument("--organism_index", type=int, default=1, help="AlphaGenome organism index; 1 is mouse in the local wrapper.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_targets", default=",".join(ALPHAGENOME_LORA_TARGETS))
    parser.add_argument("--total_weight", type=float, default=0.2)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--disable_early_stopping", action="store_true")
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


def apply_lora_to_alphagenome_embedding(
    model: grelu.lightning.LightningModel,
    targets: list[str],
    rank: int,
    alpha: int,
) -> int:
    from alphagenome_pytorch.extensions.finetuning.adapters import apply_lora

    freeze_embedding(model)
    before = sum(1 for module in model.model.embedding.model.modules() if module.__class__.__name__ == "LoRA")
    apply_lora(model.model.embedding.model, target_modules=targets, rank=rank, alpha=alpha)
    after = sum(1 for module in model.model.embedding.model.modules() if module.__class__.__name__ == "LoRA")
    n_targets = after - before
    if n_targets == 0:
        raise ValueError(f"No AlphaGenome LoRA targets matched: {targets}")
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

    default_lr = 3e-4 if args.finetune_mode == "headonly" else 1e-5
    lr = args.lr if args.lr is not None else default_lr
    run_name = f"alphagenome_{args.split_name}_seq{args.seq_len}_bin{args.bin_size}_{args.target_mode}_{args.finetune_mode}"
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
        "freeze_embedding_eval": True,
        "freeze_embedding_norm_stats": True,
        "patience": 3,
        "monitor": "val_loss",
        "mode": "min",
    }
    train_params.update(extra_train_params)

    model_params = {
        "model_type": "AlphaGenomeFinetuneModel",
        "n_tasks": len(TASK_NAMES),
        "resolution": args.resolution,
        "weights_path": str(Path(args.weights_path).expanduser()),
        "dtype_policy": DtypePolicy.full_float32(),
        "organism_index": args.organism_index,
        "num_organisms": 2,
        "gradient_checkpointing": args.gradient_checkpointing,
        "final_pool_func": None,
    }
    model = grelu.lightning.LightningModel(
        model_params=model_params,
        train_params=train_params,
    )

    if args.finetune_mode == "headonly":
        freeze_embedding(model)
        lora_targets = 0
    else:
        lora_targets = apply_lora_to_alphagenome_embedding(
            model,
            targets=[item for item in args.lora_targets.split(",") if item],
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("model: alphagenome")
    print("finetune_mode:", args.finetune_mode)
    print("seq_len:", args.seq_len)
    print("label_len:", args.label_len)
    print("bin_size:", args.bin_size)
    print("resolution:", args.resolution)
    print("organism_index:", args.organism_index)
    print("gradient_checkpointing:", args.gradient_checkpointing)
    print("lr:", lr)
    print("early_stopping:", not args.disable_early_stopping)
    print("freeze_embedding_eval:", True)
    print("freeze_embedding_norm_stats:", True)
    if args.target_mode == "poisson_multinomial":
        print("total_weight:", args.total_weight)
    if args.finetune_mode == "lora":
        print("lora_targets:", lora_targets)
        print("lora_rank:", args.lora_rank)
        print("lora_alpha:", args.lora_alpha)
    print(f"trainable_params: {trainable_params:,} / {total_params:,}")

    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    shape_device = None
    if args.devices != "cpu" and torch.cuda.is_available():
        shape_device = torch.device("cuda:0")
        model.to(shape_device)
        x = x.to(shape_device)
        y = y.to(shape_device)
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
    if shape_device is not None:
        model.to("cpu")
        torch.cuda.empty_cache()

    trainer = model.train_on_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_path=args.checkpoint_path,
    )
    print("best checkpoint:", trainer.checkpoint_callback.best_model_path)
    print("last checkpoint:", trainer.checkpoint_callback.last_model_path)


if __name__ == "__main__":
    main()
