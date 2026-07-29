#!/usr/bin/env python
"""Fine-tune AlphaGenome embeddings on Saijou pseudobulk bigWig tracks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import grelu.lightning
from alphagenome_pytorch.config import DtypePolicy
from borzoi_mmap_dataset import build_dataset_pair
from saijou_tasks import TASK_NAMES


DEFAULT_BIGWIG_DIR = "/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig"
DEFAULT_GENOME = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
DEFAULT_WEIGHTS = (
    "~/.cache/huggingface/hub/models--gtca--alphagenome_pytorch/"
    "snapshots/b01c0ffa73e07c053491f3b5ea8bcf67d93b9920/model_fold_0.safetensors"
)
ALPHAGENOME_LORA_TARGETS = ["mha", "mlp"]
ALPHAGENOME_ACTIVE_LINEAR_TARGETS = ["tower."]
ALPHAGENOME_128BP_CONV_TARGETS = ["encoder.", "embedder_128bp."]
ALPHAGENOME_1BP_CONV_TARGETS = [
    "encoder.",
    "decoder.",
    "embedder_128bp.",
    "embedder_1bp.",
]


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
    parser.add_argument("--head_hidden_channels", type=int, default=512)
    parser.add_argument("--head_hidden_layers", type=int, default=1)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--disable_head_norm", action="store_true")
    parser.add_argument("--organism_index", type=int, default=1, help="AlphaGenome organism index; 1 is mouse in the local wrapper.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        help="Lightning trainer precision, e.g. '32-true' or 'bf16-mixed'. "
        "bf16-mixed runs the frozen AlphaGenome trunk under autocast for speed/memory.",
    )
    parser.add_argument("--devices", default="0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_conv_rank", type=int, default=8)
    parser.add_argument("--lora_conv_alpha", type=int, default=16)
    parser.add_argument(
        "--lora_preset",
        default="active",
        choices=["tower", "active", "all"],
        help=(
            "tower: original mha/mlp Linear targets only; active: all active "
            "tower Linear plus encoder/embedder Conv1d for the selected resolution; "
            "all: every Linear and Conv1d in the AlphaGenome model."
        ),
    )
    parser.add_argument("--lora_targets", default=",".join(ALPHAGENOME_LORA_TARGETS))
    parser.add_argument(
        "--lora_conv_targets",
        default=None,
        help="Comma-separated Conv1d target substrings. Defaults are chosen by --lora_preset and --resolution.",
    )
    parser.add_argument("--total_weight", type=float, default=0.2)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--disable_early_stopping", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--skip_shape_check",
        action="store_true",
        help="Skip the pre-training forward shape check. Use this for DDP runs after a matching dry-run has passed.",
    )
    return parser.parse_args()


def parse_devices(value: str):
    if value == "cpu":
        return "cpu"
    devices = [int(item) for item in value.split(",")]
    return devices[0] if len(devices) == 1 else devices


def freeze_embedding(model: grelu.lightning.LightningModel) -> None:
    for param in model.model.embedding.parameters():
        param.requires_grad = False


def split_targets(value: str | None) -> list[str]:
    if value is None:
        return []
    if value.strip().lower() in {"", "none"}:
        return []
    return [item for item in value.split(",") if item]


def choose_lora_targets(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.lora_preset == "tower":
        linear_targets = split_targets(args.lora_targets)
        conv_targets = split_targets(args.lora_conv_targets)
    elif args.lora_preset == "active":
        linear_targets = split_targets(args.lora_targets)
        if linear_targets == ALPHAGENOME_LORA_TARGETS:
            linear_targets = ALPHAGENOME_ACTIVE_LINEAR_TARGETS
        conv_targets = split_targets(args.lora_conv_targets)
        if args.lora_conv_targets is None:
            conv_targets = (
                ALPHAGENOME_1BP_CONV_TARGETS
                if args.resolution == 1
                else ALPHAGENOME_128BP_CONV_TARGETS
            )
    elif args.lora_preset == "all":
        # An empty target matches every eligible module.
        linear_targets = [""]
        conv_targets = [""]
    else:
        raise ValueError(args.lora_preset)
    return linear_targets, conv_targets


def count_matching_modules(root: nn.Module, targets: list[str], module_type: type[nn.Module]) -> int:
    if not targets:
        return 0
    return sum(
        1
        for name, module in root.named_modules()
        if isinstance(module, module_type) and any(target in name for target in targets)
    )


# The upstream model does not provide a Conv1d LoRA adapter.
class Conv1dLoRA(nn.Module):
    """LoRA-style adapter for Conv1d layers, including padding='same' layers."""

    def __init__(self, original_layer: nn.Conv1d, rank: int, alpha: int) -> None:
        super().__init__()
        if original_layer.groups != 1:
            raise ValueError(f"Conv1dLoRA only supports groups=1, got {original_layer.groups}")
        if rank > original_layer.out_channels:
            raise ValueError(
                f"LoRA conv rank {rank} must be <= output channels {original_layer.out_channels}"
            )
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        self.scale = alpha / rank
        self.lora_down = nn.Conv1d(
            original_layer.in_channels,
            rank,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            dilation=original_layer.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv1d(rank, original_layer.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.original_layer(x)
        adapted = self.lora_up(self.lora_down(x)) * self.scale
        if adapted.shape[-1] != original.shape[-1]:
            delta = original.shape[-1] - adapted.shape[-1]
            if delta > 0:
                left = delta // 2
                adapted = F.pad(adapted, (left, delta - left))
            else:
                start = (-delta) // 2
                adapted = adapted[..., start : start + original.shape[-1]]
        return original + adapted


def get_parent_and_attr(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_conv1d_lora(root: nn.Module, targets: list[str], rank: int, alpha: int) -> int:
    wrapped = 0
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Conv1d):
            continue
        if not any(target in name for target in targets):
            continue
        parent, attr_name = get_parent_and_attr(root, name)
        if isinstance(parent, Conv1dLoRA) or ".original_layer" in name:
            continue
        setattr(parent, attr_name, Conv1dLoRA(module, rank=rank, alpha=alpha))
        wrapped += 1
    return wrapped


def apply_lora_to_alphagenome_embedding(
    model: grelu.lightning.LightningModel,
    linear_targets: list[str],
    conv_targets: list[str],
    linear_rank: int,
    linear_alpha: int,
    conv_rank: int,
    conv_alpha: int,
) -> dict[str, int]:
    from alphagenome_pytorch.extensions.finetuning.adapters import apply_lora

    freeze_embedding(model)
    ag_model = model.model.embedding.model
    n_linear_targets = count_matching_modules(ag_model, linear_targets, nn.Linear)
    n_conv_targets = count_matching_modules(ag_model, conv_targets, nn.Conv1d)
    if n_linear_targets == 0 and n_conv_targets == 0:
        raise ValueError(
            "No AlphaGenome adapter targets matched: "
            f"linear={linear_targets}, conv={conv_targets}"
        )

    before_lora = sum(1 for module in ag_model.modules() if module.__class__.__name__ == "LoRA")
    before_locon = sum(1 for module in ag_model.modules() if module.__class__.__name__ == "Locon")
    if linear_targets:
        apply_lora(ag_model, target_modules=linear_targets, rank=linear_rank, alpha=linear_alpha)
    wrapped_convs = 0
    if conv_targets:
        wrapped_convs = apply_conv1d_lora(
            ag_model,
            targets=conv_targets,
            rank=conv_rank,
            alpha=conv_alpha,
        )
    after_lora = sum(1 for module in ag_model.modules() if module.__class__.__name__ == "LoRA")
    after_locon = sum(1 for module in ag_model.modules() if module.__class__.__name__ == "Locon")
    return {
        "linear_targets": n_linear_targets,
        "conv_targets": n_conv_targets,
        "lora_wrappers": after_lora - before_lora,
        "locon_wrappers": (after_locon - before_locon) + wrapped_convs,
    }


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
    run_name = (
        f"alphagenome_{args.split_name}_seq{args.seq_len}_label{args.label_len}"
        f"_bin{args.bin_size}_res{args.resolution}_{args.target_mode}"
        f"_{args.finetune_mode}_{args.lora_preset}"
        f"_h{args.head_hidden_channels}x{args.head_hidden_layers}"
    )
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
        "precision": args.precision,
        "early_stopping": not args.disable_early_stopping,
        "freeze_embedding_eval": True,
        "freeze_embedding_norm_stats": True,
        "patience": 3,
        "monitor": "val_loss",
        "mode": "min",
        "finetune_mode": args.finetune_mode,
        "lora_preset": args.lora_preset,
        "lora_targets": args.lora_targets,
        "lora_conv_targets": args.lora_conv_targets,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_conv_rank": args.lora_conv_rank,
        "lora_conv_alpha": args.lora_conv_alpha,
    }
    train_params.update(extra_train_params)

    model_params = {
        "model_type": "AlphaGenomeFinetuneModel",
        "n_tasks": len(TASK_NAMES),
        "resolution": args.resolution,
        "label_len": args.label_len,
        "bin_size": args.bin_size,
        "head_hidden_channels": args.head_hidden_channels,
        "head_hidden_layers": args.head_hidden_layers,
        "head_dropout": args.head_dropout,
        "head_norm": not args.disable_head_norm,
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
        lora_result = {
            "linear_targets": 0,
            "conv_targets": 0,
            "lora_wrappers": 0,
            "locon_wrappers": 0,
        }
    else:
        linear_targets, conv_targets = choose_lora_targets(args)
        lora_result = apply_lora_to_alphagenome_embedding(
            model,
            linear_targets=linear_targets,
            conv_targets=conv_targets,
            linear_rank=args.lora_rank,
            linear_alpha=args.lora_alpha,
            conv_rank=args.lora_conv_rank,
            conv_alpha=args.lora_conv_alpha,
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("model: alphagenome")
    print("finetune_mode:", args.finetune_mode)
    print("seq_len:", args.seq_len)
    print("label_len:", args.label_len)
    print("bin_size:", args.bin_size)
    print("resolution:", args.resolution)
    print("head_hidden_channels:", args.head_hidden_channels)
    print("head_hidden_layers:", args.head_hidden_layers)
    print("head_dropout:", args.head_dropout)
    print("head_norm:", not args.disable_head_norm)
    print("organism_index:", args.organism_index)
    print("gradient_checkpointing:", args.gradient_checkpointing)
    print("lr:", lr)
    print("early_stopping:", not args.disable_early_stopping)
    print("freeze_embedding_eval:", True)
    print("freeze_embedding_norm_stats:", True)
    if args.target_mode == "poisson_multinomial":
        print("total_weight:", args.total_weight)
    if args.finetune_mode == "lora":
        print("lora_preset:", args.lora_preset)
        print("lora_linear_targets:", ",".join(linear_targets))
        print("lora_conv_targets:", ",".join(conv_targets))
        print("lora_linear_target_count:", lora_result["linear_targets"])
        print("lora_conv_target_count:", lora_result["conv_targets"])
        print("lora_wrappers:", lora_result["lora_wrappers"])
        print("locon_wrappers:", lora_result["locon_wrappers"])
        print("lora_rank:", args.lora_rank)
        print("lora_alpha:", args.lora_alpha)
        print("lora_conv_rank:", args.lora_conv_rank)
        print("lora_conv_alpha:", args.lora_conv_alpha)
    print(f"trainable_params: {trainable_params:,} / {total_params:,}")

    shape_device = None
    if not args.skip_shape_check:
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        x, y = next(iter(loader))
        if args.devices != "cpu" and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            shape_device = torch.device(f"cuda:{local_rank}")
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
    else:
        print("shape_check: skipped")
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
