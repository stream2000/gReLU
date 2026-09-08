#!/usr/bin/env python
"""Frozen NTv3 profile qualification and training on the existing Saijou stack."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch
from pytorch_lightning import Callback, seed_everything
from torch.utils.data import DataLoader

from borzoi_mmap_dataset import build_dataset_pair, manifest_matches
from saijou_tasks import TASK_NAMES
from train_borzoi import DEFAULT_BIGWIG_DIR, DEFAULT_GENOME
from grelu.lightning import LightningModel
from grelu.model.trunks.ntv3 import CHECKPOINT, CODE_REVISION, SEQ_LEN, LABEL_LEN, BIN_SIZE


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


class TrainingAudit(Callback):
    """Audit/timing only: training and metrics remain owned by LightningModel."""

    def __init__(self, out, checksum):
        self.out = Path(out)
        self.checksum = checksum
        self.seconds = []
        self.started = time.perf_counter()

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if module.device.type == "cuda":
            torch.cuda.synchronize(module.device)
        self.batch_start = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if module.device.type == "cuda":
            torch.cuda.synchronize(module.device)
        self.seconds.append(time.perf_counter() - self.batch_start)
        if not torch.isfinite(outputs["loss"]).all():
            raise FloatingPointError(f"Nonfinite loss at batch {batch_idx}")

    def on_before_optimizer_step(self, trainer, module, optimizer):
        if any(p.grad is not None for p in module.model.embedding.parameters()):
            raise RuntimeError("Frozen NTv3 received gradients")
        if any(p.grad is None or not torch.isfinite(p.grad).all()
               for p in module.model.head.parameters()):
            raise FloatingPointError("Missing or nonfinite head gradients")

    def on_train_epoch_end(self, trainer, module):
        if state_checksum(module.model.embedding) != self.checksum:
            raise RuntimeError("Frozen NTv3 parameters/non-cache buffers changed")
        if not trainer.is_global_zero:
            return
        write_json(self.out, dict(
            status="training", epoch=int(trainer.current_epoch),
            global_step=int(trainer.global_step), batches=len(self.seconds),
            world_size=trainer.world_size,
            median_train_batch_seconds=float(np.median(self.seconds)),
            elapsed_seconds=time.perf_counter() - self.started,
            metrics={k: float(v) for k, v in trainer.callback_metrics.items()},
        ))


def configure_audit(self):
    return [self._ntv3_audit]


def make_validation_loader(self, dataset, batch_size=None, num_workers=None):
    # Keep validation's mean-of-batch-loss definition unchanged while tuning
    # training batch size; final exact evaluation also uses one window/batch.
    return LightningModel.make_test_loader(self, dataset, batch_size=1, num_workers=num_workers)


def state_checksum(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require_existing_cache(args, bw_files):
    """Do not let the shared builder silently rebuild a mismatched cache."""
    prefix = f"{args.split_name}__poisson_multinomial"
    candidates = [
        Path(args.cache_dir) / f"{prefix}__bin32",
        Path(args.cache_dir) / f"{prefix}__seq524288__label196608__bin32",
    ]
    for cache in candidates:
        if manifest_matches(
            cache / "manifest.json", args.split_name, "poisson_multinomial",
            args.genome, SEQ_LEN, LABEL_LEN, BIN_SIZE, bw_files, TASK_NAMES,
        ):
            for split in ("train", "val"):
                bed = Path(args.split_dir) / args.split_name / f"{split}_intervals.bed"
                with bed.open() as handle:
                    rows = sum(bool(line.strip()) for line in handle)
                labels = np.load(cache / f"{split}_labels.npy", mmap_mode="r")
                if labels.shape != (rows, 4, 6144):
                    raise ValueError(f"Invalid {split} cache shape: {labels.shape}")
            return cache
    raise ValueError("No matching existing label cache; automatic rebuild is disabled")


def track_means(labels):
    total = np.zeros(4, dtype=np.float64)
    for start in range(0, len(labels), 16):
        batch = np.asarray(labels[start:start + 16])
        if not np.isfinite(batch).all() or (batch < 0).any():
            raise ValueError("Training labels must be finite and nonnegative")
        total += batch.sum(axis=(0, 2), dtype=np.float64)
    if not len(labels):
        raise ValueError("Empty training labels")
    return total / (len(labels) * labels.shape[2])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split_name", default="split_chr10_chr11_tiny50",
                        choices=["split_chr10_chr11_tiny50", "split_chr10_chr11"])
    parser.add_argument("--bigwig_dir", default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--genome", default=DEFAULT_GENOME)
    parser.add_argument("--split_dir", default="dataset_store/splits")
    parser.add_argument("--cache_dir", default="dataset_store/mmap_caches")
    parser.add_argument("--devices", default="0", help="Comma-separated CUDA indices or cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, choices=[17, 29, 43], default=17)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.num_workers < 0 or args.batch_size < 1 or args.accumulate_grad_batches < 1:
        parser.error("Invalid epochs/workers")
    if args.out.exists():
        parser.error("--out already exists; choose a new experiment output")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    devices = "cpu" if args.devices == "cpu" else [int(v) for v in args.devices.split(",")]
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu" if devices == "cpu" else f"cuda:{devices[local_rank]}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Stage 1 requires a bf16-capable CUDA device")
    bw_files = [str(Path(args.bigwig_dir) / f"{task}.CPM.mapq10.bw") for task in TASK_NAMES]
    cache = require_existing_cache(args, bw_files)
    # Load gated model before touching any sequence or opening data handles.
    params = dict(model_type="NTv3PretrainedProfileModel", n_tasks=4,
                  checkpoint=CHECKPOINT, revision=args.revision,
                  use_bfloat16_compute=device.type == "cuda")
    model = LightningModel(params, dict(task="regression", loss="poisson_multinomial",
                          total_weight=0.2, freeze_embedding_eval=True,
                          freeze_embedding_norm_stats=True)).to(device)
    datasets = build_dataset_pair(
        split_name=args.split_name, target_mode="poisson_multinomial", bw_files=bw_files,
        task_names=TASK_NAMES, genome=args.genome, split_dir=Path(args.split_dir),
        cache_dir=Path(args.cache_dir), seq_len=SEQ_LEN, label_len=LABEL_LEN, bin_size=BIN_SIZE,
    )
    try:
        means = track_means(datasets[0].labels)
        head = model.model.head
        with torch.no_grad():
            head.output_projection.weight.normal_(0, 1e-3)
            head.output_projection.bias.copy_(torch.as_tensor(np.log(np.maximum(means, 1e-4)), device=device))
        # Forward-oriented validation window; no RC randomness in qualification.
        x, y = next(iter(DataLoader(datasets[1], batch_size=1, num_workers=0)))
        x, y = x.to(device), y.to(device)
        if not torch.isfinite(y).all() or (y < 0).any():
            raise ValueError("Invalid validation labels")
        before = state_checksum(model.model.embedding)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.train()
        features = model.model.embedding(x)
        logits = head(features)
        loss = model.loss(logits, y)
        if logits.shape != y.shape or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("Invalid prediction shape or nonfinite logits/loss")
        loss.backward()
        if any(p.grad is not None for p in model.model.embedding.parameters()):
            raise RuntimeError("Frozen trunk received gradients")
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
            raise RuntimeError("Missing or nonfinite head gradients")
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        after = state_checksum(model.model.embedding)
        if before != after:
            raise RuntimeError("Frozen trunk state changed")
        import transformers
        result = dict(
            status="stage1_passed", checkpoint=CHECKPOINT, revision=args.revision,
            code_revision=CODE_REVISION, seed=args.seed,
            runtime_cache_exclusions=["cos_cached", "sin_cached"],
            split=args.split_name, window="val row 0", task_names=TASK_NAMES,
            label_cache=str(cache), x_shape=list(x.shape), feature_shape=list(features.shape),
            y_shape=list(y.shape), prediction_shape=list(logits.shape), loss=loss.item(),
            forward_backward_seconds=elapsed, train_track_means=means.tolist(),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
            trunk_checksum_before=before, trunk_checksum_after=after,
            torch_version=torch.__version__, transformers_version=transformers.__version__,
            device=str(device), compute="component-bf16" if device.type == "cuda" else "fp32",
            gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            stage2="not_run",
        )
        if args.dry_run:
            if local_rank == 0:
                write_json(args.out, result)
            print(json.dumps(result, indent=2))
            return
        run_dir = args.out.parent / f"seed{args.seed}"
        model.train_params.update(dict(
            optimizer="adam", lr=3e-4, batch_size=args.batch_size, num_workers=args.num_workers,
            validation_batch_size=1,
            devices=devices,
            precision="bf16-mixed" if device.type == "cuda" else "32-true",
            accumulate_grad_batches=args.accumulate_grad_batches, clip=1.0, early_stopping=True, patience=3,
            monitor="val_loss", mode="min", max_epochs=args.max_epochs,
            logger="csv", save_dir=str(run_dir), name="training",
            checkpoint=dict(dirpath=str(run_dir / "checkpoints"), filename="epoch{epoch:02d}",
                            monitor="val_loss", mode="min", save_top_k=1, save_last=True),
        ))
        model.save_hyperparameters(dict(model_params=model.model_params,
                                       train_params=dict(model.train_params)))
        model.zero_grad(set_to_none=True)
        del features, logits, loss
        for dataset in datasets:
            dataset.close_handles()
        audit = TrainingAudit(run_dir / "progress.json", before)
        model._ntv3_audit = audit
        model.configure_callbacks = MethodType(configure_audit, model)
        model.make_test_loader = MethodType(make_validation_loader, model)
        trainer = model.train_on_dataset(datasets[0], datasets[1], args.checkpoint_path)
        if state_checksum(model.model.embedding) != before:
            raise RuntimeError("NTv3 changed during training")
        trainer.strategy.barrier()
        if not trainer.is_global_zero:
            return
        last_path = trainer.checkpoint_callback.last_model_path
        best_path = trainer.checkpoint_callback.best_model_path
        best_score = float(trainer.checkpoint_callback.best_model_score)
        # A new checkpoint directory resets ModelCheckpoint's top-k history.
        # Keep the pre-migration best eligible, without consulting test metrics.
        if args.checkpoint_path:
            prior = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
            for state in prior.get("callbacks", {}).values():
                if state.get("best_model_score") is not None and state.get("best_model_path"):
                    if float(state["best_model_score"]) < best_score:
                        best_score = float(state["best_model_score"])
                        best_path = state["best_model_path"]
        # Verify exact fixed-window predictions after strict last-checkpoint reload.
        model.eval().to(device)
        with torch.no_grad():
            expected = model(x, logits=True).cpu()
        restored = LightningModel.load_from_checkpoint(last_path, map_location="cpu").to(device).eval()
        with torch.no_grad():
            reloaded = restored(x, logits=True).cpu()
        torch.testing.assert_close(expected, reloaded, rtol=0, atol=0)
        if state_checksum(restored.model.embedding) != before:
            raise RuntimeError("NTv3 changed after checkpoint reload")
        result.update(
            status="training_passed", stage2="complete", max_epochs=args.max_epochs,
            completed_epochs=int(trainer.current_epoch), best_checkpoint=best_path,
            last_checkpoint=last_path, best_val_loss=best_score,
            world_size=trainer.world_size, batch_size=args.batch_size,
            accumulate_grad_batches=args.accumulate_grad_batches,
            effective_batch_size=args.batch_size * args.accumulate_grad_batches * trainer.world_size,
            resumed_from=args.checkpoint_path,
            checkpoint_prediction_max_abs_diff=float((expected - reloaded).abs().max()),
            median_train_batch_seconds=float(np.median(audit.seconds)),
            estimated_full_train_epoch_hours=float(np.median(audit.seconds)) * 11201 / trainer.world_size / args.batch_size / 3600,
            training_wall_seconds=time.perf_counter() - audit.started,
            train_windows=datasets[0].n_seqs, val_windows=datasets[1].n_seqs,
            train_batches=len(audit.seconds), trunk_checksum_after=state_checksum(restored.model.embedding),
        )
        write_json(args.out, result)
        print(json.dumps(result, indent=2))
    finally:
        for dataset in datasets:
            dataset.close_handles()


if __name__ == "__main__":
    main()
