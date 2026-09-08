"""Adapter for Saijou fine-tuned AlphaGenome checkpoints."""

from __future__ import annotations

import sys
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch

from .base import TrackSpec
from .utils import sequences_to_tensor, track_indices

REPO_ROOT = Path(__file__).resolve().parents[5]
FT_SCRIPT_DIR = REPO_ROOT / "src" / "ft-scripts"
if str(FT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FT_SCRIPT_DIR))

from saijou_tasks import TASK_NAMES  # noqa: E402


class AlphaGenomeFinetunedAdapter:
    """Load a fine-tuned AlphaGenome LoRA checkpoint and emit 4-track profiles."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str = "cuda",
        lora_preset: str | None = None,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        lora_conv_rank: int | None = None,
        lora_conv_alpha: int | None = None,
        lora_targets: str | None = None,
        lora_conv_targets: str | None = None,
        input_length_bp: int | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.requested_device = device
        self._device = torch.device("cpu")
        self._model = None
        self._metadata: dict[str, object] = {}
        self._lora_overrides = {
            "lora_preset": lora_preset,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_conv_rank": lora_conv_rank,
            "lora_conv_alpha": lora_conv_alpha,
            "lora_targets": lora_targets,
            "lora_conv_targets": lora_conv_targets,
        }
        self._input_length_override = input_length_bp
        self.model_id = "alphagenome_finetuned_saijou"
        self.input_length_bp = 0
        self.output_resolution_bp = 0
        self.output_length_bins = 0
        self.track_specs: list[TrackSpec] = []

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def setup(self) -> None:
        import grelu.lightning
        from train_alphagenome import (
            apply_lora_to_alphagenome_embedding,
            choose_lora_targets,
        )

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        hparams = ckpt["hyper_parameters"]
        model = grelu.lightning.LightningModel(
            model_params=hparams["model_params"],
            train_params=hparams["train_params"],
        )
        train_params = hparams.get("train_params", {})
        model_params = hparams.get("model_params", {})
        finetune_mode = train_params.get("finetune_mode", "headonly")
        lora_result = {
            "linear_targets": 0,
            "conv_targets": 0,
            "lora_wrappers": 0,
            "locon_wrappers": 0,
        }
        if finetune_mode == "lora":
            lora_args = SimpleNamespace(
                lora_preset=self._override("lora_preset", train_params, "active"),
                resolution=int(model_params.get("resolution", 128)),
                lora_targets=self._override("lora_targets", train_params, "mha,mlp"),
                lora_conv_targets=self._override("lora_conv_targets", train_params, None),
            )
            linear_targets, conv_targets = choose_lora_targets(lora_args)
            lora_result = apply_lora_to_alphagenome_embedding(
                model,
                linear_targets=linear_targets,
                conv_targets=conv_targets,
                linear_rank=int(self._override("lora_rank", train_params, 8)),
                linear_alpha=int(self._override("lora_alpha", train_params, 16)),
                conv_rank=int(self._override("lora_conv_rank", train_params, 8)),
                conv_alpha=int(self._override("lora_conv_alpha", train_params, 16)),
            )

        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

        device = self.requested_device
        if device != "cpu" and not torch.cuda.is_available():
            print(
                f"[alphagenome_finetuned] CUDA unavailable, falling back to cpu "
                f"(requested {device})",
                flush=True,
            )
            device = "cpu"
        self._device = torch.device(device)
        model.eval()
        model.to(self._device)
        self._model = model
        self.input_length_bp = self._infer_input_length(model, train_params)
        self.output_resolution_bp = int(model_params.get("bin_size", model_params.get("resolution", 128)))
        label_len = int(model_params.get("label_len", 0))
        self.output_length_bins = label_len // self.output_resolution_bp
        self.track_specs = [
            TrackSpec(
                track_id=task,
                channel_index=index,
                task_name=task,
                cell_type=task,
                modality="10x_scRNAseq_3prime_pseudobulk",
                resolution_bp=self.output_resolution_bp,
                source="alphagenome_finetuned",
                group=f"{task}_finetuned_10x",
                description=f"Saijou {task} 10X scRNA-seq pseudobulk",
            )
            for index, task in enumerate(TASK_NAMES)
        ]
        self._metadata = {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_global_step": ckpt.get("global_step"),
            "finetune_mode": finetune_mode,
            "input_length_bp": self.input_length_bp,
            "output_resolution_bp": self.output_resolution_bp,
            "output_length_bins": self.output_length_bins,
            "device": str(self._device),
            **lora_result,
        }

    def _infer_input_length(self, model, train_params: dict) -> int:
        if self._input_length_override is not None:
            return int(self._input_length_override)
        data_len = int(model.data_params.get("train", {}).get("seq_len", 0))
        if data_len > 0:
            return data_len
        run_name = str(train_params.get("name", ""))
        match = re.search(r"seq(\d+)", run_name)
        if match:
            return int(match.group(1))
        raise ValueError(
            "Cannot infer AlphaGenome input length from checkpoint; pass input_length_bp"
        )

    def _override(self, key: str, train_params: dict, default):
        value = self._lora_overrides.get(key)
        if value is not None:
            return value
        return train_params.get(key, default)

    def predict_profiles(
        self,
        sequences: Sequence[str],
        tracks: Sequence[str] | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("AlphaGenomeFinetunedAdapter.setup() has not been called")
        x = sequences_to_tensor(
            sequences, expected_length=self.input_length_bp
        ).to(self._device)
        with torch.no_grad():
            pred = self._model.forward(x).detach().cpu().numpy().astype(np.float32)
        if pred.ndim != 3:
            raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")
        if pred.shape[-1] != self.output_length_bins:
            raise RuntimeError(
                f"Expected {self.output_length_bins} AlphaGenome bins, got {pred.shape[-1]}"
            )
        indices = track_indices(self.track_specs, tracks)
        return pred[:, indices, :]
