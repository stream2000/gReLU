"""Adapter for Saijou fine-tuned Borzoi checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .base import TrackSpec
from .utils import sequences_to_tensor, track_indices

REPO_ROOT = Path(__file__).resolve().parents[5]
FT_SCRIPT_DIR = REPO_ROOT / "src" / "ft-scripts"
if str(FT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FT_SCRIPT_DIR))

TASK_NAMES = ["hsc", "mac", "lsec", "chol"]


class BorzoiFinetunedAdapter:
    """Load a Saijou four-track Borzoi checkpoint and emit profile predictions."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str = "cuda",
        input_length_bp: int = 524_288,
        output_resolution_bp: int = 32,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.requested_device = device
        self.input_length_bp = int(input_length_bp)
        self.output_resolution_bp = int(output_resolution_bp)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = lora_target_modules
        self.model_id = "borzoi_finetuned_saijou"
        self.output_length_bins = 0
        self.track_specs: list[TrackSpec] = []
        self._device = torch.device("cpu")
        self._model = None
        self._metadata: dict[str, object] = {}

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def setup(self) -> None:
        import grelu.lightning
        from train_borzoi import (
            BORZOI_LORA_TARGET_MODULES,
            apply_lora_to_borzoi_embedding,
        )

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        hparams = checkpoint["hyper_parameters"]
        model_params = hparams["model_params"]
        train_params = hparams["train_params"]
        model = grelu.lightning.LightningModel(
            model_params=model_params,
            train_params=train_params,
        )

        state_dict = checkpoint["state_dict"]
        is_lora = any("lora_" in key for key in state_dict)
        lora_targets = 0
        if is_lora:
            lora_targets = apply_lora_to_borzoi_embedding(
                model,
                target_modules=self.lora_target_modules or BORZOI_LORA_TARGET_MODULES,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

        device = self.requested_device
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
        self._device = torch.device(device)
        model.eval()
        model.to(self._device)
        self._model = model

        crop_len = int(model_params.get("crop_len", 0))
        native_bins = self.input_length_bp // self.output_resolution_bp
        self.output_length_bins = native_bins - (2 * crop_len)
        if self.output_length_bins <= 0:
            raise ValueError(
                "Invalid Borzoi output geometry: "
                f"input={self.input_length_bp}, resolution={self.output_resolution_bp}, "
                f"crop_len={crop_len}"
            )
        self.track_specs = [
            TrackSpec(
                track_id=task,
                channel_index=index,
                task_name=task,
                cell_type=task,
                modality="10x_scRNAseq_3prime_pseudobulk",
                resolution_bp=self.output_resolution_bp,
                source="borzoi_finetuned",
                group=f"{task}_finetuned_10x",
                description=f"Saijou {task} 10X scRNA-seq pseudobulk",
            )
            for index, task in enumerate(TASK_NAMES)
        ]
        self._metadata = {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "finetune_mode": "lora" if is_lora else "headonly",
            "lora_targets": int(lora_targets),
            "lora_rank": self.lora_rank if is_lora else None,
            "lora_alpha": self.lora_alpha if is_lora else None,
            "input_length_bp": self.input_length_bp,
            "output_resolution_bp": self.output_resolution_bp,
            "output_length_bins": self.output_length_bins,
            "crop_len_bins_each_side": crop_len,
            "target_mode": train_params.get("loss"),
        }

    def predict_profiles(
        self,
        sequences: Sequence[str],
        tracks: Sequence[str] | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("BorzoiFinetunedAdapter.setup() has not been called")
        x = sequences_to_tensor(
            sequences, expected_length=self.input_length_bp
        ).to(self._device)
        with torch.inference_mode():
            pred = self._model.forward(x).detach().cpu().numpy().astype(np.float32)
        if pred.ndim != 3:
            raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")
        if pred.shape[-1] != self.output_length_bins:
            raise RuntimeError(
                f"Expected {self.output_length_bins} Borzoi bins, got {pred.shape[-1]}"
            )
        indices = track_indices(self.track_specs, tracks)
        return pred[:, indices, :]
