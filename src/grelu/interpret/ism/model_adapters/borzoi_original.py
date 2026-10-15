"""Original mouse Borzoi adapter for targeted Saijou ISM comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .base import TrackSpec
from .utils import sequences_to_tensor, slug_identifier, track_indices


DEFAULT_CHECKPOINT = Path(
    "~/.cache/huggingface/hub/models--Genentech--borzoi-model/"
    "snapshots/9d9e6915f21aba8bfeccca4af2c72deef291af8d/mouse_rep0.ckpt"
).expanduser()


def build_borzoi_mouse_reference_panel(tasks: pd.DataFrame) -> pd.DataFrame:
    """Select exact HSC CAGE plus liver and mesenchymal proxy tracks."""

    t = tasks.copy().reset_index(names="model_channel_index")
    assay = t["assay"].fillna("").astype(str)
    sample = t["sample"].fillna("").astype(str)
    description = t["description"].fillna("").astype(str)
    rules = [
        ("hsc_cage", assay.eq("CAGE") & sample.str.contains("hepatic stellate", case=False)),
        (
            "smooth_muscle_cage",
            assay.eq("CAGE") & sample.str.contains("aortic smooth muscle", case=False),
        ),
        (
            "mesenchymal_cage",
            assay.eq("CAGE") & sample.str.contains("mesenchymal stem", case=False),
        ),
        ("fibroblast_cage", assay.eq("CAGE") & sample.str.contains("fibroblast", case=False)),
        (
            "liver_cage",
            assay.eq("CAGE")
            & sample.str.contains("liver", case=False)
            & sample.str.contains("adult", case=False),
        ),
        ("liver_rna", assay.eq("RNA") & sample.str.contains("liver", case=False)),
        ("fibroblast_rna", assay.eq("RNA") & sample.str.contains("fibroblast", case=False)),
        (
            "liver_accessibility",
            assay.isin(["ATAC", "DNASE"]) & sample.str.contains("liver", case=False),
        ),
        (
            "fibroblast_accessibility",
            assay.isin(["ATAC", "DNASE"]) & sample.str.contains("fibroblast", case=False),
        ),
        (
            "liver_active_chromatin",
            assay.eq("CHIP")
            & sample.str.contains("liver", case=False)
            & sample.str.contains("adult", case=False)
            & description.str.contains("H3K27ac|H3K4me1|H3K4me3", case=False, regex=True),
        ),
        (
            "fibroblast_active_chromatin",
            assay.eq("CHIP")
            & sample.str.contains("fibroblast", case=False)
            & description.str.contains("H3K27ac|H3K4me1|H3K4me3", case=False, regex=True),
        ),
        (
            "liver_tf",
            assay.eq("CHIP")
            & sample.str.contains("liver", case=False)
            & description.str.contains("CTCF|GATA4", case=False, regex=True),
        ),
        (
            "fibroblast_tf",
            assay.eq("CHIP")
            & sample.str.contains("fibroblast", case=False)
            & description.str.contains("CTCF|POLR2A", case=False, regex=True),
        ),
    ]
    selected = []
    for group, mask in rules:
        part = t.loc[mask].copy()
        part["track_group"] = group
        selected.append(part)
    panel = pd.concat(selected, ignore_index=True)
    if panel.empty:
        raise ValueError("Borzoi mouse reference panel selected no tracks")
    duplicated = panel.duplicated("model_channel_index", keep=False)
    if duplicated.any():
        raise ValueError(
            "Borzoi reference panel contains duplicated model channels: "
            f"{panel.loc[duplicated, 'model_channel_index'].tolist()[:10]}"
        )
    return panel


class BorzoiOriginalAdapter:
    """Emit selected published Borzoi mouse tracks at 32 bp resolution."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        input_length_bp: int = 524_288,
        output_resolution_bp: int = 32,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.requested_device = device
        self.input_length_bp = int(input_length_bp)
        self.output_resolution_bp = int(output_resolution_bp)
        self.output_length_bins = 0
        self.model_id = "borzoi_original_mouse_rep0"
        self.track_specs: list[TrackSpec] = []
        self.track_manifest = pd.DataFrame()
        self._model_indices: list[int] = []
        self._model = None
        self._device = torch.device("cpu")
        self._metadata: dict[str, object] = {}

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def setup(self) -> None:
        from grelu.lightning import LightningModel

        if self.requested_device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested for Borzoi original: {self.requested_device}")
        model = LightningModel.load_from_checkpoint(
            self.checkpoint_path,
            map_location="cpu",
        )
        tasks = pd.DataFrame(model.data_params["tasks"])
        panel = build_borzoi_mouse_reference_panel(tasks)
        panel["track_id"] = [
            f"borzoi_mouse_{int(row.model_channel_index)}_{slug_identifier(row.name)}"
            for row in panel.itertuples(index=False)
        ]
        self.track_specs = [
            TrackSpec(
                track_id=str(row.track_id),
                channel_index=index,
                task_name=str(row.name),
                cell_type=str(row.sample),
                modality=str(row.assay).lower(),
                resolution_bp=self.output_resolution_bp,
                source="borzoi_original_mouse_rep0",
                group=str(row.track_group),
                strand=("+" if str(row.name).endswith("+") else "-" if str(row.name).endswith("-") else "."),
                description=str(row.description),
            )
            for index, row in enumerate(panel.itertuples(index=False))
        ]
        self._model_indices = panel["model_channel_index"].astype(int).tolist()
        self.track_manifest = panel
        crop_len = int(model.model_params.get("crop_len", 0))
        self.output_length_bins = (
            self.input_length_bp // self.output_resolution_bp - 2 * crop_len
        )
        if self.output_length_bins <= 0:
            raise ValueError("Invalid Borzoi original output geometry")
        self._device = torch.device(self.requested_device)
        model.eval()
        model.to(self._device)
        self._model = model
        self._metadata = {
            "model_id": self.model_id,
            "checkpoint_path": str(self.checkpoint_path),
            "organism": "mouse",
            "replicate": 0,
            "input_length_bp": self.input_length_bp,
            "output_resolution_bp": self.output_resolution_bp,
            "output_length_bins": self.output_length_bins,
            "crop_len_bins_each_side": crop_len,
            "selected_tracks": len(self.track_specs),
            "track_groups": sorted(panel["track_group"].unique().tolist()),
            "device": str(self._device),
        }

    def predict_profiles(
        self,
        sequences: Sequence[str],
        tracks: Sequence[str] | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("BorzoiOriginalAdapter.setup() has not been called")
        x = sequences_to_tensor(
            sequences, expected_length=self.input_length_bp
        ).to(self._device)
        with torch.inference_mode():
            full = self._model.forward(x)
            if full.ndim != 3 or full.shape[-1] != self.output_length_bins:
                raise RuntimeError(f"Unexpected Borzoi original output shape: {tuple(full.shape)}")
            selected = full[:, self._model_indices, :]
            if tracks is not None:
                selected = selected[:, track_indices(self.track_specs, tracks), :]
            return selected.detach().float().cpu().numpy().astype(np.float32, copy=False)
