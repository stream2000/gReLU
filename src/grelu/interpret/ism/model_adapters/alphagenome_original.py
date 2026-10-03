"""Original mouse AlphaGenome adapter for targeted Saijou ISM comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .base import TrackSpec
from .utils import sequences_to_tensor, slug_identifier, track_indices


HEAD_ORDER = ("atac", "dnase", "cage", "rna_seq", "chip_tf", "chip_histone")
DEFAULT_WEIGHTS = Path(
    "~/.cache/huggingface/hub/models--gtca--alphagenome_pytorch/"
    "snapshots/b01c0ffa73e07c053491f3b5ea8bcf67d93b9920/"
    "model_fold_0.safetensors"
).expanduser()
DEFAULT_METADATA = (
    Path(__file__).resolve().parents[4]
    / "alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_mouse.parquet"
)


def build_alphagenome_mouse_reference_panel(metadata: pd.DataFrame) -> pd.DataFrame:
    """Select liver and mesenchymal proxy tracks before model inference."""

    t = metadata.copy()
    biosample = t["biosample_name"].fillna("").astype(str)
    output = t["output_type"].fillna("").astype(str)
    histone = t["histone_mark"].fillna("").astype(str)
    tf = t["transcription_factor"].fillna("").astype(str)
    rules = [
        ("liver_rna", output.eq("rna_seq") & biosample.eq("liver")),
        ("fibroblast_rna", output.eq("rna_seq") & biosample.eq("embryonic fibroblast")),
        ("liver_cage", output.eq("cage") & biosample.eq("liver")),
        ("fibroblast_cage", output.eq("cage") & biosample.eq("fibroblast")),
        (
            "smooth_muscle_cage",
            output.eq("cage") & biosample.str.contains("aortic smooth muscle", case=False),
        ),
        (
            "mesenchymal_cage",
            output.eq("cage") & biosample.str.contains("mesenchymal stem", case=False),
        ),
        (
            "liver_accessibility",
            output.isin(["atac", "dnase"]) & biosample.eq("liver"),
        ),
        (
            "fibroblast_accessibility",
            output.isin(["atac", "dnase"])
            & biosample.str.contains("fibroblast", case=False),
        ),
        (
            "liver_active_chromatin",
            output.eq("chip_histone")
            & biosample.eq("liver")
            & histone.isin(["H3K27ac", "H3K4me1", "H3K4me3"]),
        ),
        (
            "fibroblast_active_chromatin",
            output.eq("chip_histone")
            & biosample.eq("embryonic fibroblast")
            & histone.isin(["H3K27ac", "H3K4me1", "H3K4me3"]),
        ),
        (
            "liver_tf",
            output.eq("chip_tf") & biosample.eq("liver") & tf.isin(["CTCF", "EP300"]),
        ),
        (
            "fibroblast_tf",
            output.eq("chip_tf")
            & biosample.eq("embryonic fibroblast")
            & tf.isin(["CTCF", "POLR2A"]),
        ),
    ]
    selected = []
    for group, mask in rules:
        part = t.loc[mask].copy()
        part["track_group"] = group
        selected.append(part)
    panel = pd.concat(selected, ignore_index=True)
    if panel.empty:
        raise ValueError("AlphaGenome mouse reference panel selected no tracks")
    duplicated = panel.duplicated(["output_type", "track_index"], keep=False)
    if duplicated.any():
        raise ValueError(
            "AlphaGenome reference panel contains duplicated head/track indices: "
            f"{panel.loc[duplicated, ['output_type', 'track_index']].to_dict('records')[:5]}"
        )
    return panel


class AlphaGenomeOriginalAdapter:
    """Emit selected original AlphaGenome mouse tracks at 128 bp resolution."""

    def __init__(
        self,
        *,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        metadata_path: str | Path = DEFAULT_METADATA,
        device: str = "cuda",
        input_length_bp: int = 1_048_576,
    ) -> None:
        self.weights_path = Path(weights_path).expanduser()
        self.metadata_path = Path(metadata_path)
        self.requested_device = device
        self.input_length_bp = int(input_length_bp)
        self.output_resolution_bp = 128
        self.output_length_bins = self.input_length_bp // self.output_resolution_bp
        self.model_id = "alphagenome_original_mouse"
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
        from alphagenome_pytorch.config import DtypePolicy
        from grelu.lightning import LightningModel

        if self.requested_device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested for AlphaGenome original: {self.requested_device}")
        metadata = pd.read_parquet(self.metadata_path)
        panel = build_alphagenome_mouse_reference_panel(metadata)
        heads = tuple(head for head in HEAD_ORDER if head in set(panel["output_type"]))
        model = LightningModel(
            model_params={
                "model_type": "AlphaGenomeModel",
                "output_key": heads,
                "weights_path": str(self.weights_path),
                "dtype_policy": DtypePolicy.mixed_precision(),
                "resolution": self.output_resolution_bp,
                "organism_index": 1,
                "num_organisms": 2,
            },
            train_params={"task": "regression", "loss": "mse"},
        )
        offsets: dict[str, int] = {}
        offset = 0
        for head in heads:
            offsets[head] = offset
            offset += int(model.model.embedding.model.heads[head].num_tracks)
        panel = panel.copy()
        panel["model_channel_index"] = [
            offsets[str(row.output_type)] + int(row.track_index)
            for row in panel.itertuples(index=False)
        ]
        panel["track_id"] = [
            f"ag_mouse_{row.output_type}_{int(row.track_index)}_{slug_identifier(row.track_name)}"
            for row in panel.itertuples(index=False)
        ]
        self.track_specs = [
            TrackSpec(
                track_id=str(row.track_id),
                channel_index=index,
                task_name=str(row.track_name),
                cell_type=str(row.biosample_name),
                modality=str(row.output_type),
                resolution_bp=self.output_resolution_bp,
                source="alphagenome_original_mouse",
                group=str(row.track_group),
                strand=str(row.strand),
                description=str(row.assay_title),
            )
            for index, row in enumerate(panel.itertuples(index=False))
        ]
        self._model_indices = panel["model_channel_index"].astype(int).tolist()
        self.track_manifest = panel
        self._device = torch.device(self.requested_device)
        model.eval()
        model.to(self._device)
        self._model = model
        self._metadata = {
            "model_id": self.model_id,
            "weights_path": str(self.weights_path),
            "metadata_path": str(self.metadata_path),
            "organism": "mouse",
            "organism_index": 1,
            "input_length_bp": self.input_length_bp,
            "output_resolution_bp": self.output_resolution_bp,
            "output_length_bins": self.output_length_bins,
            "computed_heads": list(heads),
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
            raise RuntimeError("AlphaGenomeOriginalAdapter.setup() has not been called")
        x = sequences_to_tensor(
            sequences, expected_length=self.input_length_bp
        ).to(self._device)
        with torch.inference_mode():
            full = self._model.forward(x)
            if full.ndim != 3 or full.shape[-1] != self.output_length_bins:
                raise RuntimeError(f"Unexpected AlphaGenome original output shape: {tuple(full.shape)}")
            selected = full[:, self._model_indices, :]
            if tracks is not None:
                selected = selected[:, track_indices(self.track_specs, tracks), :]
            return selected.detach().float().cpu().numpy().astype(np.float32, copy=False)
