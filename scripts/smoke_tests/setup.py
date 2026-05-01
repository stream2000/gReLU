"""Model initialisation and shared app state for all benchmark experiments.

GreluTutorialApp owns:
  - sequence setup for a target gene (setup)
  - model loading (setup_borzoi, setup_ag_rna, setup_ag_cage)
  - brain-vs-liver specificity transforms (get_borzoi_transform, get_ag_transform)
  - GPU memory cleanup (_cleanup)

Experiment logic (inference, ISM, eval, eQTL) lives in the individual
run_*.py scripts and calls these methods via an app instance.
"""
import os
from dataclasses import dataclass

import gc
import numpy as np
import pandas as pd
import torch
from alphagenome_pytorch.config import DtypePolicy

import grelu.io
import grelu.resources
import grelu.sequence
import grelu.transforms
import grelu.transforms.prediction_transforms
from grelu.lightning import LightningModel
from scripts.smoke_tests.config import (
    BORZOI_INPUT_LEN, BORZOI_BIN_SIZE,
    AG_INPUT_LEN, AG_BIN_SIZE, AG_META_PATH,
    WEIGHTS_PATH, ISM_HALF_WIDTH,
)


class GreluTutorialApp:
    def __init__(self, gene="SRSF11", genome="hg38", devices="0,1,2,3", num_workers=1):
        self.gene = gene
        self.genome = genome

        if devices == "cpu":
            self.devices = "cpu"
            self.inference_device = "cpu"
        else:
            self.devices = [int(x) for x in devices.split(',')]
            self.inference_device = self.devices[0]

        self.num_workers = num_workers

        # Data cache
        self.exons = None
        self.input_seqs = None
        self.ag_seqs = None
        self.target_exons = None
        self._model_cfg: "ModelConfig | None" = None

    def _cleanup(self):
        """Forced clearing of GPU memory and memory references"""
        print("Explicitly clearing GPU memory and garbage collecting...")
        if hasattr(self, 'borzoi'): del self.borzoi
        if hasattr(self, 'ag_rna'): del self.ag_rna
        if hasattr(self, 'ag_cage'): del self.ag_cage

        self.borzoi = None
        self.ag_rna = None
        self.ag_cage = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Cleanup complete.")

    # TODO: Read this
    def setup(self):
        if self.input_seqs is not None:
            return

        print(f"Setting up sequences for gene {self.gene}...")
        if self.exons is None:
            self.exons = grelu.io.genome.read_gtf(self.genome, features="exon")

        self.target_exons = self.exons[self.exons.gene_name == self.gene].copy()
        if len(self.target_exons) == 0:
            raise ValueError(f"Gene {self.gene} not found.")

        self.chrom = self.target_exons.chrom.iloc[0]
        self.ism_center = int(self.target_exons.start.min())

        self.borzoi_start_coord = self.ism_center - BORZOI_INPUT_LEN // 2

        self.ag_start_coord = self.ism_center - AG_INPUT_LEN // 2
        self.ag_end_coord = self.ag_start_coord + AG_INPUT_LEN

        b_intervals = pd.DataFrame({"chrom": [self.chrom], "start": [self.borzoi_start_coord],
                                    "end": [self.borzoi_start_coord + BORZOI_INPUT_LEN], "strand": ["+"]})
        self.input_seqs = \
        grelu.sequence.format.convert_input_type(b_intervals, output_type="strings", genome=self.genome)[0]

        ag_intervals = pd.DataFrame(
            {"chrom": [self.chrom], "start": [self.ag_start_coord], "end": [self.ag_end_coord], "strand": ["+"]})
        self.ag_seqs = \
        grelu.sequence.format.convert_input_type(ag_intervals, output_type="strings", genome=self.genome)[0]

        self.ism_region = {
            "start": self.ism_center - ISM_HALF_WIDTH,
            "end": self.ism_center + ISM_HALF_WIDTH,
        }

    def setup_borzoi(self):
        print("Loading Borzoi model...")
        self.borzoi = grelu.resources.load_model(repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt")
        self._model_cfg = ModelConfig(name="borzoi", seq_len=BORZOI_INPUT_LEN, bin_size=BORZOI_BIN_SIZE)

    def _setup_alpha_genome(self, output_key: str) -> "LightningModel":
        """Instantiate one AlphaGenome head and update self._model_cfg.

        Single source of truth for AG model construction — used by
        ``setup_ag_rna``, ``setup_ag_cage``, and ``run_genome_wide_eval``.

        Args:
            output_key: AlphaGenome head to activate (e.g. 'rna_seq', 'cage').

        Returns:
            Configured ``LightningModel`` instance (not yet moved to GPU).
        """
        print(f"Loading AlphaGenome {output_key} model...")
        ag_params = dict(weights_path=WEIGHTS_PATH, dtype_policy=DtypePolicy.mixed_precision(), resolution=128)
        model = LightningModel(
            model_params={"model_type": "AlphaGenomeModel", "output_key": output_key, **ag_params},
            train_params={"task": "regression", "loss": "mse"},
        )
        model.data_params["train"] = {"seq_len": AG_INPUT_LEN, "bin_size": AG_BIN_SIZE}
        model.model_params["crop_len"] = 0
        self._model_cfg = ModelConfig(name=f"alphagenome_{output_key}", seq_len=AG_INPUT_LEN, bin_size=AG_BIN_SIZE)
        return model

    # What's the difference between cage and rna seq?
    def setup_ag_rna(self):
        self.ag_rna = self._setup_alpha_genome("rna_seq")

    def setup_ag_cage(self):
        self.ag_cage = self._setup_alpha_genome("cage")

    def get_borzoi_transform(self):
        tasks_b = pd.DataFrame(self.borzoi.data_params["tasks"])
        b_on = tasks_b[
            (tasks_b.assay == "RNA") & tasks_b["sample"].str.contains("brain", case=False, na=False)].index.tolist()
        b_off = tasks_b[
            (tasks_b.assay == "RNA") & tasks_b["sample"].str.contains("liver", case=False, na=False)].index.tolist()
        b_bins = self.borzoi.input_intervals_to_output_bins(self.target_exons, start_pos=self.borzoi_start_coord)
        b_pos = sorted(list(set(sum([list(range(row.start, row.end)) for row in b_bins.itertuples()], []))))
        # Application of grelu functions
        return grelu.transforms.prediction_transforms.Specificity(
            on_tasks=b_on, off_tasks=b_off, positions=b_pos, on_aggfunc="mean", off_aggfunc="mean",
            length_aggfunc="mean", compare_func="divide"
        )

    def get_ag_transform(self, model):
        ag_meta = pd.read_parquet(AG_META_PATH)
        rna_meta_h = ag_meta[ag_meta.output_type == "rna_seq"]
        ag_on = rna_meta_h[rna_meta_h.biosample_name.str.contains("brain", case=False, na=False)].track_index.tolist()
        ag_off = rna_meta_h[rna_meta_h.biosample_name.str.contains("liver", case=False, na=False) & (
                    rna_meta_h.assay_title == "polyA plus RNA-seq")].track_index.tolist()
        ag_bins = model.input_intervals_to_output_bins(self.target_exons[
                                                           (self.target_exons.start < self.ag_start_coord + 131072) & (
                                                                       self.target_exons.end > self.ag_start_coord)],
                                                       start_pos=self.ag_start_coord)
        ag_pos = sorted(
            list(set(sum([list(range(max(0, row.start), min(1024, row.end))) for row in ag_bins.itertuples()], []))))
        # Application of grelu functions
        return grelu.transforms.prediction_transforms.Specificity(
            on_tasks=ag_on, off_tasks=ag_off, positions=ag_pos, on_aggfunc="mean", off_aggfunc="mean",
            length_aggfunc="mean", compare_func="divide"
        )


@dataclass
class ModelConfig:
    """Per-model constants required by every downstream task.

    Adding support for a new model means defining one ``ModelConfig`` instance
    in the corresponding ``setup_*`` method.  Task methods automatically adapt
    through ``seq_len`` / ``bin_size`` rather than per-model ``if`` branches.
    """
    name: str
    seq_len: int
    bin_size: int

    @property
    def output_bins(self) -> int:
        """Total output bins: seq_len // bin_size."""
        return self.seq_len // self.bin_size
