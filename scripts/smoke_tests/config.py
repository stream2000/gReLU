"""Global constants shared across all benchmark experiments.

All experiment scripts import from here so there is a single source of
truth for biological / model-spec constants.  Changing any value here
changes the assay definition — update experiments accordingly.
"""
import os

# ── Output directory ──────────────────────────────────────────────────────────
ISM_RESULTS_DIR = "ism_results"
os.makedirs(ISM_RESULTS_DIR, exist_ok=True)

# ── Model weights / metadata ──────────────────────────────────────────────────
WEIGHTS_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gtca--alphagenome_pytorch/"
    "snapshots/b01c0ffa73e07c053491f3b5ea8bcf67d93b9920/model_fold_0.safetensors"
)

AG_META_PATH = (
    "src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet"
)

# ── Architecture constants (biological / model-spec requirements) ─────────────
BORZOI_INPUT_LEN = 524_288   # Borzoi receptive field in bp
BORZOI_BIN_SIZE  = 32        # Borzoi output bin resolution in bp

AG_INPUT_LEN  = 131_072      # AlphaGenome receptive field in bp
AG_BIN_SIZE   = 128          # AlphaGenome output bin resolution in bp
AG_OUTPUT_BINS = AG_INPUT_LEN // AG_BIN_SIZE  # 1024

ISM_HALF_WIDTH = 100         # ±100 bp window around ISM centre
