# gReLU + AlphaGenome — Quick Start

All shared materials live in `/work/gReLU/`:
- `env/` — reference conda environment (~7 GB)
- `weights/` — pre-downloaded model weights (Borzoi + AlphaGenome + Enformer, ~3 GB)

Each user sets up their **own** git clone and conda env following the steps below **in order**.

---

## Quick verification (no setup needed)

To check that the shared environment and GPU work before committing to the full setup:

```bash
source /work/miniconda3/etc/profile.d/conda.sh
conda activate /work/gReLU/env
export HF_HOME=/work/gReLU/weights
_NVJIT="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nvjitlink/lib"
[[ -d "$_NVJIT" ]] && export LD_LIBRARY_PATH="${_NVJIT}:${LD_LIBRARY_PATH:-}"
python -c "
import torch, grelu, alphagenome_pytorch
print('grelu:', grelu.__version__)
print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())
"
```

If that passes, proceed with the full setup below.

---

## 1. Clone the code

`--recurse-submodules` is required — without it `src/alphagenome_pytorch/` will be empty and installs will fail.

```bash
git clone --recurse-submodules -b alphagenome \
    https://github.com/stream2000/gReLU.git ~/gReLU
```

## 2. Clone the shared conda env

```bash
source /work/miniconda3/etc/profile.d/conda.sh
conda create -p ~/.conda/envs/grelu_dev --clone /work/gReLU/env -y
conda activate grelu_dev
```

## 3. Reinstall editable packages (point to your clone)

The cloned env's internal package paths still point to the original developer's home directory, which you cannot read. This step fixes them.

```bash
pip uninstall -y alphagenome-pytorch gReLU
pip install -e ~/gReLU/src/alphagenome_pytorch
pip install -e ~/gReLU
```

## 4. Configure shared genome data (one-time)

Scripts need hg38 genome sequences and annotations (FASTA + GTF). Point genomepy to the shared copy:

```bash
genomepy config set genomes_dir /work/gReLU/genomes
```

## 6. Activate every session

```bash
source ~/gReLU/activate.sh
export HF_HOME=/work/gReLU/weights
```

---

## Run examples

All scripts must be run from `~/gReLU/`:

```bash
cd ~/gReLU
mkdir -p ~/results
```

### Gene expression inference
```bash
python -m scripts.run_inference --gene SRSF11 --devices 0
# Output: comparison_cage.png, comparison_specificity.png
```

### eQTL AUROC — single tissue
```bash
python scripts/eqtl/run_eqtl_auroc.py \
    --model alphagenome --tissue brain_cortex \
    --score sum --rc --rf --devices 0 \
    --manifest /work/gReLU/eqtl_data/paper_vcfs/manifest.tsv \
    --output ~/results/ag_brain_cortex.tsv
```

The `--tissue` argument does substring matching (`brain`, `liver`, `heart` all work).

### Genome-wide Pred-vs-Obs evaluation
```bash
python -m scripts.run_eval \
    --eval alphagenome \
    --eval_bigwig /path/to/signal.bw \
    --eval_track_idx 0 \
    --eval_chroms chr1 chr8 chr21 \
    --devices 0 \
    --eval_output ~/results/eval_ag.json
```

---

## Notes

- GPU memory: AlphaGenome uses ~12 GB per GPU.
- Write all outputs to `~/results/` — `/work/gReLU/` is read-only for non-owners.
- If you get `HF_HUB_OFFLINE` errors: `unset HF_HUB_OFFLINE HF_HUB_ENABLE_HF_TRANSFER`.
- Multi-GPU: replace `--devices 0` with `--devices 0,1,2,3` as needed.
