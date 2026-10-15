#!/usr/bin/env bash
# Usage: source activate.sh

# This script should be sourced, not executed directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Please run this script using: source activate.sh"
    exit 1
fi

# Ensure conda is initialized (resolves CommandNotFoundError in non-interactive shells).
# CONDA_SH and GRELU_CONDA_ENV can override these defaults for host-specific runs.
_CONDA_SH="${CONDA_SH:-}"
if [[ -z "$_CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
    _CONDA_BASE="$(conda info --base)"
    _CONDA_SH="${_CONDA_BASE}/etc/profile.d/conda.sh"
fi
if [[ -z "$_CONDA_SH" ]]; then
    _CONDA_SH="/work/miniconda3/etc/profile.d/conda.sh"
fi
if [[ ! -r "$_CONDA_SH" ]]; then
    echo "Could not find conda initialization script: ${_CONDA_SH}" >&2
    return 1
fi
source "$_CONDA_SH"

_GRELU_CONDA_ENV="${GRELU_CONDA_ENV:-${HOME}/.conda/envs/grelu_dev}"
if [[ -x "${_GRELU_CONDA_ENV}/bin/python" ]]; then
    conda activate "$_GRELU_CONDA_ENV"
else
    conda activate grelu_dev
fi

# Prefer this checkout's source tree while reusing the shared conda env.
_GRELU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ALPHAGENOME_SRC="${_GRELU_ROOT}/src/alphagenome_pytorch/src"
if [[ -d "$_ALPHAGENOME_SRC" ]]; then
    export PYTHONPATH="${_GRELU_ROOT}/src:${_ALPHAGENOME_SRC}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${_GRELU_ROOT}/src:${PYTHONPATH:-}"
fi

# CUDA library fix: pip-installed nvidia packages need to take precedence over
# system /usr/local/cuda/lib64, whose libnvJitLink.so.12 may be too old.
_NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
if [[ -d "$_NVIDIA_LIB_ROOT" ]]; then
    for _d in "$_NVIDIA_LIB_ROOT"/*/lib; do
        [[ -d "$_d" ]] && export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
    done
fi
unset _d _NVIDIA_LIB_ROOT _ALPHAGENOME_SRC _GRELU_ROOT _GRELU_CONDA_ENV _CONDA_BASE _CONDA_SH

echo "Activated conda environment: ${CONDA_PREFIX}"
