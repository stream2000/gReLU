#!/usr/bin/env bash
# Usage: source activate.sh

# This script should be sourced, not executed directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Please run this script using: source activate.sh"
    exit 1
fi

# Ensure conda is initialized (resolves CommandNotFoundError in non-interactive shells)
source /work/miniconda3/etc/profile.d/conda.sh

conda activate grelu_dev

# Prefer this checkout's source tree while reusing the shared conda env.
_GRELU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_GRELU_ROOT}/src:${PYTHONPATH:-}"

# nvjitlink fix: pip-installed nvidia packages need to take precedence over
# system /usr/local/cuda/lib64, whose libnvJitLink.so.12 is too old.
_NVJIT="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nvjitlink/lib"
if [[ -d "$_NVJIT" ]]; then
    export LD_LIBRARY_PATH="${_NVJIT}:${LD_LIBRARY_PATH:-}"
fi
unset _NVJIT _GRELU_ROOT

echo "✅ Activated conda environment: grelu_dev"
