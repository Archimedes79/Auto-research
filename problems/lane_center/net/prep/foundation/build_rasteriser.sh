#!/usr/bin/env bash
# The Gaussian rasteriser of the GaussianCaR clone, built in place for the active python environment (gcar: its export holds nvcc 12.9 and
# gcc 13; the rasteriser itself is no package of the export):
#     conda activate gcar && bash build_rasteriser.sh <clone>          (or GCAR_REPO)
# 1. adds `#include <cstdint>` to the six headers of cuda_rasterizer/ that lack it (gcc 13 does not compile them otherwise; the only change of
#    the clone), 2. builds diff_gaussian_rasterization/_C*.so next to its __init__.py, where gcar_features.py looks for it, 3. imports it.
# TORCH_CUDA_ARCH_LIST (default 8.6: the RTX 3060) and MAX_JOBS (default 2) can be set.
set -euo pipefail
G="${1:-${GCAR_REPO:-}}"
[ -n "$G" ] || { echo "build_rasteriser.sh: name the GaussianCaR clone (argument or GCAR_REPO)" >&2; exit 2; }
R="$G/gaussiancar/ops/diff-gaussian-rasterization"
[ -d "$R/cuda_rasterizer" ] || { echo "build_rasteriser.sh: no rasteriser source at $R" >&2; exit 1; }
{ [ -n "${CONDA_PREFIX:-}" ] && command -v nvcc >/dev/null; } || {
  echo "build_rasteriser.sh: activate the environment first (conda activate gcar): no nvcc" >&2; exit 1; }
cd "$R/cuda_rasterizer"
for f in rasterizer_impl.h auxiliary.h forward.h backward.h rasterizer.h config.h; do
  grep -q "<cstdint>" "$f" && continue
  if grep -q "#pragma once" "$f"; then sed -i '0,/#pragma once/s//#pragma once\n#include <cstdint>/' "$f"; else sed -i '1i #include <cstdint>' "$f"; fi
  echo "patched: $f"
done
cd "$R"
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" MAX_JOBS="${MAX_JOBS:-2}"
INC="$CONDA_PREFIX/targets/x86_64-linux/include"
for lib in cublas cusparse cusolver curand cuda_runtime cuda_nvrtc; do INC="$SP/nvidia/$lib/include:$INC"; done      # the CUDA headers of torch's pip packages
export CPATH="$INC${CPATH:+:$CPATH}"
python setup.py build_ext --inplace
python -c "import sys; sys.path.insert(0, '.'); import torch, diff_gaussian_rasterization as d; print('rasteriser ok:', d.__file__)"
