#!/usr/bin/env bash
# Schritt 2 der Umgebung 'gcar': nvcc (conda-forge) installieren, Gauss-Rasterizer bauen, Umgebung pruefen. Protokoll: out/gcar_env2.log
source "$HOME/miniconda3/etc/profile.d/conda.sh"
echo "=== $(date +%H:%M:%S) nvcc installieren"
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/scripts/gcar_nvcc.sh
conda activate gcar
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=6
export CPATH="$SP/nvidia/cublas/include:$SP/nvidia/cusparse/include:$SP/nvidia/cusolver/include:$SP/nvidia/curand/include:$SP/nvidia/cuda_runtime/include:$SP/nvidia/cuda_nvrtc/include:$CONDA_PREFIX/targets/x86_64-linux/include"
which nvcc; nvcc --version | tail -2
echo "=== $(date +%H:%M:%S) Rasterizer bauen"
cd "$HOME/work/gaussiancar"
pip install --no-build-isolation ./gaussiancar/ops/diff-gaussian-rasterization 2>&1 | tail -15
echo "=== $(date +%H:%M:%S) Pruefung"
python - <<'PY'
import torch, spconv.pytorch, torch_scatter, timm, diff_gaussian_rasterization
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("spconv ok, torch_scatter ok, timm", timm.__version__, "rasterizer ok")
PY
echo "GCAR-SCHRITT2 FERTIG $(date +%H:%M:%S)"
