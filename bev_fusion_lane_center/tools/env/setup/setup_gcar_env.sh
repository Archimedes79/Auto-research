#!/usr/bin/env bash
# Eigene Umgebung 'gcar' fuer GaussianCaR (ICRA 2026; Python 3.12, PyTorch 2.8 + CUDA 12.9). 'bev' (BEVFusion) und 'crn' (CRN) bleiben unberuehrt.
# Fortschritt: out/gcar_env.log   (Rasterizer wird mit nvcc 12.9 aus conda gebaut; flash-attn entfaellt, PTv3 laeuft mit enable_flash=False)
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
step "conda-Umgebung"
conda env list | grep -q '^gcar ' || conda create -y -n gcar --override-channels -c conda-forge python=3.12
conda activate gcar
python -m pip install -U pip wheel "setuptools>=77" 2>&1 | tail -1
step "PyTorch 2.8.0 + cu129"
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -2
step "CUDA-Compiler 12.9 (nur zum Bauen des Rasterizers)"
conda install -y -n gcar --override-channels -c nvidia/label/cuda-12.9.1 -c conda-forge cuda-nvcc cuda-cudart-dev cuda-cccl cuda-libraries-dev 2>&1 | tail -3
export CUDA_HOME="$CONDA_PREFIX"
ls "$CUDA_HOME/bin/nvcc" && "$CUDA_HOME/bin/nvcc" --version | tail -2
step "Python-Pakete"
pip install -f https://data.pyg.org/whl/torch-2.8.0+cu129.html "torch-scatter==2.1.2+pt28cu129" 2>&1 | tail -2
pip install spconv-cu126 addict efficientnet-pytorch einops fvcore hydra-core jaxtyping lightning "numpy<2" nuscenes-devkit peft pillow rootutils segmentation-models-pytorch timm transformers matplotlib opencv-python scikit-image imgaug scipy tqdm pyquaternion 2>&1 | tail -3
step "Gauss-Rasterizer bauen"
cd "$HOME/work/gaussiancar"
export TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=6
pip install --no-build-isolation ./gaussiancar/ops/diff-gaussian-rasterization 2>&1 | tail -5
step "Pruefung"
python - <<'PY'
import torch, spconv.pytorch, torch_scatter, timm, diff_gaussian_rasterization
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("spconv ok, torch_scatter ok, timm", timm.__version__, "rasterizer ok")
PY
echo "GCAR-UMGEBUNG FERTIG $(date +%H:%M:%S)"
