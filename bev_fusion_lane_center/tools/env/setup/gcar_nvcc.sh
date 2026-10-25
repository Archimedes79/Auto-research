#!/usr/bin/env bash
# nvcc 12.9 + Header nur zum Bauen des Gauss-Rasterizers in die Umgebung 'gcar' (conda-forge, ohne Meta-Pakete); cublas/cusparse/cusolver-Header kommen aus den pip-Paketen von PyTorch
source "$HOME/miniconda3/etc/profile.d/conda.sh"
mv -f "$HOME/work/gaussiancar_ckpt/nuscenes_map.ckpt.part" "$HOME/work/gaussiancar_ckpt/nuscenes_map.ckpt" 2>/dev/null
conda install -y -n gcar --override-channels -c conda-forge "cuda-version=12.9" "cuda-nvcc=12.9" "cuda-cudart-dev" "cuda-cccl" "cuda-nvrtc-dev" "gxx_linux-64=13" "$@" 2>&1 | tail -25
ls -la "$HOME/miniconda3/envs/gcar/bin/nvcc" 2>&1
