#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh -c "print(1)" 2>&1 | tail -1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate gcar
export BFLC_ROOT=$PWD
python - <<'PY'
import py_compile, sys
py_compile.compile("training/train/bench_token_step.py", doraise=True)
sys.path.insert(0, "training/train")
import importlib.util
spec = importlib.util.spec_from_file_location("bts", "training/train/bench_token_step.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("import ok", m.code_state())
PY
