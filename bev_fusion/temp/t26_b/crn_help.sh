cd ~/work/CRN || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
timeout 300 nice -n 15 ~/miniconda3/envs/crn/bin/python exps/det/CRN_r50_256x704_128x128_4key.py --help 2>&1 | head -40
echo "exit: $?"
~/miniconda3/envs/crn/bin/python - <<'PY' 2>&1 | head
import torch
print(torch.__version__, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
PY
