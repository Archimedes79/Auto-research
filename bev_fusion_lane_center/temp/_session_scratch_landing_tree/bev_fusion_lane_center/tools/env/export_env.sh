#!/bin/bash
# write the package list of the working environment 'gcar' next to this script (read-only look at the environment)
D="$(cd "$(dirname "$0")" && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
{
  echo "# environment gcar - exported $(date '+%Y-%m-%d %H:%M')"
  echo "# python $(python -V 2>&1 | cut -d' ' -f2)   torch $(python -c 'import torch;print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null)"
  echo "# GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  echo "# --- pip freeze"
  pip freeze 2>/dev/null
} > "$D/environment_gcar.txt"
conda env export --no-builds 2>/dev/null > "$D/environment_gcar.yml" || true
wc -l "$D/environment_gcar.txt" "$D/environment_gcar.yml" 2>/dev/null
