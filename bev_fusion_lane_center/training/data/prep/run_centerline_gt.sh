#!/usr/bin/env bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bev
python $BFLC/training/data/prep/centerline_gt.py \
  --dataroot "$HOME/data/nuscenes" --version v1.0-mini \
  --out $REPO/out/centerline_gt --debug 8 2>&1 | grep -v -i -E 'warning|np\.|deprecat' | tail -20
