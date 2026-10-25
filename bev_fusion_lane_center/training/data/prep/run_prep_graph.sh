#!/usr/bin/env bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bev
cd "$HOME/work/bevfusion"
export PYTHONWARNINGS=ignore
python $BFLC/training/data/prep/prep_graph.py "$HOME/exp/graph" 2>&1 | grep -v -i -E 'warn|np\.|deprecat|Loaded|=====|^Done|seconds'
