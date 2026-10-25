#!/bin/bash
# run a project script in the gcar env:  _py14.sh <script.py | folder/script.py> [args...]
# the script is found by name in training/ evaluation/ tools/ reports/build/ temp/ (a relative or absolute path works too); cwd = the folder of the script
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # the problem folder, from the location of this script (the repository can be renamed or moved)
export BFLC_ROOT="$R"
s="$1"; shift
if [ ! -f "$s" ]; then
  n="${s##*../}"
  case "$n" in */*) f=$(find $R/training $R/evaluation $R/tools $R/reports/build $R/temp -path "*/$n" -type f 2>/dev/null | head -1);; *) f=$(find $R/training $R/evaluation $R/tools $R/reports/build $R/temp -name "$n" -type f 2>/dev/null | head -1);; esac
  [ -n "$f" ] && s="$f" || { echo "_py14.sh: script not found: $s" >&2; exit 2; }
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
source $R/tools/env/bflc_env.sh
cd "$(dirname "$s")"
python "$(basename "$s")" "$@"
