#!/bin/bash
# run a project script in the gcar env:  _py14.sh <script.py | folder/script.py> [args...]
# the script is found by name in training/ evaluation/ tools/ reports/build/ temp/ (a relative or absolute path works too); cwd = the folder of the script
# a name found more than once is refused (name its folder, e.g. lane_grid/selftest.py); the resolved path goes to stderr, so it lands in the run log
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # the problem folder, from the location of this script (the repository can be renamed or moved)
export BFLC_ROOT="$R"
s="$1"; shift
if [ ! -f "$s" ]; then
  n="${s##*../}"
  case "$n" in */*) f=$(find $R/training $R/evaluation $R/tools $R/reports/build $R/temp -path "*/$n" -type f ! -path '*_session_scratch_*' 2>/dev/null);; *) f=$(find $R/training $R/evaluation $R/tools $R/reports/build $R/temp -name "$n" -type f ! -path '*_session_scratch_*' 2>/dev/null);; esac
  [ -n "$f" ] || { echo "_py14.sh: script not found: $s" >&2; exit 2; }
  [ "$(echo "$f" | wc -l)" -eq 1 ] || { echo "_py14.sh: $s is ambiguous, name its folder:" $(echo "$f" | sed "s#$R/##") >&2; exit 2; }
  s="$f"
fi
echo "_py14.sh: ${s#$R/}" >&2
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
source $R/tools/env/bflc_env.sh
cd "$(dirname "$s")"
python "$(basename "$s")" "$@"
