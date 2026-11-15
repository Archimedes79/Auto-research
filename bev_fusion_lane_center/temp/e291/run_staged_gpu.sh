#!/bin/bash
# E291: run a project script with the STAGED copies of the frozen lane files in front of the real ones (the E343 code digest is frozen), on the GPU (nothing disabled, no nice: this is the wrapper of the run.py lines):
#   bash temp/e291/run_staged_gpu.sh <script.py | basename> [args ...]        (cwd = the problem folder; run.py starts it)
# the environment is the one of tools/env/_py14.sh (gcar, bflc_env.sh PYTHONPATH); the finder is temp/e347/stage_launch.py (E347's, generic) with BFLC_STAGE = temp/_session_scratch_e291/stage.
# the stage folder lies in a `_session_scratch_` folder so that `_py14.sh` and `bflc_env.sh` of the other chats do not see two copies of one script name; the launcher prints the staged modules that were loaded
# (`run_staged: stage = ... | staged modules loaded: ...` on stderr, so it is in the run log) and refuses a module with a staged copy that was loaded from elsewhere.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BFLC_ROOT="$R"
export BFLC_STAGE="${BFLC_STAGE:-$R/temp/_session_scratch_e291/stage}"
[ "$BFLC_STAGE" = "none" ] || [ -f "$BFLC_STAGE/training/grid/jends.py" ] || { echo "run_staged_gpu.sh: no staged jends.py in $BFLC_STAGE: this run would use the REAL files" >&2; exit 2; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
CACHE="$R/temp/_session_scratch_e291/pythonpath.txt"
if [ "$E291_FAST" = "1" ] && [ -s "$CACHE" ]; then      # tests only: the PYTHONPATH of the last full call (bflc_env.sh walks the tree over the slow /mnt/c mount, about 55 s); never for a training
  export PYTHONPATH="$(cat "$CACHE")"
else
  source "$R/tools/env/bflc_env.sh"
  echo "$PYTHONPATH" > "$CACHE"
fi
exec python "$R/temp/e347/stage_launch.py" "$@"
