"""Apply the role-based code layout (training / evaluation / tools / reports/build / temp), rewrite references, write the new runner.  Run: _run.sh layout_apply.py [--go]
Every move is logged to MIGRATION_LOG.csv (step 'layout')."""
import json
import os
import re
import shutil
import sys
import time

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
RS = "/mnt/c/ML-Experiment/restructure"
SC = f"{R}/scripts"
GO = "--go" in sys.argv
plan = json.load(open(f"{RS}/layout.json"))
OVR = {"p16_pipeline_eval.py": "evaluation/drivers", "p16_route_diag.py": "evaluation/drivers", "p16_triage.py": "training/search", "p16_ctrl.py": "training/train", "p16_common.py": "training/data",
       "line_tracer.py": "training/pipeline", "test_p16_spec.py": "evaluation/tests", "p16_ref.py": "training/models", "p16_asm_grid.py": "temp/analysis", "p16_tok_resample.py": "training/pipeline"}
plan.update(OVR)
log = []


def mv(src, dst):
    if not os.path.lexists(src):
        return
    if os.path.lexists(dst):
        print("EXISTS", dst)
        return
    log.append((src, dst))
    if GO:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        open(f"{RS}/MIGRATION_LOG.csv", "a", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},layout,{src},{dst}\n")


where = {}  # file name -> new relative dir (for reference rewriting)
for f, d in sorted(plan.items()):
    if d == "temp":
        d = "temp/launchers" if re.match(r"^(_run_|_smoke_|_reeval|_py\.)", f) else "temp/analysis"
    where[f] = d
    mv(f"{SC}/{f}", f"{R}/{d}/{f}")
# archive and data_prep
if os.path.isdir(f"{SC}/_archive"):
    for g in sorted(os.listdir(f"{SC}/_archive")):
        for f in sorted(os.listdir(f"{SC}/_archive/{g}")):
            where[f] = f"temp/archive/{g}"
        mv(f"{SC}/_archive/{g}", f"{R}/temp/archive/{g}")
if os.path.isdir(f"{SC}/data_prep"):
    for f in os.listdir(f"{SC}/data_prep"):
        where[f] = "training/data/prep"
    mv(f"{SC}/data_prep", f"{R}/training/data/prep")
mv(f"{SC}/README.md", f"{R}/temp/_old_scripts_README.md")
mv(f"{SC}/__pycache__", f"{R}/_to_delete/scripts_junk/__pycache__2")
json.dump(where, open(f"{RS}/where.json", "w"), indent=0)
print(len(log), "moves", "done" if GO else "planned")

# ------------------------------------------------------------------ runners
if GO:
    os.makedirs(f"{R}/tools/env", exist_ok=True)
    open(f"{R}/tools/env/bflc_env.sh", "w", newline="\n").write('''# source this after "conda activate gcar": module search path of the project (flat imports: import graph_eval as GE ...)
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
_P=""
for _d in $(find $R/training $R/evaluation $R/tools $R/reports/build $R/temp -maxdepth 4 -type d ! -name __pycache__ 2>/dev/null); do _P="$_P:$_d"; done
export PYTHONPATH="${_P#:}:${PYTHONPATH:-}"
''')
    open(f"{R}/tools/env/_py14.sh", "w", newline="\n").write('''#!/bin/bash
# run a project script in the gcar env:  _py14.sh <script.py | folder/script.py> [args...]
# the script is found by name in training/ evaluation/ tools/ reports/build/ temp/ (a relative or absolute path works too); cwd = the folder of the script
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
s="$1"; shift
if [ ! -f "$s" ]; then
  n="${s##*../}"
  case "$n" in */*) f=$(find $R/training $R/evaluation $R/tools $R/reports/build $R/temp -path "*/$n" -type f 2>/dev/null | head -1);; *) f=$(find $R/training $R/evaluation $R/tools $R/reports/build $R/temp -name "$n" -type f 2>/dev/null | head -1);; esac
  [ -n "$f" ] && s="$f" || { echo "_py14.sh: script not found: $s" >&2; exit 2; }
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
source $R/tools/env/bflc_env.sh
cd "$(dirname "$s")"
python "$(basename "$s")" "$@"
''')
    open(f"{R}/tools/env/_pyr.sh", "w", newline="\n").write('''#!/bin/bash
# like _py14.sh, but repeats the run (up to 4 times) when it dies with a CUDA 'unknown error' (seen while a training shares the GPU): _pyr.sh script.py [args]
D="$(cd "$(dirname "$0")" && pwd)"
for i in 1 2 3 4; do
  bash $D/_py14.sh "$@" > /tmp/_pyr_$$.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then cat /tmp/_pyr_$$.log; rm -f /tmp/_pyr_$$.log; exit 0; fi
  if grep -q "CUDA error" /tmp/_pyr_$$.log; then echo "[retry $i after CUDA error]"; sleep 5; continue; fi
  cat /tmp/_pyr_$$.log; rm -f /tmp/_pyr_$$.log; exit $rc
done
cat /tmp/_pyr_$$.log; exit 1
''')
    open(f"{R}/tools/env/_py.sh", "w", newline="\n").write('''#!/bin/bash
# same as _py14.sh (kept for old wrappers); output of stderr merged
D="$(cd "$(dirname "$0")" && pwd)"
bash $D/_py14.sh "$@" 2>&1
''')
    for f in ("_py14.sh", "_pyr.sh", "_py.sh"):
        os.chmod(f"{R}/tools/env/{f}", 0o755)
