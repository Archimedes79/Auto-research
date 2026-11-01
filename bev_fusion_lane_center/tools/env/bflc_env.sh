# source this after "conda activate gcar": module search path of the project (flat imports: import graph_eval as GE ...)
R="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # the problem folder, from the location of this file
export BFLC_ROOT="$R"
_P=""
for _d in $(find $R/training $R/evaluation $R/tools $R/reports/build $R/temp -maxdepth 4 -type d ! -name __pycache__ ! -path '*_session_scratch_*' 2>/dev/null); do _P="$_P:$_d"; done   # session scratch (copies of modules, folders named like packages: coverage/) stays off the path
export PYTHONPATH="${_P#:}:${PYTHONPATH:-}"
