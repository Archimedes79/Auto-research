#!/usr/bin/env bash
# run a script of this problem with the interpreter that has torch:  py.sh <script.py> [args]     (cwd = the problem folder)
#   TOY_PY    the interpreter (default: the gcar environment of this machine, autoresearch/server.md row 2)
#   AR_ROOT   autoresearch/ (trainkit, Evaluation; default: ../../autoresearch from this problem; a copy of the problem elsewhere sets it)
HERE=$(cd "$(dirname "$0")/.." && pwd)
AR=${AR_ROOT:-$(cd "$HERE/../../autoresearch" && pwd)}
PY=${TOY_PY:-$HOME/miniconda3/envs/gcar/bin/python}
export PYTHONPATH="$HERE:$AR${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
cd "$HERE" || exit 2
exec "$PY" "$@"
