#!/usr/bin/env bash
# scratch: the final checks after the last commit; prints one line per check
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="$HOME/miniconda3/envs/gcar/bin/python"
one() { local name=$1; shift; local out; out=$("$@" 2>&1); local rc=$?; echo "$name rc=$rc | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-170)"; }
one "equivalence" "$PY" tests/test_trainkit_equivalence.py
one "trainkit selftest" bash -c "cd ../autoresearch && $PY -m trainkit.selftest"
one "trainsuite selftest" bash -c "cd ../autoresearch && python3 -m trainsuite selftest"
one "startable" "$PY" training/startable.py
one "layers" "$PY" tests/test_layers.py
one "frozen frame" "$PY" tests/test_frozen.py
one "slots" "$PY" tests/test_slots.py
echo ALLDONE
