#!/bin/bash
# every selftest of the system on the final tree (the commands of the Tests block of autoresearch/README.md, plus the jobrunner and the drawings of the problems)
R=/mnt/c/Auto-research
L=$R/bev_fusion/temp/cleanup/selftests
mkdir -p $L
PY=$HOME/miniconda3/envs/gcar/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
run() {  # run <name> <dir> <command...>
  local name=$1 dir=$2; shift 2
  local t0=$(date +%s)
  ( cd $dir && nice -n 10 "$@" ) > $L/$name.log 2>&1
  local rc=$?
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s | $(grep -v -i warn $L/$name.log | tail -1 | cut -c1-150)" | tee -a $L/summary.txt
}
: > $L/summary.txt
date +%T | tee -a $L/summary.txt
run archdraw_autoresearch $R/autoresearch bash -c 'for s in architecture.json overview.json suites.json; do python3 tools/archdraw.py check $s || exit 1; done'
run archdraw_problems $R/autoresearch bash -c 'for s in ../bev_fusion/training/architecture.json ../bev_fusion_lane_center/training/architecture.json; do python3 tools/archdraw.py check $s || exit 1; done'
run install_agents $R/autoresearch python3 tools/install_agents.py --check
run suitecli $R/autoresearch python3 suitecli.py --selftest
run trainkit $R/autoresearch $PY -m trainkit.selftest
run jobrunner $R/autoresearch python3 -m jobrunner selftest
run evalsuite $R/autoresearch $PY -m evalsuite selftest
run tools_selftest $R python3 autoresearch/tools/selftest.py
run trainsuite $R/autoresearch python3 -m trainsuite selftest
run GtGenerator $R $PY -m GtGenerator selftest
date +%T | tee -a $L/summary.txt
