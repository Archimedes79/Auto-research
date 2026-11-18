#!/bin/bash
# E387 after the training: (1) the QUIET head-alone timing f vs r vs z (clause 6; seed-0 models), (2) the stream dumps of both f arms in the two slots, (3) the CPU read-outs (every3, sweep, own cell, scores) of both arms in parallel
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
say() { echo "$(date '+%m-%d %H:%M:%S') [e387 after] $*" >> temp/e387/chain.log; }
RUN="python3 ../autoresearch/tools/run.py"
say "timing f / r / z"
$RUN --id diag.e387_time --seed 0 --gpu --ram-mb 6000 --retries 0 --no-card --split val --timeout 15 -- bash tools/env/_pyr.sh jpast_eval.py time --runs e387_f_s0,e368_r_s0,e368_z_s0 --frames 24 --repeats 4 >> temp/e387/after.out 2>&1
say "timing exit $?"
bash temp/e387/read387.sh dump f_s0 >> temp/e387/after.out 2>&1 &
bash temp/e387/read387.sh dump f_s1 >> temp/e387/after.out 2>&1 &
wait
say "dumps done"
bash temp/e387/read387.sh cpu f_s0 >> temp/e387/after.out 2>&1 &
bash temp/e387/read387.sh cpu f_s1 >> temp/e387/after.out 2>&1 &
wait
say "cpu read-outs done"
