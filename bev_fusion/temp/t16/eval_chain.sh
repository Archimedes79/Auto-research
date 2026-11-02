#!/bin/bash
# scratch of the t16 bring-up: the evaluation chain of a finished run for the report, on an idle GPU (it runs as a job of the GPU line):
# timing (idle guard), the numbers on 1000 val frames, the module probe, three dashboard videos. usage: eval_chain.sh runs/<name>
cd "$(dirname "$0")/../.." || exit 1
RUN=$1
PY=/home/ml/miniconda3/envs/gcar/bin/python
echo "== timing $(date +%T)"
$PY evaluation/bev_suite.py "$RUN" timing --n 30 || echo "timing refused or failed (not fatal)"
echo "== numbers $(date +%T)"
$PY evaluation/bev_suite.py "$RUN" --curves --fusion --velocity eval --n 1000 --prior 500 --images 8 || exit 1
echo "== generalisation (train against val) $(date +%T)"
for try in 1 2; do $PY evaluation/bev_suite.py "$RUN" generalisation --n 1000 && break; echo "generalisation failed (try $try)"; rm -rf "$RUN/eval/suite/quant/eval_train_n1000"; done
echo "== modules $(date +%T)"
$PY evaluation/bev_suite.py "$RUN" modules --frames 4 --device cuda || echo "modules failed (not fatal)"
echo "== videos $(date +%T)"
$PY evaluation/bev_suite.py "$RUN" qual videos 3 --seed 1 || echo "videos failed (not fatal)"
echo "== done $(date +%T)"
