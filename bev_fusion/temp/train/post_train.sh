#!/usr/bin/env bash
# One-off: when the combined training has finished (.done), run the whole evaluation chain without a gap:
#   1 latency of both runs on the idle GPU  2 full VAL evaluation of the combined run (maps, boxes, prior, images)
#   3 warm / cold memory test  4 training curves  5 result tables.
# Called as `bash temp/train/post_train.sh t1_r_post`: the argument keeps "t1_r" in the command line, so queued jobs of other
# chats (they wait for pgrep -f "[t]1_r" to find nothing) keep waiting until this pipeline is finished.
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
until [ -f runs/t1_r3_combined/.done ] || ! pgrep -f "train.py.*t1_r3_combined" > /dev/null; do sleep 10; done
echo "training ended $(date +%T); .done: $([ -f runs/t1_r3_combined/.done ] && echo yes || echo NO)"
[ -f runs/t1_r3_combined/.done ] || { echo "NO .done: stopping the pipeline"; exit 1; }
sleep 20
echo "== latency $(date +%T)"
$PY -u evaluation/evaluate_run.py runs/t1_r0 --time-only > temp/train/post_time_r0.log 2>&1
$PY -u evaluation/evaluate_run.py runs/t1_r3_combined --time-only > temp/train/post_time_final.log 2>&1
echo "== full VAL evaluation $(date +%T)"
$PY -u evaluation/evaluate_run.py runs/t1_r3_combined --prior 500 --images 8 --workers 6 --batch 4 > temp/train/post_eval_final.log 2>&1
echo "== memory test $(date +%T)"
$PY -u evaluation/eval_memory.py runs/t1_r3_combined --clips 600 --workers 6 > temp/train/post_memory.log 2>&1
echo "== curves and tables $(date +%T)"
mkdir -p experiments/t1_full_chain/results experiments/t1_full_chain/images
$PY temp/train/plot_curves.py experiments/t1_full_chain/images/curves.png t1_r0 t1_r3_combined > temp/train/post_curves.log 2>&1
$PY temp/train/build_tables.py runs/t1_r0 runs/t1_r3_combined > experiments/t1_full_chain/results/tables.md 2> temp/train/post_tables.log
echo "pipeline finished $(date +%T)"
