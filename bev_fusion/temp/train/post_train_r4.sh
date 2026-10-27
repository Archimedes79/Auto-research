#!/usr/bin/env bash
# One-off: when t1_r4_soa has finished (.done): latency, full VAL evaluation cold and warm (carried memory over whole scenes),
# memory test, curves, tables against the earlier runs, architecture drawing. Called as `bash temp/train/post_train_r4.sh t1_r_post4`
# (the argument keeps "t1_r" in the command line: other chats' GPU waiters stay gated).
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
R=runs/t1_r4_soa
until [ -f $R/.done ] || ! pgrep -f "train.py.*t1_r4_soa" > /dev/null; do sleep 10; done
echo "training ended $(date +%T); .done: $([ -f $R/.done ] && echo yes || echo NO)"
[ -f $R/.done ] || { echo "NO .done: stopping the pipeline"; exit 1; }
sleep 20
E=experiments/t1_r4_soa
mkdir -p $E/results $E/images
echo "== cold VAL evaluation $(date +%T)"
$PY -u evaluation/evaluate_run.py $R --prior 500 --images 8 --workers 6 --batch 4 > temp/train/post4_eval_cold.log 2>&1
echo "== warm VAL evaluation $(date +%T)"
$PY -u evaluation/evaluate_run.py $R --warm --workers 6 > temp/train/post4_eval_warm.log 2>&1
echo "== memory test $(date +%T)"
$PY -u evaluation/eval_memory.py $R --clips 600 --workers 6 > temp/train/post4_memory.log 2>&1
echo "== latency (needs an idle GPU) $(date +%T)"
$PY -u evaluation/evaluate_run.py $R --time-only > temp/train/post4_time.log 2>&1
echo "== curves, tables, drawing $(date +%T)"
$PY temp/train/plot_curves.py $E/images/curves.png t1_r0 t1_r3_combined t1_r4_soa > temp/train/post4_curves.log 2>&1
$PY temp/train/build_tables.py runs/t1_r0 $R > $E/results/tables_vs_r0.md 2> temp/train/post4_tables.log
$PY temp/train/build_tables.py runs/t1_r3_combined $R > $E/results/tables_vs_r3.md 2>> temp/train/post4_tables.log
$PY temp/train/draw_architecture_r4.py $E/images/architecture.png > /dev/null 2>&1
echo "pipeline finished $(date +%T)"
