#!/bin/bash
# t21 (D51): the numbers of one finished arm through the front door, then the timing on the idle GPU (the run.py slot is held, the next arm waits for this run):
#   bash evaluation/eval_arm.sh runs/<name>      (from anywhere; run.py starts it as t21.<arm>.eval with --after <arm>)
cd "$(dirname "$0")/.." || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
run=$1
$PY evaluation/evaluate.py "$run" --quant --curves --fusion --velocity --n 1000
rc1=$?
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
$PY evaluation/evaluate.py "$run" --time
rc2=$?
echo "numbers exit $rc1, timing exit $rc2 $(date +%T)"
# the carried state: (1) the memory command (cold against warm on the same frames: the clip-trained condition, the state of the previous seq_len - 1 keyframes
# first), (2) a warm evaluation on whole scenes (the state carried along the scene: the unbounded stream; every 7th val scene, 12 scenes; compare with --tag _warm)
$PY evaluation/bev_suite.py "$run" memory --clips 400 > $run/eval/memory_cmd.txt 2>&1
echo "memory exit $? (eval/memory.json)"
scenes=$($PY -c "import json; v = json.load(open('$run/config.json'))['val_scenes']; print(' '.join(v[::7][:12]))")
$PY evaluation/bev_suite.py "$run" --fusion --velocity eval --warm --scenes $scenes > $run/eval/warm_cmd.txt 2>&1
echo "warm eval exit $? ($scenes)"; tail -n 3 $run/eval/warm_cmd.txt | cut -c1-200
# the stream diagnostics of the buffered arms (evaluation/stream_drift.py, stream_timing.py): the drift of the stream against the batch references on a val scene, once the
# synthetic timing of the stream paths (random weights, idle GPU); a failure here is listed but does not change the exit code of the numbers
buf=$($PY -c "import json,sys; print(json.load(open('$run/config.json'))['model'].get('cam_buffer', 'window'))")
if [ "$buf" != "window" ]; then
  ck=$(ls -t $run/ckpt_*.pt | head -1)
  $PY evaluation/stream_drift.py --config $run/config.json --ckpt $ck --scene scene-0003 --frames 50 --device cuda > $run/eval/stream_drift.txt 2>&1
  echo "stream drift exit $? ($buf, $ck)"; tail -n 16 $run/eval/stream_drift.txt | cut -c1-160
fi
mkdir -p temp/stream
if [ ! -f temp/stream/timing_full_cuda.json ]; then
  $PY evaluation/stream_timing.py --size full --device cuda > temp/stream/timing_full_cuda.txt 2>&1
  echo "stream timing exit $?"; tail -n 14 temp/stream/timing_full_cuda.txt | cut -c1-160
fi
[ $rc1 -eq 0 ] && exit $rc2
exit $rc1
