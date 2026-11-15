#!/bin/bash
# scratch: gate C0 after the two evaluations: the cards (signature, frame hash, frames, split), the stand-in --check, the GPU / process state
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
date
for r in t22_standin_all_pair4 t22_B_all; do
  echo "== $r"
  ls -la runs/$r/eval/ 2>&1 | head -12
  $PY - <<EOF
import json
c = json.load(open("runs/$r/eval/card.json"))
for k in ("gt_signature", "frame_hash", "frames", "n_frames", "split", "ckpt", "checkpoint"):
    if k in c:
        print(" ", k, c[k])
print("  keys:", sorted(c)[:30])
EOF
done
echo "== stand-in --check"
$PY temp/t22/chain/make_standin_run.py runs/t21_pair4_s0 runs/t22_standin_all_pair4 --check 2>&1 | tail -8
echo "== processes"
ps -eo pid,etime,args | grep -E 'evaluate|run.py|train.py|bev_suite' | grep -v grep | cut -c1-150
