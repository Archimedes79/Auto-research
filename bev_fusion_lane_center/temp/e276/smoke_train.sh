#!/bin/bash
# scratch (E276 step 5): CPU smoke of the trainer with the tier switches (20 steps per phase, phases 3 and 4), arm t and arm s (sparse), on the stored l4 frames. Output: temp/e276/smoke_train.log
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
COMMON="--seed 0 --smoke --workers 0 --phases 3 4 --no-gates --resume-from $PWD/runs/e273_p_s0_p2/model.pt --tier-w 3 1.5 1 --vis-w 0 --neg-w 2 1.5 1 1 --link-tier-w 3 1.5 1"
{
  echo "== arm t smoke $(date +%H:%M)"
  nice -n 12 bash tools/env/_py14.sh qgraph_train.py --name e276_smoke_t $COMMON 2>&1 | tail -12
  echo "== arm s smoke $(date +%H:%M)"
  nice -n 12 bash tools/env/_py14.sh qgraph_train.py --name e276_smoke_s $COMMON --sparse 0.3 7.5 2>&1 | tail -12
  echo "== done $(date +%H:%M)"
} > temp/e276/smoke_train.log 2>&1
