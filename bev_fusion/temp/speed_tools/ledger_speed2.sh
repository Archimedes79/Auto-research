#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
L="python3 ../autoresearch/tools/ledger.py"
D="diagnostic, no verdict (Mirko's mini measurement)"
$L note speed.a --result "70 steps, 7.7 s/step median on a contended box (other jobs): NOT a baseline, replaced by speed.a3" --decision "$D" | tail -n 1
$L note speed.a3 --result "baseline HEAD fp32 on a calm box: 2.66 s/step median (first rows 2.0-2.1), data wait 4 %, GPU sm util 45 % mean (nvidia-smi dmon), peak 8.8 GB, grad_norm median 34" --decision "$D" | tail -n 1
$L note speed.b --result "HEAD + one read-back per clip + 30-step torch.profiler: same loss trajectory as a3 (first loss 21.60 both, last 6 mean 22.23 vs 22.27), steps 50-70 2.27-2.65 s vs a3 2.44-3.14 (inside the spread); profile: GPU kernels busy 33 % (34.9 s of 104 s; the 67 % in its profile.txt double counts, fixed in the profiler), main thread CPU busy 85 %, per step 691 .item(), 1283 stream syncs, 1779 memcpy; the profile window cost ~3 min to close and 22 GB RAM, I killed the process in teardown after the checkpoint (not run.py's exit)" --decision "$D" | tail -n 1
$L note speed.c --result "launcher stopped by me before a run started; repeated as speed.c2" --decision "$D" | tail -n 1
$L note speed.a2 --result "launcher stopped by me before a run started; repeated as speed.a3" --decision "$D" | tail -n 1
$L note speed.c2 --result "HEAD + speed code + bf16 + cudnn.benchmark: 2.71 s/step (no speed-up against a3 2.66), peak 5.3 GB (-40 %), GPU busy 36 % in the 8-step profile, grad_norm median 865 (a3 34, max 12858), loss mean of last 6 +8 %: bf16 as it is changes the training, not usable" --decision "$D" | tail -n 1
