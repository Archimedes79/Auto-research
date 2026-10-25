#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E209 (internal e142) RERUN with checkpoints: UNSEEN failure types. Map builder B2 and no-map control C2 are trained on real perturbed TRAIN rows of
# fog, radar_off, extr3, cam_front_off, cam_front3_off ONLY (night and blur held out), then evaluated on the VAL failure series of night and blur.
# One training at a time (two trainings used 11.6 of 12 GB and the first run died with a CUDA unknown error); every training resumes from runs/e123/<name>/ckpt.pt.
R=$BFLC
L=$R/runs/e17/logs
FP=cam_front3_off,fog,radar_off,extr3,cam_front_off
RUN="python3 $R/tools/experiment/run.py --gpu-max-mb 9000 --retries 3 --basis E206,E139"
for arm in b2 c2; do
  d=$R/runs/e123/e17_map_$arm
  [ -f $d/log.txt ] && [ ! -f $d/log_crashed_run1.txt ] && [ ! -f $d/ckpt.pt ] && mv $d/log.txt $d/log_crashed_run1.txt
done
$RUN --id E209.b2 --seed 0 --hyp "map builder trained without night/blur rows still repairs night/blur" --result $R/runs/e123/e17_map_b2/status.json -- \
  bash $R/tools/env/_py14.sh p17_mapft.py --name e17_map_b2 --init_ckpt e17_map_a --real_fail 1 --share 0.5 --u 4 --lr 3e-4 --p_degrade 0.2 --fail_perts $FP > $L/e209_train_b2.log 2>&1 || exit 1
$RUN --id E209.c2 --seed 0 --hyp "no-map control for E209.b2 (same data, map always zero)" --result $R/runs/e123/e17_map_c2/status.json -- \
  bash $R/tools/env/_py14.sh p17_mapft.py --name e17_map_c2 --init_ckpt e17_map_a --real_fail 1 --share 0.5 --u 4 --lr 3e-4 --p_degrade 0.2 --p_none 1.0 --p_wrong 0.0 --fail_perts $FP > $L/e209_train_c2.log 2>&1 || exit 1
$RUN --id E209.eval --seed 0 --hyp "evaluation of E209.b2 / c2 on the held-out night and blur series" -- bash $R/temp/launchers/loop18/_eval_e209.sh > $L/e209_eval.log 2>&1 || exit 1
echo finished > $L/e209.done
