#!/bin/bash
# E338 scoring (CPU, niced): per seed the stack read-out sc (own keypoints snapped on c2's graph, tag e338), the control on B's dump (tag e338b: must reproduce the stored card of s), the caveat read-out
# (attached ends, tag e338a), the own-keypoint nogeom split (tag e338n); then the paired scene bootstraps. usage: score_all.sh
cd "$(dirname "$0")/../.."
R=$(pwd)
L=temp/e338/score_all.log
for S in 0 1; do
  C2D=$R/temp/e292/predfull_e292_c2_s$S.pkl; C2M=$R/runs/e292_c2_s$S/model.pt
  BD=$R/temp/e276/predfull_e273_p_s$S.pkl; BM=$R/runs/e273_p_s$S/model.pt
  nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e324_s_s$S --seed $S --snap --thr 0.1 --full $C2D --base $C2M --spec close:1.5@0.5 --tag e338 > temp/e338/score_sc_s$S.log 2>&1; echo "sc s$S exit $? $(date)" >> $L
  nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e324_s_s$S --seed $S --snap --thr 0.1 --full $BD --base $BM --spec close:1.5@0.5 --tag e338b > temp/e338/score_ctl_s$S.log 2>&1; echo "control s$S exit $? $(date)" >> $L
done
for S in 0 1; do
  C2D=$R/temp/e292/predfull_e292_c2_s$S.pkl; C2M=$R/runs/e292_c2_s$S/model.pt
  nice -n 19 bash tools/env/_py14.sh jdiag.py boot e324_s_s$S ends_card_e338_snap_thr0.1 e324_s_s$S ends_card_snap_thr0.1 > temp/e338/boot_sc_vs_s_s$S.log 2>&1
  nice -n 19 bash tools/env/_py14.sh jdiag.py boot e324_s_s$S ends_card_e338_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 > temp/e338/boot_sc_vs_c_s$S.log 2>&1
  echo "boots s$S $(date)" >> $L
done
for S in 0 1; do
  C2D=$R/temp/e292/predfull_e292_c2_s$S.pkl; C2M=$R/runs/e292_c2_s$S/model.pt
  nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e324_s_s$S --seed $S --attach --snap --snap-nogeom --thr 0.1 --full $C2D --base $C2M --spec close:1.5@0.5 --tag e338a > temp/e338/score_att_s$S.log 2>&1; echo "caveat attached s$S exit $? $(date)" >> $L
  nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e324_s_s$S --seed $S --snap --snap-nogeom --thr 0.1 --full $C2D --base $C2M --spec close:1.5@0.5 --tag e338n > temp/e338/score_nog_s$S.log 2>&1; echo "own nogeom s$S exit $? $(date)" >> $L
done
echo done > temp/e338/score_all.done
