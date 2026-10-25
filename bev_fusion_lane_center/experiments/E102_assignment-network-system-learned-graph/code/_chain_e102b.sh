#!/bin/bash
# E102b: the assignment network (E87 recipe) RETRAINED on learned graphs. Training graphs: the learned graphs of fold-0 train rows (the trace network saw these rows in training: in-sample, not cross-fitted).
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
LOG=$L/e102b.log
echo "start $(date)" > $LOG
bash _py.sh p10_tosrc2.py --tok tok_e94_a_oof0 --fold 0 >> $LOG 2>&1
bash _py.sh p8_dataset.py --plan train --loc tmpl --srcprefix l --perts "" --tag _segL --draws 2 --p_wrong 0.1 --p_none 0.1 --p_bias 0.5 --bias_sigmas 1,2,3 --workers 6 >> $LOG 2>&1
bash run_gcar.sh p8_assign.py --name e102b --tag _segL --dev_tag _segL --epochs 8 --abstain 0.3 --test 1 >> $LOG 2>&1
echo "done $(date)" >> $LOG
touch $L/e102b.done
