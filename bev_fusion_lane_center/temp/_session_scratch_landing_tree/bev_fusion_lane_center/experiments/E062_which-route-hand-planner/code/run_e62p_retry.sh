#!/usr/bin/env bash
# E62p: fehlgeschlagene Bedingungen wiederholen (Graphen pruefen, ggf. neu erzeugen; dann plan_eval) - stored, cam_front_off, blur, fog
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
OUTD=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
for P in stored cam_front_off blur fog; do
  if [ "$P" = "fog" ]; then until [ -f $HOME/exp_gcar_b123/pert/fog/DONE ] && [ -f $OUTD/e57_fog/graphs_test.pkl ]; do sleep 30; done; fi
  for TRY in 1 2 3; do
    [ -f $OUTD/e62p_$P/persample_test.pkl ] && break
    if ! ./run_gcar.sh _chk_pkl.py $OUTD/e57_$P/graphs_test.pkl 2>&1 | grep -q "^OK"; then
      echo "== $(date +%H:%M:%S) $P: Graphen neu (eval_pert)"
      ./run_gcar.sh eval_pert.py --pert $P --workers 3 2>&1 | grep -E "^\[|Traceback|Error"
    fi
    echo "== $(date +%H:%M:%S) plan_eval $P try $TRY"
    ./run_gcar.sh plan_eval.py --sets test --graphs e57_$P/graphs --sigma 1.0 --reg 1 --workers 3 --out e62p_$P 2>&1 | grep -E "^\[|Traceback|Error" | cut -c1-200
  done
done
echo "== $(date +%H:%M:%S) done"
