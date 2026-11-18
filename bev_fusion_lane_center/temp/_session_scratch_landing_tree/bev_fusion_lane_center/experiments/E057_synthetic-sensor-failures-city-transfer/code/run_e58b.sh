#!/usr/bin/env bash
# E58b: Feinabstimmung des Boston-Kopfes auf wenigen Singapur-Zeilen (N = 100, 400, 1600): wie viel Zieldaten schliessen die Luecke zwischen den Orten?
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
until [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e60b.done ]; do sleep 30; done
for N in 100 400 1600; do
  case $N in 100) ST=400 ;; 400) ST=800 ;; 1600) ST=1500 ;; esac
  echo "== $(date +%H:%M:%S) fine-tune B -> Singapore, N=$N, steps=$ST"
  ./run_gcar.sh dense_head.py --name e58b_N$N --init_from e58_B --fg gcar --feats dec32,logit7 --steps $ST --bs 8 --lr 2e-4 --beta2 0.98 --eps 1e-6 --workers 2 --dir 0.5 --lam 1 --wpos 1 --train_loc 1,3 --train_rows $N --eval val_b,val_s,test_b,test_s --seed 41 2>&1 | grep -E "^\[|FERTIG|Traceback|Error"
done
echo "== $(date +%H:%M:%S) done"
