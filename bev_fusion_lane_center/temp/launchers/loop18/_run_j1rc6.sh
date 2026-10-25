#!/bin/bash
# E219.j1rc6: recall weight 6 (instead of 3) on the fast j1 192 / 96, seed 0, fork pruning 0.4 at evaluation; starts when the two equal-budget controls are finished (runs next to E231.a)
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
SETJ=(reach=null spf2=false temporal=false stage_b=false stage_c=true b=3 mcl=true 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' detach_pos=false 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' warm=true revisit=false 'constr="L"' D=1 M=192 K0=96 seed_gscale=1.0 tolerant=false local_r=12.0 junc=true)
until [ -f $R/runs/e18/rc_eval/f1eq.json ] && [ -f $R/runs/e18/rc_eval/j1eqs1.json ]; do sleep 20; done
sleep 30
python3 $R/tools/experiment/run.py --id E219.j1rc6 --gpu-max-mb 9000 --retries 1 --basis E219.j1rc --seed 0 \
  --hyp "recall weight 6 on the fast j1 closes more of the lanes-found gap to the control (0.800) than weight 3 (0.736) without losing Q" \
  --pred "lanes found 0.76 +- 0.02, Q 0.737 +- 0.008 at fork pruning 0.4, VAL" \
  --rule "supersedes j1rc as the fast recall variant if Q >= j1rc + 0.005 and lanes found >= 0.76; otherwise j1rc stays (rc2 on f1 bought recall with precision, Q flat)" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_j1rc6 --u 4 --seed 0 --init e123_D7_r3_j1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set "${SETJ[@]}" --w rec=6.0 --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_j1rc6.log 2>&1
bash $R/tools/env/_py14.sh p18_variant_prune.py --name e18_D7_j1rc6 --M 192 --K0 96 --weight 6 --tag j1rc6 --thr 0.3,0.4 --json_thr 0.4 > $R/runs/e18/rc_eval_j1rc6.log 2>&1
cat $R/runs/e18/rc_eval_j1rc6.log
