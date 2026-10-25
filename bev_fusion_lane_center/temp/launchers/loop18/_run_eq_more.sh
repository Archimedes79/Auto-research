#!/bin/bash
# equal-budget controls, two jobs in parallel: E219.f1eq (control of rc1: f1 + the same 4 u fine-tune at the default recall weight 1) and E219.j1eqs1 (seed 1 of the j1 control j1eq); each is evaluated pruned on VAL after its training
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
SETJ=(reach=null spf2=false temporal=false stage_b=false stage_c=true b=3 mcl=true 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' detach_pos=false 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' warm=true revisit=false 'constr="L"' D=1 M=192 K0=96 seed_gscale=1.0 tolerant=false local_r=12.0 junc=true)
SETF=(reach=null spf2=false junc=true temporal=true stage_b=false stage_c=true b=3 mcl=true 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' detach_pos=false 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' warm=true revisit=false 'constr="L"' D=1 seed_gscale=1.0 tolerant=false local_r=12.0 M=288 K0=144)
(
python3 $R/tools/experiment/run.py --id E219.f1eq --gpu-max-mb 9500 --retries 1 --basis E219 --seed 0 \
  --hyp "the Q / J gain of rc1 over the pruned champion is partly extra fine-tuning: f1 fine-tuned for the same 4 u at recall weight 1 gains lanes found nothing but some Q" \
  --pred "lanes found <= f1 + 0.02 (<= 0.715) at fork pruning 0.4, Q within +0.006 of f1 + pruning (0.735)" \
  --rule "report; separates the recall weight from extra fine-tuning for the 288 / 144 chain" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_f1eq --u 4 --seed 0 --init e123_D7_r3_f1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set "${SETF[@]}" --w rec=1.0 --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_f1eq.log 2>&1
bash $R/tools/env/_py14.sh p18_variant_prune.py --name e18_D7_f1eq --M 288 --K0 144 --weight 1 --tag f1eq --thr 0.3,0.4 --json_thr 0.4 > $R/runs/e18/rc_eval_f1eq.log 2>&1
) &
sleep 60
(
python3 $R/tools/experiment/run.py --id E219.j1eqs1 --gpu-max-mb 9000 --retries 1 --basis E219.j1eq --seed 1 \
  --hyp "the equal-budget control j1eq is reproducible (second fine-tune seed)" \
  --pred "lanes found 0.674 +- 0.02, Q 0.735 +- 0.005, J 0.417 +- 0.02 at fork pruning 0.4" \
  --rule "report; j1rc - j1eq is then judged on two control seeds" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_j1eqs1 --u 4 --seed 1 --init e123_D7_r3_j1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set "${SETJ[@]}" --w rec=1.0 --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_j1eqs1.log 2>&1
bash $R/tools/env/_py14.sh p18_variant_prune.py --name e18_D7_j1eqs1 --M 192 --K0 96 --weight 1 --tag j1eqs1 --thr 0.3,0.4 --json_thr 0.4 > $R/runs/e18/rc_eval_j1eqs1.log 2>&1
) &
wait
grep prune $R/runs/e18/rc_eval_f1eq.log $R/runs/e18/rc_eval_j1eqs1.log
