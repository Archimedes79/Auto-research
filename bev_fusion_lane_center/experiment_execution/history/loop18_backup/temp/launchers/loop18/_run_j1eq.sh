#!/bin/bash
# E219.j1eq: equal-budget control of j1rc - the same 4 u fine-tune of j1 192 / 96 (same recipe, seed 0, warm_frac 0.4) with the DEFAULT recall weight 1; separates the recall weight from the extra fine-tune steps. Evaluation chained below.
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
SETJ=(reach=null spf2=false temporal=false stage_b=false stage_c=true b=3 mcl=true 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' detach_pos=false 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' warm=true revisit=false 'constr="L"' D=1 M=192 K0=96 seed_gscale=1.0 tolerant=false local_r=12.0 junc=true)
python3 $R/tools/experiment/run.py --id E219.j1eq --gpu-max-mb 9000 --retries 1 --basis E219.j1rc --seed 0 \
  --hyp "the gain of j1rc comes from the recall weight, not from the extra fine-tune steps: j1 fine-tuned for the same 4 u at recall weight 1 gains much less" \
  --pred "lanes found <= j1 + 0.02 (<= 0.69) and Q <= j1 + 0.005 at fork pruning 0.4, VAL" \
  --rule "report; if lanes found of j1eq >= j1rc - 0.02 the gain is a fine-tune effect and the report says so (j1rc stays a valid fast variant but not a recall-weight result)" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_j1eq --u 4 --seed 0 --init e123_D7_r3_j1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set "${SETJ[@]}" --w rec=1.0 --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_j1eq.log 2>&1
bash $R/tools/env/_py14.sh p18_variant_prune.py --name e18_D7_j1eq --M 192 --K0 96 --weight 1 --tag j1eq --thr 0.3,0.4 --json_thr 0.4 > $R/runs/e18/rc_eval_j1eq.log 2>&1
cat $R/runs/e18/rc_eval_j1eq.log
