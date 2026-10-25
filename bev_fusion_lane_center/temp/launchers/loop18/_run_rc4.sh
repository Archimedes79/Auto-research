#!/bin/bash
# E219.rc4: recall-weighted fine-tune with an extra weight on ground-truth points beyond 25 m (P18_FAR_W = 3: weight 1 + 3 x ramp(25 -> 40 m)); recall weight 3 as rc1
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
export P18_FAR_W=3.0
python3 $R/tools/experiment/run.py --id E219.rc4 --gpu-max-mb 9000 --retries 1 --basis E219,E217 --seed 0 \
  --hyp "a recall weight that grows with the range closes the far-range recall gap with less precision loss than a uniform weight" \
  --pred "lanes found >= 0.78 with Q >= 0.740 and P@1 >= 0.87 (fork pruning 0.4, VAL)" \
  --rule "adopt as the recall variant if Q >= rc1 + 0.005 (0.745) and lanes found >= 0.78; otherwise rc1 stays" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_rc4 --u 4 --seed 0 --init e123_D7_r3_f1 --lr 0.0005 --stem_lr 0.3 --workers 4 \
  --set 'reach=null' 'spf2=false' 'junc=true' 'temporal=true' 'stage_b=false' 'stage_c=true' 'b=3' 'mcl=true' 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' 'detach_pos=false' 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' 'warm=true' 'revisit=false' 'constr="L"' 'D=1' 'seed_gscale=1.0' 'tolerant=false' 'local_r=12.0' 'M=288' 'K0=144' \
  --w rec=3.0 --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_rc4.log 2>&1
