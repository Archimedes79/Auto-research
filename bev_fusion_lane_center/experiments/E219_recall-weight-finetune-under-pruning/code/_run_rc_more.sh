#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E219.j1rc: recall-weighted fine-tune (weight 3, 4 u) of the single-frame fast configuration j1 192 / 96; E219.rc1s1: seed-1 replicate of rc1 (noise of the recall variant)
R=$BFLC
SETJ=(reach=null spf2=false temporal=false stage_b=false stage_c=true b=3 mcl=true 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' detach_pos=false 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' warm=true revisit=false 'constr="L"' D=1 M=192 K0=96 seed_gscale=1.0 tolerant=false local_r=12.0 junc=true)
SETF=(reach=null spf2=false junc=true temporal=true stage_b=false stage_c=true b=3 mcl=true 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' detach_pos=false 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' warm=true revisit=false 'constr="L"' D=1 seed_gscale=1.0 tolerant=false local_r=12.0 M=288 K0=144)
python3 $R/tools/experiment/run.py --id E219.j1rc --gpu-max-mb 9000 --retries 1 --basis E219,E215 --seed 0 \
  --hyp "recall weight 3 on the fast single-frame configuration j1 192 / 96 gives a fast variant with more lanes found" \
  --pred "lanes found >= 0.72 (j1 + pruning 0.671) with Q >= 0.735 (0.728) at fork pruning 0.4, VAL" \
  --rule "adopt as the fast recall variant if Q >= j1 + pruning + 0.005 and lanes found >= j1 + pruning + 0.04; otherwise j1 stays" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_j1rc --u 4 --seed 0 --init e123_D7_r3_j1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set "${SETJ[@]}" --w rec=3.0 --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_j1rc.log 2>&1 &
sleep 60
python3 $R/tools/experiment/run.py --id E219.rc1s1 --gpu-max-mb 9500 --retries 1 --basis E219 --seed 1 \
  --hyp "the recall variant rc1 is reproducible: a second training seed gives the same Q and lanes found within the noise" \
  --pred "Q 0.740 +- 0.005, lanes found 0.767 +- 0.02 (fork pruning 0.4)" --rule "report; no selection" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_rc1s1 --u 4 --seed 1 --init e123_D7_r3_f1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set "${SETF[@]}" --w rec=3.0 --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_rc1s1.log 2>&1 &
wait
