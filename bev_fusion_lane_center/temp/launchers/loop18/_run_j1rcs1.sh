#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E219.j1rcs1: seed-1 replicate of j1rc (the adopted fast recall variant had one training seed); evaluation chained below
R=$BFLC
SETJ=(reach=null spf2=false temporal=false stage_b=false stage_c=true b=3 mcl=true 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' detach_pos=false 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' warm=true revisit=false 'constr="L"' D=1 M=192 K0=96 seed_gscale=1.0 tolerant=false local_r=12.0 junc=true)
python3 $R/tools/experiment/run.py --id E219.j1rcs1 --gpu-max-mb 9000 --retries 1 --basis E219.j1rc --seed 1 \
  --hyp "the fast recall variant j1rc is reproducible: a second fine-tune seed gives the same gains over j1" \
  --pred "lanes found 0.736 +- 0.02, Q 0.739 +- 0.005, J 0.447 +- 0.02 at fork pruning 0.4, VAL" \
  --rule "report; no selection (decides whether j1rc stays adopted: it stays if lanes found >= j1 + 0.04 and Q >= j1 + 0.005 on both seeds)" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_j1rcs1 --u 4 --seed 1 --init e123_D7_r3_j1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set "${SETJ[@]}" --w rec=3.0 --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_j1rcs1.log 2>&1
bash $R/tools/env/_py14.sh p18_variant_prune.py --name e18_D7_j1rcs1 --M 192 --K0 96 --weight 3 --tag j1rcs1 --thr 0.3,0.4 --json_thr 0.4 > $R/runs/e18/rc_eval_j1rcs1.log 2>&1
cat $R/runs/e18/rc_eval_j1rcs1.log
