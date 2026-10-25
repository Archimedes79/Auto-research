#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E219.j1eq8: the plain fine-tune of j1 at twice the budget (8 u = 9600 steps, recall weight 1): does extra training keep helping the plain fast variant? Evaluation chained.
R=$BFLC
SETJ=(reach=null spf2=false temporal=false stage_b=false stage_c=true b=3 mcl=true 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' detach_pos=false 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' warm=true revisit=false 'constr="L"' D=1 M=192 K0=96 seed_gscale=1.0 tolerant=false local_r=12.0 junc=true)
python3 $R/tools/experiment/run.py --id E219.j1eq8 --gpu-max-mb 9000 --retries 1 --basis E219.j1eq --seed 0 \
  --hyp "extra fine-tuning keeps helping the plain fast variant: j1 fine-tuned at weight 1 for 8 u gains Q over the 4 u control j1eq" \
  --pred "Q >= j1eq + 0.003 (0.738), lanes found within +-0.01 of j1eq (0.674), J >= 0.417" \
  --rule "if Q >= j1eq + 0.005 the 4 u budget was short and 8 u becomes the plain fast variant (report); otherwise j1eq stays (report only)" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_j1eq8 --u 8 --seed 0 --init e123_D7_r3_j1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set "${SETJ[@]}" --w rec=1.0 --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_j1eq8.log 2>&1
bash $R/tools/env/_py14.sh $R/experiments/E219_recall-weight-finetune-under-pruning/code/p18_variant_prune.py --name e18_D7_j1eq8 --M 192 --K0 96 --weight 1 --tag j1eq8 --thr 0.3,0.4 --json_thr 0.4 > $R/runs/e18/rc_eval_j1eq8.log 2>&1
cat $R/runs/e18/rc_eval_j1eq8.log
