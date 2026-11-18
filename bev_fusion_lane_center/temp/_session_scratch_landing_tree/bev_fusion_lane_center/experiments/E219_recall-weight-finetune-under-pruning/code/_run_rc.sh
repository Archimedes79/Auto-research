#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# R2.6 / E219: recall-weighted fine-tune of the champion re-opened under fork pruning (loop 16b run rc1: weight 3, S 0.731 / J 0.182 / lanes found 0.768; the J loss was false forks, which pruning removes).
# usage: _run_rc.sh <name suffix> <rec weight>   -> runs/e123/e18_D7_<suffix>/model.pt, same recipe as rc1 (4 u from the champion f1, lr 5e-4, stem lr 0.3, beam 288 / 144)
R=$BFLC
sfx=$1; w=$2
python3 $R/tools/experiment/run.py --id E219.$sfx --gpu-max-mb 9000 --retries 2 --basis E184,E215 --seed 0 \
  --hyp "recall weight $w during the fine-tune lifts lanes found and Q once pruning removes the false forks" -- \
  bash $R/tools/env/_py14.sh p16_train.py --design D7 --name e18_D7_$sfx --u 4 --seed 0 --init e123_D7_r3_f1 --lr 0.0005 --stem_lr 0.3 --workers 4 \
  --set 'reach=null' 'spf2=false' 'junc=true' 'temporal=true' 'stage_b=false' 'stage_c=true' 'b=3' 'mcl=true' 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' 'detach_pos=false' 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' 'warm=true' 'revisit=false' 'constr="L"' 'D=1' 'seed_gscale=1.0' 'tolerant=false' 'local_r=12.0' 'M=288' 'K0=144' \
  --w rec=$w --warm_frac 0.4 --eval val250 > $R/runs/e17/logs/e219_$sfx.log 2>&1
