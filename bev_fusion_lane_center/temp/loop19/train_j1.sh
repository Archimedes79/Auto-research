#!/bin/bash
# One 4 u fine-tune of j1 192/96 with the recipe of E219.j1eq / j1rc (lr 5e-4, stem lr 0.3, warm_frac 0.4), then its evaluation on VAL 873 (evaluate.py, registry name).
#   bash temp/loop19/train_j1.sh <run name> <seed> <rec weight> <registry name> [more p16_train.py args...]
# Environment passes through (LCR_DIST_DIR for the exact distance map, P19_REC_CLAMP, P18_FAR_W). --save_at 0.25,0.5,0.75 keeps the intermediate weights (E241).
set -o pipefail
cd "$(dirname "$0")/../.."
NAME=$1; SEED=$2; REC=$3; REG=$4; shift 4
SETJ=(reach=null spf2=false temporal=false stage_b=false stage_c=true b=3 mcl=true 'ver_mode="survival"' 'link_mode="learned"' 'pi_mode="min"' detach_pos=false 'dedup="geo"' 'ctx="global"' 'pattern="teardrop"' 'seeds="lattice"' warm=true revisit=false 'constr="L"' D=1 M=192 K0=96 seed_gscale=1.0 tolerant=false local_r=12.0 junc=true)
mkdir -p runs/e19/logs
echo "env: LCR_DIST_DIR=${LCR_DIST_DIR:-default} P19_REC_CLAMP=${P19_REC_CLAMP:-3.0} P18_FAR_W=${P18_FAR_W:-0}"
bash tools/env/_py14.sh p16_train.py --design D7 --name "$NAME" --u 4 --seed "$SEED" --init e123_D7_r3_j1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set "${SETJ[@]}" --w rec=$REC --warm_frac 0.4 --eval val250 --save_at 0.25,0.5,0.75 "$@" > runs/e19/logs/train_$NAME.log 2>&1 || { tail -n 15 runs/e19/logs/train_$NAME.log; exit 1; }
tail -n 2 runs/e19/logs/train_$NAME.log | cut -c1-300
if [ -n "$REG" ]; then bash tools/env/_py14.sh evaluate.py "$REG" 2>&1 | tail -n 14; fi
