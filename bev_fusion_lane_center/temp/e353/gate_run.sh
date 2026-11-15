#!/bin/bash
# diagnostic null gate (by hand, no run.py): dump the null of seed 1 into temp/e352/diag_null_s1, score the null cards of both seeds (snap exists for seed 0), links runs/e352diag_null_s<S> -> the diag folders
cd "$(dirname "$0")/../.."
R=$PWD
[ -f temp/e352/diag_null_s1/point.json ] || nice bash tools/env/_py14.sh jprune_b.py dump --seed 1 --point null --out temp/e352/diag_null_s1 > temp/e353/gate_dump_s1.log 2>&1
for S in 0 1; do [ -e runs/e352diag_null_s$S ] || ln -s ../temp/e352/diag_null_s$S runs/e352diag_null_s$S; done
sc() { # <seed> <readout flags...>
  local S=$1; shift
  nice bash tools/env/_py14.sh jends_eval.py score --run e352diag_null_s$S --seed $S --full $R/runs/e352diag_null_s$S/predfull_null_s$S.pkl --base $R/runs/e273_p_s$S/model.pt --spec close:1.5@0.5 --tag null "$@" --thr 0.1 >> temp/e353/gate_score_s$S.log 2>&1
}
[ -f temp/e352/diag_null_s1/ends_card_null_snap_thr0.1.json ] || sc 1 --snap
[ -f temp/e352/diag_null_s0/ends_card_null_snap_nogeom_attach_thr0.1.json ] || sc 0 --attach --snap --snap-nogeom
[ -f temp/e352/diag_null_s0/ends_card_null_attach_thr0.1.json ] || sc 0 --attach
[ -f temp/e352/diag_null_s1/ends_card_null_snap_nogeom_attach_thr0.1.json ] || sc 1 --attach --snap --snap-nogeom
[ -f temp/e352/diag_null_s1/ends_card_null_attach_thr0.1.json ] || sc 1 --attach
echo "gate run done $(date)" > temp/e353/gate_run.done
