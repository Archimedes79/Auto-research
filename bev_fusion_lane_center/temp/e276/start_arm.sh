#!/bin/bash
# E276: one job through the launcher.  usage: start_arm.sh c|t|s|tq|sq SEED
#   c  = score the baseline B (E273 run) on GT l4 with the tier numbers       t  = train arm t (tier loss)            tq = score arm t
#   s  = train arm s (tier loss + sparse node target)                           sq = score arm s (against the sparse GT nodes)
cd "$(dirname "$0")/../.."
ARM=$1; S=$2; shift 2
HYP="E276: the E273 lane head trained with the l4 tier labels in the loss (positives x tier weight, unseen GT nodes not demanded, false nodes near the own lane cost more; arm t) and on a sparse node target (arm s) delivers a better graph in the own lane and its neighbours that a camera sees, at a lower node density"
PRED="t12_F1_1 (F1 at 1.0 m on tier 1 + 2, visible) +0.04 over B on both seeds, tier-1 recall within 0.02, global topo F1 -0.01 to -0.05; s: delivered nodes per 100 m 31.5 -> 20 or below with t12_F1_1 within 0.02 of t; false nodes in zone 1 fall by 25 %; end / start events in tier 1 +0.05"
RULE="E276 card section 0: t against B KEEP when t12_F1_1 gains > +0.02 on both seeds and t1_R_1 >= c - 0.02, topo_F1 >= c - 0.05, assignment lane_acc_exact p-g >= -0.01 and lane_acc_err p-g >= +0.03; s against t KEEP when the delivered nodes per 100 m fall >= 25 % on both seeds with t12_F1_1 within 0.02 below t and the same guards; by hand (verdict276), VAL 873, final checkpoint, in-sample weights, no TEST"
TRAIN_COMMON="--workers 3 --phases 3 4 5 6 7 --resume-from $PWD/runs/e273_p_s${S}_p2/model.pt --gate-phases 3 --gate-node-prec 0.75 --gate-node-rec 0.5 --tier-w 3 1.5 1 --vis-w 0 --neg-w 2 1.5 1 1 --link-tier-w 3 1.5 1"
case $ARM in
  c)  ID=E276.c_s$S;  CMD="bash tools/env/_py14.sh qgraph_run_eval.py score --run e273_p_s$S --seed $S --exp e276 --dump-full";;
  t)  ID=E276.t_s$S;  CMD="bash tools/env/_pyr.sh qgraph_train.py --name e276_t_s$S --seed $S $TRAIN_COMMON";;
  tq) ID=E276.tq_s$S; CMD="bash tools/env/_py14.sh qgraph_run_eval.py score --run e276_t_s$S --seed $S --exp e276 --dump-full";;
  s)  ID=E276.s_s$S;  CMD="bash tools/env/_pyr.sh qgraph_train.py --name e276_s_s$S --seed $S $TRAIN_COMMON --sparse 0.3 7.5";;
  sq) ID=E276.sq_s$S; CMD="bash tools/env/_py14.sh qgraph_run_eval.py score --run e276_s_s$S --seed $S --exp e276 --sparse 0.3 7.5 --dump-full";;
  mt) ID=E276.mt_s$S; CMD="bash tools/env/_pyr.sh qgraph_train.py --name e276_smoke_t_s$S --seed $S --smoke --no-gates --workers 2 --phases 3 4 5 6 7 --resume-from $PWD/runs/e273_p_s${S}_p2/model.pt --tier-w 3 1.5 1 --vis-w 0 --neg-w 2 1.5 1 1 --link-tier-w 3 1.5 1"; NOCARD=1;;
  ms) ID=E276.ms_s$S; CMD="bash tools/env/_pyr.sh qgraph_train.py --name e276_smoke_s_s$S --seed $S --smoke --no-gates --workers 2 --phases 3 4 5 6 7 --resume-from $PWD/runs/e273_p_s${S}_p2/model.pt --tier-w 3 1.5 1 --vis-w 0 --neg-w 2 1.5 1 1 --link-tier-w 3 1.5 1 --sparse 0.3 7.5"; NOCARD=1;;
  *) echo "arm c, t, tq, s, sq, mt or ms"; exit 2;;
esac
[ -f runs/e273_p_s${S}_p2/model.pt ] || { echo "missing runs/e273_p_s${S}_p2/model.pt"; exit 3; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [ "$NOCARD" = 1 ]; then
  python3 ../autoresearch/tools/run.py --id $ID "$@" --seed $S --gpu --retries 0 --no-card -- $CMD            # a diagnostic smoke: never feeds a verdict
else
  python3 ../autoresearch/tools/run.py --id $ID "$@" --seed $S --gpu --retries 0 --require-card --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- $CMD
fi
