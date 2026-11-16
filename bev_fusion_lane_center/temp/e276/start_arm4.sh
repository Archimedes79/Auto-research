#!/bin/bash
# E279 jobs (the soft close-node step at radius 1.25 m), the night's scheduler.  usage: start_arm4.sh w|wq SEED
#   w  = train the head with the soft step at 1.25 m (phases 3-7 from the E273 phase-2 checkpoint, the E273 recipe, --soft-nms 1.25 0.1 3)
#   wq = score it: read-outs plain@0.5 (the judged one) and close:1.5@0.5 (reported)
cd "$(dirname "$0")/../.."
ARM=$1; S=$2; shift 2
HYP="E279: the E273 lane head trained with the soft close-node step at radius 1.25 m (smoothness 0.1 m, 3 passes) delivers at e >= 0.5, without a hard step, a graph as good as the baseline with the hard step of E275 and keeps more recall than the 1.5 m step of E277: topo F1 and junction F1 above the baseline, the guards (F1@1.0, lanes found, assignment) held, topo F1 not below the hard step by more than 0.01"
PRED="topo F1 0.62 +- 0.01 on both seeds (baseline + 0.025); junction F1 0.20 or more; lanes found 0.62 - 0.65; F1@1.0 within 0.01 of the baseline; vertices per 100 m 20 - 24; exact-map assignment p-g >= -0.01"
RULE="E279 card section 0 = the E277 rule unchanged: verdict277 (topo F1 gain > +0.02 and junction F1 gain > +0.03 on both seeds over the baseline, guards held: lane_acc_exact p-g >= -0.01, lane_acc_err p-g >= +0.03, F1@1.0 >= c - 0.01, lanes_found >= c - 0.03; topo F1 not below the hard step by more than 0.01 on either seed): KEEP / TIE / CUT / INCONCLUSIVE; by hand, VAL 873, final checkpoint, in-sample, no TEST"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
case $ARM in
  w)  ID=E279.w_s$S
      [ -f runs/e273_p_s${S}_p2/model.pt ] || { echo "missing runs/e273_p_s${S}_p2/model.pt"; exit 3; }
      CMD="bash tools/env/_pyr.sh qgraph_train.py --name e279_w_s$S --seed $S --workers 3 --phases 3 4 5 6 7 --resume-from $PWD/runs/e273_p_s${S}_p2/model.pt --gate-phases 3 --gate-node-prec 0.75 --gate-node-rec 0.5 --soft-nms 1.25 0.1 3";;
  wq) ID=E279.wq_s$S
      CMD="bash tools/env/_py14.sh qgraph_run_eval.py score --run e279_w_s$S --seed $S --exp e279 --readouts plain@0.5 close:1.5@0.5";;
  *) echo "arm w or wq"; exit 2;;
esac
python3 ../autoresearch/tools/run.py --id $ID "$@" --seed $S --gpu --retries 0 --require-card --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- $CMD
