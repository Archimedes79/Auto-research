#!/bin/bash
# E277 jobs (the soft close-node step), added to the night's scheduler.  usage: start_arm3.sh u|uq SEED
#   u  = train the head with the soft step (phases 3-7 from the E273 phase-2 checkpoint, the E273 recipe, no tier weights, --soft-nms 1.5 0.1 3)
#   uq = score it: read-outs plain@0.5 (the judged one: the step is inside the network) and close:1.5@0.5 (reported)
cd "$(dirname "$0")/../.."
ARM=$1; S=$2; shift 2
HYP="E277: the E273 lane head trained with the close-node suppression as a soft step in its forward path (radius 1.5 m, smoothness 0.1 m, 3 passes) delivers at e >= 0.5, without a hard step, a graph as good as the baseline with the hard step of E275: topo F1 and junction F1 above the baseline read out as before, topo F1 not below the hard step by more than 0.01, the assignment guards held"
PRED="topo F1 0.62 +- 0.01 on both seeds (baseline + 0.025, hard step 0.626 / 0.625); junction F1 0.22 or more; lanes found 0.60 - 0.64; vertices per 100 m 18 - 22; exact-map assignment p-g >= -0.01"
RULE="E277 card section 0: verdict277 = verdict275 (soft step against the baseline: topo F1 gain > +0.02 and junction F1 gain > +0.03 on both seeds, guards held: lane_acc_exact p-g >= -0.01, lane_acc_err p-g >= +0.03, F1@1.0 >= c - 0.01, lanes_found >= c - 0.03) and the topo F1 not below the hard step by more than 0.01 on either seed: KEEP (replaces the hard step) / TIE (helps, hard step stays) / CUT / INCONCLUSIVE; by hand, VAL 873, final checkpoint, in-sample, no TEST"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
case $ARM in
  u)  ID=E277.u_s$S
      [ -f runs/e273_p_s${S}_p2/model.pt ] || { echo "missing runs/e273_p_s${S}_p2/model.pt"; exit 3; }
      CMD="bash tools/env/_pyr.sh qgraph_train.py --name e277_u_s$S --seed $S --workers 3 --phases 3 4 5 6 7 --resume-from $PWD/runs/e273_p_s${S}_p2/model.pt --gate-phases 3 --gate-node-prec 0.75 --gate-node-rec 0.5 --soft-nms 1.5 0.1 3";;
  uq) ID=E277.uq_s$S
      CMD="bash tools/env/_py14.sh qgraph_run_eval.py score --run e277_u_s$S --seed $S --exp e277 --readouts plain@0.5 close:1.5@0.5";;
  *) echo "arm u or uq"; exit 2;;
esac
python3 ../autoresearch/tools/run.py --id $ID "$@" --seed $S --gpu --retries 0 --require-card --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- $CMD
