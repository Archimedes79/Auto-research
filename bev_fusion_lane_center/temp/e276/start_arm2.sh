#!/bin/bash
# E276 night, second part: jobs added after arm s failed its gate (the first script, start_arm.sh, is not edited: a job may be running it).  usage: start_arm2.sh qp|... SEED
#   qp = diagnostic E278: score the phase-6 checkpoint of baseline B (the model before the joint refinement, phase 7), read-outs plain@0.5 and close:1.5@0.5, never a verdict (--no-card)
cd "$(dirname "$0")/../.."
ARM=$1; S=$2; shift 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
case $ARM in
  qp) ID=E278.qp_s$S
      [ -f runs/e273_p_s${S}_p6/model.pt ] || { echo "missing runs/e273_p_s${S}_p6/model.pt"; exit 3; }
      python3 ../autoresearch/tools/run.py --id $ID --seed $S --gpu --retries 0 --no-card -- bash tools/env/_py14.sh qgraph_run_eval.py score --run e273_p_s${S}_p6 --seed $S --exp e278 --readouts plain@0.5 close:1.5@0.5;;
  *) echo "arm qp"; exit 2;;
esac
