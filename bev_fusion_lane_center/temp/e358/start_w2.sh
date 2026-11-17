#!/bin/bash
# E358 arm w, re-queued after the faithful live check (starts only after BOTH faithful GPU live checks exited 0 = ACCEPTED): the learned wiring head on the ATTACHED ends of E324 s, one seed through the launcher (GPU slot: the TRAIN records are a live pass of B and the pair run, then the heads train on the CPU).
#   usage: wsl bash temp/e358/start_w.sh <seed> [dry]        ('dry' as an ARGUMENT: an env flag does not pass from Git Bash through wsl)
cd "$(dirname "$0")/../.."
S=$1; MODE=$2
[ -n "$S" ] || { echo "usage: start_w.sh <seed> [dry]"; exit 2; }
[ -e runs/e358_w_s$S ] && { echo "start_w.sh: runs/e358_w_s$S exists: REFUSED (move it away by hand)"; exit 4; }
LINE=(python3 ../autoresearch/tools/run.py --id E358.w_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --timeout 5400 --require-card --split val --after E358.live_faithful_s0,E358.live_faithful_s1
  --hyp "E358: the E325 wiring head (545 parameters) trained on TRAIN records of E324 s with the ATTACHED ends and read out at the end-to-end read-out replaces the nearest-piece distance rule there: topo F1 at least E324 s PRIMARY - 0.005, guards held (training run, seed $S)"
  --pred "w - c = +0.001 topo F1 at the end-to-end read-out (hit |w - c| <= 0.003 on both seeds; the attached-end oracle gap is +0.0009 / +0.0019); real-choice agreement above the rule's"
  --rule "E358 card section 0 (by hand: jwire_verdict.py --read-out primary; KEEP when topo F1 of w >= c - 0.005 on both seeds with the six guards held, TIE kept and refined (D87) otherwise, CUT only by the kill: the head does not learn)"
  -- bash tools/env/_pyr.sh jwire_train.py run --seed $S --pair-run e324_s_s$S --ends att --name e358_w_s$S --heads 3 --epochs 60 --workers 4)
if [ "$MODE" = "dry" ]; then printf '%q ' "${LINE[@]}"; echo; else exec "${LINE[@]}"; fi
