#!/bin/bash
# E358 scoring job of one seed through the launcher (CPU, after the training run of the same seed): usage: wsl bash temp/e358/start_sc.sh <seed> [dry]
cd "$(dirname "$0")/../.."
S=$1; MODE=$2
LINE=(python3 ../autoresearch/tools/run.py --id E358.sc_s$S --seed $S --retries 0 --require-card --split val --after E358.w_s$S
  --hyp "E358: scoring of the heads of seed $S (heads 0 - 2 and the no-jitter ablation) at the end-to-end read-out on the staged scorer, plus the paired bootstrap against the stored PRIMARY card"
  --pred "w - c = +0.001 topo F1 (hit |w - c| <= 0.003 on both seeds)"
  --rule "E358 card section 0 (by hand: jwire_verdict.py --read-out primary)"
  -- bash temp/e358/score_heads.sh $S)
if [ "$MODE" = "dry" ]; then printf '%q ' "${LINE[@]}"; echo; else exec "${LINE[@]}"; fi
