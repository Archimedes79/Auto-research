#!/usr/bin/env bash
# the evaluation of t10.dual_long_s0 chained behind its training through run.py (--after waits for the done flag): cold VAL 1000 frames of 82 scenes, as for t10.dual_s0 / s1.
# Run it in the foreground (no nohup, the WSL session must stay alive) next to start_long.sh; it waits for the GPU slot and < 3000 MB in use.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/run.py --id t10.dual_long_s0.eval --after t10.dual_long_s0 --gpu --gpu-max-mb 3000 --split val \
  --hyp "evaluation of t10.dual_long_s0 on cold VAL 1000 (82 scenes) with the call used for t10_dual_s0 and s1, so that the card's numbers (sections 3b, 3c) can be read" \
  --pred "exit 0; eval/metrics.json + card.json with GT signature 0.5_4_15efbde2d1; the numbers of the card's table (lane divider 0.41 (0.36-0.46), vehicle skill 0.55 (0.35-0.70), front mAP2 0.355 (0.33-0.385))" \
  --rule "decide.py --band 0.029 on the lane divider against 0.3352,0.3498 and the clauses of card 3c; a missing card or another GT signature = not comparable" \
  -- bash temp/t10/eval_long.sh > runs/t10_dual_long_s0.eval.launcher.log 2>&1
echo "eval launcher exit $?"
