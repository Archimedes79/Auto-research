#!/bin/bash
# E351 speed half: jtiming.py graph (g1 against the eager forward c, alternating blocks, 10 rounds x 30 forwards), run id E351.g1,
# started ONLY on a quiet host (load < 3) with an idle GPU. The caller has stopped the E345 scorers and announced the block.
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
python3 ../autoresearch/tools/run.py --id E351.g1b --seed 0 --retries 0 --split val --gpu --timeout 60 \
  --hyp "E351 g1 repeat (the first run E351.g1 labelled itself NOT QUIET because its GPU after-sample read its own tail; the sample is now taken 2.5 s after the last timed kernel): one CUDA graph of the junction front (pyramid + maps head + B's node head) with the pair head eager, against the eager forward, ms per frame on a quiet host, E312 and E324 s (seed-0 checkpoints)" \
  --pred "E312 -4 ms (-24 %), E324 s -9 ms (-35 %), both beyond the minimum effect of 15 % of the control median (2.6 / 3.7 ms); outputs identical" \
  --rule "E351 card section 0: ADOPT when the equality holds (tier 1, all 873 frames of both seeds) AND the drop is at least 15 % of the c median with a paired 95 % interval over the rounds that excludes 0 on a quiet host (load < 3 and an idle GPU before and after), else TIE / INCONCLUSIVE; label by jtiming.py graph; by hand: replay" \
  -- bash tools/env/_py14.sh jtiming.py graph --runs e312_p_s0,e324_s_s0 --tag e351_graph_20261115b
echo "timing exit $? $(date +%H:%M:%S)"
