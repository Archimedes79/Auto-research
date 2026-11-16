#!/bin/bash
# E351 speed half, two more quiet repeats of the same protocol (E351.g1c, E351.g1d): the first two runs disagree on E324 s (-10.3 % and -15.3 %)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
for X in c d; do
  python3 ../autoresearch/tools/run.py --id E351.g1$X --seed 0 --retries 0 --split val --gpu --timeout 60 \
    --hyp "E351 g1 repeat $X (the runs E351.g1 and E351.g1b disagree on E324 s: -10.3 % and -15.3 % of the control median, the minimum effect is 15 %): the same protocol again on a quiet host to get the run-to-run spread of the paired difference" \
    --pred "E324 s between -10 % and -16 % again (a run-to-run spread of about 3 points); E312 about -12 %" \
    --rule "by hand: pooled repeats; the card's rule judges one quiet run, so with four runs the verdict is the MEDIAN of the per-run paired differences against the minimum effect 15 % of the control median (written 2026-11-15 11:33 after the first two runs); a median below the minimum effect is a TIE, a spread of the per-run drops that straddles the minimum effect is reported as INCONCLUSIVE" \
    -- bash tools/env/_py14.sh jtiming.py graph --runs e312_p_s0,e324_s_s0 --tag e351_graph_20261115$X
  echo "timing $X exit $? $(date +%H:%M:%S)"
done
echo "timing cd done"
