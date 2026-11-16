#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 ../autoresearch/tools/ledger.py note E354.post_s0 --notes "2026-11-16 05:20 lead (numbers check): the three flip-rate comparisons of seed 0 were re-run OUTSIDE run.py by temp/e354/redo_flip.sh (23:53-23:54, exit 0) after the jflip.py compare crash (fixed 76c5fb22); the values in the card (-1.41 [-3.69, +0.44] etc.) come from that re-run, not from this row (hard rule 3 named; they feed a TIE verdict by the written rule, not a number of a run that was never recorded: the rate steps of this row ran under run.py)"
