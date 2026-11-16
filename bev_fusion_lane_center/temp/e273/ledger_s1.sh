#!/bin/bash
# one-off: ledger judgement columns of E273.p_s1 and the final decision of E273.p_s0 (both seeds scored)
cd "$(dirname "$0")/../.."
DEC="CUT (by hand: card rule 4 and runs/e273 verdict of both seeds on the same side: lane_acc_err p-g +0.0714 (mean of 2 seeds) > +0.03 and the exact-map guard held (-0.0066 > -0.01), but the graph guards topo F1 0.596 and F1@1.0 0.827 are more than 0.03 below d (0.7405 / 0.8781); back to the connectivity step (nodes + links), not a new card)"
python3 ../autoresearch/tools/ledger.py note E273.p_s0 --decision "$DEC"
python3 ../autoresearch/tools/ledger.py note E273.p_s1 \
  --result "VAL 873, seed 1: lane_acc with map error 1-2 m p 0.9501 vs g 0.8795 (p-g +0.0706 [+0.0551,+0.0853]); exact map p 0.9797 vs g 0.9858 (-0.0061 [-0.0144,+0.0023]); graph: topo F1 0.595 (d 0.7405), F1@1.0 0.821 (d 0.8781), junction F1 0.120 (d 0.273), lanes found 0.630 (d 0.781), Q 0.631 (d 0.742); 20.5 node ids / 31.5 DP vertices per 100 m; pieces per GT piece 1.60; 23.2 ms per frame (head only, idle GPU)" \
  --decision "$DEC" \
  --notes "all six phases in one launch on the final code (set-attention link head, commit 59ad3009 code), all with --no-gates by hand (for seed 0 on the same code the gate of phase 4 failed (junction F1 0.181) and so did the gate of phase 5 (section start accuracy 0.383 vs 0.95, 2.85 delivered pieces per GT piece); after phases 6-7 the delivered graphs of both seeds have 1.60 pieces per GT piece); same side as seed 0 in every number. Result folder results/E273.p_s1/, card runs/e273/e273_p_s1/card.json, timing runs/e273/e273_p_s1/timing.json"
python3 ../autoresearch/tools/ledger.py note E273.p_s0 --notes "ms per frame (head only, batch 1, bf16, cached features, idle GPU, median of 60): total 24.2 = pixel maps 4.3 + nodes 2.9 + links 5.5 + sections 2.8 + map encoder and assignment 8.3 (runs/e273/e273_p_s0/timing.json); the CPU read-out is not included"
python3 ../autoresearch/tools/ledger.py tail 2 | grep E273 | cut -c1-260
