#!/bin/bash
# one-off: ledger judgement columns of E273.p_s0
cd "$(dirname "$0")/../.."
python3 ../autoresearch/tools/ledger.py note E273.p_s0 \
  --result "VAL 873, seed 0: lane_acc with map error 1-2 m p 0.9489 vs g 0.8767 (p-g +0.0722 [+0.0563,+0.0871]); exact map p 0.9799 vs g 0.9870 (-0.0072 [-0.0151,+0.0004]); graph: topo F1 0.597 (d 0.7405), F1@1.0 0.8325 (d 0.8781), junction F1 0.130 (d 0.273), lanes found 0.644 (d 0.781), Q 0.636 (d 0.742); 18.2 node ids / 28.9 DP vertices per 100 m; pieces per GT piece 1.61" \
  --decision "CUT (by hand: card rule 4, the assignment lever holds (+0.072 > +0.03, exact-map guard held) but the graph guards topo F1 and F1@1.0 (d-0.03) are broken; seed 1 pending, both seeds must lie on the same side)" \
  --notes "attempts: the phase-4 gate (links) failed on 6 attempts before this run (junction F1 0.02 / F1@1.0 0.647 at attempt 2; row-softmax successor distribution of attempt 5 was worse and reverted); the model of this run uses the set-attention link head with one-to-many labels (commit 59ad3009 code). By hand: phases 5-7 ran with --no-gates (phase 5 gate: section start accuracy 0.47 vs 0.95 needed). Phases 2-3 of this seed ran on earlier code than phases 4-7 (phase 2/3 code identical to what seed 1 used). Scorer reproduces d exactly. Result folder results/E273.p_s0/, card runs/e273/e273_p_s0/card.json"
python3 ../autoresearch/tools/ledger.py tail 2 | grep E273 | cut -c1-300
