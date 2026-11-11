#!/bin/bash
# one-off (E274): the judgement columns of the four runs, from the scored cards (VAL 873) and the verdict output of temp/e274/chain.log
cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py note"
$L E274.r_s0 --result 'VAL 873: topo F1 0.558 (c s0 0.597), F1@1.0 0.865 (0.832), junction F1 0.121 (0.130), lanes found 0.777 (0.644), node R/P 0.889/0.811 (0.833/0.873), dup 0.363 (0.232); assignment p-g exact -0.0173, map-error gain +0.0625 (c +0.0722)' \
  --decision 'CUT by hand (written rule of the card): topo F1 gain -0.039 on both seeds (band 0.02), exact-map guard broken (-0.0173 and -0.0160, limit -0.01); F1@1.0 +0.033 and the map-error gain hold. r sits on the c threshold curve: lanes found and F1@1.0 up, topo F1 down' \
  --notes 'prediction missed: topo +0.05 (got -0.039), junction F1 0.16 (got 0.121); hit: node recall 0.889 (range 0.87-0.90), F1@1.0 0.865, node precision 0.811 (>0.80). scored by qgraph_run_eval.py score, card runs/e274/e274_r_s0, verdict274 by hand. Read-out diagnostics in-sample on VAL: see E275 card.'
$L E274.r_s1 --result 'VAL 873: topo F1 0.557 (c s1 0.595), F1@1.0 0.855 (0.821), junction F1 0.121 (0.120), lanes found 0.756 (0.630), node R/P 0.876/0.814 (0.833/0.868), dup 0.340 (0.192); assignment p-g exact -0.0160, map-error gain +0.0585 (c +0.0706)' \
  --decision 'CUT by hand: same rule and same side as seed 0 (topo -0.038, exact-map guard -0.0160); see E274.r_s0' \
  --notes 'seed twin of E274.r_s0; card runs/e274/e274_r_s1'
$L E274.p_s0 --result 'VAL 873: topo F1 0.582 (r s0 0.558, c s0 0.597), F1@1.0 0.863, junction F1 0.124 (r 0.121), lanes found 0.752, node R/P 0.888/0.811, long-link P/R 0.216/0.486 (c 0.167/0.503), dup 0.321; assignment p-g exact -0.0164, map-error gain +0.0618' \
  --decision 'CUT by hand: p against r junction F1 gain +0.003 (band 0.03), topo against r +0.025; p against c topo -0.015 (inside the band) with the exact-map guard broken (-0.0164)' \
  --notes 'prediction missed: junction F1 +0.06 over r (got +0.003), topo unchanged within 0.01 (got +0.025); partly hit: long-link precision 0.216 (range 0.18-0.30). Dual-softmax loss raised long-link precision and lowered duplicates but did not lift junction F1. Gate caution: the live phase-end GATE line is a different model than the phase-7 record (see HANDOVER of 2026-11-05)'
$L E274.p_s1 --result 'VAL 873: topo F1 0.577 (r s1 0.557, c s1 0.595), F1@1.0 0.852, junction F1 0.122 (r 0.121), lanes found 0.736, node R/P 0.876/0.814, long-link P/R 0.221/0.469 (c 0.191/0.470), dup 0.281; assignment p-g exact -0.0156, map-error gain +0.0590' \
  --decision 'CUT by hand: same rule and same side as seed 0 (junction gain over r +0.0002, topo against c -0.018, exact-map guard -0.0156); see E274.p_s0' \
  --notes 'seed twin of E274.p_s0; card runs/e274/e274_p_s1'
python3 ../autoresearch/tools/ledger.py tail --loop 0 -n 6 | cut -c1-200
