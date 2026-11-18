#!/bin/bash
# E372 ledger notes after the verdict (by hand, the card's rule of v3.2): one note per arm row
cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py note"
$L E372.s4_s0 --result "VAL 873 D115 live: topo F1 0.6682 (s4-z -0.0003 [-0.0043,+0.0036]; s4-ref +0.0002), junction 0.2538 (-0.0015 / +0.0021), F1@1.0 0.8459, lanes 0.7232" --decision "S4-HOLDS (by hand: the card's rule v3.2, both seeds); not S4-GAINS" --notes "update share 0.19-0.40; live-diff vs parity: all frames' cell lists changed, median |dpos| 0.048 m; no moving-label flag; no kill, 0 resumes"
$L E372.s4_s1 --result "VAL 873 D115 live: topo F1 0.6813 (s4-z +0.0049 [+0.0020,+0.0079]; s4-ref +0.0051 [+0.0019,+0.0083]), junction 0.2481 (+0.0028 / +0.0029), F1@1.0 0.8476, lanes 0.7214" --decision "S4-HOLDS (by hand: the card's rule v3.2, both seeds); not S4-GAINS" --notes "update share 0.21-0.29; live-diff vs parity: median |dpos| 0.046 m, GT node coverage 0.9544 vs 0.9540; no moving-label flag; no kill, 0 resumes"
$L E372.z_s0 --result "VAL 873 D115 live: topo F1 0.6685, junction 0.2553 (z-ref +0.0006 / +0.0036)" --decision "control (never adopted)"
$L E372.z_s1 --result "VAL 873 D115 live: topo F1 0.6763, junction 0.2454 (z-ref +0.0002 / +0.0001)" --decision "control (never adopted)"
$L E372.b_s0 --result "VAL 873 D115 live: topo F1 0.6663, junction 0.2516 (b-z -0.0022 / -0.0038; s4-b +0.0019)" --decision "control (never adopted)"
$L E372.b_s1 --result "VAL 873 D115 live: topo F1 0.6787, junction 0.2467 (b-z +0.0023 / +0.0014; s4-b +0.0026)" --decision "control (never adopted)"
echo notes done
