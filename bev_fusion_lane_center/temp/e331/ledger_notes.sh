#!/bin/bash
cd "$(dirname "$0")/../.."
DEC="by hand: replay of the rule of the card (jrule_removal.py, adopt = loss <= 0.01 on both seeds and both read-outs): NOT ADOPTED, the evidence features stay: the unsnapped (differentiable) read-out loses 0.0232 / 0.0124 topo F1; snapped within bounds (-0.0029 / -0.0007); pair precision at E312's recall +0.028 / +0.022 (the loss is at the fixed operating point)"
python3 ../autoresearch/tools/ledger.py note E331.e_s0 --result "VAL p>=0.1: unsnapped topo F1 0.6206 (E312 0.6438, -0.0232), F1@1.0 -0.0183; snapped 0.6796 (0.6825, -0.0029); maps kp F1 0.7060 (E307 0.7366)" --decision "$DEC"
python3 ../autoresearch/tools/ledger.py note E331.e_s1 --result "VAL p>=0.1: unsnapped topo F1 0.6357 (E312 0.6481, -0.0124), F1@1.0 -0.0122; snapped 0.6857 (0.6864, -0.0007); maps kp F1 0.7063 (E307 0.7268)" --decision "$DEC"
