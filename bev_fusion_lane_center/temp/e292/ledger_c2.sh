#!/bin/bash
cd "$(dirname "$0")/../.."
for S in 0 1; do
python3 ../autoresearch/tools/ledger.py note E292.c2_s$S --result "VAL snapped p>=0.1 with E312's connectors on the delivered graph of B' (B's final checkpoint fine-tuned, pyramid frozen, no layer, hard step): see card section 8; precondition on topo F1 holds (c2 - c: seed 0 +0.0065, seed 1 +0.0020; guards inside their bounds); node gate passed" --decision "by hand: precondition of the card holds (c2 passes against c): not a verdict; the comparison of arm a is not void; the schedule itself improves B (graph-only +0.0087 / +0.0122)"
done
