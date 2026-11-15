#!/bin/bash
cd "$(dirname "$0")/../.."
for S in 0 1; do
python3 ../autoresearch/tools/ledger.py note E324.s_s$S --notes "E338 stack check (2026-11-14, by hand: replay, card E338 section 5): the own-keypoint snapped read-out of this run on the delivered graph of c2 (B fine-tuned 8250 steps, pyramid frozen) scores topo F1 $( [ $S = 0 ] && echo '0.7066 (D +0.0085 over this run on B, +0.0242 over E312)' || echo '0.7016 (D +0.0006 over this run on B, +0.0152 over E312)' ); additive to c2's own gain (all pre-registered numbers hit); TIE by the stack rule (seed 1 below +0.005)"
done
