#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
OUT=experiments/E258_assembly-profile-and-speedup/variants_step1b.txt
nohup python3 ../autoresearch/tools/run.py --id E258.a0 --retries 0 \
  --hyp "E258: the assembly (7.5 - 8.9 ms of the chain) has a stage above 15 percent whose change saves at least 1.0 ms with identical graphs" \
  --pred "reloc off saves 1.5 ms (tol 0.8) and costs J 0.01; anchoring off saves 1.0 ms and costs J 0.02" \
  --rule "drop a stage only if Q loss <= 0.003, J loss <= 0.01, vert100 change <= 0.3 on both models and saving >= 0.8 ms (relative timing)" \
  -- bash -c "bash tools/env/_py14.sh temp/loop21/e258_variants.py > $OUT 2>&1" > /dev/null 2>&1 &
sleep 2; echo launched
