#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
OUT=experiments/E258_assembly-profile-and-speedup/variants_step2_seeds.txt
nohup python3 ../autoresearch/tools/run.py --id E258.b0 --retries 0 \
  --hyp "E258 step 2: switching the junction anchoring off is free for the j1 family (four seeds of the recall fine-tune)" \
  --pred "Q loss <= 0.003, J loss <= 0.01, vert100 change <= 0.3 on at least 3 of 4 seeds" \
  --rule "anchoring off is a candidate for the j1 family only if it holds on all four seeds; else the stage stays" \
  -- bash -c "bash tools/env/_py14.sh temp/loop21/e258_variants.py e24_j1rc_s0 e24_j1rc_s1 e24_j1rc_s2 e24_j1rc_s3 > $OUT 2>&1" > /dev/null 2>&1 &
sleep 2; echo launched
