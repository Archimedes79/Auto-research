#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
OUT=experiments/E259_assembly-cleanup-ablation/cputime.txt
nohup python3 ../autoresearch/tools/run.py --id E259.b0 --retries 0 \
  --hyp "E259 step 2: the CPU time of the assembly stages is measurable on a loaded machine and shows which stages are worth removing for speed" \
  --pred "anchoring saves 1.3 ms CPU, relocation 1.5, clean-up 0.9, the dp stage costs +0.3" \
  --rule "a stage is worth removing for speed only if its CPU-time saving exceeds 0.5 ms and the range over the repetitions excludes zero (provisional until an idle window)" \
  -- bash -c "bash tools/env/_py14.sh temp/loop21/e259_cputime.py > $OUT 2>&1" > /dev/null 2>&1 &
sleep 2; echo launched
