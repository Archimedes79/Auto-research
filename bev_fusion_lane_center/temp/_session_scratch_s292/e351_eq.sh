#!/bin/bash
# E351 step 1: equality of the CUDA-graph forward (g1) with the eager forward on all 873 VAL frames of four checkpoints, with dumps.
# Each run goes through run.py (--gpu slot, ledger row); the chain stops at the first failure.
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
for R in e312_p_s0 e312_p_s1 e324_s_s0 e324_s_s1; do
  python3 ../autoresearch/tools/run.py --id E351.eq_$R --seed 0 --retries 0 --split val --gpu --timeout 90 \
    --hyp "E351 step 1: the CUDA-graph forward (g1) against the eager forward on all 873 VAL frames of $R, bf16, batch 1, with the dumps for the score comparison" \
    --pred "TIER 1: every tensor of P, out and the pair-head output identical (dtype, shape, bits) on all 873 frames and the null control (eager against eager) clean; the probe saw 528 / 672 tensors identical on 24 frames" \
    --rule "E351 card section 0 tier 1 by the verdict of jgraph.py check (TIER 1 / NOT IDENTICAL / NULL NOT CLEAN); by hand: replay, the scored-card equality is a second step" \
    -- bash tools/env/_py14.sh jgraph.py check --run $R --dump --workers 4
  rc=$?
  echo "eq $R exit $rc $(date +%H:%M:%S)"
  if [ $rc -ne 0 ]; then echo "chain stopped"; exit $rc; fi
done
echo "eq chain done $(date +%H:%M:%S)"
