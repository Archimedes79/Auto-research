#!/bin/bash
# E321: the one TEST use of E312 (D84), through the launcher with the gate. Card experiments/E321_test-e312/experiment.md; log row in evaluation/TEST_USAGE_LOG.md names E321.t.
cd "$(dirname "$0")/../.."
python3 ../autoresearch/tools/run.py --id E321.t --retries 0 --require-card --split test --timeout 360 \
  --hyp "E312 keeps its VAL gain over the plain graph on TEST (one logged use, D84)" \
  --pred "snapped topo F1 +0.045 over the plain graph on TEST, not snapped +0.015" \
  --rule "E321 card: report only, winner minus control with the paired scene bootstrap, nothing selected or tuned on TEST" \
  -- bash tools/env/_py14.sh evaluation/jtest_eval.py all --split test --test-id E321.t --seeds 0 1 --device cpu
