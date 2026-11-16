#!/bin/bash
# the GPU dumps of the E343 rehearsal (GPU branch, card E343 section 2 criterion 2): the pair head of E324 s and c2's B on the GPU for both seeds, to the default paths `all` looks at (temp/e343/val/ends_e324_s_s<S>.pkl,
# predfull_e292_c2_s<S>.pkl), each with its provenance sidecar (device cuda); the dumps wait for GPU room by themselves (evaluation/gpu_gate.py). One run.py job with a GPU slot (day28.sh); stops at the first failure.
cd "$(dirname "$0")/../.."
for S in 0 1; do
  bash tools/env/_py14.sh jtest_eval.py dump-ends --split val --seed $S --run e324_s_s$S --device cuda || exit 1
  bash tools/env/_py14.sh jtest_eval.py dump-b --split val --seed $S --b-run runs/e292_c2_s$S --device cuda || exit 1
done
