#!/bin/bash
# E363 fixture dump (data preparation): the pair run of E324 s with B's OWN node head (runs/e363_b_s0, 0 tensors changed): its dump must reproduce the stored runs/e324_s_s0/ends_val.pkl
cd "$(dirname "$0")/../.."
exec python3 ../autoresearch/tools/run.py --id E363.dump_b_s0 --seed 0 --gpu --ram-mb 6000 --retries 0 --timeout 1800 --no-card --split val -- bash tools/env/_pyr.sh jends_eval.py dump --run e363_b_s0
