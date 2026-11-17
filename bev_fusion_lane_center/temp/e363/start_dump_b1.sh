#!/bin/bash
cd "$(dirname "$0")/../.."
exec python3 ../autoresearch/tools/run.py --id E363.dump_b_s1 --seed 1 --gpu --ram-mb 6000 --retries 0 --timeout 1800 --no-card --split val -- bash tools/env/_pyr.sh jends_eval.py dump --run e363_b_s1
