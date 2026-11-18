#!/bin/bash
# E358: the timing of the wiring stage (rule vs learned head) as a run.py job, host side, QUIET host required (the script labels it); one process
cd "$(dirname "$0")/../.."
exec python3 ../autoresearch/tools/run.py --id E358.time_wiring --seed 0 --retries 0 --timeout 1500 --no-card --split val -- bash tools/env/_py14.sh time_wiring.py --seeds 0 1 --rounds 5 --out temp/e358/timing_wiring.json
