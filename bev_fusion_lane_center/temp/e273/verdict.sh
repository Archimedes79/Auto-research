#!/bin/bash
# one-off: finish seed 1 (package), verdict of both seeds by the card's rule, timing of seed 1
cd "$(dirname "$0")/../.."
python3 temp/e273/finish_run.py e273_p_s1 E273.p_s1 | cut -c1-250
bash tools/env/_py14.sh qgraph_run_eval.py verdict --runs e273_p_s0 e273_p_s1 2>&1 | tail -30
bash temp/e273/timing.sh e273_p_s1 4
python3 temp/e273/make_table.py e273_p_s0 e273_p_s1 > temp/e273/table.md
cat temp/e273/table.md
