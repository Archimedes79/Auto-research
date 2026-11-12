#!/bin/bash
cd "$(dirname "$0")/.."
python3 ../autoresearch/tools/run.py --id E273.smoke2 --seed 0 --gpu --retries 0 --no-card -- bash tools/env/_pyr.sh qgraph_train.py --name e273_smoke2 --seed 0 --smoke --workers 2
