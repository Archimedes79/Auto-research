#!/bin/bash
# ms per frame of the E273 head on an idle GPU: $1 = run name
cd "$(dirname "$0")/../.."
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
bash tools/env/_py14.sh qgraph_timing.py --run $1 --n 60 2>&1 | tail -${2:-15}
