#!/bin/bash
# E348 CPU smoke of the exact command (attach + nopaint + scales + e2e + geometry loss), 120 steps on the CPU, scratch run name e348_smoke_cpu_s0 (a diagnostic: no verdict)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= BFLC_SIG=1.0
nice -n 5 bash tools/env/_pyr.sh jends_train.py --name e348_smoke_cpu_s0 --seed 0 --maps $(pwd)/runs/e332_ma_s0/model.pt --e2e --scales --maps-w 1.0 --steps 120 --geom-w 0.5 --attach --workers 2 > temp/e348/smoke_cpu.log 2>&1
echo "smoke exit $? $(date)" >> temp/e348/smoke_cpu.log
