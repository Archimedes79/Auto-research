#!/bin/bash
# E363: the layer-faithful dumps of arm a (GPU, data preparation, --no-card) and the node checks of both arms (GPU, small): run.py jobs.   usage: wsl bash temp/e363/start_faithful.sh [dry]
cd "$(dirname "$0")/../.."
MODE=$1
run() { local id=$1; shift; local line=(python3 ../autoresearch/tools/run.py --id $id --seed 0 --gpu --ram-mb 6000 --retries 0 --timeout 1800 --no-card --split val -- bash tools/env/_pyr.sh "$@"); if [ "$MODE" = "dry" ]; then printf '%q ' "${line[@]}"; echo; else "${line[@]}" > temp/e363/launch_$id.log 2>&1 & sleep 3; fi; }
run E363.dumpL_a_s0 faithful.py dump --run e363_a_s0
run E363.dumpL_a_s1 faithful.py dump --run e363_a_s1
run E363.nodes_a_s0 faithful.py check --seed 0 --g a
run E363.nodes_a_s1 faithful.py check --seed 1 --g a
run E363.nodes_c2_s0 faithful.py check --seed 0 --g c2
run E363.nodes_c2_s1 faithful.py check --seed 1 --g c2
wait
