#!/bin/bash
# E364: the four CPU jobs through run.py, one chain (fixture, n seed 0, n seed 1, mask ceiling), nice, 2 threads, --retries 0.   wsl bash temp/e364/start_chain.sh [dry]    ('dry' as an ARGUMENT)
cd "$(dirname "$0")/../.."
MODE=$1; PREV=""
job() {   # job <id> <seed> <command...>
  local id=$1 S=$2; shift 2
  local after=""; [ -n "$PREV" ] && after="--after $PREV"
  local line=(python3 ../autoresearch/tools/run.py --id $id --seed $S --require-card --retries 0 --split val $after
    --hyp "E364: what the long-link cover drop (R4) and the pair mask (R6) carry on the end-to-end read-out of E324 s (measurement, CPU): job $id"
    --pred "R4: topo F1 -0.003 without the cover drop (card guess); R6: 0.995 of the true connectors admitted (card guess)"
    --rule "E364 card section 0 (reading rules FREE / CARRIES / HARMS / MIXED for R4, FREE / COSTS for R6; by hand: D87 replay)"
    -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh "$@")
  if [ "$MODE" = "dry" ]; then printf '%q ' "${line[@]}"; echo; else "${line[@]}" > "temp/e364/launch_$id.log" 2>&1 & sleep 3; fi
  PREV=$id
}
job E364.fx_s0 0 score_nocover.py --run e324_s_s0 --seed 0 --fixture
job E364.fx_s1 1 score_nocover.py --run e324_s_s1 --seed 1 --fixture
job E364.n_s0  0 score_nocover.py --run e324_s_s0 --seed 0
job E364.n_s1  1 score_nocover.py --run e324_s_s1 --seed 1
job E364.m     0 mask_ceiling.py
wait
