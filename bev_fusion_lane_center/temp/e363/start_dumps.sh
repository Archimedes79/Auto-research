#!/bin/bash
# E363: the FAITHFUL dumps (data preparation, run.py --gpu --no-card): the pair run of E324 s with the node head of B' (c2 / a), 873 val_clean frames, written into the new run folders runs/e363_<g>_s<S>/ (the stored runs are untouched).   usage: wsl bash temp/e363/start_dumps.sh [dry]
cd "$(dirname "$0")/../.."
MODE=$1
for G in c2 a; do
  PREV=""
  for S in 0 1; do
    AFTER=""; [ -n "$PREV" ] && AFTER="--after $PREV"
    LINE=(python3 ../autoresearch/tools/run.py --id E363.dump_${G}_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --timeout 1800 --no-card --split val $AFTER
      -- bash tools/env/_pyr.sh jends_eval.py dump --run e363_${G}_s$S)
    if [ "$MODE" = "dry" ]; then printf '%q ' "${LINE[@]}"; echo; else "${LINE[@]}" > temp/e363/launch_dump_${G}_s$S.log 2>&1 & sleep 3; fi
    PREV=E363.dump_${G}_s$S
  done
done
wait
