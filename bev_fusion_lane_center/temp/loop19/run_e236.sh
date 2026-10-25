#!/bin/bash
# E236 (CPU only, cached raw outputs): tau sweep at fork pruning 0.4 for champion, recall (rc1), its seed twin, fast, fast_recall; then the matched evaluations.
#   bash temp/loop19/run_e236.sh sweep       cards runs/eval/<name>/val_sw<tau>.json (tau 0.55 ... 0.90), 4 in parallel
#   bash temp/loop19/run_e236.sh matched     python3 temp/loop19/analyse_e236.py picks tau per model (vert100 14.3 +- 0.15), then real evaluations with --tag v143
cd "$(dirname "$0")/../.."
case "$1" in
  sweep)
    for m in champion recall recall_s1 fast fast_recall; do for t in 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90; do echo "$m $t"; done; done | \
      xargs -P 4 -L 1 bash -c 'bash tools/env/_py14.sh evaluate.py $0 --asm fork_prune=0.4 --tau $1 --tag sw$1 > /dev/null 2>&1 || echo "FAILED $0 $1"'
    echo "sweep done" ;;
  sweepvx)
    for m in champion recall recall_s1 fast fast_recall; do for v in 1.5 1.7 1.9 2.1 2.3 2.5 2.7; do echo "$m $v"; done; done |       xargs -P 4 -L 1 bash -c 'bash tools/env/_py14.sh evaluate.py $0 --asm fork_prune=0.4 vtx_r=$1 --tag vx$1 > /dev/null 2>&1 || echo "FAILED $0 $1"'
    echo "vtx_r sweep done" ;;
  matched)
    python3 temp/loop19/analyse_e236.py pick > runs/e19/e236_picks.txt && cat runs/e19/e236_picks.txt
    while read m t; do bash tools/env/_py14.sh evaluate.py $m --asm fork_prune=0.4 vtx_r=$t --tag v143 2>&1 | tail -n 3; done < <(grep -E '^PICK' runs/e19/e236_picks.txt | awk '{print $2, $3}')
    python3 temp/loop19/analyse_e236.py table ;;
  *) echo "usage: run_e236.sh sweep|matched"; exit 2 ;;
esac
