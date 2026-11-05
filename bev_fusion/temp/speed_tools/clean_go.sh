#!/bin/bash
# unlink the symlinks of my two scratch trees (links only, never the targets), then tag + move the trees with clean_temp.py (--force: only the 6-hour guard; no job runs)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
if pgrep -f "train.py|bev_suite.py|run_ar.sh" > /dev/null; then echo "a training or evaluation process runs: stop here"; pgrep -af "train.py|bev_suite.py|run_ar.sh" | cut -c1-120; exit 1; fi
for t in speed speed_base; do
  for l in $t/bev_fusion/data $t/autogt $t/autoresearch; do
    [ -L "temp/$l" ] && unlink "temp/$l" && echo "unlinked temp/$l"
  done
done
echo "targets still there: data=$(ls data | wc -l) entries, autogt=$(ls ../autogt | wc -l), autoresearch=$(ls ../autoresearch | wc -l)"
for n in speed speed_base; do
  echo "=== clean_temp $n --move --force"
  python3 ../autoresearch/tools/clean_temp.py $n --move --force 2>&1 | tail -10 | cut -c1-220
done
