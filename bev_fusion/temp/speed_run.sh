#!/usr/bin/env bash
# scratch: one mini-measurement run of train.py from a scratch copy of the code (temp/speed_base = HEAD, temp/speed = HEAD + speed changes)
# usage (from bev_fusion/, under run.py): bash temp/speed_run.sh <tree: speed_base|speed> <config.json>
cd "$(dirname "$0")/.." || exit 1
tree=$1; cfg=$2; name=$(basename "$cfg" .json)
mkdir -p "runs/$name"
echo "=== $(date +%F_%T) $name from temp/$tree" >> "runs/$name/train.log"
~/miniconda3/envs/gcar/bin/python -u "temp/$tree/bev_fusion/training/train.py" "$cfg" >> "runs/$name/train.log" 2>&1
rc=$?
tail -n 15 "runs/$name/train.log"
[ $rc -eq 0 ] && touch "runs/$name/.done"
echo "$name exit $rc $(date +%T)"
exit $rc
