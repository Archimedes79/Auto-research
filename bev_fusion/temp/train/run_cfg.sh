#!/usr/bin/env bash
# One-off launcher of this loop: train one config, log to runs/<name>/train.log, mark .done only on exit code 0.
cd "$(dirname "$0")/../.." || exit 1
cfg=$1; name=$(basename "$cfg" .json)
mkdir -p runs/$name
~/miniconda3/envs/gcar/bin/python -u training/train.py "$cfg" > runs/$name/train.log 2>&1
rc=$?
[ $rc -eq 0 ] && touch runs/$name/.done
echo "$name exit $rc $(date +%T)"
