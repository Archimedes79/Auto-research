#!/usr/bin/env bash
# One-off: smoke run of a short config from the working tree (functionality, memory, speed): log to temp/train/<name>.log.
cd "$(dirname "$0")/../.." || exit 1
cfg=$1; name=$(basename "$cfg" .json)
rm -rf temp/train/runs/$name
~/miniconda3/envs/gcar/bin/python -u training/train.py "$cfg" > temp/train/$name.log 2>&1
echo "$name exit $? $(date +%T)"
