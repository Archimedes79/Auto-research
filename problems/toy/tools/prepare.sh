#!/usr/bin/env bash
# the data a training needs, as ONE item of the training block's "prepare": the raw shards, then their truth (the truth needs the shards, and prepare
# items cannot wait for each other).     prepare.sh have|make <config.json>     exit 0 done / present, 1 missing, 2 error
cd "$(dirname "$0")/.." || exit 2
mode=$1
cfg=$2
case "$mode" in
  have) bash tools/py.sh data/make_data.py have --config "$cfg" && bash tools/py.sh gt/generate.py have --config "$cfg" ;;
  make) bash tools/py.sh data/make_data.py make --config "$cfg" && bash tools/py.sh gt/generate.py generate --config "$cfg" ;;
  *) echo "usage: prepare.sh have|make <config.json>" >&2; exit 2 ;;
esac
