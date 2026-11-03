#!/usr/bin/env bash
# scratch: the trainsuite start of the 10-step smoke of the trainkit move (preview unless --go is given)
cd "$(dirname "$0")/../../../autoresearch" || exit 1
export AR_PROJECT=../bev_fusion
exec python3 -m trainsuite start ../bev_fusion/training/configs/t16_full.json --id trainkit.smoke --preset smoke --preset small_set --set steps=10 --set eval_every=10 --set save_every=5 --no-card "$@"
