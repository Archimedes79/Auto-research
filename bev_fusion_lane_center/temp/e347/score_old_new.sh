#!/bin/bash
# E347 scratch (WSL): the extended scorer against the old one on a stored maps run, a few VAL frames, CPU only. The old scorer (real file) and the new one (staged) both score a COPY of runs/<run> (the scorer writes jmaps_val.json into the run dir:
# the stored file must not be overwritten); every key of the old result must come back unchanged in the new one.   bash temp/e347/score_old_new.sh [run] [frames]
RUN="${1:-e332_ma_s0}"; N="${2:-24}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D="$R/temp/_session_scratch_e347/score_old_new/$RUN"
mkdir -p "$D"
cp "$R/runs/$RUN/model.pt" "$D/model.pt"
export CUDA_VISIBLE_DEVICES=""
REL="../temp/_session_scratch_e347/score_old_new/$RUN"
rm -f "$D/jmaps_val.json"
BFLC_STAGE=none E347_FAST=1 bash "$R/temp/e347/run_staged.sh" evaluation/jmaps_eval.py score --run "$REL" --max "$N" --workers 0 --base "$R/runs/e273_p_s0/model.pt" > "$D/old.log" 2>&1 && mv "$D/jmaps_val.json" "$D/old.json"
E347_FAST=1 bash "$R/temp/e347/run_staged.sh" evaluation/jmaps_eval.py score --run "$REL" --max "$N" --workers 0 --base "$R/runs/e273_p_s0/model.pt" > "$D/new.log" 2>&1 && mv "$D/jmaps_val.json" "$D/new.json"
tail -3 "$D/old.log" "$D/new.log" | cut -c1-200
