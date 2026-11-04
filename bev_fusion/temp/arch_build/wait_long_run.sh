#!/usr/bin/env bash
# one-off: wait until the 6000-step run's evaluation is done or one of the two runs ends otherwise (at most 11 h), then print their state.
# The architecture build (training/architecture_target.md) starts only after that.
cd "$(dirname "$0")/../.." || exit 1
R=records/runs
for i in $(seq 1 660); do
  [ -f $R/t10.dual_long_s0.eval/done ] && break
  st=$(grep -ho '"status": "[a-z_]*"' $R/t10.dual_long_s0/run.json $R/t10.dual_long_s0.eval/run.json 2>/dev/null | grep -E 'failed|timeout|stalled|gpu_busy|blocked')
  [ -n "$st" ] && break
  sleep 60
done
date +%T
for r in t10.dual_long_s0 t10.dual_long_s0.eval; do echo "$r: $(ls $R/$r 2>/dev/null | tr '\n' ' ') $(grep -ho '"status": "[a-z_]*"\|"exit": [0-9-]*' $R/$r/run.json 2>/dev/null | tr '\n' ' ')"; done
