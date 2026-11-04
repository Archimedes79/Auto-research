#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
P=/home/ml/miniconda3/envs/gcar/bin/python
for t in test_speed_flags test_grad_scaler test_freeze_probe test_host_overhead; do
  printf "%s: " $t; OMP_NUM_THREADS=2 $P tests/$t.py 2>&1 | tail -1 | cut -c1-120
done
# startable: the split on t16_full composes; a typo is a problem before any data
python3 - <<'PY'
import json
c = json.load(open("training/configs/t16_full.json"))
c["grad_clip_split"] = ["motion_block"]; json.dump(c, open("temp/t16/split_ok.json", "w"))
c["grad_clip_split"] = ["motion_blok"]; json.dump(c, open("temp/t16/split_typo.json", "w"))
PY
for f in split_ok split_typo; do echo "== startable $f"; $P training/startable.py temp/t16/$f.json 2>&1 | tail -4 | cut -c1-200; done
