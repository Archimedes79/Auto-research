#!/bin/bash
# scratch: writes training/configs/t22_B2_s0.json and runs startable on the resolved config (the raw file is refused for the key 'split' as t21_pair4 is; trainsuite resolves it first)
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY temp/t22/make_config_B2.py || exit 1
diff <(python3 -c "import json;d=json.load(open('training/configs/t22_B_s0.json'));print(json.dumps(d,indent=1,sort_keys=True))") <(python3 -c "import json;d=json.load(open('training/configs/t22_B2_s0.json'));print(json.dumps(d,indent=1,sort_keys=True))")
cd /mnt/c/Auto-research/autoresearch && AR_PROJECT=../bev_fusion python3 -m trainsuite start ../bev_fusion/training/configs/t22_B2_s0.json --id t22.B2_s0 --hyp x --pred x --rule x 2>&1 | cut -c1-300 | head -30
