#!/bin/bash
# one-off: rewrite the result packages of the bev_fusion runs that have one, with the evaluation and experiment fields
cd "$(dirname "$0")/../../../bev_fusion" || exit 9
export AR_PROJECT=.
for r in smoke.d32 t10.dual_s0 t10.dual_s1 t11.vel_hurdle t11.vel_hurdle_acc2 t12.hurdle_ctl t12.mb_s0; do
  timeout 60 python3 ../autoresearch/tools/package.py package "$r"
done
python3 - <<'EOF'
import json
d = json.load(open("results/t10.dual_s0/package.json"))
print(d["evaluation"])
print(d["experiment"], d["tier"], d["gt"], d["deep"]["bytes"], d["deleted"])
EOF
