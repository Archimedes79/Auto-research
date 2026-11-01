#!/bin/bash
# one-off: refresh the manifests (torch version) and packages of the seven finished bev_fusion runs
cd "$(dirname "$0")/../../../bev_fusion" || exit 9
export AR_PROJECT=.
for r in smoke.d32 t10.dual_s0 t10.dual_s1 t11.vel_hurdle t11.vel_hurdle_acc2 t12.hurdle_ctl t12.mb_s0; do
  python3 ../autoresearch/tools/provenance.py manifest "$r" --force | cut -c1-200
  python3 ../autoresearch/tools/package.py package "$r" | cut -c1-200
done
python3 -c "import json; d=json.load(open('results/t10.dual_s0/provenance.json')); print(d['environment'])"
