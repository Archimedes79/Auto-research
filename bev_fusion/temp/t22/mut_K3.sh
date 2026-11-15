#!/bin/bash
# scratch: the one mutant that was set up wrongly in mut_K2: zero_cam WITHOUT lane_img_old (the code, not the selftest) must fail the bev_modules selftest; on the private copy ~/mutk2
cd ~/mutk2/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
F=evaluation/bev_modules.py
cp $F temp/t22/bev_modules.keep
python3 - <<'EOF'
p = "evaluation/bev_modules.py"
s = open(p, encoding="utf-8").read()
old = '"lane_img", "lane_img_old", "img") if k in batch'
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, '"lane_img", "img") if k in batch'))
EOF
$PY evaluation/bev_modules.py --selftest > temp/t22/mut_K3.out 2>&1
rc=$?
cp temp/t22/bev_modules.keep $F
if [ $rc -ne 0 ]; then echo "KILLED   zero_cam without lane_img_old fails the bev_modules selftest (exit $rc): $(tail -1 temp/t22/mut_K3.out | cut -c1-160)"; else echo "SURVIVED zero_cam without lane_img_old"; fi
$PY evaluation/bev_modules.py --selftest 2>&1 | tail -1 | cut -c1-100
