#!/usr/bin/env bash
cd /mnt/c/Auto-research || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
L=/mnt/c/Auto-research/bev_fusion_lane_center/training/lanegraph
$PY reporter/winner.py export e324s_lanegraph --src "$L/**/*.py" --src "$L/README.md" --strip "$L" \
  --title "E324 s: the end-to-end lane graph (inference package, no weights)" \
  --numbers "VAL 873 frames, topo F1 seed 0 / seed 1: end-to-end read-out with --attached-ends 0.6912 / 0.6948 (rule wiring); the default read-out (snapped, with a rule step) 0.6981 / 0.7010. The learned wiring head (E358, 0.6931 / 0.6964) is not in this package." \
  --command "bash tools/env/_py14.sh training/lanegraph/scripts/lg_infer.py --seed 0 --first 20 --attached-ends --out /abs/path/graphs.json   # run from bev_fusion_lane_center/; tests: training/lanegraph/tests/run_lg_tests.py" \
  --weights "runs/e324_s_s0 and runs/e324_s_s1 (the junction system) and runs/e273_p_s0 / s1 (lane head B): in the repository runs, not in git"
$PY reporter/winner.py verify winners/e324s_lanegraph
ls winners/e324s_lanegraph
