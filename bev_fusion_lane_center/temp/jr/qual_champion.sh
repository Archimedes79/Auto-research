#!/usr/bin/env bash
# qualitative pictures of the token champion: derive the renderer, select the frames by the written rules, render (CPU), copy into the report folder
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export BFLC_ROOT=/mnt/c/Auto-research/bev_fusion_lane_center
/home/ml/miniconda3/envs/gcar/bin/python temp/jr/make_render_champion.py || exit 1
bash tools/env/_py14.sh temp/qual_report_1117/select_champion_q1117.py || exit 1
mkdir -p temp/qual_report_1117/final
bash tools/env/_py14.sh temp/qual_report_1117/render_champion_q1117.py temp/qual_report_1117/spec_champion_q1117.json temp/qual_report_1117/final || exit 1
ls -la temp/qual_report_1117/final
