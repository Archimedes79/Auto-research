#!/bin/bash
# scratch: rebuild the big report and copy it with its videos and tables into the result folder Mirko asked for (2026-11-03): bev_fusion/results/BEV_result_report/
# (not in git: binary results). Re-run whenever a video or a number is added.
cd /mnt/c/Auto-research || exit 1
nice -n 15 /home/ml/miniconda3/envs/gcar/bin/python bev_fusion/temp/t16/build_big_report.py || exit 1
D=bev_fusion/results/BEV_result_report
mkdir -p $D/videos $D/tables
cp bev_fusion/results/t16/report_big.html $D/report.html
cp -r bev_fusion/results/t16/videos/. $D/videos/
cp -r bev_fusion/results/t16/tables/. $D/tables/
cat > $D/README.txt <<'EOF'
BEV camera + radar system - result report (made 2026-11-02 / 03 by the BEV system chat for Mirko)

Open report.html in a browser (Chrome / Edge). It needs the folder videos/ next to it for the video players; fonts come from the internet if available, otherwise the browser's default fonts are used.

  report.html      the report: what was tried (23 experiments), results (tables, paired comparisons), lane markings, comparison with the state of the art, pictures, videos, cost, limits
  videos/<run>/    result videos (mp4): <scene>_overview.mp4 = camera with true boxes and map lines, true BEV, prediction, error maps incl. lane dividers; <scene>_dashboard.mp4 = every stage of the chain
                   t16_full_s0 = the baseline (full network, 428 scenes), t17_data = the same recipe on 700 scenes, t18_fix = the bundle arm (cut by its rule; the baseline t16_full_s0 is the delivered system)
  tables/          the stored evaluation tables of the runs (one line per metric): <run>_table_n1000.txt (validation, 1000 frames, ground truth v8), <run>_table_train.txt (train scenes, for the generalisation gap)

The same report as a private web page (needs your Claude login): https://claude.ai/artifact/KoJDweCbGxz4TAzMfntGNo
The working records are in the repository (github.com/Archimedes79/Auto-research): bev_fusion/experiments/ (cards), bev_fusion/records/ (ledger).
EOF
find $D -type f | wc -l
du -sh $D | cut -f1
ls $D $D/videos
