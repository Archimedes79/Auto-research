cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 ../autoresearch/tools/ledger.py note E261.g \
  --result "rc1 threshold-only frontier (VAL 873, in-sample): best admissible point tau 0.2 / pruning 0.4: lanes found 0.7778, R@1 far 0.7811, Q 0.7355, DP 13.48, dup 0.0831; tau 0.1 / 0.5 reaches 0.7801 but fails the dup guard (0.1102 > 0.1063); tau <= 0.1 at 0.4 and 0.05 at 0.5 fail Q >= 0.735" \
  --decision "rung passes (0.7778 < kill line 0.7802; prediction 0.777-0.779 hit): E261.cv next; snapshot tag missing (run.py: code git-error)"
python3 ../autoresearch/tools/ledger.py note E265.c --result "8 control cards re-scored with tonight's code, all reproduce their stored Q / lanes found / R@1 far" --decision "controls valid; snapshot tag missing (run.py: code git-error)"
