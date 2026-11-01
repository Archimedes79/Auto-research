cd /mnt/c/Auto-research/bev_fusion_lane_center
L="python3 ../autoresearch/tools/ledger.py note"
S="python3 ../autoresearch/tools/schedule.py set"
$L E266.gold --result "layer 0 / 1 after the loop-22 code (S1 cache key, S2 seed ring + ring metric, S3 union kind): golden numbers reproduced (6 entries, cached), run_all_tests.sh all PASS (builder 18/18, simplify_dp 9/9, seed ring + union 7, evaluate guard 10, lane prototype selftest, numba union == FastUnion)" --decision "harness valid for E264 / E266 / E267"
$L E264.u23 --result "VAL 873: union s2 + s3 vs s3: lanes found +0.047 [+0.034, +0.060], R@1 far +0.040 [+0.027, +0.053], Q -0.003 [-0.010, +0.005], topo F1 +0.002, dup +0.017 [+0.012, +0.021], false length 0.056 -> 0.064 (absolute guard 0.062 exceeded by 0.002), node-id vert100 16.65" --decision "pending E264.dec (needs E264.v DP, the matched-false-length control, the exploration pair and the idle timing)"
for i in S5.commit S1.code S2.code S3.code L22.reg; do $S $i done; done
