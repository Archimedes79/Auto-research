cd /mnt/c/Auto-research/bev_fusion_lane_center
L="python3 ../autoresearch/tools/ledger.py note"
$L E266.c --result "layer 1 after the S2 code (ring metric): recall, recall_s1, fast_recall, e24_j1rc_s1 reproduce their stored Q / lanes found / R@1 far (cards val_l22r)" --decision "control cards of E266 / E267 valid"
$L E267.s --result "arm s (spurs kept, ring metric) vs c: recall lanes found +0.013 [+0.010, +0.017], R@1 far +0.008, Q +0.003; recall_s1 +0.013 [+0.010, +0.016], +0.008, +0.003 (reproduces E265.a)" --decision "arm s of the 2 x 2; E267 waits for the ring arms (E266.b0 / b1)"
