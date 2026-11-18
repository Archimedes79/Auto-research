old = "It is stage S1 with the lane head frozen; the past, the map and a joint fine-tune are not in it, VAL has been read by every arm, four output thresholds are tuned, and no TEST number exists."
new = ("The numbers above are stage 1. A second stage trains the node head of the lane head together with the token head (S4): topo F1 0.6682 / 0.6813, S4-HOLDS by its written rule, the champion since D120 (same network, same cost at inference). The past and the map are not in it (both TIE), the pyramid of the lane head stays frozen, VAL has been read by every arm, four output thresholds are tuned, and no TEST number exists.")
for f in ("E2E_Final_Report_2026-11-18", "E2E_Report_2026-11-18"):
    p = "bev_fusion_lane_center/results/%s/report.json" % f
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == 1
    s = s.replace(old, new)
    s = s.replace("status: stage S1 (S2 and S3 TIE; S4 and the E379 / E378 results: see chapters 4 and 5)", "status: two-stage champion since D120 (S4-HOLDS; S2 and S3 TIE; E379 / E378: see chapters 4 and 5)")
    open(p, "w", encoding="utf-8", newline="\n").write(s)
