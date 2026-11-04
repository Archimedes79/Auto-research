import json
import sys

d = json.load(open(sys.argv[1]))
S = d["systems"]
print(f"run {d['run']} step {d['step']}, {d['n_frames']} frames, GaussianCaR forward offset {d['offset_m']} m")
layers = ["drivable_area", "walkway", "carpark_area", "ped_crossing", "lane_divider", "road_divider"]
for tol in ("iou@0", "iou@0.5", "iou@1"):
    print(f"\n{tol}, prediction threshold 0.5 (best of 0.3/0.5/0.7 in brackets), camera cells, all ranges")
    print("%-14s %8s %8s %8s   %s" % ("layer", "prior", "ours", "GaussCaR", "ratio ours/GaussCaR"))
    for L in layers:
        row = []
        for s in ("prior", "ours", "gcar"):
            v = S[s][L]["all"]
            best = max(v[t][tol] for t in v)
            row.append((v["0.5"][tol], best))
        print("%-14s %8.3f %8.3f %8.3f   %.2f     (best thr: %.3f %.3f %.3f)" % (L, row[0][0], row[1][0], row[2][0], row[1][0] / row[2][0], row[0][1], row[1][1], row[2][1]))
print("\nper range bin, iou@0 (drivable, walkway) and iou@0.5 (dividers), threshold 0.5: prior / ours / GaussianCaR")
for L, tol in (("drivable_area", "iou@0"), ("walkway", "iou@0"), ("lane_divider", "iou@0.5"), ("road_divider", "iou@0.5")):
    print("%-14s" % L, "   ".join("%s: %.3f / %.3f / %.3f" % (b, *[S[s][L][b]["0.5"][tol] for s in ("prior", "ours", "gcar")]) for b in ("0-20", "20-40", "40-60")))
print("\nn_gt cells (threshold 0.5):", {L: S["ours"][L]["all"]["0.5"]["n_gt"] for L in layers})
