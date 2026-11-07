"""throw-away: GaussianCaR numbers of the old scoring files against the v8 run of soa_maps_v8 --no-model --rule old400. usage: from bev_fusion"""
import json
import sys

new = json.load(open(sys.argv[1]))
print("new:", new["meta"]["n_frames"], "frames, GT", new["meta"]["gt_version"], new["meta"]["gt_signature"], "sources", new["meta"]["gt_sources"])
print("gcar IoU by offset:", new["meta"].get("gcar_iou_by_offset_m"))
LAY = ["drivable_area", "walkway", "carpark_area", "ped_crossing", "lane_divider", "road_divider"]
olds = {n: json.load(open(f"temp/soa_gap/{n}.json")) for n in ("map_ref_r4soa_400", "map_ref_t10_dual_s0_400")}
for sysname in ("gcar", "prior"):
    print(f"\n== {sysname}, thr 0.5, all ranges: old(r4soa file) / old(t10 file) / v8  (n_gt old / v8)")
    for tol in ("iou@0", "iou@0.5", "iou@1"):
        for L in LAY:
            o1 = olds["map_ref_r4soa_400"]["systems"][sysname][L]["all"]["0.5"]
            o2 = olds["map_ref_t10_dual_s0_400"]["systems"][sysname][L]["all"]["0.5"]
            n = new["systems"][sysname][L]["all"]["0.5"]
            print(f"  {tol:8s} {L:14s} {o1[tol]:.4f} / {o2[tol]:.4f} / {n[tol]:.4f}   diff {n[tol] - o1[tol]:+.4f}   n_gt {o1['n_gt']} / {n['n_gt']}")
print("\nbest-of thresholds, gcar iou@0 / iou@0.5:")
for L in LAY:
    for tol in ("iou@0", "iou@0.5"):
        ob = max(olds["map_ref_r4soa_400"]["systems"]["gcar"][L]["all"][t][tol] for t in ("0.3", "0.5", "0.7"))
        nb = max(new["systems"]["gcar"][L]["all"][t][tol] for t in ("0.3", "0.5", "0.7"))
        print(f"  {L:14s} {tol:8s} old {ob:.4f} v8 {nb:.4f}")
