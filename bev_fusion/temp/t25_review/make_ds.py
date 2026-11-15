import os
import sys

ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(1, "/mnt/c/Auto-research")
os.chdir(ROOT)
import train as T  # noqa: E402

cfg = T.TrainConfig(data_root="~/data/nuscenes", version="v1.0-mini", feature_cache="", feature_cache2="", gt_cache="", radar_cache="", lane_images="",
                    n_cam=3, cam_stride=2, radar_window=0.09, model={"radar_per_frame": True})
tr, va = T.make_datasets(cfg)
it = tr[3]
print("keys has radar_seq:", "radar_seq" in it, len(it["radar_seq"]), "window", tr.radar_window, "per_frame", tr.radar_per_frame)
print("max age per frame:", [round(float(r["dt"].max()), 4) for r in it["radar_seq"]], "n points", [len(r["x"]) for r in it["radar_seq"]])
cfg0 = T.TrainConfig(data_root="~/data/nuscenes", version="v1.0-mini", feature_cache="", feature_cache2="", gt_cache="", radar_cache="", lane_images="", n_cam=3, cam_stride=2)
tr0, _ = T.make_datasets(cfg0)
it0 = tr0[3]
print("default: radar_seq in item:", "radar_seq" in it0, "window", tr0.radar_window, "max age", round(float(it0["radar"]["dt"].max()), 3))
b = T.collate([tr[3], tr[4]])
print("collated radar_seq sets", len(b["radar_seq"]), [int(r["b"].max()) for r in b["radar_seq"]])
