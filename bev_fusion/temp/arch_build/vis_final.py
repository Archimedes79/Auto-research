"""one-off: the visibility box of training/architecture.json after D36 / D37 (run from bev_fusion/)"""
import json

p = "training/architecture.json"
s = json.load(open(p, encoding="utf-8"))
for n in s["nodes"]:
    if n["id"] == "vis":
        n["label"] = "Visibility maps\ncamera, radar fan, fusion area"
        n["code"] = ["training/precompute.py::Precompute", "training/precompute.py::radar_hits", "training/lift_pd.py::camera_visibility"]
        n["note"] = ("the precompute step (precompute.py): vis_cam from calibration + grid (cached per calibration, the same function live), "
                     "vis_radar = the radar fans (D36; placeholder until autogt.coverage.radar_cells), vis_fusion = camera x radar (D37: "
                     "fusion results are scored there), radar_hits = cells with detections now (the radar memory's write mask); read by the "
                     "network, the training (D36 loss masks) and the evaluation (scoring masks)")
json.dump(s, open(p, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
