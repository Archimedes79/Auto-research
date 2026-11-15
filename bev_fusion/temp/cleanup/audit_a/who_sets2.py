"""audit scratch: for a list of (section, field, value) show the stored configs (distinct names) that set it."""
import glob
import json
import os

ROOT = "/mnt/c/Auto-research/bev_fusion"
CHECKS = [
    ("model", "radar_kind", "pillars"), ("model", "radar_kind", "off"), ("model", "radar_kind", "delta_slots"),
    ("model", "camera_features", "bevfusion"), ("model", "cam_buffer", "recurrent"), ("model", "cam_buffer", "pair"),
    ("model", "lane_branch", "rgb"), ("model", "lane_branch", "res4"),
    ("model", "rad_temporal", "window_memory"), ("model", "cam_temporal", "none"), ("model", "cam_temporal", "ego_gru"),
    ("model", "fusion", "gate"), ("model", "fusion", "cmx"), ("model", "lift", "s"), ("model", "lift", "p"),
    ("model", "student_frames", "all"), ("model", "camera_student", "resnet18"), ("model", "camera_student", "evit_b1"),
    ("model", "radar_stationary", False), ("model", "centreline_1m", True), ("model", "object_moving", True),
    ("model", "motion_block", True), ("model", "box_vel_from_field", True),
    ("model", "share_mover", True), ("model", "splat_fused", True), ("model", "sync_free", True),
    ("model", "trunk_pretrained", ""), ("model", "head_channels", 96), ("model", "trunk_detail", 0),
    ("loss", "hygiene_2026_11_02", True), ("loss", "line_bce", 10.0), ("loss", "radar_stationary", False),
    ("loss", "w_obj_vel", 10.0), ("loss", "sync_free", True),
]
cfgs = {}
for p in sorted(glob.glob(f"{ROOT}/training/configs/*.json") + glob.glob(f"{ROOT}/runs/*/config.json")):
    try:
        c = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    name = os.path.basename(os.path.dirname(p)) if "/runs/" in p else os.path.splitext(os.path.basename(p))[0]
    cfgs.setdefault(name, c)
print("distinct config names:", len(cfgs))
for sec, f, v in CHECKS:
    names = sorted(n for n, c in cfgs.items() if isinstance(c, dict) and (c.get(sec) or {}).get(f, "__none__") == v)
    print(f"{sec}.{f}={v!r}: {len(names)} distinct: " + ", ".join(names[:14]) + (" ..." if len(names) > 14 else ""))
