"""One-off (t3_improve_loop): configs of the tries that need training beyond the swap study. Base = t2_ref_s0 (428 train / 82 val scenes).
Camera-only arms (radar_kind off) are the sensitive test of a camera fix: the radar hides camera gains in the full system.
t3_cam_adapter  camera only + trainable context adapter (try 6)      t3_cam_zsum  camera only + height-sum collapse (try 7)
t3_adapter      full system + adapter                                  t3_zsum      full system + height-sum collapse
t3_ref_long     full system, 6000 steps (control of try 8)
Usage: python3 temp/study/make_t3_configs.py (run from bev_fusion/)"""
import copy
import json
import os

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
base = json.load(open(os.path.join(root, "training/configs/t2_ref_s0.json")))
ARMS = {"t3_cam_adapter": ({"radar_kind": "off", "cam_adapter": True}, {}),
        "t3_cam_zsum": ({"radar_kind": "off", "cam_collapse": "zsum"}, {}),
        "t3_adapter": ({"cam_adapter": True}, {}), "t3_zsum": ({"cam_collapse": "zsum"}, {}),
        "t3_ref_long": ({}, {"steps": 6000, "eval_every": 2000})}
for name, (model, top) in ARMS.items():
    c = copy.deepcopy(base)
    c["name"] = name
    c["model"].update(model)
    c.update(top)
    json.dump(c, open(os.path.join(root, f"training/configs/{name}.json"), "w"), indent=1)
    print(name, model, top)
