"""One-off (t3_improve_loop): configs of the tries that need training beyond the swap study.
t3_adapter   try 6: ref_s0 + trainable context adapter (cam_adapter)
t3_ref_long  try 7: ref_s0 with 6000 steps (does the schedule limit the system?)
Usage: python3 temp/study/make_t3_configs.py (run from bev_fusion/)"""
import copy
import json
import os

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
base = json.load(open(os.path.join(root, "training/configs/t2_ref_s0.json")))
ARMS = {"t3_adapter": ({"cam_adapter": True}, {}), "t3_ref_long": ({}, {"steps": 6000, "eval_every": 2000})}
for name, (model, top) in ARMS.items():
    c = copy.deepcopy(base)
    c["name"] = name
    c["model"].update(model)
    c.update(top)
    json.dump(c, open(os.path.join(root, f"training/configs/{name}.json"), "w"), indent=1)
    print(name, model, top)
