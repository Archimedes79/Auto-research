"""One-off (experiment t2_block_swaps): write training/configs/t2_<arm>.json = the reference config with ONE block exchanged.
Usage: python3 temp/study/make_configs.py [steps]   (default 4000)"""
import copy
import json
import os
import sys

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
base = json.load(open(os.path.join(root, "training/configs/t1_r4_soa.json")))
base.update({"accum": 1, "steps": steps, "save_every": 500, "keep": 2, "eval_every": 1000, "workers": 6})

# reference model = t1_r4_soa: DualBEV image heads (frozen) -> lift PD -> window memory -> radar raster -> concat fusion ->
# DualBEV BEV encoder (pre-trained, width 1.0, detail 32) -> 96-ch light heads
ARMS = {
    "ref_s0": ({}, 0),
    "ref_s1": ({}, 1),
    "lift_p": ({"lift": "p"}, 0),                                     # pull without depth weight (Simple-BEV / Fast-BEV)
    "lift_s": ({"lift": "s"}, 0),                                     # forward splat (LSS / BEVPoolV2)
    "camtemp_none": ({"cam_temporal": "none"}, 0),                    # current frame only
    "camtemp_gru": ({"cam_temporal": "ego_gru"}, 0),                  # FIERY-style ego warp + ConvGRU
    "radar_off": ({"radar_kind": "off"}, 0),                          # camera-only system
    "radar_pillars": ({"radar_kind": "pillars"}, 0),                  # point MLP + scatter-max (PointPillars family)
    "cam_off": ({"camera_off": True}, 0),                             # radar-only system
    "fusion_gate": ({"fusion": "gate"}, 0),                           # GaussianCaR-style rectify gates
    "fusion_cmx": ({"fusion": "cmx"}, 0),                             # CMX four-stage fuser (replaces fusion + trunk)
    "trunk_rand": ({"trunk_pretrained": ""}, 0),                      # same encoder, random init
    "trunk_small": ({"trunk_pretrained": "", "trunk_width": 0.5}, 0),  # width 0.5, random init
    "trunk_second": ({"trunk_kind": "second", "trunk_pretrained": "", "trunk_detail": 0}, 0),   # SECOND + SECONDFPN
}
for name, (mod, seed) in ARMS.items():
    c = copy.deepcopy(base)
    c["name"] = f"t2_{name}"
    c["seed"] = seed
    c["model"].update(mod)
    if name == "trunk_second":
        c["model"].pop("trunk_pretrained", None)
        c["model"]["trunk_pretrained"] = ""
    json.dump(c, open(os.path.join(root, f"training/configs/t2_{name}.json"), "w"), indent=1)
print(f"wrote {len(ARMS)} configs, {steps} steps each")
