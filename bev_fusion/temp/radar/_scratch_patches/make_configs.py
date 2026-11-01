"""writes the six radar-slot comparison configs (radar-only, camera off) next to the existing ones; run from the bev_fusion folder."""
import copy
import json

base = json.load(open("training/configs/t3_radar_only.json", encoding="utf-8"))
ARMS = {"raster32": ({}, {}),                                                        # shipped radar block, 32 channels (the reference)
        "raster64": ({"c_rad": 64}, {}),                                             # matched width: the encoder must beat more channels
        "slots64": ({"radar_kind": "delta_slots", "c_rad": 64}, {"radar_aux": 1.0, "radar_aux_floor": 0.05, "radar_aux_half": 0.2})}
for arm, (model_kw, train_kw) in ARMS.items():
    for seed in (0, 1):
        c = copy.deepcopy(base)
        c["name"] = f"r2_rad_{arm}_s{seed}"
        c["seed"] = seed
        c["steps"] = 1500                                  # accum 2: the same 9000 samples as the 3000 steps of t3_radar_only
        c["accum"] = 2                                    # effective batch 6: every run with 3 had a dead velocity branch (t11)
        c["warmup"] = 150
        c["eval_every"] = 500
        c["model"]["object_moving"] = True                # the t11 hurdle: moving / static logit, velocity on moving centres only
        c["model"].update(model_kw)
        c.update(train_kw)
        with open(f"training/configs/{c['name']}.json", "w", encoding="utf-8") as f:
            json.dump(c, f, indent=1)
        print(c["name"], c["steps"], c["model"].get("radar_kind", "raster"), c["model"].get("c_rad", 32), c.get("radar_aux", 0))
