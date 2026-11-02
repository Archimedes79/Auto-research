"""scratch of the t16 bring-up (2026-11-02): writes training/configs/t16_full.json and presets/small_set.json from t10_dual_long_s0.json.
Run from the repository root: python3 bev_fusion/temp/t16/make_configs.py"""
import json
import os

CFG = "bev_fusion/training/configs"
MINI_TRAIN = ["scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-1077", "scene-1094", "scene-1100"]
MINI_VAL = ["scene-0103", "scene-0916"]

with open(f"{CFG}/t10_dual_long_s0.json", encoding="utf-8") as f:
    base = json.load(f)

full = {"_about": ("t16 full network, the bring-up of the complete chain on the five-layer framework and GT v8 (Mirko, 2026-11-02: generate a full "
                   "network with fusion, train it on a small set, time it, probe every module, fix, retrain, then the large set as the baseline): "
                   "the recipe of t10_dual_long_s0 (dual camera features, 6000 steps, accumulation 2, velocity hurdle from step 0, no init_from) "
                   "plus every block that is built and has a passing gate: the radar delta-slot encoder R (c_rad 64, side loss radar_aux), the "
                   "radar memory with the Doppler static gate, the motion block M in its joint form with the object head's velocity read from "
                   "its field; probe on (grad.<module>, upd.<module>, t_data / t_step in every logged row). The small-set run is this file with "
                   "--preset small --preset small_set (the ten scenes of autogt's small set).")}
for k, v in base.items():
    if k != "_about":
        full[k] = v
full["name"] = "t16_full"
model = dict(full["model"])
model.update(radar_kind="delta_slots", c_rad=64, rad_temporal="window_memory", rad_static_gate=True,
             motion_block=True, motion_strides=[4, 2, 1], box_vel_from_field=True)
full["model"] = model
full.update(radar_aux=1.0, radar_aux_floor=0.05, radar_aux_half=0.2, probe=True)
with open(f"{CFG}/t16_full.json", "w", encoding="utf-8") as f:
    json.dump(full, f, indent=1)
    f.write("\n")

small_set = {"_about": "the ten scenes of autogt's small set (sets/small.txt; nuScenes mini's split: 8 train, 2 val) for a bring-up on the small data: "
                       "their feature shards are linked in data/cache/small/ (cam_feats: DualBEV, cam_feats_bevfusion: BEVFusion; "
                       "temp/t16/prepare_small.sh builds the missing ones), GT and radar caches are the shared folders (keyed by signature)",
             "feature_cache": "data/cache/small/cam_feats", "feature_cache2": "data/cache/small/cam_feats_bevfusion",
             "train_scenes": MINI_TRAIN, "val_scenes": MINI_VAL, "eval_batches": 20}
with open(f"{CFG}/presets/small_set.json", "w", encoding="utf-8") as f:
    json.dump(small_set, f, indent=1)
    f.write("\n")
print("written", f"{CFG}/t16_full.json", f"{CFG}/presets/small_set.json")
print({k: full["model"][k] for k in ("camera_features", "radar_kind", "rad_temporal", "motion_block", "object_moving")}, full["steps"], full["accum"])

# the control of the baseline (S7) and of the small-set A/B: the recipe of t10_dual_long_s0 unchanged, probe on, GT v8 through the data
ctl = {"_about": ("t16 control: the matched reference of t16_full (experiments/t16_full_network): t10_dual_long_s0's recipe with none of the new blocks "
                  "(no delta slots, no radar memory, no motion block), same seed, same data, probe on (logging only); the GT is whatever the live "
                  "generator is (v8, signature c643a7539d)")}
for k, v in base.items():
    if k != "_about":
        ctl[k] = v
ctl["name"] = "t16_ctl"
ctl["probe"] = True
with open(f"{CFG}/t16_ctl.json", "w", encoding="utf-8") as f:
    json.dump(ctl, f, indent=1)
    f.write("\n")
print("written", f"{CFG}/t16_ctl.json")
