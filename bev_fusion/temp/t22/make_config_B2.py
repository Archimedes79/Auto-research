"""scratch (t22 refinement C, stage B2): write training/configs/t22_B2_s0.json = t22_B_s0.json with model.student_frames 'all' (the student also makes the older frame of the P4 pair), nothing else changed.

    python temp/t22/make_config_B2.py
"""
import json
import os

base, out = "training/configs/t22_B_s0.json", "training/configs/t22_B2_s0.json"
if os.path.exists(out):
    raise SystemExit(f"{out} exists already")
cfg = json.load(open(base))
cfg["_about"] = ("t22 refinement C, stage B2 (card experiments/t22_student_encoder): t22_B_s0.json (the EfficientViT-B1 student in the t21 P4 pair chain, warm start runs/t20_base_s0, 1500 steps) with "
                 "model.student_frames 'all': the student also makes the OLDER frame of the pair (a CAM_FRONT sweep 0.333 s earlier, its image from data/cache/img_front/sweeps, under no_grad), so the pair is "
                 "made by one source in training and in the stream. Differences from t22_B_s0: name, model.student_frames. The student trains by the current frame as before.")
cfg["name"] = "t22_B2_s0"
assert cfg["model"]["student_frames"] == "current"
cfg["model"]["student_frames"] = "all"
json.dump(cfg, open(out, "w"), indent=1)
print("wrote", out)
