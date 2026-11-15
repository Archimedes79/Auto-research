"""scratch (t22 stage B): write training/configs/t22_B_s0.json = t21_pair4.json with the student switched on (the lines of the card), nothing else changed.

    python temp/t22/make_config_B.py [--anchor 0.5] [--student runs/t22_A2_s0/ckpt_00006108.pt] [--base training/configs/t21_pair4.json] [--out training/configs/t22_B_s0.json]
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="training/configs/t21_pair4.json")
ap.add_argument("--out", default="training/configs/t22_B_s0.json")
ap.add_argument("--student", default="runs/t22_A2_s0/ckpt_00006108.pt")
ap.add_argument("--anchor", type=float, default=0.5)
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(f"{a.out} exists already")
if not os.path.isfile(a.student):
    raise SystemExit(f"{a.student}: a checkpoint FILE (a pinned one, not a run folder)")
cfg = json.load(open(a.base))
base_name = cfg["name"]
cfg["_about"] = (f"t22 stage B (card experiments/t22_student_encoder): {os.path.basename(a.base)} ({base_name}: the t21 one-frame pair chain, warm start runs/t20_base_s0, 1500 steps) with the EfficientViT-B1 "
                 f"student on the key-frame JPEG replacing the cached teacher outputs of the current frame (DualBEV and BEVFusion depth + context, and the depth the res4 lane branch lifts with); older frames keep the "
                 f"cached teacher outputs. Differences from {base_name}: name, model.camera_student / student_init / student_trainable, lr_groups student 1e-5 (0.1 x lr), grad_clip_split += student, "
                 f"student_anchor {a.anchor} (loss scale: the t21 total is about 7.5 at the end, the student's own distillation total about 1.2, so the anchor is about {a.anchor * 1.2 / 7.5 * 100:.0f} % of the total at the start and "
                 f"{a.anchor * 0.1 * 1.2 / 7.5 * 100:.1f} % at its floor), floor 0.1, until 0.5 of the steps. The student is the A2 distillation checkpoint (student_init); BatchNorm stays in eval mode.")
cfg["name"] = "t22_B_s0"
cfg["model"].update({"camera_student": "evit_b1", "student_init": a.student, "student_trainable": True, "student_frames": "current"})
cfg["lr_groups"] = {"student": {"lr": cfg["lr"] * 0.1}}
cfg["grad_clip_split"] = list(cfg.get("grad_clip_split", [])) + ["student"]
cfg["student_anchor"] = a.anchor
cfg["student_anchor_floor"] = 0.1
cfg["student_anchor_until"] = 0.5
json.dump(cfg, open(a.out, "w"), indent=1)
print("wrote", a.out)
