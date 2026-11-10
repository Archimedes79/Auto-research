"""scratch: the three configs of the twin / R-arm chain of 2026-11-09 (D67), derived from the committed stage-B and control configs; only the listed keys differ.
t21_pair4_s1  = t21_pair4 with seed 1 (the control of the twin)
t22_B_s1      = t22_B_s0 with seed 1 (the student of the twin)
t22_R_s0      = t22_B_s0 with camera_student resnet18 and student_init = the ResNet-18 distillation run t22_A2R_s0 (final step 6108 = 6 epochs of 16,289 samples at batch 16)"""
import json
import os

C = "training/configs/"


def load(n):
    return json.load(open(C + n, encoding="utf-8"))


def save(n, cfg):
    s = json.dumps(cfg, indent=1, ensure_ascii=False) + "\n"
    open(C + n, "w", encoding="utf-8", newline="\n").write(s)
    print("wrote", C + n)


ctl = load("t21_pair4.json")
ctl["name"] = "t21_pair4_s1"
ctl["seed"] = 1
ctl["_about"] = ("Seed twin of the control (D67, 2026-11-09; card experiments/t22_student_encoder, 'Seed twin and R arm'): t21_pair4.json with seed 1 and nothing else changed: "
                 "the matched partner of t22_B_s1. " + ctl["_about"])
save("t21_pair4_s1.json", ctl)

b = load("t22_B_s0.json")
s1 = json.loads(json.dumps(b))
s1["name"] = "t22_B_s1"
s1["seed"] = 1
s1["_about"] = ("Seed twin of stage B (D67, 2026-11-09; card experiments/t22_student_encoder, 'Seed twin and R arm'): t22_B_s0.json with seed 1 and nothing else changed "
                "(same A2 student checkpoint, same recipe). " + b["_about"])
save("t22_B_s1.json", s1)

r = json.loads(json.dumps(b))
r["name"] = "t22_R_s0"
r["model"]["camera_student"] = "resnet18"
r["model"]["student_init"] = "runs/t22_A2R_s0/ckpt_00006108.pt"
r["_about"] = ("R arm of the t22 card (D67, 2026-11-09; card experiments/t22_student_encoder, 'Seed twin and R arm'): t22_B_s0.json with the ResNet-18 student "
               "(model.camera_student resnet18, student_init = the distillation run t22_A2R_s0, same recipe as A2: 6 epochs, batch 16, lr 1e-3, backbone 0.1 x); "
               "everything else, the anchor and the learning rates included, as stage B. " + b["_about"])
save("t22_R_s0.json", r)

for n in ("t21_pair4_s1.json", "t22_B_s1.json", "t22_R_s0.json"):
    base = {"t21_pair4_s1.json": "t21_pair4.json"}.get(n, "t22_B_s0.json")
    a, c = load(base), load(n)
    diff = sorted(k for k in set(a) | set(c) if k != "_about" and a.get(k) != c.get(k))
    mdiff = sorted(k for k in set(a.get("model", {})) | set(c.get("model", {})) if a.get("model", {}).get(k) != c.get("model", {}).get(k))
    print(n, "differs from", base, "in", diff, "and model keys", mdiff)
