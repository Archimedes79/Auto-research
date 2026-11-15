"""scratch (t22 gate C0, diagnostic b): the run folder that evaluates the STAGE-B checkpoint (runs/t22_B_s0, a student that trained with the cached teachers on the older frame)
with `student_frames all` (the student makes the older frame too). NEW/config.json = B's config with name and model.student_frames changed (asserted: nothing else differs);
gt.json is copied; the checkpoints stay in runs/t22_B_s0 (--ckpt-dir).

    python temp/t22/chain/make_b_all.py runs/t22_B_s0 runs/t22_B_all
"""
import json
import os
import shutil
import sys

src_run, new = sys.argv[1], sys.argv[2]
if os.path.exists(new):
    sys.exit(f"{new} exists already")
cfg = json.load(open(os.path.join(src_run, "config.json")))
src = json.loads(json.dumps(cfg))
if not cfg["model"].get("camera_student") or not cfg["model"].get("student_trainable"):
    sys.exit("the source run has no trainable camera student")
cfg["name"] = os.path.basename(os.path.normpath(new))
cfg["model"]["student_frames"] = "all"
top = {k for k in set(cfg) | set(src) if cfg.get(k) != src.get(k)}
mod = {k for k in set(cfg["model"]) | set(src["model"]) if cfg["model"].get(k) != src["model"].get(k)}
assert top == {"name", "model"} and mod == {"student_frames"}, (top, mod)
os.makedirs(new)
json.dump(cfg, open(os.path.join(new, "config.json"), "w"), indent=1)
json.dump({"source_run": src_run, "mode": "stage-B checkpoint, student_frames all", "code": "the live training/"}, open(os.path.join(new, "standin.json"), "w"), indent=1)
if os.path.exists(os.path.join(src_run, "gt.json")):
    shutil.copy(os.path.join(src_run, "gt.json"), os.path.join(new, "gt.json"))
print(f"made {new} (differs from {src_run} in: name, model.student_frames)")
print(f"  python evaluation/evaluate.py {new} --quant --curves --fusion --velocity --n 1000 --ckpt-dir {src_run}")
