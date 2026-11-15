"""scratch (t22 gate A (ii), the stand-in test): write the run folder of an evaluation that runs an existing chain run's checkpoint under the LIVE code, with the camera student in place of
the cached teachers (the stand-in) or with the cached teachers (the twin: the live-code control).

    python temp/t22/chain/make_standin_run.py CHAIN_RUN NEW --student PATH [--kind evit_b1]    the stand-in: PATH = a distillation checkpoint file (runs/t22_A2_s0/ckpt_00001234.pt) or a run
                                                                      folder (its newest checkpoint is resolved NOW and the file written down)
    python temp/t22/chain/make_standin_run.py CHAIN_RUN NEW --twin                              the twin: the same recipe, no student keys
    python temp/t22/chain/make_standin_run.py CHAIN_RUN NEW --check                             after the evaluation: NEW/eval/card.json against CHAIN_RUN's (GT signature, frame hash, frames, split)

NEW/config.json = CHAIN_RUN/config.json with name = NEW's name and, for the stand-in only, model.camera_student = kind, model.student_init = the resolved checkpoint FILE,
model.student_trainable = false (asserted: nothing else differs); NEW/standin.json = what was resolved (the file, its step, the source, the expected card values); gt.json is copied. No `code/`
folder: the evaluation imports the live training/, for the stand-in and the twin alike (the reference CHAIN_RUN evaluated with its own snapshot; the twin is the control of that difference).
The checkpoints stay in CHAIN_RUN (--ckpt-dir). Prints the evaluation command and the card values the new run must show (CHAIN_RUN's own). Nothing is overwritten."""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("chain_run")
ap.add_argument("new")
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--student", metavar="PATH")
g.add_argument("--twin", action="store_true")
g.add_argument("--check", action="store_true")
ap.add_argument("--kind", default="evit_b1", choices=["evit_b1", "resnet18"])
ap.add_argument("--frames", default="current", choices=["current", "all"], help="all: the student also makes the OLDER frames of the pair (model.student_frames; needs <lane_images>/sweeps)")
a = ap.parse_args()


def card_values(run):
    """(gt_signature, frame_hash, frames, split) of RUN/eval/card.json, None when there is none"""
    p = os.path.join(run, "eval", "card.json")
    if not os.path.exists(p):
        return None
    c = json.load(open(p))
    return c["protocol"]["gt_signature"], c["provenance"]["data"]["frame_hash"], c["frames"], c["split"]


want = card_values(a.chain_run)
if a.check:
    got = card_values(a.new)
    if want is None or got is None:
        sys.exit(f"no eval/card.json in {a.chain_run if want is None else a.new}")
    for label, w, g_ in zip(("gt_signature", "frame_hash", "frames", "split"), want, got):
        print(f"{'ok  ' if w == g_ else 'DIFF'} {label}: {a.new} {g_!r}, {a.chain_run} {w!r}")
    sys.exit(0 if want == got else 1)

cfg = json.load(open(os.path.join(a.chain_run, "config.json")))
src = json.loads(json.dumps(cfg))
if cfg["model"].get("camera_features") != "dual":
    sys.exit("the stand-in needs a chain with camera_features 'dual'")
if not cfg.get("lane_images"):
    sys.exit("the chain config has no lane_images: the student has no image to read")
if os.path.exists(a.new):
    sys.exit(f"{a.new} exists already")
cfg["name"] = os.path.basename(os.path.normpath(a.new))
info = {"source_run": a.chain_run, "mode": "twin" if a.twin else "student", "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_config_sha256": hashlib.sha256(open(os.path.join(a.chain_run, "config.json"), "rb").read()).hexdigest()[:16],
        "code": "the live training/ (no code/ folder in the run)", "expected_card": dict(zip(("gt_signature", "frame_hash", "frames", "split"), want)) if want else None}
if a.student:
    if os.path.isdir(a.student):
        files = sorted(glob.glob(os.path.join(a.student, "ckpt_*.pt")))
        if not files:
            sys.exit(f"{a.student} holds no ckpt_*.pt")
        file = files[-1]
    elif os.path.isfile(a.student):
        file = a.student
    else:
        sys.exit(f"{a.student} does not exist")
    import torch
    ck = torch.load(file, map_location="cpu", weights_only=False)
    cfg["model"].update(camera_student=a.kind, student_init=file, student_trainable=False)
    if a.frames == "all":
        cfg["model"]["student_frames"] = "all"
    info.update(student_given=a.student, student_init=file, student_step=ck["step"], student_backbone=ck["cfg"]["backbone"], kind=a.kind, student_frames=a.frames)
    if (ck["cfg"]["backbone"], a.kind) not in (("efficientvit_b1", "evit_b1"), ("resnet18", "resnet18")):
        sys.exit(f"the checkpoint is a {ck['cfg']['backbone']} student, --kind {a.kind} does not fit it")
# the produced config differs from the source's only in the name and, for the stand-in, the three student keys of the model
top = {k for k in set(cfg) | set(src) if cfg.get(k) != src.get(k)}
mod = {k for k in set(cfg["model"]) | set(src["model"]) if cfg["model"].get(k) != src["model"].get(k)}
assert top == ({"name", "model"} if a.student else {"name"}), top
assert mod == ({"camera_student", "student_init", "student_trainable"} | ({"student_frames"} if a.frames == "all" else set()) if a.student else set()), mod
os.makedirs(a.new)
json.dump(cfg, open(os.path.join(a.new, "config.json"), "w"), indent=1)
json.dump(info, open(os.path.join(a.new, "standin.json"), "w"), indent=1)
if os.path.exists(os.path.join(a.chain_run, "gt.json")):
    shutil.copy(os.path.join(a.chain_run, "gt.json"), os.path.join(a.new, "gt.json"))
print(f"made {a.new} ({info['mode']}" + (f": {info['student_init']}, step {info['student_step']}" if a.student else "") + "); differs from the source in: name" + (", model.camera_student / student_init / student_trainable" if a.student else ""))
print("evaluate with (from bev_fusion/, in WSL; the reference ran --curves --fusion --velocity eval --n 1000):")
print(f"  python evaluation/evaluate.py {a.new} --quant --curves --fusion --velocity --n 1000 --ckpt-dir {a.chain_run}")
if want:
    print(f"the card of {a.new} must show gt_signature {want[0]}, frame_hash {want[1]}, frames {want[2]}, split {want[3]}; after the evaluation: python temp/t22/chain/make_standin_run.py {a.chain_run} {a.new} --check")
else:
    print(f"note: {a.chain_run} has no eval/card.json: no card values to check against")
