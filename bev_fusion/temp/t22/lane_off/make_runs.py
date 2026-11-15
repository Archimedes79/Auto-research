"""scratch (L4 check, 2026-11-08): write the two diagnostic run folders that evaluate the checkpoint of runs/t20_base_s0 under the LIVE code
   runs/diag_t20_L4on   the twin: exactly runs/t20_base_s0's config (lane branch on), evaluated with the live evaluator (it also shows whether the stored numbers reproduce)
   runs/diag_t20_L4off  the same config; temp/t22/lane_off/suite_lane_off.py drops the lane branch right after the checkpoint is loaded (model.lane = None: the forward then skips
                        `cam = cam + lane`, which is what a network without the branch computes; the trained rest of the network was NOT trained without it)
Each folder = config.json (the source's, only `name` changed, asserted), gt.json (copied), ablation.json (what it is). The checkpoints stay in runs/t20_base_s0 (--ckpt-dir). Nothing is overwritten.
    python temp/t22/lane_off/make_runs.py"""
import datetime
import json
import os
import shutil

SRC = "runs/t20_base_s0"
cfg = json.load(open(os.path.join(SRC, "config.json")))
assert cfg["model"].get("lane_branch") == "res4", cfg["model"].get("lane_branch")
for new, mode in (("runs/diag_t20_L4on", "twin: lane branch on, live code"), ("runs/diag_t20_L4off", "lane branch dropped after the checkpoint load (suite_lane_off.py)")):
    if os.path.exists(new):
        raise SystemExit(f"{new} exists already")
    out = json.loads(json.dumps(cfg))
    out["name"] = os.path.basename(new)
    top = {k for k in set(out) | set(cfg) if out.get(k) != cfg.get(k)}
    assert top == {"name"}, top
    os.makedirs(new)
    json.dump(out, open(os.path.join(new, "config.json"), "w"), indent=1)
    if os.path.exists(os.path.join(SRC, "gt.json")):
        shutil.copy(os.path.join(SRC, "gt.json"), os.path.join(new, "gt.json"))
    json.dump({"source_run": SRC, "mode": mode, "created": datetime.datetime.now().isoformat(timespec="seconds"),
               "code": "the live training/ and evaluation/ (no code/ folder in the run)"}, open(os.path.join(new, "ablation.json"), "w"), indent=1)
    print("made", new, "-", mode)
