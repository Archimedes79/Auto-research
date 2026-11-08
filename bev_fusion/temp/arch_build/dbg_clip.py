"""one-off: step 2 of dryrun.py on the CPU (an index error names its line there): one clip of 3 keyframes through train.run_clip,
flip forced on (argument 'noflip' for the control)"""
import dataclasses
import os
import sys
import traceback

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.environ.get("DBG_CODE", os.path.join(ROOT, "training")))      # old snapshot or the live tree
DEV, AMP = os.environ.get("DBG_DEVICE", "cpu"), os.environ.get("DBG_AMP", "none")
os.chdir(ROOT)
import train as T  # noqa: E402
from feature_cache import collate_cached  # noqa: E402

flip = 0.0 if "noflip" in sys.argv else 1.0
cfg = dataclasses.replace(T.read_config("runs/t10_dual_long_s0/config.json"), flip_aug=flip)
model, loss, _ = T.build(cfg)
ck = torch.load("runs/t10_dual_long_s0/ckpt_00005030.pt", map_location="cpu", weights_only=False)
model.load_state_dict({k[6:]: v for k, v in ck["state"].items() if k.startswith("model.")}, strict=False)
val = T.make_datasets(cfg)[1]
clips = T.ClipDataset(val, 3)
clip = T.collate_clips([clips[0], clips[len(clips) // 2]], collate_cached)
model, loss = model.to(DEV).train(), loss.to(DEV)
print("code:", sys.path[0], "device", DEV, "amp", AMP, flush=True)
try:
    acc, finite, _, totals = T.run_clip(model, loss, clip, 0.5, DEV, AMP, cfg, backward=True, augment=True)
    print("flip", flip, "finite", finite, "totals", totals)
except Exception:
    traceback.print_exc()
