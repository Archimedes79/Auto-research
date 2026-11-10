"""t21 scratch: init_from of the four configs on the CPU - the model train.build makes from the composed config takes the t20_base_s0 checkpoint
(EMA tensors) through trainkit.checkpoint.warm_start with no fresh key, and its state_dict keys / shapes are the checkpoint's."""
import os
import subprocess
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
import train as T  # noqa: E402
from trainkit import checkpoint  # noqa: E402

torch.set_num_threads(2)
comp = os.path.join(ROOT, "temp", "t21", "composed")
ck = None
for name in ("t21_pair", "t21_pair4", "t21_rec", "t21_ctl"):
    path = os.path.join(comp, name + ".json")
    cfg = T.read_config(path)
    model, loss, cached = T.build(cfg)
    if ck is None:
        ck = torch.load(os.path.join(ROOT, cfg.init_from), map_location="cpu", weights_only=False, mmap=True)
    own = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    ref = {k[6:]: tuple(v.shape) for k, v in ck["state"].items() if k.startswith("model.")}
    fresh = checkpoint.warm_start(cfg.init_from, model, loss, ck=ck)
    print(f"{name}: cam_buffer {cfg.model['cam_buffer']!r}, n_cam {cfg.n_cam}, stride {cfg.cam_stride}; {len(own)} tensors, checkpoint {len(ref)}; "
          f"keys equal: {own == ref}; fresh keys: {fresh}; cached chain: {cached}")
    assert own == ref and not fresh
    del model, loss
print("init_from loads with no fresh key in all four configs")
