"""t21 scratch: golden outputs of the 'window' camera path.  `python temp/t21/golden.py make` BEFORE the model.py edit stores them in
temp/t21/golden_window.pt; `python temp/t21/golden.py check` after the edit asserts that every tensor is bit-equal (the default and the
explicit cam_buffer='window')."""
import dataclasses
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for sub in ("tests", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from feature_cache import CachedBEVFusionModel  # noqa: E402
from model import ModelConfig  # noqa: E402
from test_slots import batch_for  # noqa: E402

torch.set_num_threads(2)
GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}
BASE = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24,
            trunk_width=0.25, p_drop=0.0, dense_outputs=True)
CASES = {"dual_motion": dict(camera_features="dual", motion_block=True, box_vel_from_field=True, object_moving=True),
         "plain": dict(), "ego_gru": dict(cam_temporal="ego_gru"), "no_temporal": dict(cam_temporal="none")}
FILE = os.path.join(ROOT, "temp", "t21", "golden_window.pt")


def flat(prefix, x, out):
    if torch.is_tensor(x):
        out[prefix] = x.detach().clone()
    elif isinstance(x, dict):
        for k, v in x.items():
            flat(f"{prefix}.{k}", v, out)


def run(extra):
    res = {}
    for name, kw in CASES.items():
        cfg = ModelConfig(**dict(BASE, **kw, **extra))
        torch.manual_seed(0)
        m = CachedBEVFusionModel(cfg).eval()
        with torch.no_grad():                                    # zero-start layers get small values: every path carries a signal
            for q in m.parameters():
                if q.requires_grad and q.abs().max() == 0:
                    q.normal_(0, 1e-2)
        for Fn in ((3, 2) if kw.get("motion_block") else (3, 2, 1)):
            b = batch_for(cfg, B=2, Fn=Fn)
            with torch.no_grad():
                r1 = m(b, dts=(-0.5, 0.0, 0.5))
                step = (torch.tensor([0.5, 0.45]), torch.tensor([8.0, 10.0]), torch.tensor([0.05, -0.1]))
                r2 = m(b, dts=(0.0, 0.5), state=r1["state"], step=step)
            for tag, r in (("call1", r1), ("call2", r2)):
                flat(f"{name}.F{Fn}.{tag}.out", r["out"], res)
                flat(f"{name}.F{Fn}.{tag}.state", r["state"], res)
                res[f"{name}.F{Fn}.{tag}.features"] = r["features"].detach().clone()
        res[f"{name}.keys"] = [f"{k} {tuple(v.shape)}" for k, v in m.state_dict().items()]
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "make":
        g = run({})
        torch.save(g, FILE)
        print("stored", len(g), "entries")
    else:
        ref = torch.load(FILE, weights_only=False)
        for extra in ({}, {"cam_buffer": "window"}):
            got = run(extra)
            assert set(got) == set(ref), set(got) ^ set(ref)
            bad = [k for k in ref if (ref[k] != got[k] if isinstance(ref[k], list) else not torch.equal(ref[k], got[k]))]
            print("extra", extra, "entries", len(ref), "differing:", bad[:5])
            assert not bad
        print("window path bit-identical")
