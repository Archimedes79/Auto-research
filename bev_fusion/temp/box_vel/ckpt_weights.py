"""t11 box velocity, step 2a: weight norms of the object head's branches and of the motion head in stored checkpoints.

A branch whose last layer is ~0 outputs ~its bias everywhere; with a ReLU hidden layer whose units are all off it outputs exactly
the bias. Also: the fraction of hidden units of the vel branch that are ever on (needs features, see probe_features.py).
Run (from bev_fusion/): ~/miniconda3/envs/gcar/bin/python temp/box_vel/ckpt_weights.py runs/t8_ctl_s0/ckpt_00003000.pt ...
"""
import sys

import torch


def norms(sd, prefix):
    out = {}
    for name in ("hm", "off", "size", "rot", "rad", "vel", "dir"):
        k = f"{prefix}obj_head.head.{name}.2.weight"
        if k in sd:
            w2, b2 = sd[k], sd[f"{prefix}obj_head.head.{name}.2.bias"]
            w0 = sd[f"{prefix}obj_head.head.{name}.0.weight"]
            out[name] = (w0.norm().item(), w2.norm().item(), b2.abs().max().item())
    m2 = sd[f"{prefix}mover.motion.f.net.2.weight"]
    out["motion"] = (sd[f"{prefix}mover.motion.f.net.0.weight"].norm().item(), m2.norm().item(),
                     sd[f"{prefix}mover.motion.f.net.2.bias"].abs().max().item())
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        print(f"== {path} (step {ck.get('step')})")
        for which, sd, prefix in (("ema", ck["ema"], ""), ("state", ck["state"], "model.")):
            print(f"  {which}: branch  |W_hidden|  |W_last|  max|bias_last|")
            for k, (a, b, c) in norms(sd, prefix).items():
                print(f"    {k:7s} {a:9.3f} {b:9.5f} {c:10.2e}")
