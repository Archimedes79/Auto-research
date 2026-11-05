import os, sys, glob, numpy as np
from autogt.cli import load_keyframe
d = "data/autogt/15efbde2d1"
fs = sorted(os.listdir(d))[:3]
g = load_keyframe(d, fs[0][:-4])
print({k: (v.shape, v.dtype) for k, v in g.items() if hasattr(v, "shape")})
m = g["meta"]
print(m.keys())
print(m["K"], m["res"], m["cam_file"])
b = m["boxes"][0]
print({k: (v if not hasattr(v, "shape") else v.shape) for k, v in b.items()})
print(len(m["boxes"]))
zs = [(bb["center"][2] - bb["size"][2]/2, bb["center"][2] + bb["size"][2]/2, bb["size"][2]) for bb in m["boxes"]]
print(zs[:10])
