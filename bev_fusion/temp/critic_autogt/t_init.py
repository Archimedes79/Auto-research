import os, glob, tempfile, numpy as np, shutil
from autogt import cli, scene_tokens
# 1) arrays of a stored keyframe
f = sorted(glob.glob("data/autogt/fefd519974/*.npz"))[0]
z = np.load(f, allow_pickle=True)
dt = {k: str(z[k].dtype) for k in z.files if k != "meta"}
print(len(dt), "arrays;", sum(v == "float16" for v in dt.values()), "float16;", {k: v for k, v in dt.items() if v != "float16"})
# 2) the worker path of generate: _init(..., scenes=grp) then _work, in this process (what a pool worker does)
root = os.path.expanduser("~/data/nuscenes")
d = tempfile.mkdtemp(dir="bev_fusion/temp/critic_autogt")
toks = scene_tokens("scene-0061", "v1.0-mini", root)[::20]
cli._init(root, "v1.0-mini", 0.5, 4, {}, ["scene-0061"])
print("worker devkit scenes:", [s["name"] for s in cli._STATE["gt"].nusc.scene], "light:", len(cli._STATE["gt"].nusc.sample_data))
print([cli._work((t, d)) for t in toks], sorted(os.listdir(d)) == sorted(t + ".npz" for t in toks))
# 3) scenes=[] means all
print("scenes=[] -> names:", "all" if not [] else "none")
shutil.rmtree(d)
