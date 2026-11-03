import os, time, pickle, numpy as np
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
grid = GridSpec()
poses = pickle.load(open("bev_fusion/temp/critic_coverage/dump_new.pkl", "rb"))
poses = next(iter(poses.values()))["_radar_poses"]
d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(9)
worst = (0, None); times = []
for i in rng.choice(len(files), 400, replace=False):
    g = load_keyframe(d, files[i][:-4]); m = g["meta"]
    boxes = m["boxes"]
    hiding = [b for b in boxes if b["observed"]]
    prisms = [C.prism_of(b) for b in hiding]
    place = {id(b): n for n, b in enumerate(hiding)}
    K = m["K"]; T = m["T_c2e"]
    t0 = time.time()
    for b in boxes:
        C.object_shares(b, place.get(id(b)), prisms, T[:3, 3], K, T, (1600, 900), poses, grid)
    dt = time.time() - t0
    times.append((dt, len(boxes), len(hiding)))
    if dt > worst[0]: worst = (dt, (files[i][:-4], len(boxes), len(hiding)))
ts = np.array([t[0] for t in times])
print("object_shares per frame: mean %.3f s, p95 %.3f s, max %.3f s" % (ts.mean(), np.quantile(ts, .95), ts.max()), "worst", worst)
