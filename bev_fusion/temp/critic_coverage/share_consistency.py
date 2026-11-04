import os, pickle, numpy as np
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
grid = GridSpec()
rp = next(iter(pickle.load(open("bev_fusion/temp/critic_coverage/dump_new.pkl", "rb")).values()))["_radar_poses"]
d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(12)
rows = []
for i in rng.choice(len(files), 150, replace=False):
    g = load_keyframe(d, files[i][:-4]); m = g["meta"]; K = m["K"]; T = m["T_c2e"]
    boxes = m["boxes"]; hiding = [b for b in boxes if b["observed"]]; prisms = [C.prism_of(b) for b in hiding]; place = {id(b): n for n, b in enumerate(hiding)}
    cobs = C.camera_observed(g); fus = C.fusion_cells(g)
    for b in boxes:
        sh = C.object_shares(b, place.get(id(b)), prisms, T[:3, 3], K, T, (1600, 900), rp, grid)
        f_ground = C.footprint_share(b["corners"], cobs, grid); f_fus_ground = C.footprint_share(b["corners"], fus, grid)
        rows.append((sh["camera"], f_ground, sh["fusion"], f_fus_ground, b["group"] == "pedestrian", b["observed"]))
r = np.array(rows, float)
dcam = r[:, 0] - r[:, 1]
print("boxes", len(r), "| object_shares camera minus ground-plane share: mean %.3f, p1 %.3f p99 %.3f, share with < -0.1: %.4f, > +0.1: %.4f" % (dcam.mean(), np.quantile(dcam, .01), np.quantile(dcam, .99), (dcam < -0.1).mean(), (dcam > 0.1).mean()))
dfus = r[:, 2] - r[:, 3]
print("fusion: mean %.3f, p1 %.3f p99 %.3f; class flips at 0.5: %d, at 0.1: %d (of %d)" % (dfus.mean(), np.quantile(dfus, .01), np.quantile(dfus, .99), int(((r[:, 2] >= .5) != (r[:, 3] >= .5)).sum()), int(((r[:, 2] >= .1) != (r[:, 3] >= .1)).sum()), len(r)))

print("--- outliers with object_shares camera < ground share - 0.1")
rng = np.random.default_rng(12)
for i in rng.choice(len(files), 150, replace=False):
    g = load_keyframe(d, files[i][:-4]); m = g["meta"]; K = m["K"]; T = m["T_c2e"]
    boxes = m["boxes"]; hiding = [b for b in boxes if b["observed"]]; prisms = [C.prism_of(b) for b in hiding]; place = {id(b): n for n, b in enumerate(hiding)}
    cobs = C.camera_observed(g)
    for b in boxes:
        sh = C.object_shares(b, place.get(id(b)), prisms, T[:3, 3], K, T, (1600, 900), rp, grid)
        fg = C.footprint_share(b["corners"], cobs, grid)
        if sh["camera"] < fg - 0.1:
            fp, lo, hi = C.prism_of(b)
            print(files[i][:8], b["group"], "obs", b["observed"], "centre xy", fp.mean(0).round(1), "z", round(lo, 2), round(hi, 2), "obj %.2f ground %.2f" % (sh["camera"], fg))
