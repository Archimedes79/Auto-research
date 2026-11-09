"""scratch: what a stored GT frame holds (keys, camera model, map lines) and where the front image of a token is"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
files = sorted(glob.glob(os.path.join(ROOT, "data", "cache", "gt", "*_6a7aa7864a.npz")))
print(len(files), "frames of the GT signature t20 trained on (6a7aa7864a)")
f = files[len(files) // 2]
g = np.load(f, allow_pickle=True)["gt"].item()
print(os.path.basename(f))
print("keys:", sorted(g.keys())[:40])
g0 = g["cur"]
print("cur keys:", sorted(g0.keys()))
m = g0["meta"]
print("meta keys:", sorted(m.keys()))
print("K", np.asarray(m["K"]).round(1).tolist())
print("T_c2e", np.asarray(m["T_c2e"]).round(3).tolist())
for layer, pls in m.get("vectors", {}).items():
    print(layer, len(pls), "polylines; first:", np.asarray(pls[0]).shape if len(pls) else None)
for k in ("cam_file", "image_size", "scene", "sample", "timestamp", "annotation_timestamp", "res"):
    print(k, m[k])
# a frame with lane dividers
for f2 in files[::97][:40]:
    g2 = np.load(f2, allow_pickle=True)["gt"].item()["cur"]["meta"]
    v = g2["vectors"].get("lane_divider", [])
    if len(v):
        print(os.path.basename(f2), "lane_divider", len(v), "first", np.asarray(v[0]).round(1).tolist()[:4], "layers", {k: len(x) for k, x in g2["vectors"].items()})
        break
from nuscenes.utils.splits import create_splits_scenes
sp = create_splits_scenes()
print({k: len(x) for k, x in sp.items()}, sp["val"][:3])
print(os.path.isdir(os.path.expanduser("~/data/nuscenes_full")), os.path.exists(os.path.join(os.path.expanduser("~/data/nuscenes_full"), m["cam_file"])))
