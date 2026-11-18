import os, sys, numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.join(R, sub))
import lane_map_data as LMD, lane_gt_data as LD
ds = LMD.MapData("val", 0, "e292_a", None, "exact")
lg, _ = LD.lane_store()
idx = np.linspace(0, len(ds) - 1, 40).astype(int)
turn, n_all = [], 0
for i in idx:
    g = lg.load(ds.tokens[i])
    for p in g["pieces"]:
        pts = np.asarray(p["pts"], float)
        for s in p["segments"]:
            if s["kind"] == "connector" and s["in_intersection"] and s["i1"] - s["i0"] >= 3:
                q = pts[s["i0"]:s["i1"] + 1]
                a = np.arctan2(*(q[2] - q[0])[::-1]); b = np.arctan2(*(q[-1] - q[-3])[::-1])
                turn.append((np.degrees(abs((b - a + np.pi) % (2 * np.pi) - np.pi)), len(q)))
t = np.array(turn)
print("in-intersection connector segments:", len(t), "points:", int(t[:, 1].sum()))
for th in (10, 20, 45):
    print(f"share of points on segments turning > {th} deg:", float(t[t[:, 0] > th, 1].sum() / t[:, 1].sum()))
