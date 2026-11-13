import pickle, numpy as np, os
R = "/mnt/c/Auto-research/bev_fusion_lane_center"
nb = pickle.load(open(f"{R}/temp/e321/val/smoke_b.pkl", "rb"))
sb = pickle.load(open(f"{R}/temp/e276/predfull_e273_p_s0.pkl", "rb"))[:len(nb)]
for k in ("e", "pos", "dirv", "flags", "s", "link_p", "link_a", "link_b"):
    w = 0.0; same = True
    for x, y in zip(nb, sb):
        a, b = np.asarray(x[k], float), np.asarray(y[k], float)
        if a.shape != b.shape:
            same = False; continue
        w = max(w, float(np.abs(a - b).max()) if a.size else 0.0)
    print("B", k, "max abs diff", w, "shapes equal" if same else "SHAPES DIFFER")
print("B nodes per frame new/stored", [(len(x["e"]), len(y["e"])) for x, y in zip(nb[:6], sb[:6])])
ne = pickle.load(open(f"{R}/temp/e321/val/smoke_ends.pkl", "rb"))
se = pickle.load(open(f"{R}/runs/e312_p_s0/ends_val.pkl", "rb"))[:len(ne)]
for k in ("kp_pos", "kp_dir", "kp_p", "logit", "scale"):
    print("ends", k, "max abs diff", max(float(np.abs(np.asarray(x[k], float) - np.asarray(y[k], float)).max()) for x, y in zip(ne, se)))
print("valid equal", all((x["valid"] == y["valid"]).all() for x, y in zip(ne, se)), "allowed equal", all((x["allowed"] == y["allowed"]).all() for x, y in zip(ne, se)))
