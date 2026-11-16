"""scratch: extra numbers for the qualitative text (all from the stored counts pickles / pass1 maneuver counts / nuScenes scene.json); front-camera brightness of the 22 VAL scenes."""
import os, sys, json, pickle
import numpy as np
R = os.environ["BFLC_ROOT"]; H = f"{R}/temp/qual_report_1115"
for sub in ("tools/viz", "evaluation", "training/grid", "training", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
c324 = pickle.load(open(f"{R}/runs/e324_s_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
c312 = pickle.load(open(f"{R}/runs/e312_p_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
P1 = pickle.load(open(f"{H}/pass1.pkl", "rb"))
rows = json.load(open(f"{H}/perframe_table.json"))
sets = {"B": c324["base"], "E312": c312["head"], "E324s": c324["head"]}
sc = np.array(c324["scenes"])
f1 = lambda c: (lambda t: 2 * t[0] / (t[1] + t[2]))(c["topo"]["topo"])
print("frames with topo F1 < 0.5:", {k: sum(1 for r in rows if r[k + "_topo"] < 0.5) for k in sets})
print("frames with own lane R < 0.8:", {k: sum(1 for r in rows if r[k + "_ownR"] < 0.8) for k in sets}, " < 0.5:", {k: sum(1 for r in rows if r[k + "_ownR"] < 0.5) for k in sets})
print("frame-mean own lane R:", {k: round(float(np.mean([r[k + "_ownR"] for r in rows])), 3) for k in sets})
print("frames ownP < 0.8:", {k: sum(1 for r in rows if r[k + "_ownP"] < 0.8) for k in sets})
for k in sets:
    fr = P1["frames"]
    none = sum(1 for f in fr if f["n_man"] > 0 and f["sys"][k]["man_ok"] == 0); allr = sum(1 for f in fr if f["n_man"] > 0 and f["sys"][k]["man_ok"] == f["n_man"]); nf = sum(1 for f in fr if f["n_man"] > 0)
    print(k, "frames with maneuvers", nf, "none realised", none, "all realised", allr)
d = np.array([r["E324s_topo"] - r["E312_topo"] for r in rows])
print("E324s vs E312 per frame: better >0.02:", int((d > .02).sum()), "worse >0.02:", int((d < -.02).sum()), "within:", int((abs(d) <= .02).sum()))
d = np.array([r["E324s_topo"] - r["B_topo"] for r in rows])
print("E324s vs B per frame: better >0.02:", int((d > .02).sum()), "worse >0.02:", int((d < -.02).sum()), "within:", int((abs(d) <= .02).sum()))
def pooled_own(cs): r = sum(c["tier"]["rec"][0] for c in cs); return r[0] / r[1]
def pooled_ownp(cs): r = sum(c["tier"]["prec"][1] for c in cs); return r[0] / r[1]
def pooled_topo(cs): t = sum(c["topo"]["topo"] for c in cs); return 2 * t[0] / (t[1] + t[2])
for s in (31, 42, 11, 15, 1, 23):
    ix = np.where(sc == s)[0]
    print("scene", s, {k: dict(topo=round(float(pooled_topo([cs[i] for i in ix])), 3), ownR=round(float(pooled_own([cs[i] for i in ix])), 3), ownP=round(float(pooled_ownp([cs[i] for i in ix])), 3)) for k, cs in sets.items()})
# conditions
words = ("night", "rain", "dark", "wet", "fog", "snow", "dusk", "dawn")
print("VAL scene descriptions with a weather / night word:", [(s, v["name"], v["description"]) for s, v in P1["scenes"].items() if any(w in v["description"].lower() for w in words)])
import qgraph_view as QV, cv2
infos = QV._infos()
res = []
for sid, v in P1["scenes"].items():
    vals = []
    for tk in v["tokens"][::10]:
        fn = os.path.join(QV.NUSC, infos[tk]["cam_infos"]["CAM_FRONT"]["filename"])
        im = cv2.imread(fn)
        if im is not None:
            vals.append(float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean()))
    res.append((sid, v["name"], float(np.mean(vals)) if vals else float("nan"), len(vals)))
print("front-camera mean gray level (0-255) per VAL scene, 4 frames each, darkest first:")
for r in sorted(res, key=lambda r: r[2])[:6]:
    print("  scene %d %s mean %.1f (n %d)" % r)
print("  brightest:", sorted(res, key=lambda r: -r[2])[0][:3])
