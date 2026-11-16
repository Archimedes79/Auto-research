"""scratch: pooled numbers from the stored counts pickles (own lane, far, dup, maneuvers) and per-scene pooled topo F1; picked-frame denominators."""
import os, sys, json, pickle
import numpy as np
R = os.environ["BFLC_ROOT"]
H = f"{R}/temp/qual_report_1115"
c324 = pickle.load(open(f"{R}/runs/e324_s_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
c312 = pickle.load(open(f"{R}/runs/e312_p_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
P1 = pickle.load(open(f"{H}/pass1.pkl", "rb"))
sets = {"B": c324["base"], "E312": c312["head"], "E324s": c324["head"]}
sc = np.array(c324["scenes"])
def pooled(cs):
    t = sum(c["topo"]["topo"] for c in cs); P, Rr = t[0] / t[1], t[0] / t[2]
    rng = sum(c["topo"]["rng"] for c in cs); u = sum(c["topo"]["dup"] for c in cs)
    tr = sum(c["tier"]["rec"] for c in cs); tp = sum(c["tier"]["prec"] for c in cs)
    return dict(topoF1=2 * P * Rr / (P + Rr), topoP=P, topoR=Rr, far_R=rng[2][0] / rng[2][1], far_n=rng[2][1] / 2, near_R=rng[0][0] / rng[0][1], mid_R=rng[1][0] / rng[1][1], dup=u[0] / u[1],
                ownR_vis=tr[0][0] / tr[0][1], ownR_all=tr[0][2] / tr[0][3], ownP=tp[1][0] / tp[1][1], t2R=tr[1][0] / tr[1][1], t3R=tr[2][0] / tr[2][1], z0n=tp[0][1] / 2)
for k, cs in sets.items():
    print(k, {a: round(float(b), 4) for a, b in pooled(cs).items()})
# maneuvers
for k in sets:
    ok = sum(f["sys"][k]["man_ok"] for f in P1["frames"]); n = sum(f["sys"][k]["man_n"] for f in P1["frames"])
    jh = sum(f["sys"][k]["junc_hit"] for f in P1["frames"]); jn = sum(f["sys"][k]["junc_near"] for f in P1["frames"])
    print(k, "maneuvers realised", ok, n, round(ok / n, 4), "junction precision", round(jh / jn, 4))
print("\nper scene pooled topo F1 (B / E312 / E324s), ownR (E324s), maneuvers realised share E324s, n frames")
rows = []
for s in sorted(set(sc.tolist())):
    ix = np.where(sc == s)[0]
    p = {k: pooled([cs[i] for i in ix]) for k, cs in sets.items()}
    mo = sum(P1["frames"][i]["sys"]["E324s"]["man_ok"] for i in ix); mn = sum(P1["frames"][i]["sys"]["E324s"]["man_n"] for i in ix)
    rows.append((s, p["B"]["topoF1"], p["E312"]["topoF1"], p["E324s"]["topoF1"], p["E324s"]["ownR_vis"], p["E324s"]["ownP"], p["E324s"]["far_R"], mo / max(mn, 1), len(ix), P1["scenes"][s]["name"], P1["scenes"][s]["description"][:60]))
for r in sorted(rows, key=lambda r: r[3]):
    print("sc%2d  topo %.3f %.3f %.3f | ownR %.2f ownP %.2f far %.2f man %.2f | n%3d %s %s" % r)
print("\nfar denominators of candidates")
for k in (859, 854, 861, 277, 278, 638, 640, 798, 222, 788, 789, 840):
    r = c324["head"][k]["topo"]["rng"]
    print(k, "far hit/n(E324s)", r[2][0] / 2, r[2][1] / 2, "B", c324["base"][k]["topo"]["rng"][2][0] / 2, c324["base"][k]["topo"]["rng"][2][1] / 2)
