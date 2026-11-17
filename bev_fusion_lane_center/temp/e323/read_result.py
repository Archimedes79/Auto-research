"""E323 read: the stored cards of the two directions per seed (scratch)."""
import json, os
T1 = {0: "0.083", 1: "0.09"}; T2 = {0: "0.12", 1: "0.11"}
KEYS = ("topo_F1", "dup_frac", "junction_F1", "F1@1.0", "vert100_dp", "lanes_found")
def card(run, name):
    p = f"runs/{run}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None
for S in (0, 1):
    rows = [("c(0.1) E312 snapped", card(f"e312_p_s{S}", "ends_card_snap_thr0.1.json")),
            ("u'(0.1) E312 snap-nogeom", card(f"e312_p_s{S}", "ends_card_snap_nogeom_thr0.1.json")),
            (f"c(t1={T1[S]}) E312 snapped", card(f"e312_p_s{S}", f"ends_card_snap_thr{T1[S]}.json")),
            (f"u'(t1={T1[S]}) E312 snap-nogeom", card(f"e312_p_s{S}", f"ends_card_snap_nogeom_thr{T1[S]}.json")),
            ("a(0.1) E320 PRIMARY", card(f"e320_a_s{S}", "ends_card_snap_nogeom_attach_thr0.1.json")),
            (f"a(t2={T2[S]}) E320 PRIMARY", card(f"e320_a_s{S}", f"ends_card_snap_nogeom_attach_thr{T2[S]}.json"))]
    print("seed", S, " ".join(f"{k:>11}" for k in KEYS), "  curves")
    res = {}
    for n, c in rows:
        if c is None: print(f"{n:34s} (missing)"); continue
        h = c["numbers"]["head"]; res[n[:4]] = h
        print(f"{n:34s}", " ".join(f"{h[k]:11.4f}" for k in KEYS), f"  {c['curves']}")
    def rule(a, c, u, label):
        d = lambda k: a[k] - c[k]
        print(f"  {label}: a-u' {a['topo_F1'] - u['topo_F1']:+.4f} (need >= +0.005), a-c {d('topo_F1'):+.4f}, F1@1.0 {d('F1@1.0'):+.4f} (>= -0.01), dup {d('dup_frac'):+.4f} (<= +0.01), vert {d('vert100_dp'):+.2f} (<= 1.0), junction {d('junction_F1'):+.4f} (>= -0.03)")
    try:
        rule(res["a(0."], res["c(t1"], res["u'(t"], "D1 a(0.1) vs c(t1), u'(t1)")
        rule(res["a(t2"], res["c(0."], res["u'(0"], "D2 a(t2) vs c(0.1), u'(0.1)")
        print(f"  key number: dup_frac E312 at t1 {res['c(t1']['dup_frac']:.4f} vs at 0.1 {res['c(0.']['dup_frac']:.4f}: {res['c(t1']['dup_frac'] - res['c(0.']['dup_frac']:+.4f} (H_within <= 0, H_count >= +0.004); topo F1 {res['c(t1']['topo_F1'] - res['c(0.']['topo_F1']:+.4f}, junction F1 {res['c(t1']['junction_F1'] - res['c(0.']['junction_F1']:+.4f}")
    except KeyError as e:
        print("  incomplete", e)
