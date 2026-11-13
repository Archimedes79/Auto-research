import json, os, sys
R = os.getcwd()
def card(run, name):
    p = os.path.join(R, "runs", run, name)
    return json.load(open(p)) if os.path.exists(p) else None
rows = [("E312 snapped (c)", "ends_card_snap_thr0.1.json"), ("E312 unsnapped (u)", "ends_card_thr0.1.json"),
        ("E312 snap-nogeom (u')", "ends_card_snap_nogeom_thr0.1.json"), ("E312 snap-all", "ends_card_snap_all_thr0.1.json")]
keys = ["topo_F1", "topo_P", "topo_R", "lanes_found", "junction_F1", "junction_R", "F1@1.0", "dup_frac", "vert100_dp"]
for s in (0, 1):
    print("seed", s)
    for lab, f in rows:
        c = card("e312_p_s%d" % s, f)
        if not c:
            print("  %-24s pending" % lab); continue
        h = c["numbers"]["head"]; d = c["difference"]
        print("  %-24s" % lab, " ".join("%s=%.4f(%+.4f)" % (k, h[k], d[k]) for k in keys if k in h))
