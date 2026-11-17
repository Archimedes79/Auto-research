import json
K = ("topo_F1", "F1@1.0", "dup_frac", "vert100_dp", "junction_F1", "lanes_found")
def card(run, tag):
    return json.load(open(f"runs/{run}/ends_card_{tag}thr0.1.json"))["numbers"]["head"]
for name, tag in (("snapped", "snap_"), ("PRIMARY", "snap_nogeom_attach_"), ("attach-only", "attach_"), ("unsnapped", "")):
    for S in (0, 1):
        w, c = card(f"e348_w_s{S}", tag), card(f"e324_s_s{S}", tag)
        print(f"{name:12s} s{S} w {w['topo_F1']:.4f} c {c['topo_F1']:.4f} | " + " ".join(f"{k} {w[k]-c[k]:+.4f}" for k in K))
