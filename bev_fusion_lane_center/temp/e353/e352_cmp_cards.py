import json, os, sys
ROOT = os.environ["BFLC_ROOT"]
a = json.load(open(os.path.join(ROOT, "runs/e324_s_s0/ends_card_snap_thr0.1.json")))
b = json.load(open(os.path.join(ROOT, "runs/e352_null_s0/ends_card_null_snap_thr0.1.json")))
for v in ("base", "head"):
    d = {k: b["numbers"][v][k] - a["numbers"][v][k] for k in a["numbers"][v]}
    print(v, "max abs difference of the 11 numbers:", max(abs(x) for x in d.values()), "topo_F1 stored", a["numbers"][v]["topo_F1"], "new", b["numbers"][v]["topo_F1"])
    print("  ", {k: round(x, 6) for k, x in d.items() if abs(x) > 0})
print("curves", a["curves"], b["curves"], "wired", a["wired_both_ends"], b["wired_both_ends"], "dropped", a["long_links_dropped"], b["long_links_dropped"])
