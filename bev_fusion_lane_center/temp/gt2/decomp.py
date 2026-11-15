import json, numpy as np
d = json.load(open("../../runs/e308_t_s0/maneuver_val.json"))
rows = d["rows"]
V = ["plain", "p0.5", "p0.2", "p0.1", "oracle", "oracle+fit"]
def tab(sel, name):
    n = len(sel)
    if not n:
        return
    print(f"{name:42s} n {n:6d} | " + " ".join(f"{v} {np.mean([r['real'][v] for r in sel]):.3f}" for v in V))
tab(rows, "all maneuvers")
both = [r for r in rows if r["found_e"] and r["found_x"]]
one = [r for r in rows if (r["found_e"] != r["found_x"])]
none = [r for r in rows if not r["found_e"] and not r["found_x"]]
tab(both, "both ends found")
tab(one, "one end found")
tab(none, "no end found")
print("share both / one / none:", len(both) / len(rows), len(one) / len(rows), len(none) / len(rows))
for name, f in (("turn straight (<20 deg)", lambda r: abs(r["turn"]) < 20), ("turn left (>20)", lambda r: r["turn"] > 20), ("turn right (<-20)", lambda r: r["turn"] < -20)):
    tab([r for r in rows if f(r)], name)
    tab([r for r in both if f(r)], "  both found, " + name)
for lo, hi in ((0, 10), (10, 20), (20, 35), (35, 60), (60, 200)):
    tab([r for r in rows if lo <= r["length"] < hi], f"chord length {lo}-{hi} m")
    tab([r for r in both if lo <= r["length"] < hi], f"  both found, chord {lo}-{hi} m")
for lo, hi in ((0, 20), (20, 35), (35, 80)):
    tab([r for r in rows if lo <= r["range"] < hi], f"entry range {lo}-{hi} m")
    tab([r for r in both if lo <= r["range"] < hi], f"  both found, range {lo}-{hi} m")
# of the maneuvers not realised in the oracle: how many have both ends found (curve / wiring problem) vs ends not found
nr = [r for r in rows if not r["real"]["oracle"]]
print("not realised with the oracle:", len(nr), "of which both ends found", sum(1 for r in nr if r["found_e"] and r["found_x"]), "one", sum(1 for r in nr if r["found_e"] != r["found_x"]), "none", sum(1 for r in nr if not r["found_e"] and not r["found_x"]))
