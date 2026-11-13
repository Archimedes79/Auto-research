import json, os
def head(path):
    return json.load(open(path))["numbers"]["head"] if os.path.exists(path) and os.path.getsize(path) > 0 else None
res = {}
for net in ("e287_p2", "e308_t"):
    for s in (0, 1):
        run = f"{net}_s{s}"
        c = head(f"../../temp/gt2/thr/{run}_thr0.5.json") or head(f"../../runs/{run}/ends_card.json")
        for thr in (0.3, 0.2, 0.1):
            t = head(f"../../temp/gt2/thr/{run}_thr{thr}.json")
            if not (c and t):
                continue
            d = {k: t[k] - c[k] for k in ("lanes_found", "F1@1.0", "dup_frac", "topo_F1", "junction_F1", "vert100_dp")}
            ok = d["lanes_found"] > 0.03 and d["F1@1.0"] >= -0.005 and d["dup_frac"] <= 0 and d["topo_F1"] >= -0.005 and d["junction_F1"] >= -0.02 and d["vert100_dp"] <= 1.0
            res.setdefault((net, thr), {})[s] = (ok, d)
            print(f"{net:8s} thr {thr} s{s}: {'HOLDS' if ok else 'fails'} lanes {d['lanes_found']:+.4f} F1@1.0 {d['F1@1.0']:+.4f} dup {d['dup_frac']:+.4f} topo {d['topo_F1']:+.4f} jF1 {d['junction_F1']:+.4f} vert {d['vert100_dp']:+.2f}")
for thr in (0.3, 0.2, 0.1):
    cells = [res.get((net, thr), {}).get(s) for net in ("e287_p2", "e308_t") for s in (0, 1)]
    print(f"thr {thr}: {sum(1 for c in cells if c)}/4 cells; all hold: {all(c and c[0] for c in cells)}")
