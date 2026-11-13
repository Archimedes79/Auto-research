import json, os
def card(run, name):
    p = f"../../runs/{run}/{name}"
    return json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) > 0 else None
for s in (0, 1):
    for label, run, nm in (("E312 snapped 0.1 (c)", f"e312_p_s{s}", "ends_card_snap_thr0.1.json"), ("E314 snapped 0.1", f"e314_p_s{s}", "ends_card_snap_thr0.1.json"), ("E314 unsnapped 0.5 (record)", f"e314_p_s{s}", "ends_card.json"), ("E314 snapped 0.5", f"e314_p_s{s}", "ends_card_snap_thr0.5.json")):
        c = card(run, nm)
        if not c:
            print(f"s{s} {label:30s} pending")
            continue
        n, d = c["numbers"]["head"], c["difference"]
        print(f"s{s} {label:30s} topo {n['topo_F1']:.4f} ({d['topo_F1']:+.4f}) F1@1.0 {d['F1@1.0']:+.4f} lanes {d['lanes_found']:+.4f} jF1 {d['junction_F1']:+.4f} jR {d['junction_R']:+.4f} dup {d['dup_frac']:+.4f} vert {d['vert100_dp']:+.2f} topoP {d['topo_P']:+.4f} topoR {d['topo_R']:+.4f}")
