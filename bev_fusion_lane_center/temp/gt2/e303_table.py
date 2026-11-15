import json, os
def card(run, name="ends_card.json"):
    p = f"../../runs/{run}/{name}"
    return json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) > 0 else None
rows = []
for s in (0, 1):
    for label, run, nm in (("p2 (control)", f"e287_p2_s{s}", "ends_card.json"), ("E303 scales", f"e303_s_s{s}", "ends_card.json"), ("E303 chord control", f"e303_chord_s{s}", "ends_card.json"),
                           ("E303 scales snapped 0.1", f"e303_s_s{s}", "ends_card_snap_thr0.1.json"), ("E303 chord snapped 0.1", f"e303_chord_s{s}", "ends_card_snap_thr0.1.json"),
                           ("E303 scales snapped 0.5", f"e303_s_s{s}", "ends_card_snap_thr0.5.json"), ("E303 chord snapped 0.5", f"e303_chord_s{s}", "ends_card_snap_thr0.5.json")):
        c = card(run, nm)
        if not c:
            rows.append(f"s{s} {label:28s} pending")
            continue
        n, d = c["numbers"]["head"], c["difference"]
        rows.append(f"s{s} {label:28s} topo {n['topo_F1']:.4f} ({d['topo_F1']:+.4f}) F1@1.0 {n['F1@1.0']:.4f} ({d['F1@1.0']:+.4f}) lanes {d['lanes_found']:+.4f} jF1 {d['junction_F1']:+.4f} jR {d['junction_R']:+.4f} dup {d['dup_frac']:+.4f} vert {d['vert100_dp']:+.2f}")
print("\n".join(rows))
