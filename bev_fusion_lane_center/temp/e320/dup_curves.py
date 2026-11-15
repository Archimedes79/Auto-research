"""E320: duplicates against the number of delivered connector curves over the stored snapped cards (scratch)."""
import json, os
rows = []
for S in (0, 1):
    for run in ("e312_p_s%d" % S, "e315_p_s%d" % S, "e316_p_s%d" % S, "e320_a_s%d" % S):
        for f in ("ends_card_snap_thr0.1.json", "ends_card_snap_thr0.5.json", "ends_card_thr0.1.json", "ends_card.json"):
            p = f"runs/{run}/{f}"
            if os.path.exists(p):
                c = json.load(open(p)); n = c["numbers"]["head"]
                rows.append((S, run, f.replace("ends_card", "").replace(".json", "") or "(unsnapped thr0.5)", c.get("curves"), n["dup_frac"], n["topo_F1"], n["lanes_found"]))
print("seed run readout curves dup_frac topo_F1 lanes_found")
for r in sorted(rows, key=lambda r: (r[0], r[3] or 0)):
    print(r[0], f"{r[1]:10s} {r[2]:24s} {r[3]:6d} {r[4]:.4f} {r[5]:.4f} {r[6]:.4f}")
