import json, os
H = os.path.dirname(os.path.abspath(__file__))
o = json.load(open(os.path.join(H, "pilot_outcome.json")))
print("outcome", o["outcome"], "valid", o["valid"], "s matched density ok", o["s_matched_density_ok"], "s VAL density", round(o["s_density_val"], 2), "s_passes", o["s_passes"])
print("floors m :", o["floors"]); print("floors m5:", o.get("floors_m5"))
print("bootstrap m - s(matched), free_zone_exact:", {k: round(v, 3) for k, v in o["bootstrap_m_minus_s_matched"].items()})
print("m5 bootstrap:", {k: round(v, 3) for k, v in o["bootstrap_m5_minus_s_matched"].items()})
print("d node gate:", o["arms"]["d"]["node_gate"], "| d reference free recall %.3f off %.3f" % (o["d_reference"]["free_zone_recall"], o["d_reference"]["off_zone"]))
print("| arm | read-out | nodes per 100 m | free recall | free exact | fixed recall | off-zone | winners off the GT cell at the end |")
print("|---|---|---|---|---|---|---|---|")
for k, v in o["arms"].items():
    for lab, key, t in (("e >= 0.5", "at_0_5", 0.5), ("matched e >= %.2f" % v["t_match"], "at_match", v["t_match"])):
        p = v[key]
        print("| %s | %s | %.1f | %.3f | %.3f | %.3f | %.3f | %s |" % (k, lab, p["nodes_per_100m"], p["free_zone_recall"], p["free_zone_exact"], p["fixed_zone_recall"], p["off_zone"], "%.2f" % v["win_off_gt_end"] if v.get("win_off_gt_end") is not None else "-"))
