import json
for s in (0, 1):
    d = json.load(open(f"temp/e299/heat_replay_s{s}.json"))
    print("seed", s, "pairs", d["pairs"], "sanity", d["sanity"], "baseline_ok", d.get("baseline_matches_step_0b"))
    for n, a in d["arms"].items():
        print(f"  {n:6s} gt {a['gt']} tp0 {a['base_tp']} tp1 {a['comb_tp']} gained {a['gained']} lost {a['lost']} net {a['net_gain_share']:.4f} gross {a['gross_gain_share']:.4f} new {a['new']} true {a['true_new']} false {a['false_new']} (near {a['false_new_near']} iso {a['false_new_isolated']}) vanished {a['vanished']}/{a['vanished_matched']} recall {a['base_recall']:.3f}->{a['comb_recall']:.3f} prec {a['base_precision']:.3f}->{a['comb_precision']:.3f} deliv {a['base_delivered']}->{a['comb_delivered']}")
    print("  ceiling", d["heat_visible_ceiling"])
    print("  rule", d["rule_w_star"])
