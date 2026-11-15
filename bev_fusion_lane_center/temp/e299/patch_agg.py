p = "temp/e299/heat_replay.py"
s = open(p, encoding="utf-8").read()
a = s.index("    tot = {w: None for w in WS}")
b = s.index('if __name__ == "__main__":')
new = '''    rng = np.random.RandomState(1234)
    tot, ceil = {}, dict(missed=0, visible=0, random_visible=0)
    npairs = 0
    for t0, t1 in pairs:
        if t0 not in maps or t1 not in maps:
            continue
        r = pair_stats(maps[t0], maps[t1], JTM.lane_frame(infos[t0]), JTM.lane_frame(infos[t1]), gt[frames[t1].row], rng=rng)
        npairs += 1
        for name, v in r.items():
            if name == "ceiling":
                for kk in ceil:
                    ceil[kk] += v[kk]
            elif name not in tot:
                tot[name] = dict(v)
            else:
                for kk in tot[name]:
                    tot[name][kk] += v[kk]
    out = dict(seed=a.seed, pairs=npairs, sanity=sane, w_star=WSTAR, arms={}, heat_visible_ceiling=dict(ceil, visible_share=ceil["visible"] / max(ceil["missed"], 1), random_share=ceil["random_visible"] / max(ceil["missed"], 1)))
    for name, t in tot.items():
        d = dict(t)
        d["net_gain_share"] = (t["comb_tp"] - t["base_tp"]) / t["gt"]
        d["gross_gain_share"] = t["gained"] / t["gt"]
        d["base_recall"], d["comb_recall"] = t["base_tp"] / t["gt"], t["comb_tp"] / t["gt"]
        d["base_precision"] = t["base_tp"] / max(t["base_delivered"], 1)
        d["comb_precision"] = t["comb_tp"] / max(t["comb_delivered"], 1)
        out["arms"][name] = d
    star = out["arms"][f"w{WSTAR}"]
    n1, n2 = out["arms"]["N1"]["net_gain_share"], out["arms"]["N2"]["net_gain_share"]
    out["rule_w_star"] = dict(
        net_gain_share=star["net_gain_share"], true_new=star["true_new"], false_new=star["false_new"],
        a_net_gain_share_ge_0_05=bool(star["net_gain_share"] >= 0.05), at_the_bar_0_045_0_055=bool(0.045 <= star["net_gain_share"] <= 0.055),
        temporal_excess_over_nulls=star["net_gain_share"] - max(n1, n2), a_temporal=bool(star["net_gain_share"] >= 0.05 and not (0.045 <= star["net_gain_share"] <= 0.055) and star["net_gain_share"] - max(n1, n2) >= 0.02),
        b_true_new_ge_false_new=bool(star["true_new"] >= star["false_new"]))
    print(json.dumps(out, indent=1))
    if not a.max:
        # step 0b numbers of the baseline must be reproduced (section 5b amendment): recall 0.787 / 0.790, precision 0.620 / 0.602 within 0.003
        want = {0: (0.787, 0.620), 1: (0.790, 0.602)}[a.seed]
        got = (star["base_recall"], star["base_precision"])
        out["baseline_matches_step_0b"] = bool(abs(got[0] - want[0]) <= 0.003 and abs(got[1] - want[1]) <= 0.003)
        json.dump(out, open(os.path.join(ROOT, "temp", "e299", f"heat_replay_s{a.seed}.json"), "w"), indent=1)
        print("baseline matches step 0b:", out["baseline_matches_step_0b"], got, want)


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
