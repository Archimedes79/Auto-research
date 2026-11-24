"""E390 card v5: the clause table of the verdict from the result files (runs/e390_<arm>6_s<S>/probe_card.json, clause_vs_n_base.json, runs/e390_sdgraph/nulls_val.json); applies the card's written arithmetic, prints
per seed and clause pass / fail and writes temp/e390/table390v6.md; the verdict itself is read by hand from it"""
import json
import os

os.chdir(os.environ.get("BFLC_ROOT", "."))
N = json.load(open("runs/e390_sdgraph/nulls_val.json"))["levels"]
BARS = dict(p1=0.7335, p1_ce0=0.99, p2=0.9380, p2_s2=0.9354, p3=0.7138, p6=0.9420, p6_pose2=0.8747, p6_nd=0.8444, p6_none=0.3918)
MIN = dict(p1_nodes=3000, p2_pos_dec=500, p3_pos=250, p6_proxies=20000)


def strongest(lv, k):
    return max(N[lv][f"{k}_rule"][0], N[lv][f"{k}_lr"][0])


rows, res = [], {}
for S in (0, 1):
    g = json.load(open(f"runs/e390_g6_s{S}/probe_card.json"))
    n = json.load(open(f"runs/e390_n6_s{S}/probe_card.json"))
    cl = json.load(open(f"runs/e390_g6_s{S}/clause_vs_n_base.json"))
    L = g["levels"]
    v = lambda lv, k: L[lv][k][0]  # noqa: E731
    lo = lambda lv, k: L[lv][k][1]  # noqa: E731
    c = {}
    c["P1 base"] = (v("base", "p1") >= BARS["p1"] and lo("base", "p1_minus_copy") > 0, f"{v('base', 'p1'):.4f} (bar {BARS['p1']}); - copy {L['base']['p1_minus_copy'][0]:+.4f} [{lo('base', 'p1_minus_copy'):+.4f}, {L['base']['p1_minus_copy'][2]:+.4f}]")
    c["P1 ce0"] = (v("ce0", "p1") >= BARS["p1_ce0"], f"{v('ce0', 'p1'):.4f} (bar 0.99)")
    c["P2 base"] = (v("base", "p2") >= BARS["p2"] and lo("base", "p2_minus_lr") > 0, f"{v('base', 'p2'):.4f} (bar {BARS['p2']}); - logistic {L['base']['p2_minus_lr'][0]:+.4f} [{lo('base', 'p2_minus_lr'):+.4f}, {L['base']['p2_minus_lr'][2]:+.4f}]")
    c["P2 s2"] = (v("s2", "p2") >= BARS["p2_s2"] and lo("s2", "p2_minus_lr") > 0, f"{v('s2', 'p2'):.4f} (bar {BARS['p2_s2']}); - logistic {L['s2']['p2_minus_lr'][0]:+.4f} [{lo('s2', 'p2_minus_lr'):+.4f}, {L['s2']['p2_minus_lr'][2]:+.4f}]")
    c["P3 base"] = (v("base", "p3") >= BARS["p3"] and lo("base", "p3_minus_rule") > 0, f"{v('base', 'p3'):.4f} (bar {BARS['p3']}); - rule {L['base']['p3_minus_rule'][0]:+.4f} [{lo('base', 'p3_minus_rule'):+.4f}, {L['base']['p3_minus_rule'][2]:+.4f}]")
    c["P6 base"] = (v("base", "p6") >= BARS["p6"] and lo("base", "p6_minus_rule") > 0 and v("base", "p6_false_none") <= 0.05,
                    f"{v('base', 'p6'):.4f} (bar {BARS['p6']}); - nearest {L['base']['p6_minus_rule'][0]:+.4f} [{lo('base', 'p6_minus_rule'):+.4f}, {L['base']['p6_minus_rule'][2]:+.4f}]; false-unassigned {v('base', 'p6_false_none'):.4f}; road {v('base', 'p6_road'):.4f} / intersection {v('base', 'p6_inter'):.4f}")
    c["P6 pose2"] = (v("pose2", "p6") >= BARS["p6_pose2"] and lo("pose2", "p6_minus_lr") > 0, f"{v('pose2', 'p6'):.4f} (bar {BARS['p6_pose2']}); - logistic {L['pose2']['p6_minus_lr'][0]:+.4f} [{lo('pose2', 'p6_minus_lr'):+.4f}, {L['pose2']['p6_minus_lr'][2]:+.4f}]")
    c["P6 nd"] = (v("nd", "p6") >= BARS["p6_nd"] and v("nd", "p6_none") >= BARS["p6_none"] and lo("nd", "p6_none_minus_rule") > 0 and v("nd", "p6_false_none") <= 0.05,
                  f"accuracy {v('nd', 'p6'):.4f} (bar {BARS['p6_nd']}), none recall {v('nd', 'p6_none'):.4f} (bar {BARS['p6_none']}); none - nearest {L['nd']['p6_none_minus_rule'][0]:+.4f} [{lo('nd', 'p6_none_minus_rule'):+.4f}, {L['nd']['p6_none_minus_rule'][2]:+.4f}]; false-unassigned {v('nd', 'p6_false_none'):.4f}")
    p3c, p1c, p2c, p6c = cl["p3_g_minus_n"], cl["p1_g_minus_n"], cl["p2_g_minus_n"], cl["p6_g_minus_n"]
    c["graph clause"] = (p3c[0] >= 0.05 and p3c[1] > 0 and p1c[0] >= 0 and p2c[0] >= 0 and p6c[0] >= 0,
                         f"g - n: P3 {p3c[0]:+.4f} [{p3c[1]:+.4f}, {p3c[2]:+.4f}], P1 {p1c[0]:+.4f}, P2 {p2c[0]:+.4f}, P6 {p6c[0]:+.4f}")
    p4 = g["p4"]
    c["P4"] = (p4["max_rel_change"] <= 1e-4 and p4["flips"] == 0, f"max rel change {p4['max_rel_change']:.2e}, flips {p4['flips']}")
    # P5: paired steps; g's drop <= the stronger null's drop + 0.01; beyond the TRAIN range: not below the null
    fails = []
    def null(lv, k):
        if k == "p1":
            return max(N[lv]["p1_copy"][0], N[lv]["p1_vote"][0])
        if k == "p2":
            return N[lv]["p2_lr"][0]
        if k == "p3":
            return N[lv]["p3_rule"][0]
        return strongest(lv, "p6")
    steps = [("p1", ["ce0", "ce01", "base"]), ("p1", ["base", "s1", "s2"]), ("p2", ["base", "s1", "s2"]), ("p3", ["base", "s1", "s2"]), ("p6", ["base", "s1", "s2"]), ("p6", ["base", "pose1", "pose2"])]
    for k, lvs in steps:
        for a, b in zip(lvs, lvs[1:]):
            dg, dn = v(a, k) - v(b, k), null(a, k) - null(b, k)
            if dg > dn + 0.01:
                fails.append(f"{k} {a}->{b}: drop {dg:+.4f} vs null {dn:+.4f}")
    for k, lv in (("p1", "s5"), ("p2", "s5"), ("p3", "s5"), ("p6", "s5"), ("p6", "pose5")):
        if v(lv, k) < null(lv, k):
            fails.append(f"{k} {lv}: {v(lv, k):.4f} below the null {null(lv, k):.4f}")
    c["P5"] = (not fails, "; ".join(fails) or "all steps within the null's drop + 0.01; s5 / pose5 not below the null")
    cnt = L["base"]["counts"]
    c["counts"] = (all(cnt[k] >= m for k, m in MIN.items()) and L["nd"]["counts"]["p6_unassigned"] >= 5000, json.dumps({k: cnt[k] for k in MIN}) + f", nd unassigned {L['nd']['counts']['p6_unassigned']}")
    res[S] = c
    for name, (ok, txt) in c.items():
        rows.append(f"| {S} | {name} | {'PASS' if ok else 'FAIL'} | {txt} |")
    nb = n["levels"]["base"]
    rows.append(f"| {S} | (arm n, reported) | - | P1 {nb['p1'][0]:.4f}, P2 {nb['p2'][0]:.4f}, P3 {nb['p3'][0]:.4f}, P6 {nb['p6'][0]:.4f} (base); P6 nd none recall {n['levels']['nd']['p6_none'][0]:.4f}; boot_nan g {g.get('boot_nan')} |")
hdr = ["| seed | clause | result | numbers (VAL 873; from runs/e390_<arm>6_s<S>/probe_card.json, clause_vs_n_base.json) |", "|---|---|---|---|"]
open("temp/e390/table390v6.md", "w").write("\n".join(hdr + rows) + "\n")
print("\n".join(hdr + rows))
for name in res[0]:
    print(f"{name}: s0 {'PASS' if res[0][name][0] else 'FAIL'}, s1 {'PASS' if res[1][name][0] else 'FAIL'}")
