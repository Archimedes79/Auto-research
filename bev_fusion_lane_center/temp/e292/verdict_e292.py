"""E292 verdict by hand (replay of the stored cards, the rule of the card section 0 / 1, written before the run): clauses of arm a against c2 and against c, the assignment guards from the run-eval cards, STRUCTURAL
(a with the hard step on top), the pre-registered a - c numbers. COST (ab_timing) is not in the cards: it is measured on an idle GPU later and added by hand.
usage: wsl bash tools/env/_py14.sh ../temp/e292/verdict_e292.py"""
import json
import os

R = os.environ["BFLC_ROOT"] + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")


def head(s, tag):
    name = "ends_card_snap_thr0.1.json" if tag == "c" else f"ends_card_{tag}_snap_thr0.1.json"
    return json.load(open(f"{R}e312_p_s{s}/{name}"))["numbers"]["head"]


def assign(s, which):
    if which == "c":
        p = f"{R}e275/e273_p_s{s}.close1.5_t0.5/card.json"                                   # B read out with the hard step (the graph of c)
    elif which == "c2":
        p = f"{R}e292/e292_c2_s{s}.close1.5_t0.5/card.json"
    else:
        p = f"{R}e292/e292_a_s{s}.plain_t0.5/card.json"
    c = json.load(open(p))["p"]
    return dict(exact=c["assign_exact"]["lane_acc"], err=c["assign_error"]["lane_acc"], graph_topo=c["graph"]["topo_F1"])


pred = dict(topo_F1=-0.003, lanes_found=-0.009, junction_F1=0.005, **{"F1@1.0": -0.003}, vert100_dp=-0.7, dup_frac=-0.003)
bounds = {"topo_F1": 0.01, "F1@1.0": 0.01, "dup_frac": 0.01, "vert100_dp": 1.0, "junction_F1": 0.03, "lanes_found": 0.03}                # max loss; for dup_frac and vertices a LOSS is a rise
passes = []
for s in (0, 1):
    a, c2, c, ah = head(s, "a"), head(s, "c2"), head(s, "c"), head(s, "ah")
    print(f"== seed {s}")
    for k in K:
        print(f"  {k:12s} a {a[k]:8.4f}  c2 {c2[k]:8.4f}  c {c[k]:8.4f}  | a - c2 {a[k] - c2[k]:+.4f}  a - c {a[k] - c[k]:+.4f}  | predicted a - c {pred[k]:+.4f}  ah - a {ah[k] - a[k]:+.4f}")
    ok = {}
    for ref, nm in ((c2, "c2"), (c, "c")):
        for k, b in bounds.items():
            loss = (a[k] - ref[k]) if k in ("dup_frac", "vert100_dp") else (ref[k] - a[k])
            ok[f"{k} vs {nm}"] = loss <= b
    asg = {w: assign(s, w) for w in ("c", "c2", "a")}
    for ref in ("c2", "c"):
        ok[f"lane_acc_exact vs {ref}"] = asg[ref]["exact"] - asg["a"]["exact"] <= 0.01
        ok[f"lane_acc_err vs {ref}"] = asg[ref]["err"] - asg["a"]["err"] <= 0.03
    pass_ = a["topo_F1"] >= c2["topo_F1"] - 0.01 and a["topo_F1"] >= c["topo_F1"] - 0.01 and all(ok.values())
    structural = abs(ah["topo_F1"] - a["topo_F1"]) < 0.002
    passes.append((pass_, structural))
    print(f"  assignment (B/c, c2, a): exact {asg['c']['exact']:.4f} {asg['c2']['exact']:.4f} {asg['a']['exact']:.4f}; error {asg['c']['err']:.4f} {asg['c2']['err']:.4f} {asg['a']['err']:.4f}; graph-only topo F1 {asg['c']['graph_topo']:.4f} {asg['c2']['graph_topo']:.4f} {asg['a']['graph_topo']:.4f}")
    print("  broken:", [k for k, v in ok.items() if not v] or "none", "| seed PASSES:", pass_, "| STRUCTURAL (|ah - a| < 0.002):", structural)
print("both seeds PASS:", all(p for p, _ in passes), "| STRUCTURAL on both:", all(st for _, st in passes), "| COST: measure with training/ab_timing.py on an idle GPU")
