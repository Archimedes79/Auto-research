"""Scratch (E391 card v2.1 section 5.1 item 3): the choice on the TRAIN holdout, applied mechanically BEFORE any VAL read of a fusion arm.
association arm: F1 only if its holdout assoc_acc (dec) beats P's by more than 0.01 on BOTH seeds, else P (the simpler);
attribute arm: the first of P, F1, F2, F3 that passes, on both seeds, the holdout association bars (assoc_acc, false_unassigned <= 0.05) AND the holdout attribute bars (lanes_acc, route_f1);
none: the best mean of lanes_acc and route_f1 over both seeds among those passing the holdout association bars; none of those either: the association arm.
Writes temp/e391/choice391.json and prints the arms to read on VAL (space separated)."""
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ORDER = ("P", "F1", "F2", "F3")
BARS = {0: dict(assoc_acc=0.8472, lanes_acc=0.9266, route_f1=0.8664), 1: dict(assoc_acc=0.8499, lanes_acc=0.9282, route_f1=0.8553)}


def val(arm, seed, m):
    """a killed arm has no holdout card (read refuses it): nan, which fails every bar (card v2.1 section 5.1 item 7)"""
    path = os.path.join(ROOT, "runs", f"e391_{arm}_s{seed}", "card_holdout.json")
    if not os.path.exists(path):
        return float("nan")
    return float(json.load(open(path))["levels"]["dec"][m][0])


def assoc_ok(a):
    return all(val(a, s, "assoc_acc") >= BARS[s]["assoc_acc"] and val(a, s, "false_unassigned") <= 0.05 for s in (0, 1))


def attr_ok(a):
    return all(val(a, s, m) >= BARS[s][m] for s in (0, 1) for m in ("lanes_acc", "route_f1"))


def main():
    d = {s: val("F1", s, "assoc_acc") - val("P", s, "assoc_acc") for s in (0, 1)}
    assoc = "F1" if all(d[s] > 0.01 for s in (0, 1)) else "P"
    a_ok = [a for a in ORDER if assoc_ok(a)]
    passing = [a for a in a_ok if attr_ok(a)]
    mean = {a: sum(val(a, s, m) for s in (0, 1) for m in ("lanes_acc", "route_f1")) / 4 for a in ORDER}
    if passing:
        attr, why = passing[0], "the simplest passing the holdout association and attribute bars"
    elif a_ok:
        attr, why = max(a_ok, key=lambda a: mean[a] if mean[a] == mean[a] else -1.0),"none passes the attribute bars: the best attribute mean among those passing the association bars"
    else:
        attr, why = assoc, "none passes the holdout association bars: the association arm (one head)"
    out = dict(assoc_arm=assoc, f1_minus_p_assoc=d, attr_arm=attr, attr_why=why, assoc_passing=a_ok, attr_passing=passing, attr_mean=mean, bars=BARS,
               table={a: {s: {m: val(a, s, m) for m in ("assoc_acc", "false_unassigned", "lanes_acc", "route_f1")} for s in (0, 1)} for a in ORDER})
    json.dump(out, open(os.path.join(ROOT, "temp", "e391", "choice391.json"), "w"), indent=1)
    print(" ".join(sorted({assoc, attr}, key=ORDER.index)))


if __name__ == "__main__":
    main()
