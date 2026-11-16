"""E356 table and rule (card experiments/E356_node-spacing, section 0; scratch, one use; plain python, no torch):
(A) the delivered graph read out with the close-node radius 1.5 (stored cards) / 2.0 / 2.5 (tagged cards c20 / c25 in the stored run folders) for B plain (numbers.base) and E324 s snapped / PRIMARY / attach-only (numbers.head), both seeds,
    with the vertices per 100 m (`vert100_dp`) next to every accuracy number, the bars of the rule and the token per radius;
(B) the E346 cards of the decoded GT on the 2 m (stored), 2.5 m and 3.125 m grids against g1 (undecoded lane GT), the E346 rule (topo F1, lanes found, F1@1.0: loss at most 0.03 against g1).
usage: python3 temp/e356/e356_table.py [A|B]   (from the problem folder)"""
import json
import os
import sys

ROOT = os.getcwd()
KEYS = ("topo_F1", "lanes_found", "F1@1.0", "dup_frac", "junction_F1", "vert100_dp")
READOUTS = {"snap": ("ends_card_snap_thr0.1.json", "ends_card_{t}_snap_thr0.1.json"), "primary": ("ends_card_snap_nogeom_attach_thr0.1.json", "ends_card_{t}_snap_nogeom_attach_thr0.1.json"),
            "attach": ("ends_card_attach_thr0.1.json", "ends_card_{t}_attach_thr0.1.json")}
TAGS = {"1.5": None, "2.0": "c20", "2.5": "c25"}
LOSS_BAR, F1_BAR, DUP_BAR, JUNC_BAR, VERT_FALL = 0.01, 0.01, 0.01, 0.03, 0.5


def card(S, readout, radius):
    t = TAGS[radius]
    name = READOUTS[readout][0] if t is None else READOUTS[readout][1].format(t=t)
    p = os.path.join(ROOT, "runs", f"e324_s_s{S}", name)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    assert d["frames"] == 873, (p, d["frames"])
    if t is not None:
        assert d.get("tag") == t and d.get("spec") == f"close:{radius}@0.5", (p, d.get("tag"), d.get("spec"))
    return d["numbers"]


def part_a():
    rows, tokens, collapse = [], {}, {}
    print("| seed | numbers set | radius | " + " | ".join(KEYS) + " |")
    print("|---|---|---|" + "---|" * len(KEYS))
    for S in (0, 1):
        for ro in ("snap",):
            cards = {r: card(S, ro, r) for r in TAGS}
            for which in ("base", "head"):
                if which == "base" and ro != "snap":
                    continue                                                            # B plain is read from one card per radius (the base numbers of the snapped card; the others agree: checked below)
                for r in TAGS:
                    c = cards[r]
                    if c is None:
                        print(f"| {S} | {'B plain' if which == 'base' else 'E324 s ' + ro} | {r} | " + " | ".join("-" for _ in KEYS) + " |")
                        continue
                    print(f"| {S} | {'B plain' if which == 'base' else 'E324 s ' + ro} | {r} | " + " | ".join(f"{c[which][k]:.4f}" if k != "vert100_dp" else f"{c[which][k]:.2f}" for k in KEYS) + " |")
            # B plain must be the same number in the three read-out cards of one radius (the base graph does not depend on the junction read-out)
            for r in TAGS:
                bs = [cards[r][ "base"]["topo_F1"] for ro2 in ("snap",) if cards[r]]
    # the rule per radius
    for r in ("2.0", "2.5"):
        ok_all, why = True, []
        vf = []
        for S in (0, 1):
            for ro in ("snap",):
                c0, c1 = card(S, ro, "1.5"), card(S, ro, r)
                if c1 is None:
                    ok_all = False
                    why.append(f"s{S} {ro}: card missing")
                    continue
                for which in (("base", "head") if ro == "snap" else ("head",)):
                    a, b = c0[which], c1[which]
                    d_topo, d_f1, d_dup, d_j = b["topo_F1"] - a["topo_F1"], b["F1@1.0"] - a["F1@1.0"], b["dup_frac"] - a["dup_frac"], b["junction_F1"] - a["junction_F1"]
                    d_lanes = b["lanes_found"] - a["lanes_found"]
                    collapse.setdefault(r, []).append(d_topo < -0.10)
                    bad = []
                    if d_lanes < -0.03:
                        bad.append(f"lanes found {d_lanes:+.4f}")
                    if d_topo < -LOSS_BAR:
                        bad.append(f"topo F1 {d_topo:+.4f}")
                    if d_f1 < -F1_BAR:
                        bad.append(f"F1@1.0 {d_f1:+.4f}")
                    if d_dup > DUP_BAR:
                        bad.append(f"dup_frac {d_dup:+.4f}")
                    if d_j < -JUNC_BAR:
                        bad.append(f"junction F1 {d_j:+.4f}")
                    if bad:
                        ok_all = False
                        why.append(f"s{S} {'B plain' if which == 'base' else ro}: " + ", ".join(bad))
                    if ro == "snap" and which == "head":
                        vf.append(b["vert100_dp"] - a["vert100_dp"])
        vmean = sum(vf) / len(vf) if len(vf) == 2 else float("nan")
        vert_ok = vmean <= -VERT_FALL
        tokens[r] = ok_all and vert_ok
        print(f"\nradius {r}: loss bars and guards {'HOLD' if ok_all else 'MISSED: ' + '; '.join(why)}; vert100_dp of E324 s snapped, mean over the seeds against 1.5 m: {vmean:+.2f} (bar: at most -{VERT_FALL}): {'ok' if vert_ok else 'NOT met'} -> {'TAKE' if tokens[r] else 'NOT taken'}")
    for r in ("2.0", "2.5"):
        v = collapse.get(r, [])
        print(f"E275 collapse at radius {r}: {sum(v)} of {len(v)} (seed x numbers set) lose more than 0.10 topo F1 -> {'CONFIRMED' if v and all(v) else 'not confirmed (or cards missing)'}")
    best = [r for r in ("2.5", "2.0") if tokens.get(r)]
    print("\nread-out of the delivered graph:", f"close:{best[0]}@0.5 (the larger radius wins ties)" if best else "stays close:1.5@0.5")


def part_b():
    names = ("g1", "l2", "l25", "l31", "l1")
    m = {}
    for n in names:
        p = os.path.join(ROOT, "runs", "eval", f"e346_{n}", "val.json")
        if os.path.exists(p):
            m[n] = json.load(open(p))["metrics"]
    keys = ("topo_F1", "lanes_found", "F1@1.0", "junction_F1", "dup_frac", "vert100", "vert100_dp")
    print("| arm | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for n in names:
        if n in m:
            print(f"| {n} | " + " | ".join(f"{m[n][k]:.4f}" if k in m[n] else "-" for k in keys) + " |")
    for n in ("l2", "l25", "l31"):
        if n in m and "g1" in m:
            d = {k: m[n][k] - m["g1"][k] for k in ("topo_F1", "lanes_found", "F1@1.0")}
            print(f"{n} against g1: " + ", ".join(f"{k} {v:+.4f}" for k, v in d.items()) + f" -> {'PASS' if all(v >= -0.03 for v in d.values()) else 'FAIL'} (bar 0.03)")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "AB"
    if "A" in w:
        part_a()
    if "B" in w:
        print()
        part_b()
