"""E363 verdict by hand (the rule of the card, written before any scoring): per seed the table a - c at the end-to-end read-out on the faithful graphs, the stop rules (i), (ii), (iii), (iv), (v), the guards, the bootstrap interval of a - c, the per-seed
prediction windows, the recovery share, the STACK lines (aw - a, cw - c, the stack against the references), and the token KEEP (candidate for R2, nothing adopted) / TIE / INVALID.

usage (problem folder):  wsl bash tools/env/_py14.sh verdict_e363.py [--selftest]
reads  runs/e363_{c2,a}_s<S>/ends_card_{e363c,e363a,e363cw,e363aw}_snap_nogeom_attach_thr0.1.json, runs/e292/e292_{c2,a}_s<S>.plain_t0.5/card.json (the stored graph-level cards),
       temp/e363/fixture_result.json (written by fixture.py), temp/e363/boot_a_vs_c_s<S>.log and boot_aw_vs_a_s<S>.log (`jdiag.py boot`, topo F1 block), the references of E338a / E324 s / E358 below
"""
import json
import os
import re
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = ROOT + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")
GUARDS = {"topo_F1": ("hi", 0.005), "F1@1.0": ("hi", 0.01), "dup_frac": ("lo", 0.01), "vert100_dp": ("lo", 1.0), "junction_F1": ("hi", 0.01), "lanes_found": ("hi", 0.01)}
PRED = {0: dict(topo_F1=(0.016, 0.008), dup_frac=(0.018, 0.006), junction_F1=(0.085, 0.030), lanes_found=(-0.002, 0.010)),
        1: dict(topo_F1=(0.024, 0.008), dup_frac=(0.011, 0.006), junction_F1=(0.100, 0.030), lanes_found=(-0.004, 0.010))}
WITNESS = {0: 0.0162, 1: 0.0239}                                   # E360: ak - ck topo F1 (the own-keypoint witness, independent of the node head)
SYM_TOL = 0.01
STACK_WINDOW = (0.0005, 0.0035)
STACK_PRED = {0: 0.0019, 1: 0.0016}                                  # aw - a: the gain of E358's head on B's graph, carried over (hit within 0.003)
BASE_REF = {0: dict(c=0.6060, a=0.6274), 1: dict(c=0.6037, a=0.6326)}   # graph-level topo F1 of c2 plain / a plain
CURVES = {0: 11729, 1: 11880}
E338A = {0: 0.6989, 1: 0.6949}                                        # c2 WITH the hard step, attached ends from B's nodes
E324S_PRIMARY = {0: 0.6912, 1: 0.6948}                                # B with the hard step, rule wiring
E358_W = {0: 0.6931, 1: 0.6964}                                       # B with the hard step, the E358 head
FRAMES = 873


def load(p):
    return json.load(open(p, encoding="utf-8"))


def interval(path):
    if not os.path.exists(path):
        return None
    m = re.search(r"topo_F1\s+A - B: point ([+-][0-9.]+), bootstrap mean [+-][0-9.]+, 95 % \[([+-][0-9.]+), ([+-][0-9.]+)\]", open(path).read())
    return None if m is None else (float(m.group(1)), float(m.group(2)), float(m.group(3)))


def share(card):
    w = card["ends_within_1m_of_a_delivered_piece_end"]
    return w["entry"] - w["keypoint_entry"], w["exit"] - w["keypoint_exit"]


def judge(seed, cards, base_cards, boots, surgery_ok, nodes_ok):
    """cards: dict(c, a, cw, aw) (cw / aw may be None); base_cards: the stored graph-level cards of c2 plain / a plain (None in the selftest); -> (token of the seed, lines)"""
    out = []
    c, a = cards["c"], cards["a"]
    nc, na = c["numbers"]["head"], a["numbers"]["head"]
    stop = []
    for arm, card, ref in (("c", c, BASE_REF[seed]["c"]), ("a", a, BASE_REF[seed]["a"])):
        b = card["numbers"]["base"]
        if abs(b["topo_F1"] - ref) > 1e-3:
            stop.append(f"(i) {arm}: base topo F1 {b['topo_F1']:.4f} != stored {ref:.4f}")
        if base_cards is not None:
            g = base_cards[arm]
            bad = [k for k in b if k in g and abs(float(b[k]) - float(g[k])) > 1e-3]
            if bad or not set(b) & set(g):
                stop.append(f"(i) {arm}: base block differs from the stored graph-level card in {bad or 'all keys'}")
        if card["curves"] != CURVES[seed]:
            stop.append(f"(ii) {arm}: curves {card['curves']} != {CURVES[seed]}")
        if card.get("frames") != FRAMES:
            stop.append(f"(v) {arm}: {card.get('frames')} frames, not {FRAMES}")
    for arm, w, ref in (("cw", "c", None), ("aw", "a", None)):
        wc = cards.get(arm)
        if wc is not None:
            if wc["numbers"]["base"] != cards[w]["numbers"]["base"] or wc["curves"] != CURVES[seed] or wc.get("wire") != "head":
                stop.append(f"(iv) {arm}: base block / curves differ from {w}'s or the card is not a head read-out")
    if surgery_ok is not True and surgery_ok != "n/a":
        stop.append("(ii'/iii/iv) the surgery check (arrays of the dumps, the fixture on both seeds, the node tensors, the provenance) is missing or failed")
    if nodes_ok is not True and nodes_ok != "n/a":
        stop.append("(vi) the node check (the pair model's node outputs against QGraph.node_stage of B') is missing or failed for an arm of this seed")
    sc, sa = share(c), share(a)
    out.append(f"  attached - keypoint share within 1 m (STOP RULE vii, tolerance {SYM_TOL}): c entry {sc[0]:+.4f} exit {sc[1]:+.4f} | a entry {sa[0]:+.4f} exit {sa[1]:+.4f}")
    for i_, side in enumerate(("entry", "exit")):
        if abs(sa[i_] - sc[i_]) > SYM_TOL:
            stop.append(f"(vii) symmetry {side}: a {sa[i_]:+.4f} vs c {sc[i_]:+.4f} (> {SYM_TOL})")
    d = {k: na[k] - nc[k] for k in K}
    guards = {}
    for k, (kind, bound) in GUARDS.items():
        loss = -d[k] if kind == "hi" else d[k]
        guards[k] = loss <= bound
    for k in K:
        p = PRED[seed].get(k)
        ptxt = "" if p is None else f"  pred {p[0]:+.3f} +-{p[1]:.3f}: {'HIT' if abs(d[k] - p[0]) <= p[1] else 'miss'}"
        out.append(f"  {k:12s} a {na[k]:8.4f}  c {nc[k]:8.4f}  a - c {d[k]:+.4f}  guard {'ok' if guards.get(k) else 'BROKEN'}{ptxt}")
    ct = cards.get("t")
    if ct is not None:
        nt = ct["numbers"]["head"]
        rec = (na["topo_F1"] - nc["topo_F1"]) / (nt["topo_F1"] - nc["topo_F1"]) if nt["topo_F1"] != nc["topo_F1"] else float("nan")
        out.append(f"  rule-version arm t (c2 WITH the hard step, faithful): {nt['topo_F1']:.4f}; recovery share (a - c) / (t - c) = {rec:+.2f}; a - t {na['topo_F1'] - nt['topo_F1']:+.4f}   (stale E338a {E338A[seed]:.4f}: history only)")
    else:
        out.append("  rule-version arm t: not scored")
    b = boots.get("a")
    out.append("  bootstrap a - c topo F1: " + ("missing" if b is None else f"{b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]"))
    # the stack (reported; the bounds of E358's rule against the layer graph; its own prediction)
    if cards.get("aw") is not None and cards.get("cw") is not None:
        nw, ncw = cards["aw"]["numbers"]["head"], cards["cw"]["numbers"]["head"]
        ds = {k: nw[k] - na[k] for k in K}
        gs = {k: ((-ds[k] if GUARDS[k][0] == "hi" else ds[k]) <= GUARDS[k][1]) for k in K}
        bs = boots.get("aw")
        win = STACK_WINDOW[0] <= ds["topo_F1"] <= STACK_WINDOW[1] and bs is not None and bs[1] > 0
        out.append(f"  STACK aw (layer + head) {nw['topo_F1']:.4f}: aw - a {ds['topo_F1']:+.4f} (pred {STACK_PRED[seed]:+.4f}; HIT needs the window [{STACK_WINDOW[0]:+.4f}, {STACK_WINDOW[1]:+.4f}] and a bootstrap lower bound above 0: {'HIT' if win else 'miss'}), guards against a: {'all held' if all(gs.values()) else 'BROKEN ' + str([k for k, v in gs.items() if not v])}, "
                   + ("bootstrap missing" if bs is None else f"bootstrap aw - a {bs[0]:+.4f} [{bs[1]:+.4f}, {bs[2]:+.4f}]"))
        out.append(f"  head on the control graph: cw {ncw['topo_F1']:.4f}: cw - c {ncw['topo_F1'] - nc['topo_F1']:+.4f}")
        out.append(f"  STACK against the references: aw {nw['topo_F1']:.4f} vs E324 s PRIMARY on B with the hard step {E324S_PRIMARY[seed]:.4f} ({nw['topo_F1'] - E324S_PRIMARY[seed]:+.4f}), vs E358 w {E358_W[seed]:.4f} ({nw['topo_F1'] - E358_W[seed]:+.4f}), vs E338a {E338A[seed]:.4f} ({nw['topo_F1'] - E338A[seed]:+.4f})")
    if stop:
        return "INVALID", out + ["  STOP RULES FAILED: " + "; ".join(stop)]
    witness_gap = d["topo_F1"] - WITNESS[seed]
    out.append(f"  KEEP condition (b): (a - c) {d['topo_F1']:+.4f} against E360's own-keypoint witness {WITNESS[seed]:+.4f}: difference {witness_gap:+.4f} ({'within' if abs(witness_gap) <= 0.01 else 'OUTSIDE'} 0.01)")
    keep = all(guards.values()) and b is not None and b[1] > 0 and abs(witness_gap) <= 0.01
    tok = "KEEP" if keep else "TIE"
    if abs(witness_gap) > 0.01:
        tok += " (attachment-dominated: the gain differs from the own-keypoint witness by more than 0.01)"
    return tok, out


def main():
    tokens = []
    sx = f"{ROOT}/temp/e363/surgery_check.json"
    surgery_ok = bool(load(sx).get("pass")) if os.path.exists(sx) else False
    for s in (0, 1):
        def card(g, tag):
            p = f"{R}e363_{g}_s{s}/ends_card_{tag}_snap_nogeom_attach_thr0.1.json"
            return load(p) if os.path.exists(p) else None
        cards = dict(c=card("c2", "e363c"), a=card("a", "e363a"), cw=card("c2", "e363cw"), aw=card("a", "e363aw"), t=card("c2", "e363t"))
        nn = []
        for g in ("a", "c2"):
            npth = f"{ROOT}/temp/e363/nodecheck_{g}_s{s}.json"
            nn.append(os.path.exists(npth) and bool(load(npth).get("passed")))
        nodes_ok = all(nn)
        boots = {"a": interval(f"{ROOT}/temp/e363/boot_a_vs_c_s{s}.log"), "aw": interval(f"{ROOT}/temp/e363/boot_aw_vs_a_s{s}.log")}
        base_cards = {arm: load(f"{R}e292/e292_{g}_s{s}.plain_t0.5/card.json")["p"]["graph"] for arm, g in (("c", "c2"), ("a", "a"))}
        print(f"== seed {s}")
        if cards["c"] is None or cards["a"] is None:
            print("  c or a card missing: INVALID")
            tokens.append("INVALID")
            continue
        tok, lines = judge(s, cards, base_cards, boots, surgery_ok, nodes_ok)
        print("\n".join(lines))
        print(f"  token of the seed: {tok}")
        tokens.append(tok)
    if any(x == "INVALID" for x in tokens):
        final = "INVALID (a stop rule failed; nothing is cut, re-run after the fix)"
    elif all(x == "KEEP" for x in tokens):
        final = "KEEP (end-to-end candidate for R2 on faithful graphs; NOTHING adopted)"
    else:
        final = "TIE, kept and refined (D87)"
    print("TOKEN:", final)


def selftest():
    def card(topo, dup, base_topo, curves=11729, f1=0.88, junc=0.30, lanes=0.76, vert=19.0, wire=None, frames=873, ent=0.01, ex=0.01):
        return dict(curves=curves, frames=frames, wire=wire, numbers=dict(head=dict(topo_F1=topo, dup_frac=dup, **{"F1@1.0": f1}, junction_F1=junc, lanes_found=lanes, vert100_dp=vert), base=dict(topo_F1=base_topo)),
                    ends_within_1m_of_a_delivered_piece_end=dict(pairs=curves, entry=0.4 + ent, exit=0.4 + ex, keypoint_entry=0.4, keypoint_exit=0.4))
    c0, a0 = BASE_REF[0]["c"], BASE_REF[0]["a"]
    good = dict(c=card(0.65, 0.17, c0), a=card(0.666, 0.175, a0), cw=card(0.652, 0.17, c0, wire="head"), aw=card(0.668, 0.175, a0, wire="head"))
    b_pos, b_zero, b_neg = (0.016, 0.010, 0.022), (0.001, -0.003, 0.005), (-0.01, -0.02, -0.001)
    tok, lines = judge(0, good, None, {"a": b_pos, "aw": b_pos}, "n/a", "n/a")
    assert tok == "KEEP", (tok, lines)
    assert any("STACK aw" in l and "HIT" in l for l in lines) and any("cw - c" in l for l in lines), lines
    assert judge(0, good, None, {"a": b_zero}, "n/a", "n/a")[0].startswith("TIE") and judge(0, good, None, {"a": b_neg}, "n/a", "n/a")[0].startswith("TIE") and judge(0, good, None, {"a": None}, "n/a", "n/a")[0].startswith("TIE")
    assert judge(0, dict(good, a=card(0.666, 0.19, a0)), None, {"a": b_pos}, "n/a", "n/a")[0].startswith("TIE")                      # dup_frac +0.02 against the bound 0.01
    assert judge(0, dict(good, a=card(0.640, 0.17, a0)), None, {"a": b_pos}, "n/a", "n/a")[0].startswith("TIE")                      # topo F1 loss 0.010 against the bound 0.005
    for bad, key in ((dict(good, a=card(0.666, 0.175, 0.5)), "(i)"), (dict(good, c=card(0.65, 0.17, c0, curves=11000)), "(ii)"), (dict(good, a=card(0.666, 0.175, a0, frames=800)), "(v)"),
                     (dict(good, aw=card(0.668, 0.175, a0, wire=None)), "(iv)"), (dict(good, cw=card(0.652, 0.17, 0.4, wire="head")), "(iv)")):
        tok, lines = judge(0, bad, None, {"a": b_pos}, "n/a", "n/a")
        assert tok == "INVALID" and key in lines[-1], (key, tok, lines[-1])
    tok, lines = judge(0, dict(good, a=card(0.700, 0.175, a0)), None, {"a": b_pos}, "n/a", "n/a")
    assert tok.startswith("TIE") and "attachment-dominated" in tok and any("OUTSIDE" in l for l in lines), tok               # a gain far from the own-keypoint witness is never KEEP
    tok, lines = judge(0, dict(good, a=card(0.666, 0.175, a0, ent=0.05)), None, {"a": b_pos}, "n/a", "n/a")
    assert tok == "INVALID" and "(vii)" in lines[-1], (tok, lines[-1])                                                          # the symmetry stop rule of E360 is kept
    for fx in (False, None):
        tok, lines = judge(0, good, None, {"a": b_pos}, fx, "n/a")
        assert tok == "INVALID" and "(ii'/iii/iv)" in lines[-1], (fx, tok)
        tok, lines = judge(0, good, None, {"a": b_pos}, "n/a", fx)
        assert tok == "INVALID" and "(vi)" in lines[-1], (fx, tok)
    tok, _ = judge(0, good, None, {"a": b_pos}, True, True)
    assert tok == "KEEP"
    stored = dict(c=dict(topo_F1=c0, vert100=19.0), a=dict(topo_F1=a0, vert100=19.0))
    good2 = {k: dict(good[k], numbers=dict(head=good[k]["numbers"]["head"], base=dict(topo_F1=(c0 if k in ("c", "cw") else a0), vert100=19.0))) for k in good}
    assert judge(0, good2, stored, {"a": b_pos}, "n/a", "n/a")[0] == "KEEP"
    assert judge(0, good2, dict(stored, a=dict(topo_F1=a0, vert100=25.0)), {"a": b_pos}, "n/a", "n/a")[0] == "INVALID"            # one key of the base block differs
    no_stack = dict(good, cw=None, aw=None)
    tok, lines = judge(0, no_stack, None, {"a": b_pos}, "n/a", "n/a")
    assert tok == "KEEP" and not any("STACK" in l for l in lines)
    assert interval("/nonexistent/x.log") is None
    print("verdict_e363 selftest ok (16 planted cases)")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
