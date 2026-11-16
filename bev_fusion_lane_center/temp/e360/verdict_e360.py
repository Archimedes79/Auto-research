"""E360 verdict by hand (the rule of the card, written before any number): per seed the table a - c at PRIMARY, the stop rules (i) - (iii) and (v), the guards, the bootstrap interval of a - c (parsed from the
`jdiag.py boot` log), the own-keypoint witness (ak - ck), the per-seed prediction windows, the recovery share, and the token KEEP (candidate for R2, nothing adopted) / TIE / INVALID.

usage (problem folder):  wsl bash tools/env/_py14.sh ../temp/e360/verdict_e360.py [--selftest]
reads  runs/e324_s_s<S>/ends_card_{e360c,e360a}_snap_nogeom_attach_thr0.1.json (+ ends_card_{e360ck,e360ak}_snap_nogeom_thr0.1.json, ends_card_e360t_..., E338a),
       runs/e292/e292_{c2,a}_s<S>.plain_t0.5/card.json, temp/e360/boot_a_vs_c_s<S>.log (jdiag boot, topo F1 of a - c), temp/e360/boot_ak_vs_ck_s<S>.log (optional)
"""
import json
import os
import re
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = ROOT + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")
# guards against c: (metric, 'max loss' sign, bound): loss = c - a for higher-is-better, a - c for lower-is-better
GUARDS = {"topo_F1": ("hi", 0.005), "F1@1.0": ("hi", 0.01), "dup_frac": ("lo", 0.01), "vert100_dp": ("lo", 1.0), "junction_F1": ("hi", 0.01), "lanes_found": ("hi", 0.01)}
PRED = {0: dict(topo_F1=(0.011, 0.010), dup_frac=(0.010, 0.008), junction_F1=(0.040, 0.040), lanes_found=(0.010, 0.020)),
        1: dict(topo_F1=(0.019, 0.010), dup_frac=(0.003, 0.008), junction_F1=(0.040, 0.040), lanes_found=(0.000, 0.020))}
BASE_REF = {0: dict(c=0.6060, a=0.6274), 1: dict(c=0.6037, a=0.6326)}      # graph-level topo F1 of c2 plain / a plain (the stored run-eval cards)
CURVES = {0: 11729, 1: 11880}
E338A = {0: 0.6989, 1: 0.6949}
SYM_TOL = 0.02


def load(p):
    return json.load(open(p, encoding="utf-8"))


def interval(path):
    """-> (point, lo, hi) of the topo F1 line of a jdiag boot log, None if the file is missing"""
    if not os.path.exists(path):
        return None
    m = re.search(r"topo_F1\s+A - B: point ([+-][0-9.]+), bootstrap mean [+-][0-9.]+, 95 % \[([+-][0-9.]+), ([+-][0-9.]+)\]", open(path).read())
    return None if m is None else (float(m.group(1)), float(m.group(2)), float(m.group(3)))


def sym(card):
    w = card["ends_within_1m_of_a_delivered_piece_end"]
    return w["entry"] - w["keypoint_entry"], w["exit"] - w["keypoint_exit"]


def judge(seed, cards, base_cards, boots, t_ok):
    """cards: dict(c, a, ck, ak) of the PRIMARY / own-keypoint read-outs (ck / ak may be None), base_cards: the stored graph-level topo F1 of c2 plain and a plain. -> (token of the seed, lines)"""
    out = []
    c, a = cards["c"], cards["a"]
    nc, na = c["numbers"]["head"], a["numbers"]["head"]
    stop = []
    for arm, card, ref in (("c", c, BASE_REF[seed]["c"]), ("a", a, BASE_REF[seed]["a"])):                # (i) the base block = the stored graph-level card; (ii) the curves
        b = card["numbers"]["base"]["topo_F1"]
        if abs(b - ref) > 1e-3:
            stop.append(f"(i) {arm}: base topo F1 {b:.4f} != stored {ref:.4f}")
        if card["curves"] != CURVES[seed]:
            stop.append(f"(ii) {arm}: curves {card['curves']} != {CURVES[seed]}")
    sc, sa = sym(c), sym(a)
    for i, side in enumerate(("entry", "exit")):                                                          # (iii) the symmetry of the attached-end bias
        if abs(sa[i] - sc[i]) > SYM_TOL:
            stop.append(f"(iii) symmetry {side}: a {sa[i]:+.4f} vs c {sc[i]:+.4f} (> {SYM_TOL})")
    if t_ok is False:
        stop.append("(v) arm t does not reproduce E338a")
    out.append(f"  attached - keypoint share within 1 m: c entry {sc[0]:+.4f} exit {sc[1]:+.4f} | a entry {sa[0]:+.4f} exit {sa[1]:+.4f} (stored B +0.009 / +0.008, c2 +0.010 / +0.010)")
    d = {k: na[k] - nc[k] for k in K}
    guards = {}
    for k, (kind, bound) in GUARDS.items():
        loss = -d[k] if kind == "hi" else d[k]
        guards[k] = loss <= bound
    for k in K:
        p = PRED[seed].get(k)
        ptxt = "" if p is None else f"  pred {p[0]:+.3f} +-{p[1]:.3f}: {'HIT' if abs(d[k] - p[0]) <= p[1] else 'miss'}"
        out.append(f"  {k:12s} a {na[k]:8.4f}  c {nc[k]:8.4f}  a - c {d[k]:+.4f}  guard {'ok' if guards.get(k) else 'BROKEN'}{ptxt}")
    rec = (na["topo_F1"] - nc["topo_F1"]) / (E338A[seed] - nc["topo_F1"]) if E338A[seed] != nc["topo_F1"] else float("nan")
    out.append(f"  recovery share (a - c) / (E338a - c) = {rec:+.2f}   (E338a {E338A[seed]:.4f}, c2 with the hard step)")
    b = boots.get("a")
    out.append("  bootstrap a - c topo F1: " + ("missing" if b is None else f"{b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]"))
    kp = None
    if cards.get("ck") is not None and cards.get("ak") is not None:
        kp = cards["ak"]["numbers"]["head"]["topo_F1"] - cards["ck"]["numbers"]["head"]["topo_F1"]
        out.append(f"  own-keypoint witness ak - ck topo F1 {kp:+.4f}")
    if stop:
        return "INVALID", out + ["  STOP RULES FAILED: " + "; ".join(stop)]
    keep = all(guards.values()) and b is not None and b[1] > 0 and kp is not None and kp > 0
    note = ""
    if kp is not None and (kp > 0) != (d["topo_F1"] > 0):
        note = " (PRIMARY and own-keypoint signs differ: not interpretable)"
    return ("KEEP" if keep else "TIE") + note, out


def main():
    tokens = []
    for s in (0, 1):
        t = lambda tag, suffix="snap_nogeom_attach": load(f"{R}e324_s_s{s}/ends_card_{tag}_{suffix}_thr0.1.json")  # noqa: E731
        cards = {"c": t("e360c"), "a": t("e360a")}
        for k, tag in (("ck", "e360ck"), ("ak", "e360ak")):
            p = f"{R}e324_s_s{s}/ends_card_{tag}_snap_nogeom_thr0.1.json"
            cards[k] = load(p) if os.path.exists(p) else None
        t_ok = None
        if s == 0 and os.path.exists(f"{R}e324_s_s0/ends_card_e360t_snap_nogeom_attach_thr0.1.json"):
            x, y = load(f"{R}e324_s_s0/ends_card_e360t_snap_nogeom_attach_thr0.1.json"), load(f"{R}e324_s_s0/ends_card_e338a_snap_nogeom_attach_thr0.1.json")
            t_ok = all(x[k] == y[k] for k in ("numbers", "curves", "wired_both_ends", "long_links_dropped", "ends_within_1m_of_a_delivered_piece_end"))
        boots = {"a": interval(f"{ROOT}/temp/e360/boot_a_vs_c_s{s}.log")}
        print(f"== seed {s}")
        tok, lines = judge(s, cards, None, boots, t_ok)
        print("\n".join(lines))
        print(f"  token of the seed: {tok}")
        tokens.append(tok)
    if any(x == "INVALID" for x in tokens):
        final = "INVALID (a stop rule failed; nothing is cut, re-run after the fix)"
    elif all(x == "KEEP" for x in tokens):
        final = "KEEP (end-to-end candidate for R2; NOTHING adopted; the faithful re-dump card is next)"
    else:
        final = "TIE, kept and refined (D87)"
    print("TOKEN:", final)


def selftest():
    """planted cards: KEEP, TIE by a guard, TIE by a negative interval, INVALID by a stop rule, sign mismatch note"""
    def card(topo, dup, base_topo, curves, ent=0.01, ex=0.01, f1=0.88, junc=0.30, lanes=0.76, vert=19.0):
        return dict(curves=curves, numbers=dict(head=dict(topo_F1=topo, dup_frac=dup, **{"F1@1.0": f1}, junction_F1=junc, lanes_found=lanes, vert100_dp=vert), base=dict(topo_F1=base_topo)),
                    ends_within_1m_of_a_delivered_piece_end=dict(pairs=curves, entry=0.4 + ent, exit=0.4 + ex, keypoint_entry=0.4, keypoint_exit=0.4))
    kp = lambda topo: dict(numbers=dict(head=dict(topo_F1=topo)))  # noqa: E731
    base = lambda s: (BASE_REF[s]["c"], BASE_REF[s]["a"])  # noqa: E731
    good = dict(c=card(0.65, 0.17, BASE_REF[0]["c"], 11729), a=card(0.66, 0.175, BASE_REF[0]["a"], 11729), ck=kp(0.64), ak=kp(0.65))
    b_pos, b_neg, b_zero = (0.01, 0.002, 0.02), (-0.01, -0.02, -0.001), (0.001, -0.003, 0.005)
    tok, _ = judge(0, good, None, {"a": b_pos}, None)
    assert tok == "KEEP", tok
    tok, _ = judge(0, good, None, {"a": b_zero}, None)
    assert tok == "TIE", tok                                                                             # the interval contains 0
    tok, _ = judge(0, good, None, {"a": b_neg}, None)
    assert tok == "TIE", tok                                                                             # a negative interval
    bad_dup = dict(good, a=card(0.66, 0.19, BASE_REF[0]["a"], 11729))
    tok, _ = judge(0, bad_dup, None, {"a": b_pos}, None)
    assert tok == "TIE", tok                                                                             # dup_frac +0.02 against the bound 0.01
    bad_base = dict(good, a=card(0.66, 0.175, 0.5, 11729))
    tok, lines = judge(0, bad_base, None, {"a": b_pos}, None)
    assert tok == "INVALID" and "(i)" in lines[-1], (tok, lines[-1])
    bad_curves = dict(good, c=card(0.65, 0.17, BASE_REF[0]["c"], 11000))
    tok, lines = judge(0, bad_curves, None, {"a": b_pos}, None)
    assert tok == "INVALID" and "(ii)" in lines[-1], (tok, lines[-1])
    bad_sym = dict(good, a=card(0.66, 0.175, BASE_REF[0]["a"], 11729, ent=0.05))
    tok, lines = judge(0, bad_sym, None, {"a": b_pos}, None)
    assert tok == "INVALID" and "(iii)" in lines[-1], (tok, lines[-1])
    tok, _ = judge(0, good, None, {"a": b_pos}, False)
    assert tok == "INVALID", tok                                                                         # arm t failed
    flip = dict(good, ck=kp(0.65), ak=kp(0.64))                                                          # PRIMARY positive, own-keypoint negative
    tok, _ = judge(0, flip, None, {"a": b_pos}, None)
    assert tok.startswith("TIE") and "not interpretable" in tok, tok
    no_wit = dict(good, ck=None, ak=None)
    tok, _ = judge(0, no_wit, None, {"a": b_pos}, None)
    assert tok == "TIE", tok                                                                             # no witness: never KEEP
    tok, _ = judge(0, good, None, {"a": None}, None)
    assert tok == "TIE", tok                                                                             # no bootstrap: never KEEP
    assert interval("/nonexistent/x.log") is None
    print("verdict_e360 selftest ok (10 planted cases)")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
