"""E350 (promoted from `temp/e325/verdict_e325.py`): the verdict by hand of a wiring-head card (E325, E350): snapped topo F1 and the guards of the arm w (head seed 0) against the control c on both seeds, the head seeds 1 and 2
(range), the kill and the agreement numbers of `runs/<head prefix><seed>/wire_agreement.json`, the pilot, the calibration on labelled ends, the arm x (optional), the price.

    wsl bash tools/env/_py14.sh jwire_verdict.py [--card E325] [--control-prefix e312_p_s] [--head-prefix e325_w_s] [--tag e325w] [--tag-x e350x] [--boot <pattern>]     # the defaults are E325's: reads the cards
                                                                                  # `runs/<control prefix><S>/ends_card_snap_thr0.1.json` (c), `ends_card_<tag>[1|2]_snap_thr0.1.json` (head seeds 0 | 1 | 2),
                                                                                  # `ends_card_<tag-x>_snap_thr0.1.json` (arm x), `runs/<head prefix><S>/wire_agreement.json` and the log of
                                                                                  # `jdiag.py boot` (default `temp/<card>/boot_w_vs_c_s<S>.log`, `{seed}` in a pattern)
    wsl bash tools/env/_py14.sh jwire_verdict.py selftest
    E350: --card E350 --control-prefix e324_s_s --head-prefix e350_w_s --tag e350w --tag-x e350x

The rule (E350 section 0, the rule of E325 with a sign class): KEEP when, on BOTH seeds, snapped topo F1 of w is at least c - 0.005 and every guard holds (BOUND). The class comes from the paired 95 % interval of w - c (head 0,
scene bootstrap): KEEP+ when its lower bound is above 0 on both seeds, KEEP- when its upper bound is below 0 on either seed (a loss inside the band: reported as a loss), KEEP= otherwise (no interval for a seed: the class stays
open). TIE (kept and refined, D87) when a bound or a guard is missed and no kill fired. CUT only by a kill: (a) a number is not finite, (b) the place-level agreement with the oracle label on the VAL real choices is below the
rule's by more than 0.02 on either seed. A kill that fires on the TRAIN ends only is a SHIFT FLAG, not a kill: the token then comes from topo F1 and the guards. Pilot limit: a TRAIN median keypoint error below 0.7 x the VAL
median labels the result 'shift-dominated'. INCOMPLETE without the cards of arm w and the wire_agreement.json of BOTH seeds; INVALID when a head says it was trained on another pair run or seed than the control's.
"""
import argparse
import json
import math
import os
import re
import tempfile
from types import SimpleNamespace as NS

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")
BOUND = dict(topo_F1=0.005, **{"F1@1.0": 0.01}, dup_frac=0.01, vert100_dp=1.0, junction_F1=0.01, lanes_found=0.01)   # max loss against c (dup_frac and vertices: a rise), the JSON of the card
RISE = ("dup_frac", "vert100_dp")
SEEDS = (0, 1)
KILL_MARGIN, SHIFT_LIMIT = 0.02, 0.7
BOOT = re.compile(r"topo_F1\s+A - B: point ([+-][0-9.]+), bootstrap mean [+-][0-9.]+, 95 % \[([+-][0-9.]+), ([+-][0-9.]+)\]")


def finite(x):
    """True when every number inside a nested dict / list is finite"""
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def guards(w, c):
    """-> (difference to c per key, {key: within its bound})"""
    d = {k: w[k] - c[k] for k in K}
    return d, {k: ((d[k] if k in RISE else -d[k]) <= BOUND[k]) for k in K}


def read_seed(a, s):
    """-> everything the verdict needs of one seed (None where a file is missing): c, w = [numbers of head 0, 1, 2], x, wa = wire_agreement.json, boot = (point, lower, upper)"""
    rd = f"{a.root}/runs/{a.control_prefix}{s}/"

    def card(tag):
        p = rd + f"ends_card_{tag + '_' if tag else ''}snap_thr0.1.json"
        return json.load(open(p))["numbers"]["head"] if os.path.exists(p) else None
    wp = f"{a.root}/runs/{a.head_prefix}{s}/wire_agreement.json"
    bp = os.path.join(a.root, a.boot.format(seed=s))
    m = BOOT.search(open(bp).read()) if os.path.exists(bp) else None
    return dict(c=card(None), w=[card(a.tag + (str(h) if h else "")) for h in range(3)], x=card(a.tag_x) if a.tag_x else None, wa=json.load(open(wp)) if os.path.exists(wp) else None,
                boot=tuple(float(g) for g in m.groups()) if m else None, run=f"{a.control_prefix}{s}")


def judge(data):
    """the rule of the module docstring on {seed: read_seed(...)} -> (token, why, notes)"""
    notes = []
    missing = [f"seed {s}: {what}" for s in SEEDS for what, ok in (("the control card", data[s]["c"]), ("the card of arm w", data[s]["w"][0]), ("wire_agreement.json", data[s]["wa"])) if ok is None]
    if missing:
        return "INCOMPLETE", "missing: " + "; ".join(missing) + " (a crashed run or scorer; a crash with 'the loss is not finite' in its log is the kill (a): by hand)", notes
    for s in SEEDS:
        wa = data[s]["wa"]
        if wa.get("pair_run", data[s]["run"]) != data[s]["run"] or wa.get("seed", s) != s:
            return "INVALID", f"the head of seed {s} was trained on pair run {wa.get('pair_run')} seed {wa.get('seed')}, the control is {data[s]['run']} seed {s}", notes
        if wa["pilot_train"]["err_median"] < SHIFT_LIMIT * wa["pilot_val"]["err_median"]:
            notes.append(f"seed {s}: SHIFT-DOMINATED (TRAIN median keypoint error {wa['pilot_train']['err_median']:.2f} m < {SHIFT_LIMIT} x VAL {wa['pilot_val']['err_median']:.2f} m)")
        if wa.get("kill_on_train") and not wa.get("kill_on_val"):
            notes.append(f"seed {s}: SHIFT FLAG (below the rule's agreement on the TRAIN ends with the jitter, not on VAL: not a kill, the token comes from topo F1 and the guards)")
    bad = [s for s in SEEDS if not finite(data[s]["wa"]) or not finite(data[s]["w"][0])]
    if bad:
        return "CUT", f"kill (a): a number of arm w is not finite (seed {bad})", notes
    bad = [s for s in SEEDS if data[s]["wa"].get("kill_on_val")]
    if bad:
        return "CUT", f"kill (b): the VAL real-choice agreement is below the rule's by more than {KILL_MARGIN} (seed {bad})", notes
    miss = {s: [k for k, ok in guards(data[s]["w"][0], data[s]["c"])[1].items() if not ok] for s in SEEDS}
    if any(miss.values()):
        return "TIE", f"kept and refined (D87): the head stays the differentiable replacement, a bound or a guard is missed: {miss}", notes
    iv = [data[s]["boot"] for s in SEEDS]
    if any(i is None for i in iv):
        return "KEEP", "the sign class is open: no bootstrap interval for seed " + str([s for s, i in zip(SEEDS, iv) if i is None]), notes
    if any(i[2] < 0 for i in iv):
        return "KEEP-", "the paired 95 % interval of w - c is below 0 on a seed: a loss inside the allowed band, reported as a loss", notes
    if all(i[1] > 0 for i in iv):
        return "KEEP+", "the paired 95 % interval of w - c is above 0 on both seeds", notes
    return "KEEP=", "the paired 95 % interval of w - c includes 0 on a seed", notes


def render(a, data):
    out = [f"{a.card} verdict by hand (the rule of the card section 0, written before the run): control {a.control_prefix}<S>, head {a.head_prefix}<S>, tag {a.tag}"]
    for s in SEEDS:
        sd = data[s]
        c, wa = sd["c"], sd["wa"]
        if wa is None:
            out.append(f"== seed {s}: no wire_agreement.json (the training run did not finish)")
            continue
        pt = wa["pilot_train"]
        out.append(f"== seed {s}: pilot TRAIN {{keypoint error median {pt['err_median']:.2f} m, rule real-choice agreement {pt['multi']:.3f}, kp_p mean {pt.get('kp_p_mean', float('nan')):.2f}, multi-place share {pt.get('multi_share', float('nan')):.2f}}}, jitter {wa['jitter_scale']:.2f} m")
        out.append(f"   pilot VAL keypoint error median {wa['pilot_val']['err_median']:.2f} m (TRAIN / VAL {pt['err_median'] / wa['pilot_val']['err_median']:.2f}, shift-dominated below {SHIFT_LIMIT}); provenance: pair run {wa.get('pair_run', 'not recorded (E312)')}, seed {wa.get('seed', s)}")
        for h in wa["heads"]:
            v, t = h["val"], h["train_jitter"]
            out.append(f"   head {h['head']}{' (distance-only ablation)' if h['ablation'] else ''}: VAL overall {v['head_same'] / v['oracle']:.3f} (rule {v['rule_same'] / v['oracle']:.3f}), real {v['head_multi_same'] / v['multi']:.3f} (rule {v['rule_multi_same'] / v['multi']:.3f}) | TRAIN+jitter real {t['head_multi_same'] / t['multi']:.3f} (rule {t['rule_multi_same'] / t['multi']:.3f})")
        out.append(f"   KILL fired: {wa['kill_fired']} (on VAL {wa.get('kill_on_val')}, on TRAIN {wa.get('kill_on_train')}); a fire on the TRAIN ends only is a shift flag, not a kill (the in-sample VAL cross-validation is above the rule)")
        cal = wa.get("calibration_val_labelled_ends")
        if cal:
            out.append("   calibration on labelled VAL ends (largest weight bin: ends, share correct): " + ", ".join(f"[{lo:.1f},{hi:.1f}) {n}: {r:.2f}" for lo, hi, n, r in cal if n and r is not None))
        for hh, w in enumerate(sd["w"]):
            if w is None or c is None:
                out.append(f"   head {hh}: no scored card")
                continue
            d, ok = guards(w, c)
            out.append(f"   head {hh} snapped: " + " ".join(f"{k} {w[k]:.4f} ({d[k]:+.4f})" for k in K) + f" | guards {'held' if all(ok.values()) else 'MISSED: ' + str([k for k in ok if not ok[k]])}")
        out.append(f"   bootstrap w - c topo F1: {tuple(f'{x:+.4f}' for x in sd['boot']) if sd['boot'] else 'not yet'}")
        if a.tag_x:
            x = sd["x"]
            out.append(f"   arm x ({a.tag_x}, the transfer control): " + ("no scored card" if x is None or c is None else " ".join(f"{k} {x[k]:.4f} (vs c {x[k] - c[k]:+.4f}" + (f", vs w {x[k] - sd['w'][0][k]:+.4f})" if sd["w"][0] else ")") for k in K)))
    token, why, notes = judge(data)
    out += ["price: 545 parameters, at most 0.12 MMACs per frame (the card's count; ms of the host-side candidate lists not measured here)"] + [f"NOTE {n}" for n in notes]
    out.append(f"TOKEN ({a.card}): {token} ({why})")
    return out, token


def _selftest():
    def agree(multi_head, kill_val=False, kill_train=False, train_err=1.0, **extra):
        v = dict(oracle=1000, head_same=900, rule_same=880, multi=500, head_multi_same=multi_head, rule_multi_same=380)
        return dict(dict(pilot_train=dict(err_median=train_err, multi=0.7), pilot_val=dict(err_median=1.1), jitter_scale=0.3, heads=[dict(head=0, ablation=False, val=v, train_jitter=v), dict(head=1, ablation=True, val=v, train_jitter=v)],
                         kill_fired=kill_val or kill_train, kill_on_val=kill_val, kill_on_train=kill_train, calibration_val_labelled_ends=[[0.9, 1.0, 10, 0.98]]), **extra)

    def world(t, spec):
        """spec[seed] = dict(d = {key: difference of head 0 to c}, iv = (lo, hi) or None, wa = wire_agreement.json dict or None, x = True)"""
        for s, sp in spec.items():
            rd, hd = f"{t}/runs/cs{s}", f"{t}/runs/hs{s}"
            os.makedirs(rd, exist_ok=True)
            os.makedirs(hd, exist_ok=True)
            c = dict(topo_F1=0.70, dup_frac=0.17, **{"F1@1.0": 0.87}, lanes_found=0.76, junction_F1=0.32, vert100_dp=19.0)
            json.dump(dict(numbers=dict(head=c)), open(f"{rd}/ends_card_snap_thr0.1.json", "w"))
            if sp.get("d") is not None:
                json.dump(dict(numbers=dict(head={k: c[k] + sp["d"].get(k, 0.0) for k in K})), open(f"{rd}/ends_card_tw_snap_thr0.1.json", "w"))
            if sp.get("x"):
                json.dump(dict(numbers=dict(head={k: c[k] - 0.001 for k in K})), open(f"{rd}/ends_card_tx_snap_thr0.1.json", "w"))
            if sp.get("wa") is not None:
                json.dump(sp["wa"], open(f"{hd}/wire_agreement.json", "w"))
            if sp.get("iv"):
                os.makedirs(f"{t}/temp/e000", exist_ok=True)
                open(f"{t}/temp/e000/boot_w_vs_c_s{s}.log", "w").write(f"A = x\n  topo_F1      A - B: point {sum(sp['iv']) / 2:+.4f}, bootstrap mean {sum(sp['iv']) / 2:+.4f}, 95 % [{sp['iv'][0]:+.4f}, {sp['iv'][1]:+.4f}]\n  dup_frac     A - B: point +0.0001, bootstrap mean +0.0001, 95 % [-0.0010, +0.0010]\n")

    def run_case(spec, **kw):
        with tempfile.TemporaryDirectory() as t:
            world(t, spec)
            a = NS(root=t, card="E000", control_prefix="cs", head_prefix="hs", tag="tw", tag_x=kw.get("tag_x"), boot="temp/e000/boot_w_vs_c_s{seed}.log")
            data = {s: read_seed(a, s) for s in SEEDS}
            lines, token = render(a, data)
            assert token == judge(data)[0] and lines[-1].startswith(f"TOKEN (E000): {token}"), lines[-1]
            return token, lines, judge(data)
    good = dict(d=dict(topo_F1=0.002), iv=(0.001, 0.004), wa=agree(420))
    # KEEP+ / KEEP= / KEEP- from the sign of the paired interval
    assert run_case({0: good, 1: good})[0] == "KEEP+"
    assert run_case({0: good, 1: dict(good, iv=(-0.001, 0.004))})[0] == "KEEP="                          # the interval includes 0 on a seed
    assert run_case({0: good, 1: dict(good, d=dict(topo_F1=-0.003), iv=(-0.006, -0.001))})[0] == "KEEP-"    # a loss inside the band (-0.005): upper bound below 0
    assert run_case({0: good, 1: dict(good, iv=None)})[0] == "KEEP"                                        # no bootstrap log: the class stays open
    # TIE: the topo bound or a guard missed, no kill; the head is kept
    assert run_case({0: good, 1: dict(good, d=dict(topo_F1=-0.006))})[0] == "TIE"
    assert run_case({0: good, 1: dict(good, d=dict(topo_F1=0.002, dup_frac=0.011))})[0] == "TIE"
    assert run_case({0: dict(good, d=dict(topo_F1=0.002, vert100_dp=1.5)), 1: good})[0] == "TIE"
    assert run_case({0: good, 1: dict(good, d=dict(topo_F1=0.002, lanes_found=-0.011))})[0] == "TIE"      # a loss of a metric that rises when better
    # CUT only by the kill: (b) on VAL, either seed, even when topo F1 and the guards are fine; (a) not finite
    assert run_case({0: good, 1: dict(good, wa=agree(360, kill_val=True))})[0] == "CUT"
    assert run_case({0: good, 1: dict(good, wa=agree(float("nan")))})[0] == "CUT"
    assert run_case({0: good, 1: dict(good, d=dict(topo_F1=float("nan")))})[0] == "CUT"
    # a TRAIN-only fire is a shift flag: the token comes from topo F1 and the guards (KEEP+ here, TIE when a guard is missed), and the flag is a note
    tok, lines, (_, _, notes) = run_case({0: good, 1: dict(good, wa=agree(420, kill_train=True))})
    assert tok == "KEEP+" and any("SHIFT FLAG" in n for n in notes) and not any("SHIFT-DOMINATED" in n for n in notes), (tok, notes)
    assert run_case({0: good, 1: dict(good, d=dict(topo_F1=-0.006), wa=agree(420, kill_train=True))})[0] == "TIE"
    # the pilot limit: TRAIN median error below 0.7 x VAL's labels the result
    tok, _, (_, _, notes) = run_case({0: good, 1: dict(good, wa=agree(420, train_err=0.7))})
    assert tok == "KEEP+" and any("SHIFT-DOMINATED" in n for n in notes), notes                         # 0.7 < 0.77 = 0.7 x 1.1
    # INCOMPLETE without both seeds (the card, the run's file) and INVALID for a head of another pair run
    assert run_case({0: good})[0] == "INCOMPLETE" and run_case({0: good, 1: dict(good, wa=None)})[0] == "INCOMPLETE" and run_case({0: good, 1: dict(good, d=None)})[0] == "INCOMPLETE"
    assert run_case({0: good, 1: dict(good, wa=agree(420, pair_run="cs0", seed=1))})[0] == "INVALID" and run_case({0: good, 1: dict(good, wa=agree(420, pair_run="cs1", seed=0))})[0] == "INVALID"
    assert run_case({0: good, 1: dict(good, wa=agree(420, pair_run="cs1", seed=1))})[0] == "KEEP+"
    # arm x is reported when its tag is asked for and never changes the token
    tok, lines, _ = run_case({0: dict(good, x=True), 1: dict(good, x=True)}, tag_x="tx")
    assert tok == "KEEP+" and sum("arm x (tx" in l for l in lines) == 2 and not any("no scored card" in l for l in lines if "arm x" in l), lines
    print("jwire_verdict selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="verdict", choices=("verdict", "selftest"))
    ap.add_argument("--card", default="E325", help="the card number of the heading and of the default bootstrap logs")
    ap.add_argument("--control-prefix", default="e312_p_s", help="the pair run of the control: runs/<prefix><seed> holds the control card and the head cards")
    ap.add_argument("--head-prefix", default="e325_w_s", help="the training runs of the head: runs/<prefix><seed>/wire_agreement.json")
    ap.add_argument("--tag", default="e325w", help="the tag of the head cards (head seed 0 = the arm; head seeds 1 and 2 = <tag>1, <tag>2)")
    ap.add_argument("--tag-x", default=None, help="the tag of the cards of arm x (the transfer control), reported only")
    ap.add_argument("--boot", default=None, help="the bootstrap log of `jdiag.py boot` per seed, {seed} is replaced (default temp/<card>/boot_w_vs_c_s{seed}.log)")
    a = ap.parse_args(argv)
    if a.what == "selftest":
        _selftest()
        return
    a.root = ROOT
    a.boot = a.boot or f"temp/{a.card.lower()}/boot_w_vs_c_s{{seed}}.log"
    lines, _ = render(a, {s: read_seed(a, s) for s in SEEDS})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
