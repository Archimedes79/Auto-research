"""E350 (promoted from `temp/e325/verdict_e325.py`): the verdict by hand of a wiring-head card (E325, E350): snapped topo F1 and the guards of the arm w (head seed 0) against the control c on both seeds, the head seeds 1 and 2
(range), the kill and the agreement numbers of `runs/<head prefix><seed>/wire_agreement.json`, the pilot, the calibration on labelled ends, the arm x (optional), the price.

    wsl bash tools/env/_py14.sh jwire_verdict.py [--card E325] [--control-prefix e312_p_s] [--head-prefix e325_w_s] [--tag e325w] [--tag-x e350x --x-head-prefix e325_w_s] [--read-out snap|primary] [--boot <pattern>]     # the defaults are E325's:
                                                                                  # reads the cards `runs/<control prefix><S>/ends_card_snap_thr0.1.json` (c), `ends_card_<tag>[1|2]_snap_thr0.1.json` (head seeds
                                                                                  # 0 | 1 | 2), `ends_card_<tag-x>_snap_thr0.1.json` (arm x), `runs/<head prefix><S>/wire_agreement.json` and the log of `jdiag.py boot`
                                                                                  # (default `temp/<card>/boot_w_vs_c_s<S>.log`, `{seed}` in a pattern; the LAST block of the file counts)
    wsl bash tools/env/_py14.sh jwire_verdict.py selftest
    E350: --card E350 --control-prefix e324_s_s --head-prefix e350_w_s --tag e350w --tag-x e350x
    E358: --card E358 --control-prefix e324_s_s --head-prefix e358_w_s --tag e358w --read-out primary
The READ-OUT (E358): `snap` (default, E325 / E350) reads the snapped cards `ends_card_[<tag>_]snap_thr0.1.json` and wants `attach` and `snap_nogeom` false in them; `primary` (the end-to-end read-out of E324 s: attached ends, snap wiring without the
snap geometry) reads `ends_card_[<tag>_]snap_nogeom_attach_thr0.1.json` (the control `ends_card_snap_nogeom_attach_thr0.1.json`) and wants `attach` true and `snap_nogeom` (in the card's `snap_options`) true. The head's `ends`
(`wire_agreement.json`, no field = kp) must be the read-out's kind (primary: att, snap: kp): a head of the other kind is INVALID.

The rule (E350 section 0, the rule of E325 with a sign class): KEEP when, on BOTH seeds, snapped topo F1 of w is at least c - 0.005 and every guard holds (BOUND). The class comes from the paired 95 % interval of w - c (head 0,
scene bootstrap): KEEP+ when its lower bound is above 0 on both seeds, KEEP- when its upper bound is below 0 on either seed (a loss inside the band: reported as a loss), KEEP= otherwise. TIE (kept and refined, D87) when a
bound or a guard is missed and no kill fired. CUT only by a kill: (a) a number is not finite, (b) the place-level agreement with the oracle label on the VAL real choices is below the rule's by more than 0.02 on either seed
(recomputed here from the numbers of head 0; a stored flag that disagrees is noted, not trusted). A kill that fires on the TRAIN ends only is a SHIFT FLAG, not a kill: the token then comes from topo F1 and the guards. Pilot
limit: a TRAIN median keypoint error below 0.7 x the VAL median, or a TRAIN share of the labelled ends whose nearest piece end lies within 0.5 m (`near05` of the pilot, E358) that differs from the VAL share by more than 0.10 (absolute), labels the result 'shift-dominated'. INCOMPLETE without the control card, the card of arm w and the wire_agreement.json of BOTH seeds, and without a bootstrap
interval for a KEEP (the sign class is open: never a bare KEEP). INVALID (with the reason) when what is read is not what the card is about: a card of another run, seed, read-out (snap, attach, threshold), tag or head (`wire`,
`wire_run`: w = <head prefix><S>, heads 1 and 2 = `...:model_h<k>.pt`, arm x = <x head prefix><S>), a control scored with a head, a wire_agreement.json of another pair run or seed (the key `pair_run` is required unless the
control is E312's `e312_p_s<S>`), a bootstrap log that is not w against c of this control (header line `A = <run> <card>   B = <run> <card>`) or whose point estimate is not w - c of the cards (within 1e-4).
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
KILL_MARGIN, SHIFT_LIMIT, SHARE_LIMIT, POINT_TOL = 0.02, 0.7, 0.10, 1e-4
READ_OUT_ENDS = dict(snap="kp", primary="att")                                                       # the read-outs and the end kind a head must have been trained on for each
BOOT_HEAD = re.compile(r"^A = (\S+) (\S+)\s+B = (\S+) (\S+)\s+\d+ scenes, \d+ resamples\s*$", re.M)
BOOT_TOPO = re.compile(r"^\s*topo_F1\s+A - B: point ([+-][0-9.]+), bootstrap mean [+-][0-9.]+, 95 % \[([+-][0-9.]+), ([+-][0-9.]+)\]", re.M)


def stem(read_out):
    """the part of a card's file name after the tag: `ends_card_[<tag>_]<stem>thr0.1.json` (the name rule of `qends_eval.card_name` for the two read-outs)"""
    return "snap_nogeom_attach_" if read_out == "primary" else "snap_"


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


def kills(wa):
    """-> (kill on the VAL real choices, fire on the TRAIN ends) recomputed from the numbers of head 0 (the rule of `jwire_train.kill_fired`)"""
    h = wa["heads"][0]
    below = lambda d: d["head_multi_same"] / max(d["multi"], 1) < d["rule_multi_same"] / max(d["multi"], 1) - KILL_MARGIN  # noqa: E731
    return bool(below(h["val"])), bool(below(h["train_jitter"]))


def boot_block(text):
    """-> the LAST block of a `jdiag.py boot` log (the post-processing appends): dict(a=(run, card), b=(run, card), point, lower, upper), or dict(error=...)"""
    heads = list(BOOT_HEAD.finditer(text))
    if not heads:
        return dict(error="no header line `A = <run> <card>   B = <run> <card>`")
    h = heads[-1]
    m = BOOT_TOPO.search(text, h.end())
    if not m:
        return dict(error="no topo_F1 line after the header")
    return dict(a=(h.group(1), h.group(2)), b=(h.group(3), h.group(4)), point=float(m.group(1)), lower=float(m.group(2)), upper=float(m.group(3)))


def read_seed(a, s):
    """-> everything the verdict needs of one seed (None where a file is missing): c, w = [numbers of head 0, 1, 2], x; cards = the whole cards (their provenance is checked); wa = wire_agreement.json; boot = `boot_block`"""
    rd = f"{a.root}/runs/{a.control_prefix}{s}/"

    def card(tag):
        p = rd + f"ends_card_{tag + '_' if tag else ''}{stem(a.read_out)}thr0.1.json"
        return json.load(open(p)) if os.path.exists(p) else None
    cards = dict(c=card(None), w=[card(a.tag + (str(h) if h else "")) for h in range(3)], x=card(a.tag_x) if a.tag_x else None)
    num = lambda c: c["numbers"]["head"] if c else None  # noqa: E731
    wp = f"{a.root}/runs/{a.head_prefix}{s}/wire_agreement.json"
    bp = os.path.join(a.root, a.boot.format(seed=s))
    return dict(c=num(cards["c"]), w=[num(c) for c in cards["w"]], x=num(cards["x"]), cards=cards, wa=json.load(open(wp)) if os.path.exists(wp) else None, boot=boot_block(open(bp).read()) if os.path.exists(bp) else None,
                run=f"{a.control_prefix}{s}")


def check_inputs(a, s, sd):
    """-> the reason (a string) when what was read is not what the card is about, else None (the INVALID rule of the module docstring)"""
    run, cards, wa, bt = sd["run"], sd["cards"], sd["wa"], sd["boot"]
    primary = a.read_out == "primary"
    named = [("control card", cards["c"])] + [(f"head {h} card", c) for h, c in enumerate(cards["w"])] + [("arm x card", cards["x"])]
    for what, c in named:
        if c is None:
            continue
        got = dict(c, snap_nogeom="snap_nogeom" in (c.get("snap_options") or []))
        for key, ok in (("run", c.get("run") == run), ("seed", c.get("seed") == s), ("snap", c.get("snap") is True), ("attach", bool(c.get("attach")) == primary), ("snap_nogeom", got["snap_nogeom"] == primary), ("thr", c.get("thr") == 0.1)):
            if not ok:
                return f"seed {s}: the {what} says {key} = {got.get(key)!r} (the {'primary (attached ends, snap wiring without the snap geometry)' if primary else 'snapped'} read-out at p >= 0.1 of {run} seed {s} is meant)"
    c = cards["c"]
    if c.get("wire") is not None or c.get("tag") is not None:
        return f"seed {s}: the control card is not the rule's (wire {c.get('wire')!r}, tag {c.get('tag')!r})"
    heads = [(f"head {h} card", cards["w"][h], a.tag + (str(h) if h else ""), f"{a.head_prefix}{s}" + (f":model_h{h}.pt" if h else "")) for h in range(3)]
    if a.tag_x:
        heads.append(("arm x card", cards["x"], a.tag_x, f"{a.x_head_prefix}{s}"))
    for what, c, tag, wire_run in heads:
        if c is not None and (c.get("wire") != "head" or c.get("wire_run") != wire_run or c.get("tag") != tag):
            return f"seed {s}: the {what} is wire {c.get('wire')!r} wire_run {c.get('wire_run')!r} tag {c.get('tag')!r}, expected head {wire_run!r} tag {tag!r}"
    if "pair_run" not in wa and run != f"e312_p_s{s}":
        return f"seed {s}: wire_agreement.json has no pair_run (provenance is required unless the control is e312_p_s{s})"
    if wa.get("pair_run", run) != run or wa.get("seed") != s:
        return f"seed {s}: the head was trained on pair run {wa.get('pair_run')} seed {wa.get('seed')}, the control is {run} seed {s}"
    if [h["head"] for h in wa["heads"]][:1] != [0] or wa["heads"][0]["ablation"]:
        return f"seed {s}: the first head of wire_agreement.json is not head 0"
    if wa.get("ends", "kp") != READ_OUT_ENDS[a.read_out]:
        return f"seed {s}: the head was trained on {wa.get('ends', 'kp')!r} ends (no field = kp), the {a.read_out} read-out needs a head trained on {READ_OUT_ENDS[a.read_out]!r} ends"
    if bt is not None:
        if "error" in bt:
            return f"seed {s}: the bootstrap log is not a `jdiag.py boot` log: {bt['error']}"
        want_a, want_b = (run, f"ends_card_{a.tag}_{stem(a.read_out)}thr0.1"), (run, f"ends_card_{stem(a.read_out)}thr0.1")
        if bt["a"] != want_a or bt["b"] != want_b:
            return f"seed {s}: the bootstrap log is A = {bt['a']} against B = {bt['b']}, expected {want_a} against {want_b}"
        w0, c0 = sd["w"][0], sd["c"]
        if w0 is not None and abs(bt["point"] - (w0["topo_F1"] - c0["topo_F1"])) > POINT_TOL:
            return f"seed {s}: the bootstrap point estimate {bt['point']:+.4f} is not topo F1 of w - c of the cards ({w0['topo_F1'] - c0['topo_F1']:+.4f})"
    return None


def judge(data):
    """the rule of the module docstring on {seed: read_seed(...)} -> (token, why, notes)"""
    notes = []
    missing = [f"seed {s}: {what}" for s in SEEDS for what, ok in (("the control card", data[s]["c"]), ("the card of arm w", data[s]["w"][0]), ("wire_agreement.json", data[s]["wa"])) if ok is None]
    if missing:
        return "INCOMPLETE", "missing: " + "; ".join(missing) + " (a crashed run or scorer; a crash with 'the loss is not finite' in its log is the kill (a): by hand)", notes
    a = data["args"]
    for s in SEEDS:
        why = check_inputs(a, s, data[s])
        if why:
            return "INVALID", why, notes
    kv = {}
    for s in SEEDS:
        wa = data[s]["wa"]
        kv[s], kt = kills(wa)
        for name, now in (("kill_on_val", kv[s]), ("kill_on_train", kt)):
            if wa.get(name) is not None and bool(wa[name]) != now:
                notes.append(f"seed {s}: the stored {name} is {wa[name]}, the numbers of head 0 give {now} (the recomputed one is used)")
        if wa["pilot_train"]["err_median"] < SHIFT_LIMIT * wa["pilot_val"]["err_median"]:
            notes.append(f"seed {s}: SHIFT-DOMINATED (TRAIN median keypoint error {wa['pilot_train']['err_median']:.2f} m < {SHIFT_LIMIT} x VAL {wa['pilot_val']['err_median']:.2f} m)")
        near = [wa[k].get("near05") for k in ("pilot_train", "pilot_val")]
        if None not in near and abs(near[0] - near[1]) > SHARE_LIMIT:
            notes.append(f"seed {s}: SHIFT-DOMINATED (TRAIN share of the labelled ends within 0.5 m of a piece end {near[0]:.3f} differs from VAL {near[1]:.3f} by more than {SHARE_LIMIT})")
        if kt and not kv[s]:
            notes.append(f"seed {s}: SHIFT FLAG (below the rule's agreement on the TRAIN ends with the jitter, not on VAL: not a kill, the token comes from topo F1 and the guards)")
    bad = [s for s in SEEDS if not finite(data[s]["wa"]) or not finite(data[s]["w"][0])]
    if bad:
        return "CUT", f"kill (a): a number of arm w is not finite (seed {bad})", notes
    bad = [s for s in SEEDS if kv[s]]
    if bad:
        return "CUT", f"kill (b): the VAL real-choice agreement is below the rule's by more than {KILL_MARGIN} (seed {bad})", notes
    miss = {s: [k for k, ok in guards(data[s]["w"][0], data[s]["c"])[1].items() if not ok] for s in SEEDS}
    if any(miss.values()):
        return "TIE", f"kept and refined (D87): the head stays the differentiable replacement, a bound or a guard is missed: {miss}", notes
    iv = [data[s]["boot"] for s in SEEDS]
    if any(i is None for i in iv):
        return "INCOMPLETE", "the sign class is open: no bootstrap interval (`jdiag.py boot` log) for seed " + str([s for s, i in zip(SEEDS, iv) if i is None]), notes
    if any(i["upper"] < 0 for i in iv):
        return "KEEP-", "the paired 95 % interval of w - c is below 0 on a seed: a loss inside the allowed band, reported as a loss", notes
    if all(i["lower"] > 0 for i in iv):
        return "KEEP+", "the paired 95 % interval of w - c is above 0 on both seeds", notes
    return "KEEP=", "the paired 95 % interval of w - c includes 0 on a seed", notes


def render(a, data):
    data = dict(data, args=a)
    out = [f"{a.card} verdict by hand (the rule of the card section 0, written before the run): control {a.control_prefix}<S>, head {a.head_prefix}<S>, tag {a.tag}, read-out {a.read_out}"]
    for s in SEEDS:
        sd = data[s]
        c, wa = sd["c"], sd["wa"]
        if wa is None:
            out.append(f"== seed {s}: no wire_agreement.json (the training run did not finish)")
            continue
        pt = wa["pilot_train"]
        out.append(f"== seed {s}: pilot TRAIN {{keypoint error median {pt['err_median']:.2f} m, rule real-choice agreement {pt['multi']:.3f}, kp_p mean {pt.get('kp_p_mean', float('nan')):.2f}, multi-place share {pt.get('multi_share', float('nan')):.2f}}}, jitter {wa['jitter_scale']:.2f} m")
        out.append(f"   pilot VAL keypoint error median {wa['pilot_val']['err_median']:.2f} m (TRAIN / VAL {pt['err_median'] / wa['pilot_val']['err_median']:.2f}, shift-dominated below {SHIFT_LIMIT}); labelled ends within 0.5 m of a piece end: TRAIN {pt.get('near05', float('nan')):.3f}, VAL {wa['pilot_val'].get('near05', float('nan')):.3f} (shift-dominated above a difference of {SHARE_LIMIT}); provenance: pair run {wa.get('pair_run', 'not recorded (E312)')}, ends {wa.get('ends', 'kp (not recorded)')}, seed {wa.get('seed', s)}")
        for h in wa["heads"]:
            v, t = h["val"], h["train_jitter"]
            out.append(f"   head {h['head']}{' (distance-only ablation)' if h['ablation'] else ''}: VAL overall {v['head_same'] / v['oracle']:.3f} (rule {v['rule_same'] / v['oracle']:.3f}), real {v['head_multi_same'] / v['multi']:.3f} (rule {v['rule_multi_same'] / v['multi']:.3f}) | TRAIN+jitter real {t['head_multi_same'] / t['multi']:.3f} (rule {t['rule_multi_same'] / t['multi']:.3f})")
        nj = wa.get("no_jitter_head")
        if nj:
            v, t = nj["val"], nj["train"]
            out.append(f"   no-jitter head (wn, reported only): VAL overall {v['head_same'] / v['oracle']:.3f} (rule {v['rule_same'] / v['oracle']:.3f}), real {v['head_multi_same'] / max(v['multi'], 1):.3f} (rule {v['rule_multi_same'] / max(v['multi'], 1):.3f}) | TRAIN (unperturbed) real {t['head_multi_same'] / max(t['multi'], 1):.3f} (rule {t['rule_multi_same'] / max(t['multi'], 1):.3f})")
        kv, kt = kills(wa)
        out.append(f"   KILL fired: {kv or kt} (on VAL {kv}, on TRAIN {kt}; recomputed from the numbers of head 0); a fire on the TRAIN ends only is a shift flag, not a kill (the in-sample VAL cross-validation is above the rule)")
        cal = wa.get("calibration_val_labelled_ends")
        if cal:
            out.append("   calibration on labelled VAL ends (largest weight bin: ends, share correct): " + ", ".join(f"[{lo:.1f},{hi:.1f}) {n}: {r:.2f}" for lo, hi, n, r in cal if n and r is not None))
        for hh, w in enumerate(sd["w"]):
            if w is None or c is None:
                out.append(f"   head {hh}: no scored card")
                continue
            d, ok = guards(w, c)
            out.append(f"   head {hh} {'primary' if a.read_out == 'primary' else 'snapped'}: " + " ".join(f"{k} {w[k]:.4f} ({d[k]:+.4f})" for k in K) + f" | guards {'held' if all(ok.values()) else 'MISSED: ' + str([k for k in ok if not ok[k]])}")
        b = sd["boot"]
        out.append(f"   bootstrap w - c topo F1: {tuple(f'{b[k]:+.4f}' for k in ('point', 'lower', 'upper')) if b and 'error' not in b else ('unusable: ' + b['error'] if b else 'not yet')}")
        if a.tag_x:
            x = sd["x"]
            out.append(f"   arm x ({a.tag_x}, the transfer control): " + ("no scored card" if x is None or c is None else " ".join(f"{k} {x[k]:.4f} (vs c {x[k] - c[k]:+.4f}" + (f", vs w {x[k] - sd['w'][0][k]:+.4f})" if sd["w"][0] else ")") for k in K)))
    token, why, notes = judge(data)
    out += ["price: 545 parameters, at most 0.12 MMACs per frame (the card's count; ms of the host-side candidate lists not measured here)"] + [f"NOTE {n}" for n in notes]
    out.append(f"TOKEN ({a.card}): {token} ({why})")
    return out, token


def _selftest():
    SIX = {"topo_F1": 0.005, "F1@1.0": 0.01, "dup_frac": 0.01, "vert100_dp": 1.0, "junction_F1": 0.01, "lanes_found": 0.01}   # the bounds of the JSON of the card, typed here once more
    assert BOUND == SIX and set(K) == set(SIX) and set(RISE) == {"dup_frac", "vert100_dp"}
    RUN, SEED, NONE = "@run", "@seed", "@none"                                                          # placeholders of a planted wire_agreement.json: the control run, the seed, 'leave the key out'

    def agree(val_head=420, train_head=420, train_err=1.0, pair_run=RUN, seed=SEED, ends=NONE, near=None, nj=False, **extra):
        """a wire_agreement.json; the rule's real-choice agreement is 380 / 500 = 0.76 (a kill below 0.74), the stored kill flags follow the numbers, `extra` overrides them; ends = the key `ends` (NONE: left out, E325's files),
        near = (TRAIN, VAL) share of the labelled ends within 0.5 m (left out when None), nj = a no-jitter head"""
        v = lambda h: dict(oracle=1000, head_same=900, rule_same=880, multi=500, head_multi_same=h, rule_multi_same=380)  # noqa: E731
        kv, kt = val_head / 500 < 0.74, train_head / 500 < 0.74
        return dict(dict(pair_run=pair_run, seed=seed, ends=ends, pilot_train=dict(err_median=train_err, multi=0.7, **({} if near is None else dict(near05=near[0]))), pilot_val=dict(err_median=1.1, **({} if near is None else dict(near05=near[1]))), jitter_scale=0.3, kill_fired=kv or kt, kill_on_val=kv, kill_on_train=kt,
                         heads=[dict(head=0, ablation=False, val=v(val_head), train_jitter=v(train_head)), dict(head=1, ablation=True, val=v(420), train_jitter=v(420))], calibration_val_labelled_ends=[[0.9, 1.0, 10, 0.98]], **(dict(no_jitter_head=dict(val=v(400), train=v(410))) if nj else {})), **extra)

    def boot_text(run, point, lo, hi, a=None, b=None, st="snap_"):
        a, b = a or f"{run} ends_card_tw_{st}thr0.1", b or f"{run} ends_card_{st}thr0.1"
        return f"_py14.sh: evaluation/jdiag.py\nA = {a}   B = {b}   22 scenes, 1000 resamples\n  topo_F1      A - B: point {point:+.4f}, bootstrap mean {point:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}]\n  dup_frac     A - B: point +0.0001, bootstrap mean +0.0001, 95 % [-0.0010, +0.0010]\n"

    def world(t, spec, cp, ro="snap"):
        """spec[seed] = dict(d = {key: difference of head 0 to c} (None: no card), iv = (lower, upper) of the planted bootstrap log (None: no log), wa = wire_agreement.json (None: missing), x = a card of arm x,
        over = {card: {key: value}} (c, w0, w1, w2, x), boot = the text of the log instead of the planted one); ro = the read-out: the cards of `primary` are attached and nogeom, named as the scorer names them"""
        st, prim = stem(ro), ro == "primary"
        for s, sp in spec.items():
            rd, hd, run = f"{t}/runs/{cp}{s}", f"{t}/runs/hs{s}", f"{cp}{s}"
            os.makedirs(rd, exist_ok=True)
            os.makedirs(hd, exist_ok=True)
            c = dict(topo_F1=0.70, dup_frac=0.17, **{"F1@1.0": 0.87}, lanes_found=0.76, junction_F1=0.32, vert100_dp=19.0)

            def card(numbers, name, **kw):
                over = {k: (1 - s if v == "@other" else v) for k, v in sp.get("over", {}).get(name, {}).items()}
                return dict(dict(run=run, seed=s, thr=0.1, snap=True, attach=prim, numbers=dict(head=numbers), tag=None, wire=None, **({"snap_options": ["snap_nogeom"]} if prim else {})), **dict(kw, **over))
            json.dump(card(c, "c"), open(f"{rd}/ends_card_{st}thr0.1.json", "w"))
            if sp.get("d") is not None:
                for h in range(3):
                    wn = {k: c[k] + sp["d"].get(k, 0.0) * (1 if h == 0 else 0.5) for k in K}
                    json.dump(card(wn, f"w{h}", tag="tw" + (str(h) if h else ""), wire="head", wire_run=f"hs{s}" + (f":model_h{h}.pt" if h else "")), open(f"{rd}/ends_card_tw{h or ''}_{st}thr0.1.json", "w"))
            if sp.get("x"):
                json.dump(card({k: c[k] - 0.001 for k in K}, "x", tag="tx", wire="head", wire_run=f"xs{s}"), open(f"{rd}/ends_card_tx_{st}thr0.1.json", "w"))
            if sp.get("wa") is not None:
                wa = {k: (run if v == RUN else s if v == SEED else v) for k, v in sp["wa"].items() if v != NONE}
                json.dump(wa, open(f"{hd}/wire_agreement.json", "w"))
            if sp.get("boot") is not None or sp.get("iv"):
                os.makedirs(f"{t}/temp/e000", exist_ok=True)
                open(f"{t}/temp/e000/boot_w_vs_c_s{s}.log", "w").write(sp["boot"] if sp.get("boot") is not None else boot_text(run, (sp.get("d") or {}).get("topo_F1", 0.0), *sp["iv"], st=st))

    def run_case(spec, cp="cs", ro="snap", **kw):
        with tempfile.TemporaryDirectory() as t:
            world(t, spec, cp, ro)
            a = NS(root=t, card="E000", control_prefix=cp, head_prefix="hs", tag="tw", tag_x=kw.get("tag_x"), x_head_prefix="xs", boot="temp/e000/boot_w_vs_c_s{seed}.log", read_out=ro)
            data = {s: read_seed(a, s) for s in SEEDS}
            lines, token = render(a, data)
            verdict = judge(dict(data, args=a))
            assert token == verdict[0] and lines[-1].startswith(f"TOKEN (E000): {token}"), lines[-1]
            return token, lines, verdict
    tok = lambda *a, **k: run_case(*a, **k)[0]  # noqa: E731
    why = lambda *a, **k: run_case(*a, **k)[2][1]  # noqa: E731
    good = dict(d=dict(topo_F1=0.002), iv=(0.001, 0.004), wa=agree())
    seeds = lambda bad, ok=good: [{s: bad, 1 - s: ok} for s in SEEDS]                                    # the case on seed 0 and on seed 1
    # KEEP+ / KEEP= / KEEP- from the sign of the paired interval (a loss inside the band on either seed)
    assert tok({0: good, 1: good}) == "KEEP+"
    for sp in seeds(dict(good, iv=(-0.001, 0.004))):
        assert tok(sp) == "KEEP=", sp                                                                      # the interval includes 0 on a seed
    for sp in seeds(dict(good, d=dict(topo_F1=-0.003), iv=(-0.006, -0.001))):
        assert tok(sp) == "KEEP-", sp                                                                      # upper bound below 0 on either seed
    # TIE: every guard of the card just beyond its bound (and just inside it: KEEP+), on either seed; no kill
    for k, b in SIX.items():
        beyond, inside = (b + 0.0005, b - 0.0005) if k in RISE else (-b - 0.0005, -b + 0.0005)
        for sp in seeds(dict(good, d=dict({"topo_F1": 0.002}, **{k: beyond}))):
            assert tok(sp) == "TIE", (k, sp)
        assert tok({0: good, 1: dict(good, d=dict({"topo_F1": 0.002}, **{k: inside}))}) == "KEEP+", k
    # CUT only by the kill: (b) on VAL on either seed, even when topo F1 and the guards are fine; (a) not finite
    for sp in seeds(dict(good, wa=agree(360))) + seeds(dict(good, wa=agree(float("nan")))) + seeds(dict(good, d=dict(topo_F1=float("nan")), iv=None)):    # (a card with NaN has no usable bootstrap log: INVALID comes first)
        assert tok(sp) == "CUT", sp
    # the VAL kill is recomputed from the numbers: a stored flag that says False cannot hide it, one that says True cannot make it (both noted)
    t, _, (_, _, notes) = run_case({0: good, 1: dict(good, wa=agree(360, kill_on_val=False, kill_fired=False))})
    assert t == "CUT" and any("stored kill_on_val" in n for n in notes), notes
    t, _, (_, _, notes) = run_case({0: good, 1: dict(good, wa=agree(420, kill_on_val=True, kill_fired=True))})
    assert t == "KEEP+" and any("stored kill_on_val" in n for n in notes), notes
    # a TRAIN-only fire is a shift flag: the token comes from topo F1 and the guards (KEEP+ here, TIE when a guard is missed), and the flag is a note
    for sp in seeds(dict(good, wa=agree(420, 360))):
        t, _, (_, _, notes) = run_case(sp)
        assert t == "KEEP+" and any("SHIFT FLAG" in n for n in notes) and not any("SHIFT-DOMINATED" in n for n in notes), (t, notes)
    assert tok({0: good, 1: dict(good, d=dict(topo_F1=-0.006), wa=agree(420, 360))}) == "TIE"
    # the pilot limit: TRAIN median error below 0.7 x VAL's labels the result
    t, _, (_, _, notes) = run_case({0: good, 1: dict(good, wa=agree(train_err=0.7))})
    assert t == "KEEP+" and any("SHIFT-DOMINATED" in n for n in notes), notes                              # 0.7 < 0.77 = 0.7 x 1.1
    # INCOMPLETE: a seed, a card, the run's file missing; no interval for a KEEP (never a bare KEEP); but TIE / CUT need none
    assert tok({0: good}) == "INCOMPLETE" and tok({0: good, 1: dict(good, wa=None)}) == "INCOMPLETE" and tok({0: good, 1: dict(good, d=None)}) == "INCOMPLETE"
    assert tok({0: good, 1: dict(good, iv=None)}) == "INCOMPLETE" and "sign class is open" in why({0: good, 1: dict(good, iv=None)})
    assert tok({0: good, 1: dict(good, iv=None, d=dict(topo_F1=-0.006))}) == "TIE" and tok({0: good, 1: dict(good, iv=None, wa=agree(360))}) == "CUT"
    # INVALID: what is read is not what the card is about (each planted alone, on either seed); the cards of arm x are read when its tag is asked for
    bad_cards = [("c", dict(run="other")), ("w0", dict(run="other")), ("w1", dict(seed="@other")), ("w2", dict(seed="@other")), ("x", dict(run="other")), ("c", dict(snap=False)), ("w0", dict(snap=False)), ("c", dict(attach=True)),
                 ("w0", dict(attach=True)), ("c", dict(snap_options=["snap_nogeom"])), ("w1", dict(snap_options=["snap_nogeom"])), ("c", dict(thr=0.5)), ("w0", dict(thr=0.5)), ("c", dict(wire="head")), ("c", dict(tag="tw")), ("w0", dict(wire="twin")), ("w0", dict(wire_run="xs0")), ("w0", dict(wire_run="hs9")),
                 ("w1", dict(wire_run="hs0")), ("w2", dict(wire_run="hs0:model_h1.pt")), ("w0", dict(tag="tw1")), ("w1", dict(tag="tw")), ("x", dict(wire="twin")), ("x", dict(wire_run="hs0")), ("x", dict(tag="tw"))]
    for name, over in bad_cards:
        for s in SEEDS:
            t, _, (_, reason, _) = run_case({s: dict(good, x=True, over={name: over}), 1 - s: dict(good, x=True)}, tag_x="tx")
            assert t == "INVALID" and f"seed {s}" in reason, (name, over, s, t, reason)
    assert tok({0: dict(good, x=True), 1: dict(good, x=True)}, tag_x="tx") == "KEEP+"                    # the same cards, nothing planted
    assert tok({0: dict(good, over={"w0": dict(run="other")}), 1: good}) == "INVALID" and tok({0: dict(good, over={"x": dict(run="other")}, x=True), 1: good}) == "KEEP+"   # arm x is not read without its tag
    # wire_agreement.json: the pair run and the seed are required and must be the control's; E312's legacy file (no pair_run) is valid for the control e312_p_s<S> only
    for s in SEEDS:
        for wa in (agree(pair_run=f"cs{1 - s}"), agree(seed=1 - s), agree(seed=NONE), agree(pair_run=NONE)):
            assert tok({s: dict(good, wa=wa), 1 - s: good}) == "INVALID", (s, wa["pair_run"], wa["seed"])
        assert "no pair_run" in why({s: dict(good, wa=agree(pair_run=NONE)), 1 - s: good})
        swapped = agree()
        swapped["heads"] = swapped["heads"][::-1]                                                          # the first head must be head 0, not the ablation
        assert tok({s: dict(good, wa=swapped), 1 - s: good}) == "INVALID"
    legacy = dict(good, wa=agree(pair_run=NONE))
    assert tok({0: legacy, 1: legacy}, cp="e312_p_s") == "KEEP+" and tok({0: good, 1: good}, cp="e312_p_s") == "KEEP+"
    # the bootstrap log: the header names w against c of this control, the LAST block counts, the point estimate is w - c of the cards (the printed rounding is fine, 2e-4 off is not)
    for s in SEEDS:
        def log(point=0.002, a="ends_card_tw_snap_thr0.1", b="ends_card_snap_thr0.1", run_a=None, run_b=None):
            return boot_text(f"cs{s}", point, 0.001, 0.004, f"{run_a or f'cs{s}'} {a}", f"{run_b or f'cs{s}'} {b}")
        ok_log, bad_log = log(), log(a="ends_card_other_snap_thr0.1")
        for text in (log(a="ends_card_snap_thr0.1"), log(b="ends_card_tw_snap_thr0.1"), bad_log, log(run_a="e312_p_s0"), log(run_b="e312_p_s0"), log(point=0.0022), log(point=-0.0021), ok_log.split("\n  topo_F1")[0] + "\n",
                     "no header here\n  topo_F1      A - B: point +0.0020, bootstrap mean +0.0020, 95 % [+0.0010, +0.0040]\n", ok_log + bad_log):   # the last block is the bad one
            assert tok({s: dict(good, boot=text), 1 - s: good}) == "INVALID", (s, text)
        assert tok({s: dict(good, boot=bad_log + ok_log), 1 - s: good}) == "KEEP+", "the last block is the good one"
        assert tok({s: dict(good, d=dict(topo_F1=0.00204)), 1 - s: good}) == "KEEP+", "the log prints +0.0020 for w - c = 0.00204"
    # arm x is reported when its tag is asked for and never changes the token
    t, lines, _ = run_case({0: dict(good, x=True), 1: dict(good, x=True)}, tag_x="tx")
    assert t == "KEEP+" and sum("arm x (tx" in l for l in lines) == 2 and not any("no scored card" in l for l in lines if "arm x" in l), lines
    # E358, the PRIMARY read-out (attached ends, snap wiring without the snap geometry): the same rule on the cards `ends_card_[<tag>_]snap_nogeom_attach_thr0.1.json`; the head must have been trained on attached ends
    assert stem("snap") == "snap_" and f"ends_card_{stem('primary')}thr0.1.json" == "ends_card_snap_nogeom_attach_thr0.1.json" and f"ends_card_e358w1_{stem('primary')}thr0.1.json" == "ends_card_e358w1_snap_nogeom_attach_thr0.1.json"   # the file names of E324 s (`jwire_check` pins them to `qends_eval.card_name`)
    gp = dict(good, wa=agree(ends="att", near=(0.45, 0.44), nj=True))
    ptok = lambda spec, **k: tok(spec, ro="primary", **k)  # noqa: E731
    pwhy = lambda spec, **k: why(spec, ro="primary", **k)  # noqa: E731
    pseeds = lambda bad: seeds(bad, gp)  # noqa: E731
    assert ptok({0: gp, 1: gp}) == "KEEP+"
    for sp in pseeds(dict(gp, iv=(-0.001, 0.004))):
        assert ptok(sp) == "KEEP=", sp
    for sp in pseeds(dict(gp, d=dict(topo_F1=-0.003), iv=(-0.006, -0.001))):
        assert ptok(sp) == "KEEP-", sp
    for sp in pseeds(dict(gp, d=dict(topo_F1=0.002, dup_frac=0.011))) + pseeds(dict(gp, d=dict(topo_F1=-0.0051))):
        assert ptok(sp) == "TIE", sp
    for sp in pseeds(dict(gp, wa=agree(360, ends="att"))) + pseeds(dict(gp, wa=agree(float("nan"), ends="att"))):
        assert ptok(sp) == "CUT", sp
    assert ptok({0: gp}) == "INCOMPLETE" and ptok({0: gp, 1: dict(gp, d=None)}) == "INCOMPLETE" and ptok({0: gp, 1: dict(gp, iv=None)}) == "INCOMPLETE" and ptok({0: gp, 1: dict(gp, wa=None)}) == "INCOMPLETE"
    with tempfile.TemporaryDirectory() as t:                                                           # the primary read-out reads the primary names and not the snapped ones (and the other way round)
        world(t, {0: good, 1: good}, "cs", "snap")
        a = NS(root=t, card="E000", control_prefix="cs", head_prefix="hs", tag="tw", tag_x=None, x_head_prefix="xs", boot="temp/e000/boot_w_vs_c_s{seed}.log", read_out="primary")
        assert read_seed(a, 0)["c"] is None and read_seed(a, 0)["w"] == [None] * 3 and read_seed(NS(**dict(vars(a), read_out="snap")), 0)["c"] is not None
    # INVALID: a snapped card under the primary name (attach false, no snap_nogeom, snap false), a head of the other kind (kp: key missing or 'kp'; and an att head under the snapped read-out), the bootstrap log of the snapped cards
    for name, over in (("c", dict(attach=False)), ("w0", dict(attach=False)), ("w2", dict(attach=False)), ("c", dict(snap_options=[])), ("w1", dict(snap_options=[])), ("w0", dict(snap_options=None)), ("c", dict(snap=False)), ("w0", dict(thr=0.5)),
                       ("c", dict(run="other")), ("w1", dict(seed="@other")), ("c", dict(wire="head")), ("w0", dict(wire_run="hs9"))):
        for s in SEEDS:
            t, _, (_, reason, _) = run_case({s: dict(gp, over={name: over}), 1 - s: gp}, ro="primary")
            assert t == "INVALID" and f"seed {s}" in reason, (name, over, s, t, reason)
    for s in SEEDS:
        for wa in (agree(), agree(ends="kp")):
            assert ptok({s: dict(gp, wa=wa), 1 - s: gp}) == "INVALID" and "needs a head trained on 'att' ends" in pwhy({s: dict(gp, wa=wa), 1 - s: gp}), (s, wa.get("ends"))
        assert tok({s: dict(good, wa=agree(ends="att")), 1 - s: good}) == "INVALID" and "needs a head trained on 'kp' ends" in why({s: dict(good, wa=agree(ends="att")), 1 - s: good}), "an att head under the snapped read-out"
        assert tok({s: dict(good, wa=agree(ends="kp")), 1 - s: good}) == "KEEP+", "a head that names its kind kp is E325's"
        ok_p = boot_text(f"cs{s}", 0.002, 0.001, 0.004, f"cs{s} ends_card_tw_snap_nogeom_attach_thr0.1", f"cs{s} ends_card_snap_nogeom_attach_thr0.1")
        assert ptok({s: dict(gp, boot=ok_p), 1 - s: gp}) == "KEEP+" and ptok({s: dict(gp, boot=boot_text(f"cs{s}", 0.002, 0.001, 0.004)), 1 - s: gp}) == "INVALID", "the bootstrap log must name the primary cards"
    # the pilot's second statistic: the share of labelled ends within 0.5 m differing by more than 0.10 labels the result shift-dominated (a note, the token stays); a file without it (E325's) says nothing
    for near, flagged in (((0.45, 0.56), True), ((0.56, 0.45), True), ((0.45, 0.54), False), (None, False)):
        t, _, (_, _, notes) = run_case({0: gp, 1: dict(gp, wa=agree(ends="att", near=near))}, ro="primary")
        assert t == "KEEP+" and any("SHIFT-DOMINATED (TRAIN share" in n for n in notes) == flagged, (near, notes)
    # the report names the read-out, the head's kind, the share within 0.5 m and the no-jitter head
    t, lines, _ = run_case({0: gp, 1: gp}, ro="primary")
    text = "\n".join(lines)
    assert t == "KEEP+" and "read-out primary" in text and "head 0 primary:" in text and "ends att" in text and "no-jitter head (wn, reported only)" in text and "TRAIN 0.450, VAL 0.440" in text and "snapped:" not in text, text
    t, lines, _ = run_case({0: good, 1: good})
    text = "\n".join(lines)
    assert "read-out snap" in text and "head 0 snapped:" in text and "no-jitter head" not in text and "primary" not in text, text
    print("jwire_verdict selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="verdict", choices=("verdict", "selftest"))
    ap.add_argument("--card", default="E325", help="the card number of the heading and of the default bootstrap logs")
    ap.add_argument("--control-prefix", default="e312_p_s", help="the pair run of the control: runs/<prefix><seed> holds the control card and the head cards")
    ap.add_argument("--head-prefix", default="e325_w_s", help="the training runs of the head: runs/<prefix><seed>/wire_agreement.json")
    ap.add_argument("--tag", default="e325w", help="the tag of the head cards (head seed 0 = the arm; head seeds 1 and 2 = <tag>1, <tag>2)")
    ap.add_argument("--tag-x", default=None, help="the tag of the cards of arm x (the transfer control), reported only")
    ap.add_argument("--x-head-prefix", default="e325_w_s", help="the head of arm x: its cards must say --wire-run <prefix><seed>")
    ap.add_argument("--read-out", default="snap", choices=tuple(READ_OUT_ENDS), help="snap (default, E325 / E350): the snapped cards; primary (E358): the end-to-end read-out of E324 s, the cards `..._snap_nogeom_attach_thr0.1.json`")
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
