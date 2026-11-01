"""Self-test on synthetic runs with known answers (stdlib; curves are checked when matplotlib is installed).
Run: python -m trainsuite selftest    (from autoresearch/, a few seconds, writes only to a temporary folder)"""
import ast
import codecs
import contextlib
import io
import json
import math
import os
import random
import tempfile
import time

from . import cli, control, health
from .cli import main as cli_main
from .kill import Rule, load_rules
from .runs import Run, read_log

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(HERE, "..", "..", "bev_fusion", "training", "train.py")       # the trainer that inlines the stop check


def rows(n, every=20, seed=0, f=None, t0=None, sec=1.0, mean=None):
    """n logged rows of a healthy run: loss falls from 16 to 10 with noise 1.2 (or sits at `mean`); group.lines falls from 1.6 to 1.0"""
    rng = random.Random(seed)
    out = []
    for i in range(1, n + 1):
        s = i * every
        base = (10 + 6 * math.exp(-s / 800)) if mean is None else mean
        r = {"step": s, "lr": 3e-4, "grad_norm": 12 + rng.gauss(0, 2), "s_per_step": sec, "total": base + rng.gauss(0, 1.2),
             "group.lines": 1.0 + 0.6 * math.exp(-s / 500) + rng.gauss(0, 0.05), "gpu_gb": 7.1}
        if t0 is not None:
            r["t"] = t0 + s * sec
        out.append(f(r) if f else r)
    return out


def make(tmp, name, rs, total=3000, text="", done=False, age=0.0, evals=(), partial="", honours=None):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"steps": total}, f)
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
        for e in evals:
            f.write(json.dumps({"step": e[0], "val": e[1] if isinstance(e[1], dict) else {"total": e[1]}}) + "\n")
        f.write(partial)
    with open(os.path.join(d, "train.log"), "w") as f:
        f.write(text or "".join(f"step {r['step']}: total {r['total']:.4f}\n" for r in rs[-5:]))
    if done:
        open(os.path.join(d, ".done"), "w").close()
    if honours is not None:                                     # a code snapshot with / without the stop check
        os.makedirs(os.path.join(d, "code"), exist_ok=True)
        with open(os.path.join(d, "code", "train.py"), "w") as f:
            f.write("def _stop_requested(run_dir):\n    return None\n" if honours else "def train():\n    pass\n")
    if age:
        for fn in ("log.jsonl", "train.log"):
            t = time.time() - age
            os.utime(os.path.join(d, fn), (t, t))
    return d


def codes(run, level=None, ref=None, since=None):
    return {f.code for f in health.check(run, ref, since) if level is None or f.level == level}


def run_cli(*args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main(list(args))
    return rc, buf.getvalue()


class _Stop(Exception):
    pass


def run_cli_err(*args):
    """(rc, stdout, stderr) of a trainsuite call"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli_main(list(args))
    return rc, out.getvalue(), err.getvalue()


def early_pairs(tmp, name, early, final, total=1000, steps=(100, 250, 500, 750, 1000), skip=()):
    """a root with one pair p<i>_v / p<i>_c per (early, final) gap: the control sits at 10 - step/200, the variant `val.score` above it by `early` at
    every evaluation before the last and by `final` at the last; `val.loss` = the same series negated plus offset, so lower is better. `skip` = indices
    of pairs whose variant is not finished"""
    root = os.path.join(tmp, name)
    for i, (e, fin) in enumerate(zip(early, final)):
        for tag, g in (("c", 0.0), ("v", None)):
            ev = []
            for s in steps:
                gap = g if g is not None else (fin if s == steps[-1] else e)
                ev.append((s, {"score": 10 - s / 200 + gap, "loss": 20 - (10 - s / 200 + gap)}))
            make(root, f"p{i:02d}_{tag}", rows(5), total=total, done=not (tag == "v" and i in skip), evals=ev)
    return root


def early_set(tmp, name, off_early, off_final, steps=(100, 250, 500, 750, 1000)):
    """runs g_0 .. g_n-1 of one prefix, each `val.score` = control curve + its offset (early offset before the last evaluation, final offset at it): N runs make
    N(N-1)/2 pairs that share runs"""
    root = os.path.join(tmp, name)
    for i, (e, fin) in enumerate(zip(off_early, off_final)):
        ev = [(s, {"score": 10 - s / 200 + (fin if s == steps[-1] else e)}) for s in steps]
        make(root, f"g_{i}", rows(5), total=1000, done=True, evals=ev)
    return root


def early_checks(tmp, ok):
    """trainsuite early: calibration over finished pairs and the advice for one run, on synthetic pairs with known answers"""
    from . import early as E
    # statistics with known answers: average ranks for ties, Spearman 1 / 0.8 / constant side None, Wilson bound of 8 of 8
    assert E.ranks([5, 5, 7]) == [1.5, 1.5, 3.0] and E.spearman([1, 2, 3], [10, 20, 30]) == 1.0 and abs(E.spearman([1, 2, 3, 4], [1, 3, 2, 4]) - 0.8) < 1e-9
    assert E.spearman([1, 2, 3], [4, 4, 4]) is None and abs(E.wilson_low(8, 8) - 0.6756) < 1e-3 and E.wilson_low(0, 0) == 0.0
    H, L = dict(better="higher"), dict(better="lower")

    fin = [0.1 * k * (-1) ** k for k in range(1, 9)]                         # 8 final gaps of both signs and different sizes
    # the early gap predicts the final one: sign agreement 8 of 8, rank correlation 1; a finished pair with another step total, an unfinished one
    # and two runs of another name prefix are no pairs
    root = early_pairs(tmp, "early_good", [0.5 * x for x in fin] + [0.3], fin + [0.3], skip=(8,))
    make(root, "q_v", rows(5), total=500, done=True, evals=[(100, {"score": 1.0})])
    make(root, "q_c", rows(5), total=1000, done=True, evals=[(100, {"score": 1.0})])
    pairs = E.find_pairs(root)
    assert len(pairs) == 8 and all(a.name.split("_")[0] == b.name.split("_")[0] for a, b in pairs), [(a.name, b.name) for a, b in pairs]
    assert len(E.find_pairs(root, group="all")) == 153 and len(E.find_pairs(root, explicit=[("p00_v", "p00_c"), ("p08_v", "p08_c")])) == 1
    cal = E.calibrate(pairs, ["val.score"], band=0.02, **H)
    kd = cal["keys"]["val.score"]["fractions"]
    assert cal["n_pairs"] == 8 and cal["n_runs"] == 16
    e50 = kd["0.5"]
    assert e50["calibrated"] and e50["agree"] == 8 and e50["n_sign"] == 8 and abs(e50["spearman"] - 1.0) < 1e-9 and e50["agree_band"] == 8 and e50["n_band"] == 8, e50
    assert e50["n_runs_band"] == 16 and abs(e50["frac_lo"] - 0.5) < 1e-9 and abs(kd["0.1"]["frac_lo"] - 0.1) < 1e-9
    assert kd["0.75"]["agree"] == 8
    # a band of 0.22 leaves only the 4 largest early gaps: fewer than 6 pairs beyond it
    cal_wide = E.calibrate(pairs, ["val.score"], band=0.22, **H)
    assert cal_wide["keys"]["val.score"]["fractions"]["0.5"]["n_band"] == 4 and cal_wide["keys"]["val.score"]["fractions"]["0.5"]["agree_band"] is None
    ok("early: Spearman / Wilson known answers; calibration over 8 pairs whose early gap predicts the final one: agreement 8 of 8, Spearman 1; unfinished, other-total and other-prefix runs are no pairs")

    # the early gap points the wrong way: agreement 0 of 8 (< 0.5), Spearman -1; 5 pairs or equal curves: no number
    root_bad = early_pairs(tmp, "early_bad", [-0.5 * x for x in fin], fin)
    cal_bad = E.calibrate(E.find_pairs(root_bad), ["val.score"], band=0.02, **H)
    b50 = cal_bad["keys"]["val.score"]["fractions"]["0.5"]
    assert b50["agree"] == 0 and b50["n_sign"] == 8 and abs(b50["spearman"] + 1.0) < 1e-9 and b50["agree_band"] == 0, b50
    assert "0 of 8" in E.describe(cal_bad)
    root_few = early_pairs(tmp, "early_few", [0.5 * x for x in fin[:5]], fin[:5])
    cal_few = E.calibrate(E.find_pairs(root_few), ["val.score"], band=0.02, **H)
    f50 = cal_few["keys"]["val.score"]["fractions"]["0.5"]
    assert not f50["calibrated"] and f50["agree"] is None and f50["spearman"] is None and f50["agree_band"] is None and f50["n_sign"] == 5, f50
    assert "not enough pairs to calibrate" in E.describe(cal_few) and "no threshold" in E.describe(cal_few)
    root_eq = early_pairs(tmp, "early_eq", [0.0] * 8, [0.0] * 8)
    q50 = E.calibrate(E.find_pairs(root_eq), ["val.score"], band=0.02, **H)["keys"]["val.score"]["fractions"]["0.5"]
    assert q50["n"] == 8 and q50["n_tied"] == 8 and q50["n_sign"] == 0 and not q50["calibrated"] and q50["spearman"] is None, q50
    ok("early: a misleading early gap gives agreement 0 of 8 and Spearman -1; 5 pairs or equal curves give 'not enough pairs to calibrate' and no number")

    # the rank correlation must not depend on which run of a pair is called the variant: three of the pairs swapped, same Spearman (unoriented, it differs)
    mag = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.5]
    fin_o = [0.1 * k * (-1) ** k for k in range(1, 9)]
    early_o = [0.5 * m * (1 if f > 0 else -1) for m, f in zip(mag, fin_o)]          # signs agree with the final gaps, magnitudes shuffled
    root_o = early_pairs(tmp, "early_orient", early_o, fin_o)
    want = E.spearman([abs(x) for x in early_o], [abs(x) for x in fin_o])
    flip = [-1 if i in (1, 2, 5) else 1 for i in range(8)]
    assert abs(want - 1.0) > 0.05 and abs(E.spearman([x * s for x, s in zip(early_o, flip)], [x * s for x, s in zip(fin_o, flip)]) - want) > 0.05, "test data must be sensitive to the orientation"
    sp = lambda prs: E.calibrate(prs, ["val.score"], band=0.02, **H)["keys"]["val.score"]["fractions"]["0.5"]["spearman"]
    swapped = E.find_pairs(root_o, explicit=[(f"p{i:02d}_c", f"p{i:02d}_v") if i in (1, 2, 5) else (f"p{i:02d}_v", f"p{i:02d}_c") for i in range(8)])
    assert abs(sp(E.find_pairs(root_o)) - want) < 1e-9 and abs(sp(swapped) - want) < 1e-9, (sp(E.find_pairs(root_o)), sp(swapped), want)
    ok("early: Spearman is taken on pairs oriented by their final gap: swapping variant and control of three pairs gives the same value")

    # direction and band are never guessed: refused without, and a calibration without them is not used
    v0, c0 = Run(os.path.join(root, "p00_v")), Run(os.path.join(root, "p00_c"))
    for bad in (lambda: E.calibrate(pairs, ["val.score"], band=0.02), lambda: E.calibrate(pairs, ["val.loss"], band=0.02, better="up"),
                lambda: E.advise(v0, c0, "val.score", 0.5, None, 0.02)):
        try:
            bad()
            raise AssertionError("a missing direction was accepted")
        except ValueError as e:
            assert "direction" in str(e), e
    assert E.calibrate(pairs, ["val.loss"], lower=["val.loss"], band=0.02)["keys"]["val.loss"]["better"] == "lower"
    lo = E.calibrate(E.find_pairs(root), ["val.loss"], band=0.02, **L)["keys"]["val.loss"]
    assert lo["better"] == "lower" and lo["fractions"]["0.5"]["agree"] == 8 and abs(lo["fractions"]["0.5"]["spearman"] - 1.0) < 1e-9
    mixed = E.calibrate(E.find_pairs(root), ["val.loss", "val.score"], lower=["val.loss"], band=0.02, **H)["keys"]
    assert mixed["val.loss"]["better"] == "lower" and mixed["val.score"]["better"] == "higher"
    rc, out, err = run_cli_err("--root", root, "early", "--calibrate", "--key", "val.score", "--band", "0.02")
    assert rc == 2 and "direction" in err and not out, (rc, out, err)
    rc, out, err = run_cli_err("--root", root, "early", "--calibrate", "--key", "val.score", "--better", "higher")
    assert rc == 2 and "--band is required" in err, (rc, err)
    nodir = json.loads(json.dumps(cal))
    del nodir["keys"]["val.score"]["better"]
    try:
        E.advise(v0, c0, "val.score", 0.5, "higher", 0.02, nodir)
        raise AssertionError("a calibration without a direction was used")
    except ValueError as e:
        assert "does not state the direction" in str(e), e
    ok("early: no default direction and no default band (ValueError / CLI exit 2); --lower names a loss key; a calibration without its direction is refused")

    # the band of the calibration is the band of the advice: a different --band is refused, none given takes the calibration's, none at all is refused
    cal_json = json.loads(json.dumps(cal))
    assert E.resolve_band(None, cal_json) == 0.02 and E.resolve_band(0.02, cal_json) == 0.02 and E.resolve_band(0.5, None) == 0.5
    for b_, c_ in ((0.5, cal_json), (None, None)):
        try:
            E.resolve_band(b_, c_)
            raise AssertionError("band accepted")
        except ValueError as e:
            assert "band" in str(e)
    cp = os.path.join(tmp, "calib.json")
    json.dump(cal, open(cp, "w"))
    root_one = early_pairs(tmp, "early_one", [-0.5, 0.01, 0.5], [-1.0, 0.01, 1.0])         # pair 0: variant 0.5 below the control, pair 1 inside the band, pair 2: 0.5 above
    common = ("--root", root_one, "early", "p00_v", "--vs", "p00_c", "--key", "val.score", "--better", "higher", "--at", "0.5", "--calib", cp)
    rc, out, err = run_cli_err(*common, "--band", "0.5")
    assert rc == 2 and "differs from the band" in err and not out, (rc, out, err)
    rc, out, err = run_cli_err(*common)
    assert rc == 0 and "band 0.02" in out, (rc, out)
    rc, out, err = run_cli_err("--root", root_one, "early", "p00_v", "--vs", "p00_c", "--key", "val.score", "--better", "higher")
    assert rc == 2 and "no band" in err, (rc, err)
    ok("early: advice uses the band of the calibration; a different --band or none without a calibration is refused")

    # advice for one run
    bad_run, mid_run, good_run = (Run(os.path.join(root_one, f"p0{i}_v")) for i in range(3))
    ctl = lambda i: Run(os.path.join(root_one, f"p0{i}_c"))
    big = [0.1 * k * (-1) ** k for k in range(1, 21)]
    root_big = early_pairs(tmp, "early_big", [0.5 * x for x in big], big)                   # 20 pairs from 40 runs, the early gap predicts the final one every time
    cal_big = json.loads(json.dumps(E.calibrate(E.find_pairs(root_big), ["val.score"], band=0.02, **H)))
    text, info = E.advise(bad_run, ctl(0), "val.score", 0.5, "higher", None, cal_big)
    assert info["verdict"] == "consider_stop" and "behind control by 0.5" in text and "20 of 20 pairs from 40 runs" in text, text
    assert text.endswith("not the decision metric of the card]") and "consider stopping" in text, text
    # the boundary: the printed 95 % lower bound decides, not the point estimate. 19 of 20 agree (0.95) but the bound is 0.76 < 0.8
    early_19 = [0.5 * x for x in big[:19]] + [-0.5 * big[19]]
    cal_19 = json.loads(json.dumps(E.calibrate(E.find_pairs(early_pairs(tmp, "early_19", early_19, big)), ["val.score"], band=0.02, **H)))
    text, info = E.advise(bad_run, ctl(0), "val.score", 0.5, "higher", None, cal_19)
    assert info["verdict"] == "weak" and "19 of 20" in text and "no reliable signal" in text and "consider stopping" not in text, text
    assert E.advise(bad_run, ctl(0), "val.score", 0.5, "higher", None, cal_19, min_agree=0.7)[1]["verdict"] == "consider_stop", "the bound 0.76 clears 0.7"
    # 5 of 6 (0.83 >= 0.8 as a point estimate, bound 0.44): too few pairs for any 'consider stopping'
    early_56 = [0.5 * x for x in fin[:5]] + [-0.5 * fin[5]]
    cal_56 = json.loads(json.dumps(E.calibrate(E.find_pairs(early_pairs(tmp, "early_56", early_56, fin[:6])), ["val.score"], band=0.02, **H)))
    e56 = cal_56["keys"]["val.score"]["fractions"]["0.5"]
    assert e56["agree_band"] == 5 and e56["n_band"] == 6, e56
    text, info = E.advise(bad_run, ctl(0), "val.score", 0.5, "higher", None, cal_56, min_agree=0.4)
    assert info["verdict"] == "dependent" and "not enough independent evidence" in text and "consider stopping" not in text, text
    # 15 pairs that all agree, but from 6 runs only (6 runs make 15 pairs that share runs): not independent either, however good the bound
    root_set = early_set(tmp, "early_set", [0.05 * i for i in range(6)], [0.1 * i for i in range(6)])
    cal_set = json.loads(json.dumps(E.calibrate(E.find_pairs(root_set), ["val.score"], band=0.02, **H)))
    e_set = cal_set["keys"]["val.score"]["fractions"]["0.5"]
    assert e_set["agree_band"] == 15 and e_set["n_band"] == 15 and e_set["n_runs_band"] == 6, e_set
    g0, g5 = Run(os.path.join(root_set, "g_0")), Run(os.path.join(root_set, "g_5"))
    text, info = E.advise(g0, g5, "val.score", 0.5, "higher", None, cal_set, min_agree=0.7)
    assert info["verdict"] == "dependent" and "from 6 runs" in text, text
    ok("early: 'consider stopping' needs the 95 % lower bound >= 0.8 (19 of 20 is not enough, 20 of 20 is) from >= 15 pairs and >= 8 runs (5 of 6 and 15 pairs of 6 runs are not)")

    text, info = E.advise(bad_run, ctl(0), "val.score", 0.5, "higher", 0.02, cal_json)       # the 8-pair calibration: 8 of 8 but only 8 pairs
    assert info["verdict"] == "dependent", text
    text, info = E.advise(bad_run, ctl(0), "val.loss", 0.5, "lower", 0.02, cal_json)          # the variant's val.loss is HIGHER (worse); this key is not in the calibration
    assert info["verdict"] == "uncalibrated" and text.startswith("not calibrated: gap only"), text
    assert E.advise(bad_run, ctl(0), "val.score", 0.5, "lower", 0.02, cal_json)[1]["verdict"] == "ahead", "the direction decides who is behind"
    assert E.advise(bad_run, ctl(0), "val.score", 0.5, "higher", 0.02, None)[1]["verdict"] == "uncalibrated"
    text, info = E.advise(mid_run, ctl(1), "val.score", 0.5, "higher", 0.02, cal_json)
    assert info["verdict"] == "inside" and text.startswith("inside the band: no signal"), text
    assert E.advise(good_run, ctl(2), "val.score", 0.5, "higher", 0.02, cal_json)[1]["verdict"] == "ahead"
    assert E.advise(good_run, ctl(2), "val.score", 0.5, "lower", 0.02, cal_json)[1]["verdict"] == "uncalibrated", "a calibration made for another direction is not used"
    assert E.advise(bad_run, ctl(0), "val.score", 0.1, "higher", 0.02, cal_json, min_rows=3)[1]["verdict"] == "none"          # one evaluation up to 10 %: too early
    assert E.advise(bad_run, ctl(0), "val.score", 0.5, "higher", 0.02, json.loads(json.dumps(cal_few)))[1]["verdict"] == "uncalibrated"
    ok("early: advice for one run: inside the band, ahead, direction, no calibration, calibration too small, too few rows")

    # the kill-rule line: only for 'consider stopping'; a rule the suite reads: it hits the lagging variant and not its control (both directions)
    info_stop = E.advise(bad_run, ctl(0), "val.score", 0.5, "higher", None, cal_big)[1]
    line = E.as_rule(ctl(0), info_stop)
    assert line is not None and Rule(line).evaluate(bad_run)[0] == "hit" and Rule(line).evaluate(ctl(0))[0] == "ok", line
    cal_big_lo = json.loads(json.dumps(E.calibrate(E.find_pairs(root_big), ["val.loss"], band=0.02, **L)))
    info_lo = E.advise(bad_run, ctl(0), "val.loss", 0.5, "lower", None, cal_big_lo)[1]
    line_lo = E.as_rule(ctl(0), info_lo)
    assert info_lo["verdict"] == "consider_stop" and ">" in line_lo and Rule(line_lo).evaluate(bad_run)[0] == "hit" and Rule(line_lo).evaluate(ctl(0))[0] == "ok", line_lo
    for cal_x in (cal_19, cal_56, cal_set, None):                                   # weak / dependent / uncalibrated: no rule although the run lags
        info_x = E.advise(bad_run, ctl(0), "val.score", 0.5, "higher", None if cal_x else 0.02, cal_x)[1]
        assert info_x["advantage"] < -0.02 and E.as_rule(ctl(0), info_x) is None, info_x
    cb = os.path.join(tmp, "calib_big.json")
    c19 = os.path.join(tmp, "calib_19.json")
    for path_, root_, c_ in ((cb, root_big, None), (c19, os.path.join(tmp, "early_19"), None)):
        rc, out, err = run_cli_err("--root", root_, "early", "--calibrate", "--key", "val.score", "--better", "higher", "--band", "0.02", "--out", path_)
        assert rc == 0 and json.load(open(path_))["n_pairs"] == 20, (out, err)
    args = ("--root", root_one, "early", "p00_v", "--vs", "p00_c", "--key", "val.score", "--better", "higher", "--at", "0.5", "--as-rule")
    rc, out, err = run_cli_err(*args, "--calib", cb)
    assert rc == 0 and "consider stopping" in out and "val.score@500 <" in out, out
    rc, out, err = run_cli_err(*args, "--calib", c19)
    assert rc == 0 and "no reliable signal" in out and "# no rule" in out and "val.score@500" not in out, out
    assert not os.path.exists(os.path.join(root_one, "p00_v", "kill.txt")), "the rule is printed, never written"
    assert run_cli_err("--root", root_one, "early", "p00_v")[0] == 2 and run_cli_err("--root", root_one, "early", "--calibrate", "--pair", "x")[0] == 2
    ok("early: --as-rule prints a kill.txt line only for 'consider stopping' (it hits the laggard, not the control, both directions); the CLI exits 0, usage errors 2, nothing is written")


def main():
    n = [0]

    def ok(msg):
        n[0] += 1
        print(f"  ok  {msg}")

    with tempfile.TemporaryDirectory() as tmp:
        now = time.time()
        # 1 healthy run
        d = make(tmp, "good", rows(100, t0=now - 2000, sec=1.0))
        r = Run(d)
        assert r.state() == "running" and r.step == 2000 and r.total == 3000
        assert abs(r.s_per_step() - 1.0) < 1e-6 and abs(r.eta_s() - 1000) < 1e-6, (r.s_per_step(), r.eta_s())
        bad = {c for c in codes(r) if c not in ("flat", "vs_ref")}
        assert not bad, bad
        ok("healthy run: state, progress, ETA from the row times, no warnings")

        # 2 non-finite value, spike, rising loss, level shift, grad norm, GPU
        def nan_last(rs):
            rs[-1]["total"] = float("nan")
            return rs
        assert "nonfinite" in codes(Run(make(tmp, "nan", nan_last(rows(100)))), "alert")
        ok("NaN in a logged row is an alert")
        rs = rows(100)
        rs[-1]["total"] += 40
        assert "spike" in codes(Run(make(tmp, "spike", rs)), "warn")
        rs = rows(100)
        rs[-8]["total"] += 40                                                  # a spike 8 rows back: the default look (5 rows) misses it
        d = make(tmp, "spike8", rs)
        assert "spike" not in codes(Run(d)) and "spike" in codes(Run(d), "warn", since=rs[-9]["step"])
        ok("a single value far above the recent median is a spike; watch finds one between two looks (since_step)")
        rs = rows(100)
        for x in rs:
            if x["step"] > 400:
                x["total"] += 0.02 * (x["step"] - 400)
        assert "rising" in codes(Run(make(tmp, "rise", rs)), "warn")
        early = Run(make(tmp, "rise_early", rs, total=6000))                   # the same rise at 33 % of the steps
        assert "rising" in codes(early, "info") and "rising" not in codes(early, "warn")
        ok("a loss that climbs for many rows is a rising warning (only info before 50 % of the steps: loss ramps)")
        rs = rows(150, mean=10.0)
        for x in rs[-12:]:
            x["total"] += 10                                                   # a permanent jump (2x) in the last 12 rows
        assert "shift" in codes(Run(make(tmp, "shift", rs)), "warn")
        assert "shift" not in codes(Run(make(tmp, "noshift", rows(150, mean=10.0))))
        ok("a jump of the loss that stays is a level-shift warning, a flat noisy run is not")
        rs = rows(100)
        rs[-1]["grad_norm"] = 400
        assert "grad_norm" in codes(Run(make(tmp, "gn", rs)), "warn")
        rs = rows(100)
        rs[-1]["gpu_gb"] = 11.8
        assert "gpu" in codes(Run(make(tmp, "gpu", rs)), "warn")
        ok("grad-norm blow-up and GPU above 11 GB are warnings")

        # 3 speed: slowdown, resume gap, rows without t, finished runs
        rs = rows(100, t0=now - 3000)
        for k, x in enumerate(rs[-6:], 1):
            x["t"] += 40 * k                                                   # the last intervals take 3 s/step instead of 1
        assert "slow" in codes(Run(make(tmp, "slow", rs)), "warn")
        assert "slow" not in codes(Run(make(tmp, "slow_done", rs, done=True)))
        rs = rows(100, sec=1.0, t0=now - 5000)
        for x in rs[60:]:
            x["t"] += 1200                                                     # downtime between two attempts: 20 min
        r = Run(make(tmp, "gap", rs))
        assert abs(r.s_per_step() - 1.0) < 1e-6 and abs(r.eta_s() - 1000) < 1e-6, (r.s_per_step(), r.eta_s())
        assert "slow" not in codes(r)
        rs = rows(100)
        for i, x in enumerate(rs):
            x["s_per_step"] = 1.0 + 2.0 * i / 100                              # a rising cumulative average, no row times
        r = Run(make(tmp, "notime", rs))
        assert not r.timed() and "slow" not in codes(r)
        ok("slowdown is a warning (not for finished runs); a resume gap does not spoil speed and ETA; rows without time give no slowdown alarm")

        # 4 states: stalled, failed, ended, done, new; stale limit grows with the row interval
        assert Run(make(tmp, "stall", rows(50), age=3600)).state() == "stalled"
        assert "stalled" in codes(Run(os.path.join(tmp, "stall")), "alert")
        tb = "step 40: total 11.0\nTraceback (most recent call last):\n  File x\nRuntimeError: CUDA error: unknown error\n"
        r = Run(make(tmp, "fail", rows(2), text=tb))
        assert r.state() == "failed" and "CUDA error" in r.crash_reason()
        long_tb = "step 40: total 11.0\nTraceback (most recent call last):\n" + "".join(f"  File 'f{i}.py', line {i}\n    x()\n    ^^^\n" for i in range(40)) \
            + "RuntimeError: boom\n"
        assert Run(make(tmp, "fail_long", rows(2), text=long_tb)).crash_reason() == "RuntimeError: boom"
        assert Run(make(tmp, "tb_then_steps", rows(2), text=tb + "step 60: total 10.0\n")).state() != "failed"
        handled = "it 40 loss 11.0\nTraceback (most recent call last):\n  File x\nValueError: bad sample, skipped\nit 41 loss 10.9\n"   # S4: another log format
        assert Run(make(tmp, "tb_other_fmt", rows(2), text=handled)).state() == "failed", "without a layout file only 'step ' lines count"
        with open(os.path.join(tmp, "tb_other_fmt", "trainsuite.json"), "w") as f:
            json.dump({"progress": "it "}, f)
        r = Run(os.path.join(tmp, "tb_other_fmt"))
        assert r.crash_reason() == "" and r.state() != "failed", "trainsuite.json 'progress' names the trainer's progress lines"
        r = Run(make(tmp, "tb_cuda_multiline", rows(2), text=tb + "CUDA kernel errors might be asynchronously reported\nFor debugging consider CUDA_LAUNCH_BLOCKING=1.\n"))
        assert r.state() == "failed", "a crash message over several lines is still a crash"
        assert Run(make(tmp, "end", rows(150), total=3000, age=3600)).state() == "ended"
        assert Run(make(tmp, "fin", rows(150), done=True)).state() == "done"
        os.makedirs(os.path.join(tmp, "empty"))
        assert Run(os.path.join(tmp, "empty")).state() == "new"
        slowrows = Run(make(tmp, "rowslow", rows(50, t0=now - 50 * 400, sec=20.0, every=20), age=1500))     # a row every 400 s
        assert slowrows.stale_limit() >= 4000 and slowrows.state() == "running", (slowrows.stale_limit(), slowrows.state())
        ok("states: stalled, failed (a long traceback too; not when progress lines follow it, their prefix set in trainsuite.json), ended, done, new; the stall limit grows with the row interval")

        # 5 skipped steps, resume dedupe, eval rows
        txt = "".join(f"step {i}: non-finite loss, skipped ({k} in a row)\n" for i, k in ((10, 1), (11, 2), (12, 3)))
        r = Run(make(tmp, "skip", rows(50), text=txt))
        assert r.skipped_steps() == (3, 3) and "skipped" in codes(r, "alert")
        first = rows(10)
        again = [dict(x, total=-1.0) for x in rows(8, seed=5)[5:]]            # a restart from step 100: 120, 140, 160 again
        d = make(tmp, "resume", first + again, partial='{"step": 17')
        tr, ev, rs_ = read_log(os.path.join(d, "log.jsonl"))
        steps = [x["step"] for x in tr]
        assert steps == sorted(set(steps)) and rs_ >= 1, (steps, rs_)
        assert [x["total"] for x in tr if x["step"] in (120, 140, 160)] == [-1.0] * 3 and tr[-1]["step"] == 160
        rs = rows(100)
        r = Run(make(tmp, "val", rs, evals=[(1000, 12.0), (2000, 13.5)]))
        assert [e["step"] for e in r.evals] == [1000, 2000] and r.series("val.total")[-1] == (2000.0, 13.5) and "overfit" in codes(r, "warn")
        r = Run(make(tmp, "emptyval", rows(10), evals=[(200, {})]))            # an eval row with an empty val dict is NOT the train row of step 200
        assert [x["step"] for x in r.train][-1] == 200 and len(r.evals) == 1 and r.restarts == 0, (r.train[-1], r.evals, r.restarts)
        chrono = []                                             # an attempt up to step 1000 with evaluations at 500 and 1000, then a resume from the checkpoint at 500
        for x in rows(50):
            chrono.append(x)
            if x["step"] in (500, 1000):
                chrono.append({"step": x["step"], "val": {"total": 12.0}})
        d = make(tmp, "rollback", [])
        with open(os.path.join(d, "log.jsonl"), "w") as fh:
            for x in chrono + rows(30)[25:]:                    # the later attempt's rows: steps 520 ... 600
                fh.write(json.dumps(x) + "\n")
        r = Run(d)
        assert [e["step"] for e in r.evals] == [500] and r.train[-1]["step"] == 600 and r.restarts == 1, (r.evals, r.train[-1], r.restarts)
        assert Rule("val.total@1000 > 11").evaluate(r)[0] == "pending", "an evaluation of the superseded training suffix must not trigger a kill rule"
        ok("skipped steps counted from the text log; after a resume the later attempt wins, its evaluations too; eval rows kept apart (an empty one too); over-fitting warned")

        # 6 reference run
        ref = Run(make(tmp, "ref", rows(100)))
        worse = Run(make(tmp, "worse", rows(100, f=lambda r: dict(r, total=r["total"] * 1.25))))
        twin = Run(make(tmp, "twin", rows(100, seed=1)))
        assert "vs_ref" in codes(worse, "warn", ref) and "vs_ref" not in codes(Run(os.path.join(tmp, "good")), "warn", ref)
        assert "vs_ref" not in codes(twin, "warn", ref)
        longref = Run(make(tmp, "longref", rows(100), total=6000))                   # a reference with another step total is on another schedule: named, not compared
        vs_long = [f for f in health.check(worse, longref) if f.code == "vs_ref"]
        assert len(vs_long) == 1 and vs_long[0].level == "info" and "6000 steps" in vs_long[0].text and "not compared" in vs_long[0].text, vs_long
        ok("a loss 25 % above the reference at the same step is a warning; the equal one and a seed twin are not")

        # 7 kill rules
        r = Run(os.path.join(tmp, "good"))
        assert Rule("group.lines@1000 > 1.48").evaluate(r)[0] == "ok"
        assert Rule("group.lines@1000 > 0.9  # inline comment").evaluate(r)[0] == "hit"
        assert Rule("group.lines@2500 > 0.1").evaluate(r)[0] == "pending"
        assert Rule("group.line@1000 > 0.1").evaluate(r)[0] == "unknown"
        assert Rule("total@500 < 3 window 5").evaluate(r)[0] == "ok"
        assert Rule("val.total@2000>=13").evaluate(Run(os.path.join(tmp, "val")))[0] == "hit"
        mr = Rule("objhead.vel@+0.0@1000 > 1  # a key with @ and + (the bev_fusion velocity terms)")
        assert mr.key == "objhead.vel@+0.0" and mr.step == 1000 and Rule("val.motion@+0.0@2000 > 1").key == "val.motion@+0.0"
        assert cli.split_ref("run_a:run_b") == ("run_a", "run_b") and cli.split_ref("run_a") == ("run_a", None)
        win = "C:" + chr(92) + "runs" + chr(92) + "t10"
        assert cli.split_ref(win) == (win, None) and cli.split_ref(win + ":ref") == (win, "ref")
        assert Rule("total@2000 > 1").evaluate(Run(os.path.join(tmp, "nan")))[0] in ("hit", "ok")      # NaN rows: below
        try:
            Rule("lines is bad")
            raise AssertionError("bad rule accepted")
        except ValueError:
            pass
        nanrun = Run(make(tmp, "nan2", rows(50, f=lambda x: dict(x, total=float("nan")))))
        assert Rule("total@500 > 1e9").evaluate(nanrun)[0] == "hit"
        with open(os.path.join(tmp, "good", "kill.txt"), "w") as f:
            f.write("# card kill criterion\ngroup.lines@1000 > 0.9   # inline\nthis line is no rule\n")
        rules, problems = load_rules(r, ["total@100 > 1", "nonsense"])
        assert len(rules) == 2 and len(problems) == 2 and "line 3" in problems[0], (rules, problems)
        rc, out = run_cli("--root", tmp, "status", "good")                      # a bad rule does not kill the status
        assert "kill_rule" in out and "good" in out
        ok("kill rules: ok / hit / pending / unknown key / NaN = hit, inline comments, a bad line is reported and the rest still works")

        # 7b the encodings of kill.txt: UTF-8, UTF-8 with BOM, UTF-16 with BOM (PowerShell 5.1 `>`) all work; bytes that are no text are ONE problem, no exception,
        # and `status` of a root that holds such a file still lists every run
        rule_line = "group.lines@1000 > 0.9\r\n"
        for label, data in (("utf-8", rule_line.encode("utf-8")), ("utf-8 bom", codecs.BOM_UTF8 + rule_line.encode("utf-8")),
                            ("utf-16 le bom", codecs.BOM_UTF16_LE + rule_line.encode("utf-16-le")), ("utf-16 be bom", codecs.BOM_UTF16_BE + rule_line.encode("utf-16-be"))):
            kd = make(tmp, "enc_" + label.replace(" ", "_"), rows(50))
            with open(os.path.join(kd, "kill.txt"), "wb") as f:
                f.write(data)
            rs, pr = load_rules(Run(kd))
            assert len(rs) == 1 and not pr and rs[0].key == "group.lines", (label, rs, pr)
        fd_ = make(tmp, "enc_ff", rows(50))                      # a form feed or U+2028 in a comment is no line break: line numbers stay right
        with open(os.path.join(fd_, "kill.txt"), "wb") as f:
            f.write("# a\x0c b   c\ngroup.lines@1000 > 0.9\nnot a rule\n".encode("utf-8"))
        rs, pr = load_rules(Run(fd_))
        assert len(rs) == 1 and len(pr) == 1 and pr[0].startswith("kill.txt line 3:"), (rs, pr)
        bd = make(tmp, "enc_garbage", rows(50))
        with open(os.path.join(bd, "kill.txt"), "wb") as f:
            f.write(b"\xff\xfe\xfd group.lines@1000 > 0.9\n\x80\x81")
        rs, pr = load_rules(Run(bd))
        assert rs == [] and len(pr) == 1 and "not text" in pr[0], (rs, pr)
        rc, out = run_cli("--root", tmp, "status")
        assert "enc_garbage" in out and "enc_utf-16_le_bom" in out and "good" in out, out
        ok("kill.txt: UTF-8 / BOM / UTF-16 with BOM are read, bytes that are no text are one problem and status still lists all runs")

        # 8 control: stale files are moved aside, not compared by time; the copy in train.py agrees
        d = os.path.join(tmp, "good")
        assert control.stop_requested(d) is None
        control.request_stop(d, "test")
        assert control.stop_requested(d) == "test"
        assert control.clear_stale(d) == ["CONTROL"] and control.stop_requested(d) is None and os.path.exists(os.path.join(d, "CONTROL.old"))
        control.mark_stopped(d, 2000, "test")
        assert Run(d).state() == "stopped"
        os.utime(os.path.join(d, "log.jsonl"), (time.time() + 200, time.time() + 200))       # the run went on after the stop marker
        assert Run(d).state() != "stopped"
        # a restart: the old STOPPED is still there while the new trainer imports (10-40 s); output a few seconds newer keeps 'stopped', output beyond the grace
        # period is a running run again (watch dropped such a run as terminal)
        control.mark_stopped(d, 2000, "again")
        t_stop = os.path.getmtime(os.path.join(d, "STOPPED"))
        for name in ("log.jsonl", "train.log"):
            if os.path.exists(os.path.join(d, name)):
                os.utime(os.path.join(d, name), (t_stop + 5, t_stop + 5))
        assert Run(d).state() == "stopped"
        for name in ("log.jsonl", "train.log"):
            if os.path.exists(os.path.join(d, name)):
                os.utime(os.path.join(d, name), (t_stop + 20, t_stop + 20))
        assert Run(d).state() != "stopped", "output 20 s after the marker: the run went on"
        nd = make(tmp, "nanstep", rows(20))                                    # NaN / Infinity as a step is no row (and no crash of int(step) in status)
        with open(os.path.join(nd, "log.jsonl"), "a") as fh:
            fh.write('{"step": NaN, "total": 1.0}\n{"step": Infinity, "total": 1.0}\n{"step": 420, "total": 1.0, "huge": ' + "9" * 400 + "}\n")      # a 400-digit integer: no float
        rn = Run(nd)
        assert rn.step == 420 and len(rn.train) == 21 and "huge" not in rn.train[-1], (rn.step, len(rn.train))
        rc, out = run_cli("--root", tmp, "status")
        assert rc is not None and "nanstep" in out, out
        assert control.honours_stop(os.path.join(tmp, "good")) is None
        assert control.honours_stop(make(tmp, "snap_old", rows(5), honours=False)) is False
        assert control.honours_stop(make(tmp, "snap_new", rows(5), honours=True)) is True
        if os.path.exists(TRAIN_PY):                                           # the inlined copy in the trainer behaves like this module
            tree = ast.parse(open(TRAIN_PY, encoding="utf-8").read())
            ns = {"os": os, "json": json}
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in ("_stop_requested", "_clear_control"):
                    exec(compile(ast.Module([node], []), TRAIN_PY, "exec"), ns)
            assert "_stop_requested" in ns and "_clear_control" in ns, "train.py has no inlined stop check"
            for body in (None, '{"cmd": "stop", "reason": "x"}', '{"cmd": "go"}', "not json", "[1]", "{}"):
                cd = os.path.join(tmp, "cmp")
                os.makedirs(cd, exist_ok=True)
                p = os.path.join(cd, "CONTROL")
                if os.path.exists(p):
                    os.remove(p)
                if body is not None:
                    open(p, "w").write(body)
                assert ns["_stop_requested"](cd) == control.stop_requested(cd), body
            open(os.path.join(cd, "CONTROL"), "w").write('{"cmd": "stop"}')
            open(os.path.join(cd, "STOPPED"), "w").write("{}")
            ns["_clear_control"](cd)
            assert not os.path.exists(os.path.join(cd, "CONTROL")) and os.path.exists(os.path.join(cd, "STOPPED.old"))
        ok("stop protocol: CONTROL / STOPPED of an earlier stop are moved aside, the snapshot check, the copy in train.py agrees with control.py")

        # 9 watch: enforce on every run, old snapshots refused, an ignored stop is written again then given up
        for nm, h in (("enf1", True), ("enf2", True), ("enf_old", False)):
            make(tmp, nm, rows(100, t0=now - 2000), honours=h)
        rc, out = run_cli("--root", tmp, "watch", "enf1", "enf2", "enf_old", "--kill", "group.lines@1000 > 0.9", "--enforce", "--once")
        assert os.path.exists(os.path.join(tmp, "enf1", "CONTROL")) and os.path.exists(os.path.join(tmp, "enf2", "CONTROL")), out
        assert not os.path.exists(os.path.join(tmp, "enf_old", "CONTROL")) and "NOT stopped" in out, out
        make(tmp, "noenf", rows(100, t0=now - 2000), honours=True)
        run_cli("--root", tmp, "watch", "noenf", "--kill", "group.lines@1000 > 0.9", "--once")
        assert not os.path.exists(os.path.join(tmp, "noenf", "CONTROL"))
        rc, out = run_cli("--root", tmp, "stop", "enf_old")
        assert rc == 2 and "NOT requested" in out
        real_sleep, calls = cli.time.sleep, [0]

        def stop_after(_):
            calls[0] += 1
            if calls[0] >= 3:
                raise _Stop()
        cli.time.sleep = stop_after
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    cli_main(["--root", tmp, "watch", "noenf", "--kill", "group.lines@1000 > 0.9", "--enforce", "--honour-wait", "-1", "--every", "1"])
                except _Stop:
                    pass
        finally:
            cli.time.sleep = real_sleep
        assert "written again" in buf.getvalue() and "given up" in buf.getvalue(), buf.getvalue()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):                                   # a run that does not exist yet: wait, then give up
            cli_main(["--root", tmp, "watch", "not_yet", "--give-up", "-1", "--every", "0.01"])
        assert "new at step 0" in buf.getvalue() and "no longer watched" in buf.getvalue(), buf.getvalue()
        d = make(tmp, "was_stopped", rows(30), total=3000)                       # a stopped run is waited for, not abandoned at once
        control.mark_stopped(d, 600, "x")
        os.utime(os.path.join(d, "STOPPED"), (time.time() + 5, time.time() + 5))
        rc, out = run_cli("--root", tmp, "watch", "was_stopped", "--give-up", "-1", "--every", "0.01")
        assert "stopped at step" in out and "still stopped" in out, out
        ok("watch --enforce stops every run with a hit, refuses an old snapshot, rewrites an ignored stop once; waits for runs that have not started or were stopped")

        # 10 flapping findings are reported once (latch)
        seq = iter(range(100))
        orig = health.check

        def flapping(run, ref=None, since_step=None):
            i = next(seq)
            f = orig(run, ref, since_step)
            return f + ([health.Finding("warn", "flappy", f"no output for {i} min")] if i % 2 == 0 else [])
        cli.health.check = flapping
        cli.time.sleep = stop_after
        calls[0] = 0
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    cli_main(["--root", tmp, "watch", "noenf", "--every", "1", "--give-up", "100"])
                except _Stop:
                    pass
        finally:
            cli.health.check = orig
            cli.time.sleep = real_sleep
        assert buf.getvalue().count("flappy") == 1, buf.getvalue()
        ok("a finding that comes and goes is reported once")

        # 10b a run that fails while watched and is restarted by the launcher (CUDA retry) is still watched
        d = make(tmp, "retry", rows(100, t0=now - 2000))
        tb = "step 99: total 1.0\nTraceback (most recent call last):\n  File x\nRuntimeError: CUDA error: unknown error\n"
        calls[0] = 0

        def crash_then_resume(_):
            calls[0] += 1
            with open(os.path.join(d, "train.log"), "w") as f:
                f.write(tb if calls[0] == 1 else tb + "step 100: total 1.0\n")
            if calls[0] >= 3:
                raise _Stop()
        cli.time.sleep = crash_then_resume
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    cli_main(["--root", tmp, "watch", "retry", "--every", "1"])
                except _Stop:
                    pass
        finally:
            cli.time.sleep = real_sleep
        assert calls[0] >= 3 and "failed at step" in buf.getvalue() and "running at step" in buf.getvalue().split("failed at step")[1], buf.getvalue()
        ok("watch keeps a run that failed while watched for the launcher's retry")
        rc, out = run_cli("--root", tmp, "status", "retry")                       # the events `watch` wrote are shown by the detail of one run
        assert "last events of `watch`" in out and "failed at step" in out, out
        ok("status of one run shows the last events written by watch")

        # 10c a restart from an earlier checkpoint: the watch looked at step 5000 last, the resumed run is at step 4560 and its first rows (a spike at 4520) are new
        d = make(tmp, "stepback", rows(250, every=20, mean=10.0, t0=now - 9000))
        calls[0] = 0

        def resume_with_spike(_):
            calls[0] += 1
            if calls[0] == 1:
                back = rows(226, every=20, mean=10.0, t0=now - 9000)                      # steps 20 .. 4520, then the spike row replaces the last one
                back[-1]["total"] = 80.0
                back += [dict(back[-2], step=4540 + 20 * i, t=back[-2]["t"] + 20 * (i + 2)) for i in range(2)]
                with open(os.path.join(d, "log.jsonl"), "w") as f:
                    f.write("".join(json.dumps(x) + "\n" for x in back))
            if calls[0] >= 3:
                raise _Stop()
        cli.time.sleep = resume_with_spike
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    cli_main(["--root", tmp, "watch", "stepback", "--every", "1"])
                except _Stop:
                    pass
        finally:
            cli.time.sleep = real_sleep
        assert Run(d).step == 4560 and "warn spike" in buf.getvalue(), (Run(d).step, buf.getvalue())
        ok("watch looks at the first rows after a restart from an earlier checkpoint (the cursor of the last look is reset)")

        # 10d/10e the cursor in the other cases: a restart that is past the old step again at the next look (spike in the resumed segment, not the last row),
        # and normal progress (a spike in the middle of 30 new rows is seen, the 5-row default window would miss it)
        def watch_between(name, first_sleep):
            calls[0] = 0

            def sleeper(_):
                calls[0] += 1
                if calls[0] == 1:
                    first_sleep()
                if calls[0] >= 2:
                    raise _Stop()
            cli.time.sleep = sleeper
            out_ = io.StringIO()
            try:
                with contextlib.redirect_stdout(out_):
                    try:
                        cli_main(["--root", tmp, "watch", name, "--every", "1"])
                    except _Stop:
                        pass
            finally:
                cli.time.sleep = real_sleep
            return out_.getvalue()

        def dump(path, rs):
            with open(path, "w") as f:
                f.write("".join(json.dumps(x) + "\n" for x in rs))
        old_rows = rows(250, every=20, mean=10.0, t0=now - 9000)                     # steps 20 .. 5000
        d = make(tmp, "fastback", old_rows)
        new_rows = rows(252, every=20, mean=10.0, t0=now - 9000)[224:]               # the resume: steps 4500 .. 5040, past the old 5000 at the next look
        new_rows[11]["total"] = 80.0                                                  # step 4720, 16 rows from the end
        text = watch_between("fastback", lambda: dump(os.path.join(d, "log.jsonl"), old_rows + new_rows))
        r = Run(d)
        assert r.restarts == 1 and r.resumes == [4500] and r.step == 5040 and "warn spike" in text, (r.restarts, r.resumes, r.step, text)
        d = make(tmp, "midway", old_rows)
        more = rows(280, every=20, mean=10.0, t0=now - 9000)[250:]                    # steps 5020 .. 5600
        more[5]["total"] = 80.0                                                       # step 5120, 24 rows from the end
        text = watch_between("midway", lambda: dump(os.path.join(d, "log.jsonl"), old_rows + more))
        assert Run(d).restarts == 0 and "warn spike" in text, text
        ok("watch tests every row since its last look, also after a restart that has caught up with the old step again")

        # 11 layout override, status table, default listing, run:reference
        d = os.path.join(tmp, "other")
        os.makedirs(d)
        with open(os.path.join(d, "trainsuite.json"), "w") as f:
            json.dump({"log": "m.jsonl", "text_log": "out.txt", "loss": "loss", "steps": 100}, f)
        with open(os.path.join(d, "m.jsonl"), "w") as f:
            for s in range(10, 60, 10):
                f.write(json.dumps({"step": s, "loss": 5.0 - s / 20}) + "\n")
        open(os.path.join(d, "out.txt"), "w").write("ok\n")
        r = Run(d)
        assert r.step == 50 and r.total == 100 and r.loss_key == "loss" and r.state() == "running"
        os.makedirs(os.path.join(tmp, "empty_log"))
        open(os.path.join(tmp, "empty_log", "log.jsonl"), "w").close()
        rc, out = run_cli("--root", tmp, "status", "--json")
        names = {x["run"] for x in json.loads(out)}
        assert "empty_log" not in names and "good" in names
        rc, out = run_cli("--root", tmp, "status", "--all", "--json")
        assert {x["run"] for x in json.loads(out)} >= {"good", "other", "ref", "empty_log"}
        rc, out = run_cli("--root", tmp, "status", "worse:ref")
        assert "vs_ref" in out, out
        rc, out = run_cli("--root", tmp, "status", "good")
        assert "running" in out and "2000/3000" in out, out
        ok("another layout through trainsuite.json; status as table / json, without empty logs by default, with a reference per run")

        # 11b a run that died before its first row (import error: a traceback in the text log, no metrics stream) is listed as failed; an old text-only folder is not
        c0 = make(tmp, "crash_start", [], text="Traceback (most recent call last):\n  File x\nImportError: no module named torchvision\n")
        os.remove(os.path.join(c0, "log.jsonl"))
        o0 = make(tmp, "old_text_only", [], text="loading\n", age=3 * 86400)
        os.remove(os.path.join(o0, "log.jsonl"))
        rc, out = run_cli("--root", tmp, "status", "--json")
        got = {x["run"]: x for x in json.loads(out)}
        assert got["crash_start"]["state"] == "failed" and "torchvision" in str(got["crash_start"]["findings"]), got.get("crash_start")
        assert "old_text_only" not in got and "empty_log" not in got and "good" in got
        rc, out = run_cli("--root", tmp, "status")
        assert "crash_start" in out and "failed" in out, out
        rc, out = run_cli("--root", tmp, "status", "--all", "--json")
        assert {"crash_start", "old_text_only", "empty_log"} <= {x["run"] for x in json.loads(out)}
        ok("status lists a run that crashed before its first row, not an old folder with a text log only")

        # 12 curves
        try:
            import matplotlib  # noqa: F401
            run_cli("--root", tmp, "curves", "good", "worse", "--out", os.path.join(tmp, "c.png"))
            assert os.path.getsize(os.path.join(tmp, "c.png")) > 5000
            ok("curves png of two runs")
        except ImportError:
            print("  --  curves skipped (no matplotlib)")

        # 13 early: advice from learning curves (calibration over finished pairs)
        early_checks(tmp, ok)
    print(f"trainsuite selftest: {n[0]} groups of checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
