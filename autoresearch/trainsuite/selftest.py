"""Self-test on synthetic runs with known answers (stdlib; curves are checked when matplotlib is installed).
Run: python -m trainsuite selftest    (from autoresearch/, a few seconds, writes only to a temporary folder)"""
import codecs
import contextlib
import io
import json
import math
import os
import re
import sys
import random
import tempfile
import time

from . import cli, control, health
from .cli import main as cli_main
from .kill import Rule, load_rules
from . import modules
from .runs import Run, read_log

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(HERE, "..", "..", "bev_fusion", "training", "train.py")       # the BEV trainer: its snapshot check looks for the name _stop_requested


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


def _start_project(tmp, name, prepare=True):
    """a throw-away project for `start`: records/runs, a base config, and (prepare) a prepare check that always says MISSING; -> project folder"""
    proj = os.path.join(tmp, name)
    os.makedirs(os.path.join(proj, "records", "runs"))
    json.dump({"steps": 3000, "name": "x"}, open(os.path.join(proj, "base.json"), "w"))
    open(os.path.join(proj, "chkgt.py"), "w").write("import sys\nsys.exit(1)\n")
    block = {"out": "composed", "train": "bash train.sh {config}", "runs": "runs"}
    if prepare:
        block["prepare"] = [{"name": "gt", "check": "python3 chkgt.py {config}", "build": "bash build.sh {config}"}]
    json.dump({"tag_prefix": "t", "ledger": "LEDGER.csv", "ledger_md": "LEDGER.md", "run_records": "records/runs", "trainsuite": block}, open(os.path.join(proj, "project.json"), "w"))
    return proj


def _start_go(proj, finished, *args):
    """`start base.json <args> --go --wait 0.1` with only run.py's launcher faked: a job whose id is in `finished` ends at once with exit 0, the others keep
    running; -> (exit code, output, the ids of the launched jobs in order)"""
    import subprocess as sp
    launched, real = [], sp.Popen

    class Fake:
        pid = 4242

        def __init__(self, rid):
            self.rid = rid

        def wait(self, timeout=None):
            if self.rid in finished:
                return 0
            raise sp.TimeoutExpired("run.py", timeout)

    def fake(argv, **kw):                                           # the prepare check and the check command run for real
        if len(argv) > 1 and str(argv[1]).endswith("run.py"):
            launched.append(argv[argv.index("--id") + 1])
            return Fake(launched[-1])
        return real(argv, **kw)
    old = os.environ.get("AR_PROJECT")
    os.environ["AR_PROJECT"], sp.Popen = proj, fake
    try:
        rc, out = run_cli("start", "base.json", *args, "--go", "--wait", "0.1")
    finally:
        sp.Popen = real
        if old is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old
    return rc, out, launched


def check_bad_steps(tmp):
    """a step total that is no number (config.json or trainsuite.json) is a finding of that run, not a crash of the status table of all"""
    from . import cli as cli_mod
    root = os.path.join(tmp, "bad_steps")
    make(root, "a_good", rows(5), total=3000)
    for name, bad in (("b_str", "abc"), ("c_list", [5]), ("d_nan", float("nan")), ("e_inf", float("inf"))):
        r = Run(make(root, name, rows(5), total=bad))
        assert r.total == 0 and len(r.config_problems) == 1, (name, r.total, r.config_problems)
    d = make(root, "f_opt", rows(5), total=3000)
    json.dump({"steps": "abc"}, open(os.path.join(d, "trainsuite.json"), "w"))
    assert Run(d).total == 0 and Run(d).config_problems
    rc, out = run_cli("--root", root, "status", "--all")
    assert rc == 0 and all(x in out for x in ("a_good", "b_str", "c_list", "d_nan", "e_inf", "f_opt")) and "bad_config" in out, out
    s, f, _ = cli_mod.summary(Run(os.path.join(root, "b_str")))
    assert [x.code for x in f if x.level == "warn"] == ["bad_config"] and s["total"] == 0, f
    assert Run(os.path.join(root, "a_good")).config_problems == []


def check_compose_values(tmp):
    """--set and presets: 1e3 is the int 1000; NaN / Infinity / 1e400 are refused everywhere (they would be written as invalid json); the base file keeps its floats"""
    from . import compose as C
    for text in ("steps=1e3", "steps=1000.0"):
        v = C.parse_set(text)[1]
        assert v == 1000 and type(v) is int, (text, v)
    assert C.parse_set("lr=2.5")[1] == 2.5 and C.parse_set("w=[1e2, 0.5]")[1] == [100, 0.5] and C.parse_set("w=abc")[1] == "abc"
    for bad in ("lr=NaN", "lr=Infinity", "lr=-Infinity", "lr=1e400", "lr=[1, NaN]", 'lr={"a": 1e999}'):
        try:
            C.parse_set(bad)
            raise AssertionError(f"--set {bad!r} must be refused")
        except C.ComposeError:
            pass
    d = os.path.join(tmp, "cvals")
    os.makedirs(os.path.join(d, "presets"))
    open(os.path.join(d, "presets", "sci.json"), "w").write('{"steps": 1e3}')
    open(os.path.join(d, "presets", "nan.json"), "w").write('{"lr": NaN}')
    open(os.path.join(d, "presets", "big.json"), "w").write('{"lr": 1e400}')
    open(os.path.join(d, "base.json"), "w").write('{"steps": 3000, "w": 2.0}')
    open(os.path.join(d, "base_nan.json"), "w").write('{"steps": NaN}')
    cfg, _ = C.compose(C.load(os.path.join(d, "base.json")), ["sci"], [], os.path.join(d, "presets"))
    assert type(cfg["steps"]) is int and cfg["steps"] == 1000 and type(cfg["w"]) is float, cfg
    for name in ("nan", "big"):
        try:
            C.compose({"steps": 1}, [name], [], os.path.join(d, "presets"))
            raise AssertionError(f"preset {name} must be refused")
        except C.ComposeError:
            pass
    try:
        C.load(os.path.join(d, "base_nan.json"))
        raise AssertionError("a base file with NaN must be refused")
    except C.ComposeError:
        pass
    out = os.path.join(d, "out.json")
    try:
        C.write({"a": float("nan")}, out)
        raise AssertionError("the writer must refuse a NaN")
    except C.ComposeError:
        pass
    assert not os.path.exists(out) and not os.path.exists(out + ".tmp"), "a refused config leaves no file behind"
    rc, so, se = run_cli_err("compose", os.path.join(d, "base.json"), "--set", "lr=NaN")
    assert rc == 2 and "not a json number" in se, (rc, se)


def check_folder_twins(tmp):
    """t1.a and t1_a are one run folder (runs/t1_a): the second id is refused when the first has a run record, and the first one is not"""
    from . import compose as C
    proj = _start_project(tmp, "twins", prepare=False)
    block = json.load(open(os.path.join(proj, "project.json")))["trainsuite"]
    recs = os.path.join(proj, "records", "runs")
    os.makedirs(os.path.join(recs, "t17.a"))
    card = dict(hyp="h", pred="p", rule="r", records=recs)
    try:
        C.plan_start(proj, block, "base.json", "t17_a", [], [], **card)
        raise AssertionError("t17_a shares runs/t17_a with t17.a: refused")
    except C.ComposeError as e:
        assert "also the folder of the run id t17.a" in str(e) and "runs/t17_a" in str(e), e
    assert C.plan_start(proj, block, "base.json", "t17.a", [], [], **card)["held"] is False
    assert C.plan_start(proj, block, "base.json", "t17.b", [], [], **card)["held"] is False, "another folder is no twin"
    open(os.path.join(recs, "t17.a", "HOLD"), "w").write("{}")
    try:
        C.plan_start(proj, block, "base.json", "t17_a", [], [], **card)
        raise AssertionError("the twin of a held id is refused")
    except C.ComposeError as e:
        assert "ON HOLD" in str(e), e
    assert C.plan_start(proj, block, "base.json", "t17.a", [], [], **card)["held"] is True
    assert C.same_folder_ids(None, "t17.a") == []


def check_start_hold(tmp):
    """a held training id (or the id that shares its run folder) stops `start --go` before any job: no prepare build (hours) that the held training then waits for"""
    proj = _start_project(tmp, "hold_start")
    recs = os.path.join(proj, "records", "runs")
    os.makedirs(os.path.join(recs, "t16.h"))
    open(os.path.join(recs, "t16.h", "HOLD"), "w").write("{}")
    rc, out, launched = _start_go(proj, (), "--id", "t16.h", "--no-card", "--no-check", "--prepare")
    assert rc == 1 and launched == [] and "ON HOLD" in out and "no prepare job" in out, (rc, out, launched)
    assert not os.path.exists(os.path.join(proj, "composed", "t16_h.json")) and not os.path.exists(os.path.join(proj, "runs")), "nothing is written for a held id"
    rc, out, launched = _start_go(proj, (), "--id", "t16_h", "--no-card", "--no-check", "--prepare")
    assert rc == 2 and launched == [], (rc, out, launched)                  # the twin id: the same run folder
    rc, out, launched = _start_go(proj, (), "--id", "t16.free", "--no-card", "--no-check", "--prepare")
    assert rc == 0 and launched == ["t16.free.prep_gt", "t16.free"], (rc, out, launched)


def check_start_prepare_done(tmp):
    """a prepare job that ends with exit 0 inside the wait is a success (the training goes on); an exit 0 of the training itself is still 'nothing runs'"""
    proj = _start_project(tmp, "prep_done")
    rc, out, launched = _start_go(proj, {"t16.p.prep_gt"}, "--id", "t16.p", "--no-card", "--no-check", "--prepare")
    assert rc == 0 and launched == ["t16.p.prep_gt", "t16.p"] and "started t16.p, pid 4242" in out, (rc, out, launched)
    rc, out, launched = _start_go(proj, {"t16.q"}, "--id", "t16.q", "--no-card", "--no-check", "--prepare")
    assert rc == 1 and "launcher of t16.q ended" in out, (rc, out)


def check_resolve(tmp):
    """the project's resolve command (json in, json out) fills in names before the config is written, checked and compared with a run's earlier one: what it sets is in the
    config and named in a note, a refusal or output that is no json object refuses the start, a project without the key is untouched"""
    from . import compose as C
    proj = _start_project(tmp, "resolve", prepare=False)
    prog = "import json, sys; d = json.load(sys.stdin); n = d.pop('split'); d['lists'] = [n] * 3; json.dump(d, sys.stdout)"
    block = dict(json.load(open(os.path.join(proj, "project.json")))["trainsuite"], resolve=f"{sys.executable} -c \"{prog}\"")
    json.dump({"steps": 3000, "name": "x", "split": "big"}, open(os.path.join(proj, "base.json"), "w"))
    plan = C.plan_start(proj, block, "base.json", "t18.a", [], [], no_card=True)
    assert plan["config"] == {"steps": 3000, "name": "t18_a", "lists": ["big"] * 3}, plan["config"]
    assert plan["notes"] == ["resolve: lists = 3 entries", "resolve: split removed"], plan["notes"]
    for cmd, text in ((f"{sys.executable} -c \"import sys; sys.exit('no such split')\"", "no such split"), (f"{sys.executable} -c \"print('[1]')\"", "must be a json object"),
                      (f"{sys.executable} -c \"print('nope')\"", "not json"), ("no_such_command_xyz", "cannot run")):
        try:
            C.plan_start(proj, dict(block, resolve=cmd), "base.json", "t18.b", [], [], no_card=True)
            raise AssertionError(f"resolve {cmd!r} must refuse the start")
        except C.ComposeError as e:
            assert text in str(e), (cmd, e)
    assert C.plan_start(proj, {k: v for k, v in block.items() if k != "resolve"}, "base.json", "t18.c", [], [], no_card=True)["config"]["split"] == "big", "no resolve key, no change"


def check_status_views(tmp):
    """2026-11-04: what a person looks at in a training: the gradient norm over the run (now against before, the peak), the non-finite steps the trainer skipped (how many, which
    steps), and the data the run trained on (read from the run's own config.json and gt.json, nothing dataset-specific); the table of all runs carries them"""
    from . import cli as cli_mod
    root = os.path.join(tmp, "views")
    gn = lambda base: (lambda r: {**r, "grad_norm": base * (1 + r["step"] / 2000)})                   # noqa: E731 - a gradient norm that doubles over 2000 steps
    rs = [gn(10.0)(r) for r in rows(100)]
    rs[40]["grad_norm"] = 900.0                                                                        # one spike at step 820
    skips = "".join(f"step {s}: non-finite loss, skipped ({k} in a row)\n" for k, s in ((1, 300), (2, 320), (1, 900)))
    d = make(root, "g", rs, text=skips + "".join(f"step {r['step']}: total {r['total']:.4f}\n" for r in rs[-5:]))
    json.dump({"steps": 3000, "dataset": "waymo", "train_split": "train", "val_split": "val", "train_scenes": list(range(798)), "val_scenes": list(range(202))}, open(os.path.join(d, "config.json"), "w"))
    json.dump({"gt_signature": "abc123", "gt_version": "v2"}, open(os.path.join(d, "gt.json"), "w"))
    run = Run(d)
    t = modules.grad_trend(run)
    assert t["peak"] == 900.0 and t["peak_step"] == 820 and t["rows"] == 100, t
    assert 18 < t["now"] < 20.1 and 1.2 < t["ratio"] < 1.5 and 10 < t["first"] < 12, t                # median of the last 20 rows, against the median of the rows before the window
    assert modules.grad_trend(Run(make(root, "nogn", [{k: v for k, v in r.items() if k != "grad_norm"} for r in rows(30)]))) is None, "no grad_norm, no trend"
    assert run.skipped_at() == [300, 320, 900] and run.skipped_steps() == (3, 2), (run.skipped_at(), run.skipped_steps())
    assert modules.data_of(run) == "waymo train 798 / val 202 scenes, gt v2 abc123", modules.data_of(run)
    assert modules.data_of(Run(make(root, "plain", rows(5)))) == "", "a run that names no data says nothing"
    c = json.load(open(os.path.join(d, "config.json")))
    c.pop("dataset")
    c["data_root"] = "~/data/nuscenes_full"
    json.dump(c, open(os.path.join(root, "g", "config.json"), "w"))
    assert modules.data_of(Run(d)).startswith("nuscenes_full train 798"), modules.data_of(Run(d))
    s, f, _ = cli_mod.summary(Run(d))
    assert s["grad"]["peak"] == 900.0 and s["skipped"] == 3 and s["data"].startswith("nuscenes_full"), s
    rc, out = run_cli("--root", root, "status", "g")
    assert rc == 0 and "grad" in out.splitlines()[0] and "skip" in out.splitlines()[0], out
    assert re.search(r"g\s+running\s+2000/3000.*\s3\s", out), out                                      # the row of the table: the skipped count
    assert "gradient norm: first 10 rows" in out and "peak 900" in out and "step 820" in out, out
    assert "non-finite: 3 step(s) skipped at 300, 320, 900 (up to 2 in a row)" in out, out
    assert "data: nuscenes_full train 798 / val 202 scenes, gt v2 abc123" in out, out
    # a comparison against a run on other data is flagged (a loss on another dataset is no reference)
    other = make(root, "h", rows(100))
    json.dump({"steps": 3000, "dataset": "kitti", "train_split": "train"}, open(os.path.join(other, "config.json"), "w"))
    assert "data_differs" in codes(Run(d), ref=Run(other)), "runs on different data must say so"
    assert "data_differs" not in codes(Run(d), ref=Run(d)) and "data_differs" not in codes(Run(d), ref=Run(make(root, "plain2", rows(100)))), "same data, or no data named: no finding"
    # what counts as other data: both name it and it differs; a version label over the same signature is a name, a field that only one run names is no difference, the scene LIST counts
    def with_data(name, cfg, gt=None):
        dd = make(root, name, rows(100))
        json.dump({"steps": 3000, **cfg}, open(os.path.join(dd, "config.json"), "w"))
        if gt is not None:
            json.dump(gt, open(os.path.join(dd, "gt.json"), "w"))
        return Run(dd)
    base_cfg = {"dataset": "nu", "train_split": "train", "train_scenes": [f"s{i}" for i in range(10)], "val_scenes": ["v1", "v2"]}
    a = with_data("da", base_cfg, {"gt_signature": "sig1", "gt_version": "v8"})
    assert modules.data_differs(a, with_data("db", base_cfg, {"gt_signature": "sig1", "gt_version": "v9"})) == "", "same signature, another version label: the same ground truth"
    assert "ground truth sig1 against sig2" in modules.data_differs(a, with_data("dc", base_cfg, {"gt_signature": "sig2", "gt_version": "v8"}))
    assert modules.data_differs(a, with_data("dd", base_cfg)) == "" and modules.data_differs(with_data("de", base_cfg), a) == "", "an older run without gt.json knows nothing: no difference"
    assert "ground truth version v8 against v9" in modules.data_differs(with_data("df", base_cfg, {"gt_version": "v8"}), with_data("dg", base_cfg, {"gt_version": "v9"})), "no signature on either side: the version"
    assert "other train scenes (10 each)" in modules.data_differs(a, with_data("dh", {**base_cfg, "train_scenes": [f"t{i}" for i in range(10)]}, {"gt_signature": "sig1"})), "10 other scenes are other data"
    assert modules.data_differs(a, with_data("di", {**base_cfg, "train_scenes": list(reversed(base_cfg["train_scenes"]))}, {"gt_signature": "sig1"})) == "", "the order of the scenes does not matter"
    assert "train scenes 10 against 60" in modules.data_differs(a, with_data("dj", {**base_cfg, "train_scenes": list(range(60))})) and "dataset nu against kitti" in modules.data_differs(a, with_data("dk", {**base_cfg, "dataset": "kitti"}))
    assert modules.data_differs(with_data("p1", {"train_scenes": 428}), with_data("p2", {"train_split": "trainval", "train_scenes": 428})) == "", "a split that only one run names is no difference"
    assert modules.data_of(with_data("p3", {"train_scenes": 428})) == "train 428 scenes", "...but the display says 'train' for a count"
    # odd values never crash and say nothing false: empty scene lists, a count, a bool or a list for a name, only val named, a bool / nan version
    assert modules.data_of(with_data("o1", {"train_scenes": [], "val_scenes": [], "train_split": "train"}, {})) == "train", "empty lists (a legacy run) say nothing about counts"
    assert modules.data_of(with_data("o2", {"val_scenes": ["a", "b", "c"]})) == "val 3 scenes", modules.data_of(with_data("o2", {"val_scenes": ["a", "b", "c"]}))
    assert modules.data_of(with_data("o3", {"dataset": True, "data_root": ["x"], "train_scenes": 12, "val_scenes": float("nan")}, {"gt_version": True, "gt_signature": float("nan")})) == "train 12 scenes", modules.data_of(with_data("o3", {"train_scenes": 12}))
    assert modules.data_of(with_data("o4", {"dataset": "x", "train_scenes": {"a": 1}}, {"gt_version": 8})) == "x, gt 8", "a dict is no scene list; a numeric version is text"
    # the gradient norm: an inf row, a window of non-finite rows, fewer rows than a window, a zero base
    inf_rows = [{**r, "grad_norm": 5.0} for r in rows(60)]
    inf_rows[10]["grad_norm"] = float("inf")
    t = modules.grad_trend(Run(make(root, "ginf", inf_rows)))
    assert t["peak"] == 5.0 and t["now"] == 5.0 and t["nonfinite_recent"] == 0 and t["rows"] == 60, "an inf row is no peak"
    bad_tail = [{**r, "grad_norm": 5.0 if i < 40 else float("nan")} for i, r in enumerate(rows(60))]
    t = modules.grad_trend(Run(make(root, "gnan", bad_tail)))
    assert t["now"] is None and t["nonfinite_recent"] == 20 and t["peak"] == 5.0 and t["ratio"] is None, t         # the last 20 rows say nothing finite: no calm 'now' from older rows
    rc, out = run_cli("--root", root, "status", "gnan")
    assert "non-finite" in out.splitlines()[1] and "20 of those rows are NOT finite" in out, out
    short = modules.grad_trend(Run(make(root, "gshort", [{**r, "grad_norm": 3.0} for r in rows(8)])))
    assert short["now"] == 3.0 and short["base"] is None and short["ratio"] is None and short["first"] == 3.0, short
    zero = modules.grad_trend(Run(make(root, "gzero", [{**r, "grad_norm": 0.0 if i < 40 else 2.0} for i, r in enumerate(rows(60))])))
    assert zero["base"] == 0.0 and zero["ratio"] is None, "a zero base gives no ratio (no division)"
    assert modules.grad_trend(Run(make(root, "gonly_nan", [{**r, "grad_norm": float("nan")} for r in rows(30)])))["now"] is None


def check_queue_view(tmp):
    """2026-11-04: `queue` names the machine lock and the launchers it holds back, a slot that is still loading, and the memory gate of a training start"""
    from . import compose as C
    tools = os.path.join(os.path.dirname(HERE), "tools")
    proj = os.path.join(tmp, "qview")
    os.makedirs(os.path.join(proj, "records", "runs", ".gpu_slots", ".queue"))
    json.dump({"tag_prefix": "t", "ledger": "LEDGER.csv", "ledger_md": "LEDGER.md", "run_records": "records/runs", "gpu_slots": 2, "ram_mb": 0, "gpu_settle_s": 180,
               "trainsuite": {"runs": "runs", "out": "temp/compose", "train": "python3 train.py {config}", "gpu_max_mb": 3000}}, open(os.path.join(proj, "project.json"), "w"))
    lockf = os.path.join(proj, "lock.json")
    old = {k: os.environ.get(k) for k in ("AR_MACHINE_LOCK", "AR_PROJECT", "AR_LOCK_OWNER")}
    os.environ.update(AR_MACHINE_LOCK=lockf, AR_PROJECT=proj)
    os.environ.pop("AR_LOCK_OWNER", None)
    try:
        sys.path.insert(0, tools)
        import run as R
        slot = os.path.join(proj, "records", "runs", ".gpu_slots", "0")
        os.makedirs(slot)
        json.dump(dict(id="loader", pid=os.getpid(), run_dir=os.path.join(proj, "runs", "loader"), taken=R.now()), open(os.path.join(slot, "run.json"), "w"))
        R.save_rec(os.path.join(proj, "records", "runs", ".gpu_slots", ".queue", "00000000000000000001_aa"), dict(id="waiter", pid=os.getpid(), since=R.now(), owner=""), indent=None)
        rc, out = run_cli("queue")
        assert "machine lock: free" in out and "slot 0: loader" in out and "loading:" in out and "held back" not in out, out
        assert "GPU memory:" in out and "a training starts only below 3000 MB" in out, out
        json.dump(dict(owner="timing", start="2026-01-01T00:00:00", end="2999-01-01T00:00:00", exclusive=True, note="phase timing"), open(lockf, "w"))
        rc, out = run_cli("queue")
        assert "machine lock: timing until 2999-01-01T00:00:00 (EXCLUSIVE" in out and "phase timing" in out and "1. waiter" in out and "held back by the machine lock of timing" in out, out
        json.dump(dict(owner="timing", end="2999-01-01T00:00:00", exclusive=False), open(lockf, "w"))
        rc, out = run_cli("queue")
        assert "a reservation: it holds nobody" in out and "held back" not in out, out
        # the items of the line carry what a person looks for: the gradient norm now (against before), the skipped non-finite steps, the health flags
        rs = [{**r, "grad_norm": 10.0 * (1 + r["step"] / 2000)} for r in rows(100)]
        make(os.path.join(proj, "runs"), "g", rs, text="".join(f"step {s}: non-finite loss, skipped ({k} in a row)\n" for k, s in ((1, 300), (2, 320), (1, 900))) + "step 2000: total 9.0\n")
        make(os.path.join(proj, "runs"), "clean", rows(100))
        item = lambda rid, seq: dict(id=rid, argv=["python3", "run.py", "--id", rid, "--", "x"], seq=seq, priority=5, added="2026-11-04T00:00:00", after=[], held=None, pin=None, restarts=0, note="", by="")      # noqa: E731
        json.dump(dict(paused=None, seq=3, items=[item("g", 1), item("clean", 2), item("nometrics", 3)]), open(os.path.join(proj, "records", "runs", ".gpu_slots", "queue.qview.json"), "w"))
        rc, out = run_cli("queue")
        line = {rid: next(x for x in out.splitlines() if x.strip().startswith(rid + " ")) for rid in ("g", "clean", "nometrics")}
        assert "grad 19" in line["g"] and "(x1.4)" in line["g"] and "3 skipped" in line["g"] and "flags: skipped" in line["g"], line["g"]
        assert "grad 1" in line["clean"] and "skipped" not in line["clean"] and "flags" not in line["clean"], line["clean"]
        assert "grad" not in line["nometrics"] and "flags" not in line["nometrics"], line["nometrics"]
    finally:
        sys.path.remove(tools)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    block = json.load(open(os.path.join(proj, "project.json")))["trainsuite"]
    base = os.path.join(proj, "base.json")
    json.dump({"steps": 10}, open(base, "w"))
    argv = C.plan_start(proj, block, "base.json", "t.gate", [], [], no_card=True)["argv"]
    assert "--gpu" in argv and argv[argv.index("--gpu-max-mb") + 1] == "3000" and argv.index("--gpu-max-mb") < argv.index("--"), argv
    argv = C.plan_start(proj, {k: v for k, v in block.items() if k != "gpu_max_mb"}, "base.json", "t.gate2", [], [], no_card=True)["argv"]
    assert "--gpu-max-mb" not in argv, "no key, no gate"
    for bad in (True, "3000", 2.5, -1, float("nan"), float("inf"), float("-inf"), [3000]):
        try:
            C.plan_start(proj, dict(block, gpu_max_mb=bad), "base.json", "t.gate3", [], [], no_card=True)
            raise AssertionError(f"gpu_max_mb {bad!r} must be refused (true would be a 1 MB gate: no training would ever start)")
        except C.ComposeError as e:
            assert "gpu_max_mb" in str(e), e
    assert C.gpu_gate({"gpu_max_mb": 3000.0}) == 3000 and C.gpu_gate({"gpu_max_mb": 0}) == 0 and C.gpu_gate({}) == 0, "a whole float, 0 and no key are fine"
    # the queue runner puts the same gate on every training it launches: an item queued before the gate existed, one rebuilt from a run record, one added by hand
    import types
    from . import trainqueue as TQ
    old_p = os.environ.get("AR_PROJECT")
    try:
        c = TQ.context(proj)
    finally:
        if old_p is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old_p
    plain = ["python3", "run.py", "--id", "x", "--gpu", "--split", "train", "--", "python3", "t.py"]
    gated = TQ.with_gate(c, plain)
    assert gated == ["python3", "run.py", "--id", "x", "--gpu", "--split", "train", "--gpu-max-mb", "3000", "--", "python3", "t.py"], gated
    assert TQ.with_gate(c, gated) == gated, "a command that has a gate keeps it"
    mine = plain[:5] + ["--gpu-max-mb", "9000"] + plain[5:]
    assert TQ.with_gate(c, mine) == mine, "an explicit gate of the item is not overridden"
    eq_form = plain[:5] + ["--gpu-max-mb=9000"] + plain[5:]
    assert TQ.with_gate(c, eq_form) == eq_form, "`--gpu-max-mb=N` is a gate of the item too"
    no_gpu = ["python3", "run.py", "--id", "x", "--", "python3", "t.py"]
    assert TQ.with_gate(c, no_gpu) == no_gpu, "a job that asks for no GPU is not gated"
    # the runner's own launch: the command that really starts has the gate before the `--` (a stub run.py writes the argv it got); a gate that is no number starts nothing and does not
    # mark the item refused (the cause is the project's, the view says it)
    stub = os.path.join(proj, "stub_run.py")
    open(stub, "w").write("import json, sys\njson.dump(sys.argv[1:], open(sys.argv[0] + '.argv.json', 'w'))\n")
    it = dict(id="gl", argv=["python3", stub, "--id", "gl", "--gpu", "--split", "train", "--", "python3", "t.py"], pin=None)
    ok, text = TQ.launch(c, it, wait=30)
    got = json.load(open(stub + ".argv.json"))
    assert ok and got[got.index("--gpu-max-mb") + 1] == "3000" and got.index("--gpu-max-mb") < got.index("--"), (ok, text, got)
    os.remove(stub + ".argv.json")
    it2 = dict(id="gl2", argv=list(it["argv"]), pin=None)
    ok, text = TQ.launch(types.SimpleNamespace(**{**vars(c), "block": {"gpu_max_mb": float("inf")}}), it2, wait=30)
    assert not ok and "not started" in text and "gpu_max_mb" in text and "refused" not in it2 and not os.path.exists(stub + ".argv.json"), (ok, text, it2)
    # `queue` with such a project.json says it in the view, it does not crash
    pj = os.path.join(proj, "project.json")
    good = open(pj).read()
    bad_cfg = json.loads(good)
    bad_cfg["trainsuite"]["gpu_max_mb"] = float("inf")
    json.dump(bad_cfg, open(pj, "w"))
    os.environ["AR_PROJECT"] = proj
    try:
        rc, out = run_cli("queue")
        assert rc == 0 and "GPU memory gate:" in out and "NO queue item can start" in out, out
    finally:
        open(pj, "w").write(good)
        if old_p is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old_p
    assert TQ.with_gate(types.SimpleNamespace(**{**vars(c), "block": {}}), plain) == plain, "no key in the project: nothing added"
    try:
        TQ.with_gate(types.SimpleNamespace(**{**vars(c), "block": {"gpu_max_mb": True}}), plain)
        raise AssertionError("a gate that is no number is refused")
    except TQ.QueueError:
        pass


REVIEW_CHECKS = (check_bad_steps, check_compose_values, check_folder_twins, check_start_hold, check_start_prepare_done, check_resolve, check_status_views, check_queue_view)       # review of 2026-11-02 (+ the resolve hook), then the views of 2026-11-04, one rule each


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
        from trainkit import control as engine_control                         # the trainer side lives in the engine; this module re-exports it
        assert control.stop_requested is engine_control.stop_requested and control.clear_stale is engine_control.clear_stale and control.EXIT_STOPPED == 75
        for body, want in ((None, None), ('{"cmd": "stop", "reason": "x"}', "x"), ('{"cmd": "stop"}', ""), ('{"cmd": "go"}', None), ("not json", None), ("[1]", None), ("{}", None)):
            cd = os.path.join(tmp, "cmp")
            os.makedirs(cd, exist_ok=True)
            p = os.path.join(cd, "CONTROL")
            if os.path.exists(p):
                os.remove(p)
            if body is not None:
                open(p, "w").write(body)
            assert engine_control.stop_requested(cd) == want, body
        open(os.path.join(cd, "CONTROL"), "w").write('{"cmd": "stop"}')
        open(os.path.join(cd, "STOPPED"), "w").write("{}")
        engine_control.clear_stale(cd)
        assert not os.path.exists(os.path.join(cd, "CONTROL")) and os.path.exists(os.path.join(cd, "STOPPED.old"))
        if os.path.exists(TRAIN_PY):                                           # the BEV trainer asks through a name the snapshot check looks for
            assert "_stop_requested" in open(TRAIN_PY, encoding="utf-8").read(), "train.py has no stop check"
        ok("stop protocol: CONTROL / STOPPED of an earlier stop are moved aside, the snapshot check; the suite and the trainer share the engine's implementation")

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

        # 13 readiness: phase and percentage
        d = make(tmp, "rdy", rows(75, t0=now - 1500), total=3000)                                   # step 1500 of 3000
        r = Run(d)
        assert r.phase() == "training" and r.ready_pct() == 50.0 and r.ready_at() and r.ready_at() > now, (r.phase(), r.ready_pct())
        d_end = make(tmp, "rdy_end", rows(150, t0=now - 3000), total=3000)                          # every step logged, the process has not exited with 0
        assert Run(d_end).phase() == "finishing" and Run(d_end).ready_pct() == 99.0, "100 only with .done"
        open(os.path.join(d_end, ".done"), "w").close()
        assert Run(d_end).phase() == "done" and Run(d_end).ready_pct() == 100.0 and Run(d_end).ready_at() is None
        d_new = make(tmp, "rdy_start", [], text="loading caches\n")
        os.remove(os.path.join(d_new, "log.jsonl"))
        assert Run(d_new).phase() == "starting" and Run(d_new).ready_pct() == 0.0
        assert make(tmp, "rdy_nosteps", rows(5), total=0) and Run(os.path.join(tmp, "rdy_nosteps")).ready_pct() is None
        d_old = make(tmp, "rdy_stalled", rows(30), total=3000, age=4000)
        assert Run(d_old).phase() == "stalled" and Run(d_old).ready_pct() == 20.0 and Run(d_old).ready_at() is None, "a stalled run keeps its share and has no ready time"
        d_fail = make(tmp, "rdy_failed", rows(30), total=3000, text="step 600: total 1.0\nTraceback (most recent call last):\n  File x\nRuntimeError: boom\n")
        assert Run(d_fail).phase() == "failed" and Run(d_fail).ready_pct() == 20.0
        rc, out = run_cli("--root", tmp, "status", "rdy", "rdy_end")
        assert "ready" in out and "50%" in out and "100%" in out and "last step" in out, out
        rc, out = run_cli("--root", tmp, "status", "rdy", "--json")
        j = json.loads(out)[0]
        assert j["phase"] == "training" and j["ready_pct"] == 50.0 and j["pct"] == 50.0 and j["ready_at"] > now, j
        rc, out = run_cli("--root", tmp, "status", "rdy")
        assert "phase training, ready 50 %, ETA" in out, out
        # watch prints a line for every 10 % of readiness, not for the first look
        d = make(tmp, "rdy_watch", rows(75, t0=now - 1500), total=3000)

        def more_rows():
            with open(os.path.join(d, "log.jsonl"), "w") as f:
                f.write("".join(json.dumps(x) + "\n" for x in rows(95, t0=now - 1500)))                  # step 1900 = 63 %
        text = watch_between("rdy_watch", more_rows)
        assert "ready 60 %" in text and "ready 50 %" not in text, text
        ok("readiness: phase (starting / training / finishing / done / stalled / failed), the percentage held at 99 until .done, the ready time, status table / json / detail, a watch line per 10 %")

        # 14 per-module, timing and incident observability
        def keyed(n, fn):
            rs = rows(n, t0=now - n * 20)
            for r in rs:
                r.update(fn(r))
            return rs
        good = lambda r: {"grad.fusion": 1.0, "upd.fusion": 1e-3, "grad.head": 2.0, "upd.head": 2e-3, "t_data": 0.1, "t_step": 1.0}
        assert not codes(Run(make(tmp, "mod_ok", keyed(100, good))), "warn") & {"dead_module", "grad_module", "update_ratio", "data_bound", "incident"}
        dead = lambda r: dict(good(r), **{"grad.head": 0.0})
        assert "dead_module" in codes(Run(make(tmp, "mod_dead", keyed(100, dead))), "warn"), "a module with no gradient"
        d = make(tmp, "mod_frozen", keyed(100, dead))
        json.dump({"steps": 3000, "freeze": ["head."]}, open(os.path.join(d, "config.json"), "w"))
        assert "dead_module" not in codes(Run(d)), "frozen on purpose: not asked to learn"
        json.dump({"steps": 3000, "train_only": ["fusion."]}, open(os.path.join(d, "config.json"), "w"))
        assert "dead_module" not in codes(Run(d)), "train_only: everything else is frozen"
        json.dump({"steps": 3000, "train_only": ["head."]}, open(os.path.join(d, "config.json"), "w"))
        assert "dead_module" in codes(Run(d), "warn"), "train_only head: the head is trained and has no gradient"
        assert modules._rel("lane", "lane.res.4") and modules._rel("lane.res", "lane") and modules._rel("fusion", "fusion.") and not modules._rel("lanehead", "lane"), \
            "a prefix names a module, a module inside it, or the module that holds it; a longer name is another module"
        jump = lambda r: dict(good(r), **{"grad.fusion": 1.0 if r["step"] < 1600 else 30.0})
        assert "grad_module" in codes(Run(make(tmp, "mod_jump", keyed(100, jump))), "warn"), "one module's gradient grows 30 times"
        slow = lambda r: dict(good(r), **{"upd.fusion": 1e-9})
        fast = lambda r: dict(good(r), **{"upd.fusion": 0.5})
        assert "update_ratio" in codes(Run(make(tmp, "mod_slow", keyed(100, slow))), "warn") and "update_ratio" in codes(Run(make(tmp, "mod_fast", keyed(100, fast))), "warn")
        both = lambda r: dict(good(r), **{"grad.head": 0.0, "upd.head": 0.0})
        c = codes(Run(make(tmp, "mod_both", keyed(100, both))), "warn")
        assert "dead_module" in c and "update_ratio" not in c, "a dead module is one finding, not two"
        short = lambda r: dict(good(r), **{"grad.head": 0.0}) if r["step"] > 1900 else good(r)
        assert "dead_module" not in codes(Run(make(tmp, "mod_short", keyed(100, short))), "warn"), "five zero rows are not a dead module"
        wait = lambda r: dict(good(r), **{"t_data": 0.6})
        assert "data_bound" in codes(Run(make(tmp, "mod_wait", keyed(100, wait))), "warn") and "data_bound" not in codes(Run(make(tmp, "mod_nowait", keyed(100, good))))
        assert "data_bound" not in codes(Run(make(tmp, "mod_fewrows", keyed(4, wait)))), "four rows say nothing"
        d = make(tmp, "mod_inc", keyed(100, good))
        assert "incident" not in codes(Run(d))
        json.dump({"step": 400, "kind": "nonfinite", "batch": ["a", "b"]}, open(os.path.join(d, "incident_400.json"), "w"))
        open(os.path.join(d, "incident_800.json"), "w").write("{broken")
        assert "incident" in codes(Run(d), "warn") and [i["kind"] for i in modules.incidents(Run(d))] == ["nonfinite", "unreadable"]
        d2 = make(tmp, "mod_inc2", keyed(100, good))
        for st in (1000, 90, 300):
            json.dump({"step": st, "kind": "spike"}, open(os.path.join(d2, f"incident_{st}.json"), "w"))
        assert [i["step"] for i in modules.incidents(Run(d2))] == [90, 300, 1000], "by step, not by file name"
        rc, out = run_cli("--root", tmp, "status", "mod_inc")
        assert "modules (median of the last 20 rows" in out and "fusion" in out and "time: 1.00 s per step, 0.10 s of it waiting" in out and "incidents (2" in out \
            and "step 400: nonfinite" in out and "loss groups" in out, out
        rc, out = run_cli("--root", tmp, "status", "good")
        assert "modules (median" not in out and "incidents (" not in out, "a trainer without a probe loses nothing"
        ok("observability: dead / exploding / not-moving modules (frozen ones exempt), data-bound steps, incident dumps, and the detail view of all of it")

        # 14b the readiness and module views against odd data (critic pass 2026-11-01)
        for cfg_val, exempt in (({"freeze": True}, False), ({"train_only": 1}, False), ({"freeze": "head"}, True), ({"freeze": ["head.*"]}, True), ({"freeze": ["head."]}, True),
                                ({"freeze": ["head.gate"]}, False), ({"train_only": "fusion"}, True), ({"train_only": ["fusion.", 7, None]}, True)):
            d = make(tmp, "mod_cfg", keyed(100, dead))
            json.dump(dict(steps=3000, **cfg_val), open(os.path.join(d, "config.json"), "w"))
            got = "dead_module" in codes(Run(d), "warn")                                                                # also: a freeze: true must not crash status / health
            assert got == (not exempt), (cfg_val, got)
            run_cli("--root", tmp, "status", "mod_cfg")
        pause = lambda n: keyed(n, lambda r: dict(good(r), **{"grad.head": 0.0 if r["step"] > 1500 else 2.0}))          # noqa: E731
        rs = pause(100)
        for r in rs[-5:]:
            r["t"] += 100000                                                                                         # a long pause (an evaluation, a resume) five rows before the end
        assert "dead_module" in codes(Run(make(tmp, "mod_pause", rs)), "warn"), "a pause must not cut the module window down to five rows"
        d = make(tmp, "mod_six", keyed(6, dead))
        assert "dead_module" not in codes(Run(d)), "six rows say nothing"
        rs = keyed(25, lambda r: dict(good(r), **{"grad.fusion": 1.0 if r["step"] < 120 else 50.0}))
        assert "grad_module" not in codes(Run(make(tmp, "mod_nobase", rs)), "warn"), "no earlier median without 20 earlier rows (5 earlier rows say nothing)"
        rs = keyed(100, good)
        for r in rs[-3:]:
            r["t"] += 100000                                                                                         # a pause three rows before the end
        assert modules.timing(Run(make(tmp, "mod_tpause", rs))) is not None, "a pause must not cut the timing window down to three rows"
        assert Run(make(tmp, "mod_ended", rows(150, t0=now - 9000), total=3000, age=4000)).phase() == "ended" and Run(os.path.join(tmp, "mod_ended")).ready_pct() == 99.0
        assert Run(make(tmp, "rdy_neg", rows(5), total=-100)).ready_pct() is None
        d = make(tmp, "rdy_huge", rows(30, t0=now - 600), total=10 ** 30)
        rc, out = run_cli("--root", tmp, "status", "rdy_huge")
        assert rc == 0 and "phase training" in out, out
        assert cli_main is not None and __import__("trainsuite.cli", fromlist=["x"]).clock(1e30) == "-" and __import__("trainsuite.cli", fromlist=["x"]).clock(None) == "-"
        d = make(tmp, "br[1]_run", keyed(100, good))
        json.dump({"step": 50, "kind": "spike"}, open(os.path.join(d, "incident_50.json"), "w"))
        assert [i["step"] for i in modules.incidents(Run(d))] == [50], "a run folder name with [ is no glob pattern"
        d = make(tmp, "mod_inc3", keyed(100, good))
        json.dump({"step": 300}, open(os.path.join(d, "incident_300.json"), "w"))
        json.dump([1, 2], open(os.path.join(d, "incident_400.json"), "w"))
        inc = modules.incidents(Run(d))
        assert [(i["step"], i["kind"]) for i in inc] == [(300, "?"), (400, "?")], inc
        hc = [f for f in health.check(Run(d)) if f.code == "incident"]
        assert hc and "step 400" in hc[0].text, "the finding names the LAST incident"
        zt = keyed(100, lambda r: dict(good(r), **{"t_step": 0.0}))
        assert modules.timing(Run(make(tmp, "mod_zero_t", zt))) is None, "a step time of 0 is no step time"
        onlyupd = keyed(100, lambda r: {"upd.lone": 1e-3})
        run_ = Run(make(tmp, "mod_upd_only", onlyupd))
        assert [m["module"] for m in modules.module_table(run_)] == ["lone"] and not {f.code for f in health.check(run_)} & {"dead_module", "update_ratio"}, "a module with only upd. keys is shown, not accused"
        # one look, one answer: the fields of a summary come from one state
        from . import cli as cli_mod
        d = make(tmp, "mod_oneshot", rows(150, t0=now - 3000), total=3000)
        seq = iter(["running"] + ["done"] * 50)
        real_state = Run.state
        Run.state = lambda self: next(seq)
        try:
            s_, _, _ = cli_mod.summary(Run(d))
        finally:
            Run.state = real_state
        assert s_["state"] == "running" and s_["phase"] == "finishing" and s_["ready_pct"] == 99.0, s_
        ok("odd data: configs of the wrong type, partial and starred prefixes, a pause inside the window, short runs, unknown steps, a huge step total, a run folder with [, incident files without kind, step time 0, one state per summary")

        # 15 compose (presets, --set) and start
        from . import compose as C
        cdir = os.path.join(tmp, "cmp")
        os.makedirs(os.path.join(cdir, "presets"))
        base = {"name": "a", "steps": 3000, "model": {"x": 1, "y": 2}, "train_only": [], "lr": 3e-4}
        json.dump(base, open(os.path.join(cdir, "base.json"), "w"))
        json.dump({"steps": 200, "model": {"x": 5}}, open(os.path.join(cdir, "presets", "small.json"), "w"))
        json.dump({"steps": 20, "eval_every": 10}, open(os.path.join(cdir, "presets", "smoke.json"), "w"))
        cfg, ch = C.compose(base, ["small"], ["steps=500", "model.y=7", "flip=0.5", "note=hello", 'train_only=["fusion."]', "model.new.deep=true", "lr=null"], os.path.join(cdir, "presets"))
        assert cfg == {"name": "a", "steps": 500, "model": {"x": 5, "y": 7, "new": {"deep": True}}, "train_only": ["fusion."], "lr": None, "flip": 0.5, "note": "hello"}, cfg
        assert base["steps"] == 3000 and base["model"] == {"x": 1, "y": 2}, "the base is not changed"
        assert [(k, o, n, s) for k, o, n, s in ch][:3] == [("steps", 3000, 200, "preset small"), ("model.x", 1, 5, "preset small"), ("steps", 200, 500, "--set")], ch
        assert [k for k, o, n, s in ch if o == C.MISSING] == ["flip", "note", "model.new.deep"]
        cfg2, ch2 = C.compose(base, ["smoke", "small"], [], os.path.join(cdir, "presets"))
        assert cfg2["steps"] == 200 and cfg2["eval_every"] == 10, "presets are merged in the order given, the later one wins"
        txt = "\n".join(C.describe(C.compose(base, [], ["step=500", "steps=4000"])[1], C.keys_of(base)))
        assert "NEW step = 500" in txt and "close to 'steps'" in txt and "steps: 3000 -> 4000  (--set)" in txt, txt
        for bad, why in (("steps", "no ="), ("=3", "no key"), ("a..b=1", "empty part"), ("name.sub=1", "a value is not a group")):
            try:
                C.compose(base, [], [bad])
                raise AssertionError(f"--set {bad!r} must be refused ({why})")
            except C.ComposeError:
                pass
        try:
            C.compose(base, ["smal"], [], os.path.join(cdir, "presets"))
            raise AssertionError("an unknown preset must be refused")
        except C.ComposeError as e:
            assert "did you mean 'small'" in str(e), e
        try:
            C.compose(base, ["small"], [])
            raise AssertionError("a preset without a presets folder must be refused")
        except C.ComposeError:
            pass
        out = os.path.join(cdir, "out", "c.json")
        C.write(cfg, out)
        assert json.load(open(out)) == cfg and not os.path.exists(out + ".tmp")
        rc, so = run_cli("compose", os.path.join(cdir, "base.json"), "--preset", "small", "--set", "steps=500", "--presets-dir", os.path.join(cdir, "presets"), "--out", os.path.join(cdir, "o2.json"))
        assert rc == 0 and "steps: 3000 -> 200  (preset small)" in so and "steps: 200 -> 500  (--set)" in so and json.load(open(os.path.join(cdir, "o2.json")))["steps"] == 500, so
        rc, so, se = run_cli_err("compose", os.path.join(cdir, "base.json"), "--set", "oops")
        assert rc == 2 and "key=value" in se, (rc, se)
        # start: the plan, the check, the command; nothing is started without --go
        proj = os.path.join(tmp, "proj")
        os.makedirs(os.path.join(proj, "presets"))
        os.makedirs(os.path.join(proj, "records", "runs"))
        json.dump({"steps": 3000, "name": "x"}, open(os.path.join(proj, "base.json"), "w"))
        json.dump({"steps": 50}, open(os.path.join(proj, "presets", "small.json"), "w"))
        open(os.path.join(proj, "chk.py"), "w").write("import json, sys\nd = json.load(open(sys.argv[1]))\nprint('config ok' if 'bogus' not in d else 'unknown key bogus')\nsys.exit(1 if 'bogus' in d else 0)\n")
        block = {"presets": "presets", "out": "composed", "check": "python3 chk.py {config}", "train": "bash train.sh {config}", "runs": "runs"}
        json.dump({"tag_prefix": "t", "ledger": "LEDGER.csv", "ledger_md": "LEDGER.md", "run_records": "records/runs", "trainsuite": block}, open(os.path.join(proj, "project.json"), "w"))
        card = dict(hyp="h", pred="p", rule="r")
        plan = C.plan_start(proj, block, "base.json", "t15.a", ["small"], ["lr=0.001"], extra=["--seed", "1"], **card)
        assert plan["config"]["name"] == "t15_a" and plan["config"]["steps"] == 50 and plan["config"]["lr"] == 0.001, plan["config"]
        assert plan["composed"] == os.path.join(proj, "composed", "t15_a.json")
        a = plan["argv"]
        assert a[2:8] == ["--id", "t15.a", "--gpu", "--split", "train", "--hyp"] and a[a.index("--seed") + 1] == "1" and a[a.index("--") + 1:] == ["bash", "train.sh", "composed/t15_a.json"], a
        assert a.index("--seed") < a.index("--") and plan["check"] == ["python3", "chk.py", "composed/t15_a.json"] and plan["notes"] == []
        for bad in ("name=mine", " name = mine"):
            try:
                C.plan_start(proj, block, "base.json", "t15.b", [], [bad], **card)
                raise AssertionError("the name of the run folder is the id: --set name= must be refused")
            except C.ComposeError as e:
                assert "change --id" in str(e), e
        json.dump({"steps": 1}, open(os.path.join(proj, "base_noname.json"), "w"))
        assert C.plan_start(proj, block, "base_noname.json", "t15.c", [], [], **card)["config"]["name"] == "t15_c", "a base without the key must not fall back to the default name"
        for odd in ("123", "1e3", "true", "null"):
            assert C.plan_start(proj, block, "base.json", odd, [], [], **card)["config"]["name"] == odd, "an id that looks like a json value stays a string"
        assert "name" not in C.plan_start(proj, dict(block, name_key=""), "base_noname.json", "t15.c", [], [], **card)["config"], 'name_key "" = the project has no such key'
        for bad_id in ("-x", "--no-card", ".x", "a/b", "a b", "", "../x"):
            try:
                C.plan_start(proj, block, "base.json", bad_id, [], [], **card)
                raise AssertionError(f"id {bad_id!r} must be refused")
            except C.ComposeError:
                pass
        for kw, why in ((dict(hyp="h", pred="p"), "no rule"), (dict(), "no card at all")):
            try:
                C.plan_start(project_root=proj, block=block, base_path="base.json", run_id="t15.d", **kw)
                raise AssertionError(f"start must refuse: {why}")
            except C.ComposeError:
                pass
        assert C.plan_start(proj, block, "base.json", "t15.e", [], [], no_card=True)["argv"].count("--no-card") == 1
        try:
            C.plan_start(proj, dict(block, train="bash train.sh"), "base.json", "t15.f", [], [], **card)
            raise AssertionError("a train command without {config} must be refused")
        except C.ComposeError:
            pass
        sp_block = dict(block, out="out dir", check="bash -c 'echo ${HOME} {config}'")        # a space in a folder, a ${HOME} in a command
        sp_plan = C.plan_start(proj, sp_block, "base.json", "t15.s", [], [], **card)
        assert sp_plan["check"] == ["bash", "-c", "echo ${HOME} 'out dir/t15_s.json'"] or sp_plan["check"][:2] == ["bash", "-c"], sp_plan["check"]
        assert sp_plan["argv"][-3:] == ["bash", "train.sh", "out dir/t15_s.json"], "a path with a space is one argument"
        # a run that exists: resume note, finished note, hold note, and the composed file of a run is never replaced by another config
        os.makedirs(os.path.join(proj, "runs", "t15_g"))
        open(os.path.join(proj, "runs", "t15_g", "ckpt_00000100.pt"), "w").close()
        os.makedirs(os.path.join(proj, "records", "runs", "t15.g"))
        open(os.path.join(proj, "records", "runs", "t15.g", "HOLD"), "w").write("{}")
        notes = C.plan_start(proj, block, "base.json", "t15.g", [], [], records=os.path.join(proj, "records", "runs"), **card)["notes"]
        assert len(notes) == 2 and "RESUMES" in notes[0] and "ON HOLD" in notes[1], notes
        C.write(C.plan_start(proj, block, "base.json", "t15.g", [], [], **card)["config"], os.path.join(proj, "composed", "t15_g.json"))
        assert C.plan_start(proj, block, "base.json", "t15.g", [], [], **card)["notes"], "the same config resumes"
        try:
            C.plan_start(proj, block, "base.json", "t15.g", [], ["lr=0.5"], **card)
            raise AssertionError("another config for an id that has a run must be refused")
        except C.ComposeError as e:
            assert "same config" in str(e), e
        open(os.path.join(proj, "runs", "t15_g", ".done"), "w").close()
        assert "FINISHED" in C.plan_start(proj, block, "base.json", "t15.g", [], [], **card)["notes"][0], "a finished run is not announced as a resume"
        # --set: a group of keys is never replaced by a value, an object sets its keys one by one, 1 and true are different
        gb = {"model": {"x": 1, "y": 2}, "flag": 1}
        for bad_set in ("model=5", "model=null", "model=[1]"):
            try:
                C.compose(gb, [], [bad_set])
                raise AssertionError(f"--set {bad_set} would wipe a group of keys")
            except C.ComposeError:
                pass
        assert C.compose(gb, [], ["model={}"])[0] == gb and C.compose(gb, [], ['model={"x": 9}'])[0]["model"] == {"x": 9, "y": 2}, "an object sets its keys; an empty one changes nothing"
        assert "unchanged" not in "\n".join(C.describe(C.compose(gb, [], ["flag=true"])[1])) and "unchanged" in "\n".join(C.describe(C.compose(gb, [], ["flag=1"])[1])), "1 -> true is a change"
        old_env = os.environ.get("AR_PROJECT")
        os.environ["AR_PROJECT"] = proj
        try:
            args = ["start", "base.json", "--id", "t15.h", "--preset", "small", "--hyp", "h", "--pred", "p", "--rule", "r"]
            rc, so = run_cli(*args)
            assert rc == 0 and "steps: 3000 -> 50  (preset small)" in so and "check: python3 chk.py composed/t15_h.preview.json -> exit 0" in so and "config ok" in so \
                and "(nothing started: add --go)" in so and "--id t15.h --gpu --split train" in so and "composed/t15_h.json" in so, so
            assert json.load(open(os.path.join(proj, "composed", "t15_h.preview.json")))["name"] == "t15_h" and not os.path.exists(os.path.join(proj, "composed", "t15_h.json")), \
                "a preview writes the preview file and leaves the real config alone"
            rc, so = run_cli(*args, "--set", "bogus=1")
            assert rc == 1 and "unknown key bogus" in so and "nothing started" in so and "add --go" not in so, so
            rc, so = run_cli(*args, "--set", "bogus=1", "--no-check")
            assert rc == 0 and "check:" not in so, "--no-check skips the check"
            rc, so, se = run_cli_err("start", "base.json", "--id", "t15.i", "--hyp", "h")
            assert rc == 2 and "--pred" in se and "--rule" in se, se
            rc, so = run_cli("start", "base.json", "--id", "t15.j", "--no-card", "--run-arg=--seed", "--run-arg=1", "--no-check")
            assert rc == 0 and "--no-card --seed 1 --" in so, so
            os.makedirs(os.path.join(proj, "records", "runs", "t15.k"))
            open(os.path.join(proj, "records", "runs", "t15.k", "HOLD"), "w").write("{}")
            rc, so = run_cli("start", "base.json", "--id", "t15.k", "--no-card", "--no-check")
            assert rc == 0 and "ON HOLD" in so, "the hold note comes from the project's run records"
            import subprocess as sp
            calls, real_popen = [], sp.Popen

            class FakeProc:
                pid = 4242
                returncode = None

                def wait(self, timeout=None):
                    raise sp.TimeoutExpired("run.py", timeout)

            class DeadProc(FakeProc):
                returncode = 2

                def wait(self, timeout=None):
                    return 2

            def fake_popen_of(cls):
                def fake(argv, **kw):                                  # only the launcher is faked, the check runs for real
                    if len(argv) > 1 and str(argv[1]).endswith("run.py"):
                        calls.append((argv, kw))
                        return cls()
                    return real_popen(argv, **kw)
                return fake
            sp.Popen = fake_popen_of(FakeProc)
            try:
                rc, so = run_cli(*args, "--go")
            finally:
                sp.Popen = real_popen
            assert rc == 0 and "started t15.h, pid 4242" in so and "watch t15_h" in so and len(calls) == 1, so
            (argv_, kw_), = calls
            assert argv_[2:4] == ["--id", "t15.h"] and kw_["cwd"] == proj and kw_["env"]["AR_PROJECT"] == proj and kw_["start_new_session"] is True and os.path.exists(os.path.join(proj, "runs", "t15_h.launcher.log"))
            assert json.load(open(os.path.join(proj, "composed", "t15_h.json")))["name"] == "t15_h" and "composed/t15_h.json" in argv_, "--go writes the real config"
            calls.clear()
            sp.Popen = fake_popen_of(DeadProc)
            try:
                rc, so = run_cli(*args, "--go", "--wait", "0.1")
            finally:
                sp.Popen = real_popen
            assert rc == 1 and "ended after 0.1 s with exit 2: nothing runs" in so and "started, pid" not in so, so
            os.environ["AR_PROJECT"] = os.path.relpath(proj, sys.modules["project"].AR)           # a relative AR_PROJECT: the launcher runs in the project folder, so it gets the absolute one
            calls.clear()
            sp.Popen = fake_popen_of(FakeProc)
            try:
                rc, so = run_cli(*args, "--go")
            finally:
                sp.Popen = real_popen
            assert rc == 0 and len(calls) == 1 and calls[0][1]["env"]["AR_PROJECT"] == proj, (so, calls)
            os.environ.pop("AR_PROJECT")
            rc, so = run_cli("start", "base.json", "--id", "t15.l", "--no-card", "--no-check", "--project", proj)
            assert rc == 0 and "--id t15.l" in so, "--project replaces AR_PROJECT"
            # the prepare step: data a run needs (the GT store) is checked, built as a run of its own when missing, and the training waits for it
            os.environ["AR_PROJECT"] = proj
            open(os.path.join(proj, "chkgt.py"), "w").write("import json, os, sys\njson.load(open(sys.argv[1]))\nok = os.path.exists('gt_present')\nprint('GT store present' if ok else 'GT store of the run missing')\nsys.exit(0 if ok else 1)\n")
            pblock = dict(block, prepare=[{"name": "gt", "check": "python3 chkgt.py {config}", "build": "bash build.sh {config}", "wait_min": 600}])
            json.dump({"tag_prefix": "t", "ledger": "LEDGER.csv", "ledger_md": "LEDGER.md", "run_records": "records/runs", "trainsuite": pblock}, open(os.path.join(proj, "project.json"), "w"))
            pargs = ["start", "base.json", "--id", "t15.p", "--no-card", "--no-check"]
            rc, so = run_cli(*pargs)
            assert rc == 1 and "prepare gt:" in so and "MISSING" in so and "GT store of the run missing" in so and "--prepare" in so \
                and "--id t15.p.prep_gt --no-card -- bash build.sh composed/t15_p.json" in so and "--after" not in so and "nothing started" not in so, so
            rc, so = run_cli(*pargs, "--prepare")
            assert rc == 0 and "(nothing started: add --go)" in so, so
            lines = [ln for ln in so.splitlines() if ln.startswith("cd ")]
            assert len(lines) == 2 and "--id t15.p.prep_gt" in lines[0] and "--gpu" not in lines[0] and "--id t15.p --gpu" in lines[1] \
                and "--after t15.p.prep_gt --wait-after 600 -- bash train.sh" in lines[1], lines
            open(os.path.join(proj, "gt_present"), "w").close()
            rc, so = run_cli(*pargs)
            assert rc == 0 and "(present)" in so and "--after" not in so and "prep_gt --no-card" not in so, so
            os.remove(os.path.join(proj, "gt_present"))
            rc, so = run_cli(*pargs, "--skip-prepare")
            assert rc == 0 and "prepare gt" not in so, "--skip-prepare does not look"
            calls.clear()
            sp.Popen = fake_popen_of(FakeProc)
            try:
                rc, so = run_cli(*pargs, "--prepare", "--go")
            finally:
                sp.Popen = real_popen
            assert rc == 0 and [c[0][3] for c in calls] == ["t15.p.prep_gt", "t15.p"] and "started t15.p.prep_gt, pid 4242" in so and "started t15.p, pid 4242" in so, (so, [c[0][:4] for c in calls])
            assert "--after" in calls[1][0] and "--gpu" not in calls[0][0] and os.path.exists(os.path.join(proj, "runs", "t15_p_prep_gt.launcher.log")), calls
            calls.clear()
            sp.Popen = fake_popen_of(DeadProc)
            try:
                rc, so = run_cli(*pargs, "--prepare", "--go", "--wait", "0.1")
            finally:
                sp.Popen = real_popen
            assert rc == 1 and len(calls) == 1 and "launcher of t15.p.prep_gt ended" in so and "later runs are not started" in so, (so, len(calls))
            json.dump({"tag_prefix": "t", "ledger": "LEDGER.csv", "ledger_md": "LEDGER.md", "run_records": "records/runs", "trainsuite": dict(block, prepare=[{"name": "bad name", "check": "x", "build": "y"}])},
                      open(os.path.join(proj, "project.json"), "w"))
            rc, so, se = run_cli_err(*pargs)
            assert rc == 2 and "need a name" in se, se
            os.environ["AR_PROJECT"] = proj
            json.dump({"tag_prefix": "t", "ledger": "LEDGER.csv", "ledger_md": "LEDGER.md", "run_records": "records/runs"}, open(os.path.join(proj, "project.json"), "w"))
            rc, so, se = run_cli_err(*args)
            assert rc == 2 and '"trainsuite" block' in se, se
        finally:
            if old_env is None:
                os.environ.pop("AR_PROJECT", None)
            else:
                os.environ["AR_PROJECT"] = old_env
        ok("compose: presets in order, --set as json or string, the base untouched, every change listed (NEW keys and typos marked), errors refused; start: the check, the exact run.py command, resume / hold notes, nothing started without --go")

        # 15b the review of 2026-11-02: a bad step total, NaN / 1e3 in --set, ids that share a folder, a held id, a prepare job that is done at once
        for check in REVIEW_CHECKS:
            check(tmp)
        ok("review fixes: a non-numeric steps is a finding (status goes on), --set / presets refuse NaN and read 1e3 as 1000, t1.a and t1_a do not share a run folder, "
           "start --go refuses a held id before any job, a prepare job that finishes inside --wait does not stop the training, the project's resolve command fills in names (a split) before the config is written")

        # 16 the view of run.py's queue
        qp =os.path.join(tmp, "qproj")
        for d_ in ("records/runs/.gpu_slots/0", "records/runs/.gpu_slots/.queue", "records/runs/Rheld"):
            os.makedirs(os.path.join(qp, d_))
        json.dump({"tag_prefix": "t", "ledger": "LEDGER.csv", "ledger_md": "LEDGER.md", "run_records": "records/runs", "gpu_slots": 2, "ram_mb": 5000}, open(os.path.join(qp, "project.json"), "w"))
        json.dump({"id": "Rrun", "pid": os.getpid(), "taken": "2026-11-01T22:00:00"}, open(os.path.join(qp, "records/runs/.gpu_slots/0/run.json"), "w"))
        for nm, rid, pid in (("00000000000000000001_a", "Rwait1", os.getpid()), ("00000000000000000002_b", "Rdead", 2 ** 30), ("00000000000000000003_c", "Rwait2", os.getpid())):
            json.dump({"id": rid, "pid": pid, "since": "2026-11-01T22:05:00"}, open(os.path.join(qp, "records/runs/.gpu_slots/.queue", nm), "w"))
        json.dump({"id": "Rheld", "who": "run.py", "when": "2026-11-01T21:00:00"}, open(os.path.join(qp, "records/runs/Rheld/HOLD"), "w"))
        os.makedirs(os.path.join(qp, "records/runs/.gpu_slots/1"))
        json.dump({"id": "Rghost", "pid": 2 ** 30, "taken": "2026-11-01T21:30:00"}, open(os.path.join(qp, "records/runs/.gpu_slots/1/run.json"), "w"))       # its launcher is dead
        mi_ = os.path.join(tmp, "q_meminfo")
        old_mi_ = os.environ.get("AR_MEMINFO")
        os.environ["AR_MEMINFO"] = mi_
        try:
            open(mi_, "w").write("MemAvailable:   2097152 kB\n")                                       # 2048 MB
            rc, so = run_cli("queue", "--project", qp)
            assert rc == 0 and "slot 0: Rrun  (pid" in so and "slot 1: free  (a stale owner is reclaimed by the next launcher)" in so and "Rghost" not in so and "1. Rwait1" in so and "2. Rwait2" in so and "Rdead" not in so, so
            assert "RAM: 2048 MB available, the gate asks for 5000 MB" in so and "a run waits" in so and "Rheld  (since 2026-11-01T21:00:00)" in so, so
            open(mi_, "w").write("MemAvailable:   10485760 kB\n")
            rc, so = run_cli("queue", "--project", qp)
            assert "a run could start" in so, so
            os.environ["AR_MEMINFO"] = os.path.join(tmp, "no_such_meminfo")
            rc, so = run_cli("queue", "--project", qp)
            assert "not readable here: no gate" in so, so
        finally:
            if old_mi_ is None:
                os.environ.pop("AR_MEMINFO", None)
            else:
                os.environ["AR_MEMINFO"] = old_mi_
        os.remove(os.path.join(qp, "records/runs/.gpu_slots/0/run.json"))
        rc, so = run_cli("queue", "--project", qp)
        assert "slot 0: free  (a stale owner" not in so and "slot 0: free" in so, "an empty slot is free"
        ok("queue: the slots with their holders, the line in order without a dead launcher's ticket, the RAM gate against /proc/meminfo, the runs on hold")

        # 17 the speed table: runs side by side against the first one
        sb = make(tmp, "sp_base", rows(40, sec=2.0, t0=now - 5000), total=800)                      # 2.0 s per step
        fr = rows(40, sec=1.0, t0=now - 5000)
        for r_ in fr:
            r_.update({"t_data": 0.2, "t_step": 1.0, "gpu_gb": 6.5})
        sf = make(tmp, "sp_fast", fr, total=800)
        open(os.path.join(sf, "profile.txt"), "w").write("window: 30 optimiser steps, 30.00 s wall = 1.000 s per step\nGPU kernels busy: 21.00 s of 30.00 s = 70 % (the rest is the GPU waiting)\n")
        rc, so = run_cli("--root", tmp, "speed", "sp_base", "sp_fast", "--skip", "5")
        lines_ = {ln.split()[0]: ln for ln in so.splitlines() if ln.startswith("sp_")}
        assert rc == 0 and "x1.00" in lines_["sp_base"] and "x2.00" in lines_["sp_fast"] and "2.00" in lines_["sp_base"] and "1.00" in lines_["sp_fast"], so
        assert "20 %" in lines_["sp_fast"] and "70 %" in lines_["sp_fast"] and "6.5" in lines_["sp_fast"] and "7.1" in lines_["sp_base"], so
        assert " - " in lines_["sp_base"] or "-" in lines_["sp_base"].split()[4:6], "no probe, no profile: dashes"
        short = make(tmp, "sp_short", rows(10, sec=1.0, t0=now - 5000), total=800)                   # shorter than the others: compared at the last step they share
        rc, so = run_cli("--root", tmp, "speed", "sp_base", "sp_short", "--skip", "2")
        assert "loss@200" in so and rc == 0, so
        try:
            run_cli_err("--root", tmp, "speed", "no_such_run")
            raise AssertionError("an unknown run must be refused")
        except SystemExit as e:
            assert "no run directory" in str(e), e
        ok("speed: s/step and the speed-up against the first run, data wait (probe), GPU busy (profile.txt), peak memory, the loss at the last shared step")

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

        # 14 the queue: real launchers, a real runner, a toy trainer (about a minute; TS_SELFTEST_NO_QUEUE=1 leaves it out)
        if not os.environ.get("TS_SELFTEST_NO_QUEUE"):                         # a process of its own: run.py reads AR_STALE_S when it is imported, and the groups above import it
            import subprocess
            r = subprocess.run([sys.executable, "-m", "trainsuite.selftest_queue"], cwd=os.path.dirname(HERE), capture_output=True, text=True)
            print(r.stdout, end="")
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
            ok("queue self-test (selftest_queue.py) passed")
            r = subprocess.run([sys.executable, "-m", "trainsuite.selftest_api"], cwd=os.path.dirname(HERE), capture_output=True, text=True)       # the interface (api.py, runsuite): real launchers, a toy trainer and runner
            print(r.stdout, end="")
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
            ok("interface self-test (selftest_api.py: jobs, check, run, status, stop, compare, the queue functions, the runner, the command line) passed")
    print(f"trainsuite selftest: {n[0]} groups of checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
