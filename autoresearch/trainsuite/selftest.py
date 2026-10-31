"""Self-test on synthetic runs with known answers (stdlib; curves are checked when matplotlib is installed).
Run: python -m trainsuite selftest    (from autoresearch/, a few seconds, writes only to a temporary folder)"""
import contextlib
import io
import json
import math
import os
import random
import tempfile
import time

from . import control, health
from .cli import main as cli_main
from .kill import Rule, load_rules
from .runs import Run, read_log


def rows(n, every=20, seed=0, f=None, t0=None, sec=1.0):
    """n logged rows of a healthy run: loss falls from 16 to 10 with noise 1.2; group.lines falls from 1.6 to 1.0"""
    rng = random.Random(seed)
    out = []
    for i in range(1, n + 1):
        s = i * every
        base = 10 + 6 * math.exp(-s / 800)
        r = {"step": s, "lr": 3e-4, "grad_norm": 12 + rng.gauss(0, 2), "s_per_step": sec, "total": base + rng.gauss(0, 1.2),
             "group.lines": 1.0 + 0.6 * math.exp(-s / 500) + rng.gauss(0, 0.05), "gpu_gb": 7.1}
        if t0 is not None:
            r["t"] = t0 + s * sec
        out.append(f(r) if f else r)
    return out


def make(tmp, name, rs, total=3000, text="", done=False, age=0.0, evals=(), partial=""):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"steps": total}, f)
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
        for e in evals:
            f.write(json.dumps({"step": e[0], "val": {"total": e[1]}}) + "\n")
        f.write(partial)
    with open(os.path.join(d, "train.log"), "w") as f:
        f.write(text or "".join(f"step {r['step']}: total {r['total']:.4f}\n" for r in rs[-5:]))
    if done:
        open(os.path.join(d, ".done"), "w").close()
    if age:
        for fn in ("log.jsonl", "train.log"):
            t = time.time() - age
            os.utime(os.path.join(d, fn), (t, t))
    return d


def codes(run, level=None, ref=None):
    return {f.code for f in health.check(run, ref) if level is None or f.level == level}


def main():
    n = [0]

    def ok(msg):
        n[0] += 1
        print(f"  ok  {msg}")

    with tempfile.TemporaryDirectory() as tmp:
        # 1 healthy run
        now = time.time()
        d = make(tmp, "good", rows(100, t0=now - 2000, sec=1.0))
        r = Run(d)
        assert r.state() == "running" and r.step == 2000 and r.total == 3000
        assert abs(r.s_per_step() - 1.0) < 1e-6 and abs(r.eta_s() - 1000) < 1e-6, (r.s_per_step(), r.eta_s())
        bad = codes(r) - {"vs_ref"}
        assert not {c for c in bad if c != "flat"}, bad
        ok("healthy run: state, progress, ETA from the row times, no warnings")

        # 2 non-finite logged value, spike, rising loss, grad norm blow-up, GPU, slowdown
        def nan_last(rs):
            rs[-1]["total"] = float("nan")
            return rs
        d = make(tmp, "nan", nan_last(rows(100)))
        assert "nonfinite" in codes(Run(d), "alert")
        ok("NaN in a logged row is an alert")
        rs = rows(100)
        rs[-1]["total"] += 40
        assert "spike" in codes(Run(make(tmp, "spike", rs)), "warn")
        ok("a single value far above the recent median is a spike warning")
        rs = rows(100, f=lambda r: r)
        for i, x in enumerate(rs):
            if x["step"] > 400:
                x["total"] += 0.02 * (x["step"] - 400)
        assert "rising" in codes(Run(make(tmp, "rise", rs)), "warn")
        early = Run(make(tmp, "rise_early", rs, total=6000))                # the same rise at 33 % of the steps
        assert "rising" in codes(early, "info") and "rising" not in codes(early, "warn")
        ok("a loss that climbs for many rows is a rising warning (only info before 50 % of the steps: loss ramps)")
        rs = rows(100)
        rs[-1]["grad_norm"] = 400
        assert "grad_norm" in codes(Run(make(tmp, "gn", rs)), "warn")
        rs = rows(100)
        rs[-1]["gpu_gb"] = 11.8
        assert "gpu" in codes(Run(make(tmp, "gpu", rs)), "warn")
        rs = rows(100, t0=now - 3000)
        for k, x in enumerate(rs[-6:], 1):
            x["t"] += 40 * k                                                 # the last intervals take 3 s/step instead of 1
        assert "slow" in codes(Run(make(tmp, "slow", rs)), "warn"), "slow"
        ok("grad-norm blow-up, GPU above 11 GB, recent slowdown are warnings")

        # 3 stalled, failed, ended, done, new
        assert Run(make(tmp, "stall", rows(50), age=3600)).state() == "stalled"
        assert "stalled" in codes(Run(os.path.join(tmp, "stall")), "alert")
        tb = "step 40: total 11.0\nTraceback (most recent call last):\n  File x\nRuntimeError: CUDA error: unknown error\n"
        r = Run(make(tmp, "fail", rows(2), text=tb))
        assert r.state() == "failed" and "CUDA error" in r.crash_reason()
        assert Run(make(tmp, "end", rows(150), total=3000, age=3600)).state() == "ended"      # step 3000 of 3000, no .done
        assert Run(make(tmp, "fin", rows(150), done=True)).state() == "done"
        os.makedirs(os.path.join(tmp, "empty"))
        assert Run(os.path.join(tmp, "empty")).state() == "new"
        ok("states: stalled (old output), failed (traceback at the end of the text log), ended (all steps, no .done), done, new")

        # 4 skipped steps in the text log
        txt = "".join(f"step {i}: non-finite loss, skipped ({k} in a row)\n" for i, k in ((10, 1), (11, 2), (12, 3)))
        r = Run(make(tmp, "skip", rows(50), text=txt))
        assert r.skipped_steps() == (3, 3) and "skipped" in codes(r, "alert")
        ok("non-finite steps skipped by the trainer are counted from the text log (3 in a row = alert)")

        # 5 resume: steps twice, the later attempt wins; half-written line skipped
        first = rows(10)                                  # steps 20..200
        again = rows(8, seed=5)                           # a restart from step 100: steps 20..160 again
        again = [dict(x, total=-1.0) for x in again[5:]]  # 120, 140, 160 replaced (total -1 marks the later attempt)
        d = make(tmp, "resume", first + again, partial='{"step": 17')
        tr, ev, rs_ = read_log(os.path.join(d, "log.jsonl"))
        steps = [x["step"] for x in tr]
        assert steps == sorted(set(steps)) and rs_ >= 1, (steps, rs_)
        assert [x["total"] for x in tr if x["step"] in (120, 140, 160)] == [-1.0] * 3 and tr[-1]["step"] == 160
        ok("after a resume the later attempt of a step wins, steps are unique and increasing, a half-written last line is skipped")

        # 6 eval rows, over-fitting warning
        rs = rows(100)
        d = make(tmp, "val", rs, evals=[(1000, 12.0), (2000, 13.5)])
        r = Run(d)
        assert [e["step"] for e in r.evals] == [1000, 2000] and r.series("val.total")[-1] == (2000.0, 13.5)
        assert "overfit" in codes(r, "warn")
        ok("eval rows are kept apart; val rising while the training loss falls is a warning")

        # 7 reference run
        ref = Run(make(tmp, "ref", rows(100)))
        worse = Run(make(tmp, "worse", rows(100, f=lambda r: dict(r, total=r["total"] * 1.25))))
        assert "vs_ref" in codes(worse, "warn", ref) and "vs_ref" not in codes(Run(os.path.join(tmp, "good")), "warn", ref)
        ok("a loss 25 % above the reference at the same step is a warning, the equal one is not")

        # 8 kill rules
        r = Run(os.path.join(tmp, "good"))                                # group.lines falls to about 1.0 by step 1000
        assert Rule("group.lines@1000 > 1.48").evaluate(r)[0] == "ok"
        assert Rule("group.lines@1000 > 0.9").evaluate(r)[0] == "hit"
        assert Rule("group.lines@2500 > 0.1").evaluate(r)[0] == "pending"
        assert Rule("total@500 < 3 window 5").evaluate(r)[0] == "ok"
        assert Rule("val.total@2000>=13").evaluate(Run(os.path.join(tmp, "val")))[0] == "hit"
        try:
            Rule("lines is bad")
            raise AssertionError("bad rule accepted")
        except ValueError:
            pass
        with open(os.path.join(tmp, "good", "kill.txt"), "w") as f:
            f.write("# card kill criterion\ngroup.lines@1000 > 0.9\n")
        assert len(load_rules(r, ["total@100 > 1"])) == 2
        ok("kill rules: ok / hit / pending, window, eval keys, bad text refused, rules read from kill.txt")

        # 9 control: stop request is honoured only when newer than the process start
        d = os.path.join(tmp, "good")
        since = time.time() - 1
        assert control.stop_requested(d, since) is None
        control.request_stop(d, "test")
        assert control.stop_requested(d, since) == "test" and control.stop_requested(d, time.time() + 5) is None
        control.mark_stopped(d, 2000, "test")
        os.utime(os.path.join(d, "STOPPED"), (time.time() + 10, time.time() + 10))
        assert Run(d).state() == "stopped"
        os.utime(os.path.join(d, "log.jsonl"), (time.time() + 200, time.time() + 200))       # the run went on after the stop marker
        assert Run(d).state() != "stopped"
        ok("CONTROL counts only when newer than the process start; STOPPED shows state 'stopped' until the run is active again")

        # 10 watch --enforce writes the stop request on a kill hit
        d = make(tmp, "enf", rows(100, t0=now - 2000))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli_main(["--root", tmp, "watch", "enf", "--kill", "group.lines@1000 > 0.9", "--enforce", "--once"])
        assert os.path.exists(os.path.join(d, "CONTROL")) and "STOP requested" in buf.getvalue(), buf.getvalue()
        d2 = make(tmp, "noenf", rows(100, t0=now - 2000))
        with contextlib.redirect_stdout(io.StringIO()):
            cli_main(["--root", tmp, "watch", "noenf", "--kill", "group.lines@1000 > 0.9", "--once"])
        assert not os.path.exists(os.path.join(d2, "CONTROL"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):                                 # a run that does not exist yet: wait, then give up
            cli_main(["--root", tmp, "watch", "not_yet", "--give-up", "-1", "--every", "0.01"])
        assert "new at step 0" in buf.getvalue() and "never started" in buf.getvalue(), buf.getvalue()
        ok("watch reports a kill hit; only with --enforce it asks the trainer to stop; a run that has not started is waited for, then dropped")

        # 11 layout override and status table / json
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
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli_main(["--root", tmp, "status", "--all", "--json"])
        data = json.loads(buf.getvalue())
        assert {x["run"] for x in data} >= {"good", "other", "ref"}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli_main(["--root", tmp, "status", "good"])
        assert "running" in buf.getvalue() and "2000/3000" in buf.getvalue(), buf.getvalue()
        ok("another layout through trainsuite.json; status as a table and as json")

        # 12 curves
        try:
            import matplotlib  # noqa: F401
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(["--root", tmp, "curves", "good", "worse", "--out", os.path.join(tmp, "c.png")])
            assert os.path.getsize(os.path.join(tmp, "c.png")) > 5000
            ok("curves png of two runs")
        except ImportError:
            print("  --  curves skipped (no matplotlib)")
    print(f"trainsuite selftest: {n[0]} groups of checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
