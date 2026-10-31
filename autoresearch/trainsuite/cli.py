"""Command line of the training suite: python -m trainsuite <status|health|curves|stop|watch|selftest> ... (run from autoresearch/)."""
import argparse
import json
import os
import sys
import time

from . import control, health
from .kill import load_rules
from .runs import STALE_S, Run, fmt_dur

TERMINAL = ("done", "failed", "stopped", "ended")


def resolve(arg, root):
    return arg if os.path.isdir(arg) else os.path.join(root, arg)


def open_run(arg, root, loss):
    path = resolve(arg, root)
    if not os.path.isdir(path):
        raise SystemExit(f"no run directory {arg!r} (looked in {path})")
    return Run(path, loss)


def summary(run, ref=None, extra_rules=()):
    findings = health.check(run, ref)
    rules = load_rules(run, extra_rules)
    kills = [(r, *r.evaluate(run)) for r in rules]
    loss = run.last(run.loss_key)
    return dict(run=run.name, state=run.state(), step=run.step, total=run.total,
                pct=100.0 * run.step / run.total if run.total else None, s_per_step=run.s_per_step(), eta_s=run.eta_s(), loss=loss,
                restarts=run.restarts, findings=[f._asdict() for f in findings],
                kill=[dict(rule=r.text, status=s, value=v) for r, s, v in kills]), findings, kills


def cmd_status(a):
    if a.runs:
        runs = [open_run(x, a.root, a.loss) for x in a.runs]
    else:
        names = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d))) if os.path.isdir(a.root) else []
        runs = [Run(os.path.join(a.root, d), a.loss) for d in names]
        runs = [r for r in runs if r.train or r.mtime(r.log_name)]
        if not a.all:
            runs = [r for r in runs if (r.heartbeat_age() or 1e9) < 24 * 3600 or r.state() in ("running", "stalled")]
    ref = open_run(a.vs, a.root, a.loss) if a.vs else None
    rows = [summary(r, ref, a.kill) for r in runs]
    if a.json:
        print(json.dumps([r[0] for r in rows], indent=1))
        return 0
    if not rows:
        print(f"no runs with metrics under {a.root} (use --all for the old ones)")
        return 0
    print(f"{'run':<24}{'state':<9}{'step/total':<13}{'%':>4}  {'s/step':>6}  {'ETA':>7}  {'loss':>8}  flags")
    for s, f, k in rows:
        flags = ",".join(sorted({x.code for x in f if x.level != "info"})) or "-"
        flags += "".join(f" KILL[{r.text}]" for r, st, _ in k if st == "hit")
        prog = "%d/%s" % (s["step"], s["total"] or "?")
        pct = "-" if s["pct"] is None else "%.0f" % s["pct"]
        sp = "-" if not s["s_per_step"] else "%.2f" % s["s_per_step"]
        eta = fmt_dur(s["eta_s"]) if s["state"] == "running" else "-"
        loss = "-" if s["loss"] is None else "%.4g" % s["loss"]
        print(f"{s['run']:<24}{s['state']:<9}{prog:<13}{pct:>4}  {sp:>6}  {eta:>7}  {loss:>8}  {flags}")
    if len(rows) == 1:
        return cmd_detail(runs[0], rows[0])
    return 0


def cmd_detail(run, row):
    s, findings, kills = row
    print()
    for f in findings:
        print(f"  [{f.level:<5}] {f.code}: {f.text}")
    for r, st, v in kills:
        print(f"  [kill ] {r.text}: {st}" + (f" (value {v:.4g})" if v is not None else ""))
    if run.evals:
        print("  last eval step %d: %s" % (run.evals[-1]["step"], ", ".join(
            f"{k[4:]} {v:.4g}" for k, v in run.evals[-1].items() if k.startswith("val.") and "last" not in k)))
    if run.restarts:
        print(f"  the stream was resumed {run.restarts} time(s); the later attempt of each step is used")
    return 0


def cmd_health(a):
    run = open_run(a.runs[0], a.root, a.loss)
    ref = open_run(a.vs, a.root, a.loss) if a.vs else None
    s, f, k = summary(run, ref, a.kill)
    print(f"{run.name}: {s['state']}, step {s['step']}/{s['total'] or '?'}")
    cmd_detail(run, (s, f, k))
    if not f and not k:
        print("  nothing to report")
    return 1 if any(x.level == "alert" for x in f) or any(st == "hit" for _, st, _ in k) else 0


def cmd_curves(a):
    from .curves import default_keys, plot
    runs = [open_run(x, a.root, a.loss) for x in a.runs]
    keys = a.keys or default_keys(runs[0])
    out = a.out or os.path.join(runs[0].dir, "curves.png")
    print(plot(runs, keys, out, a.smooth))
    return 0


def cmd_stop(a):
    run = open_run(a.runs[0], a.root, a.loss)
    print(f"stop requested for {run.name}: {control.request_stop(run.dir, a.reason)}")
    print("the trainer sees it at the next step boundary, saves a checkpoint and exits with code %d (runs of an older snapshot ignore it)"
          % control.EXIT_STOPPED)
    return 0


def _event(run, text, kind="info"):
    print(f"[{time.strftime('%H:%M:%S')}] {run.name}: {text}", flush=True)
    try:
        with open(run.p("trainsuite_events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(t=time.time(), kind=kind, text=text)) + "\n")
    except OSError:
        pass


def cmd_watch(a):
    """one line per CHANGE (state, finding, kill rule, new eval); returns when every run is finished"""
    runs = {x: Run(resolve(x, a.root), a.loss) for x in a.runs}          # a run that has not started yet may be watched (state `new`)
    ref = open_run(a.vs, a.root, a.loss) if a.vs else None
    seen = {}
    stopped_by = set()
    t_begin = time.time()
    while runs:
        for key in list(runs):
            run = runs[key] = Run(runs[key].dir, a.loss)
            s, findings, kills = summary(run, ref, a.kill)
            snap = seen.get(key, {"state": None, "find": set(), "kill": {}, "evals": 0})
            if s["state"] != snap["state"]:
                _event(run, f"{s['state']} at step {s['step']}/{s['total'] or '?'}", "state")
            for f in findings:
                if (f.level, f.code) not in snap["find"] and f.level != "info":
                    _event(run, f"{f.level} {f.code}: {f.text}", f.level)
            for r, st, v in kills:
                if snap["kill"].get(r.text) != st and st != "pending":
                    _event(run, f"kill rule {r.text!r}: {st} (value {v:.4g})", "kill" if st == "hit" else "info")
                if st == "hit" and a.enforce and r.text not in stopped_by and s["state"] == "running":
                    stopped_by.add(r.text)
                    control.request_stop(run.dir, f"kill rule {r.text}")
                    _event(run, f"STOP requested by kill rule {r.text!r}", "stop")
            if len(run.evals) > snap["evals"]:
                e = run.evals[-1]
                _event(run, f"eval at step {int(e['step'])}: " + ", ".join(
                    f"{k[4:]} {v:.4g}" for k, v in e.items() if k.startswith("val.") and "last" not in k), "eval")
            seen[key] = {"state": s["state"], "find": {(f.level, f.code) for f in findings}, "kill": {r.text: st for r, st, _ in kills},
                         "evals": len(run.evals)}
            if s["state"] in TERMINAL:
                del runs[key]
            elif s["state"] == "new" and time.time() - t_begin > 60 * a.give_up:
                _event(run, f"never started within {a.give_up:g} min, no longer watched", "state")
                del runs[key]
        if a.once or not runs:
            break
        time.sleep(a.every)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trainsuite", description=__doc__)
    ap.add_argument("--root", default="runs", help="folder of the run directories (default: runs)")
    ap.add_argument("--loss", default="total", help="key of the optimised loss in the metrics stream (default: total)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, runs="+", **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("runs", nargs=runs, help="run directories or names under --root")
        p.add_argument("--vs", help="reference run to compare the loss with at the same step")
        p.add_argument("--kill", action="append", default=[], help='kill rule, e.g. "group.lines@1000 > 1.48" (repeatable)')
        p.set_defaults(fn=fn)
        return p
    p = add("status", cmd_status, "*", help="table of runs (one run: details)")
    p.add_argument("--all", action="store_true", help="also runs without activity in the last 24 h")
    p.add_argument("--json", action="store_true")
    add("health", cmd_health, 1, help="health checks and kill rules of one run (exit 1 on alert or kill hit)")
    p = add("curves", cmd_curves, "+", help="loss curves of one or more runs as a png")
    p.add_argument("--keys", nargs="+")
    p.add_argument("--smooth", type=int, default=10)
    p.add_argument("--out")
    p = add("stop", cmd_stop, 1, help="ask the trainer to checkpoint and stop")
    p.add_argument("--reason", default="stopped by hand")
    p = add("watch", cmd_watch, "+", help="print one line per change until all runs are finished (for a Monitor)")
    p.add_argument("--every", type=float, default=60.0, help="seconds between looks")
    p.add_argument("--enforce", action="store_true", help="a kill-rule hit requests a stop (default: only report)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--give-up", type=float, default=180.0, help="minutes to wait for a run that has not started (default 180)")
    sub.add_parser("selftest", help="check the suite on synthetic runs").set_defaults(fn=None)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        from . import selftest
        return selftest.main()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
