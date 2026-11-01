"""Command line of the training suite: python -m trainsuite <status|health|curves|stop|resume|watch|early|selftest> ... (run from autoresearch/)."""
import argparse
import json
import os
import re
import sys
import time

from . import control, health
from .health import Finding
from .kill import load_rules
from .runs import Run, fmt_dur

TERMINAL = ("done", "failed", "stopped", "ended")
RETRY_GRACE_S = 300       # watch: a run that fails while watched may be restarted by the launcher (a CUDA retry after 15 s): wait this long
LATCH_POLLS = 5          # watch: a finding that went away must stay away this many looks before it can be reported again


def resolve(arg, root):
    return arg if os.path.isdir(arg) else os.path.join(root, arg)


def open_run(arg, root, loss):
    path = resolve(arg, root)
    if not os.path.isdir(path):
        raise SystemExit(f"no run directory {arg!r} (looked in {path})")
    return Run(path, loss)


def split_ref(arg):
    """'run' or 'run:reference' -> (run, reference or None): a reference of its own for this run"""
    skip = 2 if len(arg) > 2 and arg[1] == ":" and arg[0].isalpha() and arg[2] in "/" + chr(92) else 0     # a Windows drive letter is no separator
    name, _, ref = arg[skip:].partition(":")
    return arg[:skip] + name, ref or None


def summary(run, ref=None, extra_rules=(), since_step=None):
    findings = health.check(run, ref, since_step)
    rules, problems = load_rules(run, extra_rules)
    findings += [Finding("alert", "kill_rule", p) for p in problems]
    kills = [(r, *r.evaluate(run)) for r in rules]
    findings += [Finding("warn", "kill_key", f"kill rule {r.text!r}: the run passed step {r.step} and never logged {r.key!r}")
                 for r, st, _ in kills if st == "unknown"]
    loss = run.last(run.loss_key)
    return dict(run=run.name, state=run.state(), step=run.step, total=run.total,
                pct=100.0 * run.step / run.total if run.total else None, s_per_step=run.s_per_step(), eta_s=run.eta_s(), loss=loss,
                restarts=run.restarts, findings=[f._asdict() for f in findings],
                kill=[dict(rule=r.text, status=s, value=v) for r, s, v in kills]), findings, kills


def cmd_status(a):
    if a.runs:
        specs = [split_ref(x) for x in a.runs]
        runs = [(open_run(n, a.root, a.loss), r) for n, r in specs]
    else:
        names = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d))) if os.path.isdir(a.root) else []
        runs = [(Run(os.path.join(a.root, d), a.loss), None) for d in names]
        recent = lambda r: (r.heartbeat_age() or 1e9) < 24 * 3600

        def shown(r):       # a run that crashed in its imports or is still starting has a text log and no row yet: it is the news, not clutter (an old text-only folder is clutter)
            if a.all:
                return r.train or r.mtime(r.log_name) or r.mtime(r.text_name)
            return r.train or (r.mtime(r.text_name) and recent(r))
        runs = [x for x in runs if shown(x[0])]
        if not a.all:
            runs = [x for x in runs if recent(x[0]) or x[0].state() in ("running", "stalled")]
    rows = []
    for run, ref_name in runs:
        ref = open_run(ref_name or a.vs, a.root, a.loss) if (ref_name or a.vs) else None
        rows.append(summary(run, ref, a.kill))
    if a.json:
        print(json.dumps([r[0] for r in rows], indent=1))
        return 0
    if not rows:
        print(f"no runs with metrics or a text log under {a.root} (use --all for the old ones)")
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
        return cmd_detail(runs[0][0], rows[0])
    return 0


def events_of(run, n=5):
    """the last n events that `watch` wrote for this run (trainsuite_events.jsonl; a damaged line is skipped)"""
    out = []
    try:
        with open(run.p("trainsuite_events.jsonl"), encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if isinstance(e, dict):
                    out.append(e)
    except OSError:
        return []
    return out[-n:]


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
    ev = events_of(run, 5)
    if ev:
        print("  last events of `watch` (trainsuite_events.jsonl):")
        for e in ev:
            t_ = e.get("t") if isinstance(e.get("t"), (int, float)) else 0
            print(f"    [{time.strftime('%m-%d %H:%M:%S', time.localtime(t_))}] {e.get('kind', 'info')}: {e.get('text', '')}")
    if not run.timed() and run.train:
        print("  rows have no time `t` (trainer older than 2026-10-31): speed is the trainer's average, no slowdown check")
    return 0


def cmd_health(a):
    name, ref_name = split_ref(a.runs[0])
    run = open_run(name, a.root, a.loss)
    ref = open_run(ref_name or a.vs, a.root, a.loss) if (ref_name or a.vs) else None
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
    ok = control.honours_stop(run.dir)
    if ok is False:
        print(f"NOT requested: the code snapshot of {run.name} (code/train.py) has no stop check, it would ignore CONTROL. "
              "Stopping it is a decision about the process, not a file.")
        return 2
    print(f"stop requested for {run.name}: {control.request_stop(run.dir, a.reason)}")
    print("the trainer sees it at the next step boundary, saves a checkpoint, writes STOPPED and exits with code %d" % control.EXIT_STOPPED
          + ("" if ok else " (could not look into code/train.py: unverified)"))
    return 0


def cmd_resume(a):
    run = open_run(a.runs[0], a.root, a.loss)
    moved = control.clear_stale(run.dir)
    print(f"{run.name}: " + (f"moved {', '.join(moved)} aside (*.old)" if moved else "no CONTROL / STOPPED to clear")
          + "; start the trainer again with the same command: it resumes from the last checkpoint")
    return 0


def cmd_early(a):
    """advice only (exit 0; 2 for a usage error, also a missing direction or band, or a calibration that cannot be used): `early <run> --vs <control>`
    or `early --calibrate [root]`, see early.py"""
    try:
        return _early(a)
    except (ValueError, OSError) as e:
        print(f"early: {e}", file=sys.stderr)
        return 2


def _early(a):
    from . import early
    if a.calibrate is not None:
        explicit = []
        for x in a.pair:
            v, c = split_ref(x)
            if not c:
                print("usage: --pair variant:control", file=sys.stderr)
                return 2
            explicit.append((v, c))
        if a.band is None:
            raise ValueError("--band is required: the seed noise in the units of the key (a loss is not on the 0..1 scale of the lane S)")
        pairs = early.find_pairs(a.calibrate or a.root, a.loss, explicit, a.group)
        cal = early.calibrate(pairs, a.key, a.lower, a.better, band=a.band)
        print(early.describe(cal))
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(cal, f, indent=1)
            print(f"written: {a.out}")
        return 0
    if len(a.runs) != 1 or not a.vs:
        print("usage: early <run> --vs <control run> [--key K] [--at F]   |   early --calibrate [runs root]", file=sys.stderr)
        return 2
    cal = None
    if a.calib:
        with open(a.calib, encoding="utf-8") as f:
            cal = json.load(f)
    var, ctl = open_run(a.runs[0], a.root, a.loss), open_run(a.vs, a.root, a.loss)
    key = (a.key or ["val.total"])[0]
    better = early.direction(key, a.better, a.lower)
    text, info = early.advise(var, ctl, key, a.at, better, a.band, cal, a.min_rows, a.min_agree)
    print(f"{var.name} vs {ctl.name}: {text}")
    if a.as_rule:
        line = early.as_rule(ctl, info)
        print(line if line else f"# no rule: only the verdict 'consider stopping' gives one (this one: {info['verdict']})")
    return 0


def _event(run, text, kind="info"):
    print(f"[{time.strftime('%H:%M:%S')}] {run.name}: {text}", flush=True)
    try:
        with open(run.p("trainsuite_events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(t=time.time(), kind=kind, text=text)) + "\n")
    except OSError:
        pass


def cmd_watch(a):
    """one line per CHANGE (state, finding, kill rule, new eval); returns when every run is finished. A run that was not running when the
    watch began (not started, failed, stopped, ended) is waited for up to --give-up minutes: it may be about to be restarted."""
    specs = [split_ref(x) for x in a.runs]
    runs = {n: Run(resolve(n, a.root), a.loss) for n, _ in specs}
    refs = {n: (open_run(r or a.vs, a.root, a.loss) if (r or a.vs) else None) for n, r in specs}
    snaps = {n: dict(state=None, find={}, kill={}, evals=0, step=None, restarts=None, since=time.time(), initial=True, req=None, rewrote=False) for n in runs}
    stopped_by = set()
    while runs:
        for key in list(runs):
            run = runs[key] = Run(runs[key].dir, a.loss)
            snap = snaps[key]
            cursor = snap["step"]
            if snap["restarts"] is not None and (run.restarts > snap["restarts"] or run.step < snap["step"]):
                # resumed from an earlier checkpoint since the last look (the run may be past the old step again by now): the rows from the resume step on are new
                cursor = run.resumes[-1] - 1 if run.resumes else None
            s, findings, kills = summary(run, refs[key], a.kill, cursor)
            now = time.time()
            if s["state"] != snap["state"]:
                if snap["state"] is not None:
                    snap["initial"] = False
                snap["since"] = now
                _event(run, f"{s['state']} at step {s['step']}/{s['total'] or '?'}", "state")
            here = {(f.code, re.sub(r"[\d.]+", "#", f.text)[:60]): f for f in findings if f.level != "info"}     # one latch per finding (numbers in the text change every poll)
            for code, f in here.items():
                if code not in snap["find"]:
                    _event(run, f"{f.level} {f.code}: {f.text}", f.level)
                snap["find"][code] = 0
            for code in [c for c in snap["find"] if c not in here]:
                snap["find"][code] += 1
                if snap["find"][code] >= LATCH_POLLS:
                    del snap["find"][code]
            for r, st, v in kills:
                if snap["kill"].get(r.text) != st and st != "pending":
                    _event(run, f"kill rule {r.text!r}: {st}" + ("" if v is None else f" (value {v:.4g})"), "kill" if st == "hit" else "info")
                if st == "hit" and a.enforce and (key, r.text) not in stopped_by and s["state"] == "running":
                    stopped_by.add((key, r.text))
                    ok = control.honours_stop(run.dir)
                    if ok is False:
                        _event(run, f"kill rule {r.text!r} hit, but this run's code snapshot has no stop check: NOT stopped", "alert")
                    else:
                        control.request_stop(run.dir, f"kill rule {r.text}")
                        snap["req"] = now
                        _event(run, f"STOP requested by kill rule {r.text!r}" + ("" if ok else " (trainer not verified)"), "stop")
            if snap["req"] and s["state"] == "running" and now - snap["req"] > a.honour_wait:
                if not snap["rewrote"]:
                    control.request_stop(run.dir, "kill rule (written again)")
                    snap["rewrote"], snap["req"] = True, now
                    _event(run, f"stop NOT honoured after {a.honour_wait:g} s (busy, or CONTROL written before the trainer started): written again", "alert")
                else:
                    snap["req"] = None
                    _event(run, "stop NOT honoured a second time: the request is given up, stopping is now yours", "alert")
            if len(run.evals) > snap["evals"]:
                e = run.evals[-1]
                _event(run, f"eval at step {int(e['step'])}: " + ", ".join(
                    f"{k[4:]} {v:.4g}" for k, v in e.items() if k.startswith("val.") and "last" not in k), "eval")
            snap.update(state=s["state"], kill={r.text: st for r, st, _ in kills}, evals=len(run.evals), step=run.step, restarts=run.restarts)
            waiting = s["state"] in ("new", "stalled") or (s["state"] in TERMINAL and s["state"] != "done" and snap["initial"])
            retrying = s["state"] == "failed" and not snap["initial"] and now - snap["since"] <= RETRY_GRACE_S
            if s["state"] == "done" or (s["state"] in TERMINAL and not waiting and not retrying):
                del runs[key]
            elif waiting and now - snap["since"] > 60 * a.give_up:
                _event(run, f"still {s['state']} after {a.give_up:g} min, no longer watched", "state")
                del runs[key]
        if a.once or not runs:
            break
        time.sleep(a.every)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trainsuite", description=__doc__)
    ap.add_argument("--root", default="runs", help="folder of the run directories (default: ./runs; from autoresearch/ use ../<problem>/runs)")
    ap.add_argument("--loss", default="total", help="key of the optimised loss in the metrics stream (default: total)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, runs="+", **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("runs", nargs=runs, help="run directories or names under --root; `run:reference` gives one run its own reference")
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
    p = add("stop", cmd_stop, 1, help="ask the trainer to checkpoint and stop (refused when its code snapshot has no stop check)")
    p.add_argument("--reason", default="stopped by hand")
    add("resume", cmd_resume, 1, help="move CONTROL / STOPPED of an earlier stop aside (the trainer does it itself at its start)")
    p = add("watch", cmd_watch, "+", help="print one line per change until all runs are finished (for a Monitor)")
    p.add_argument("--every", type=float, default=60.0, help="seconds between looks")
    p.add_argument("--enforce", action="store_true", help="a kill-rule hit requests a stop (default: only report)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--give-up", type=float, default=180.0, help="minutes to wait for a run that has not started / restarted (default 180)")
    p.add_argument("--honour-wait", type=float, default=300.0, help="seconds to wait for a requested stop before writing it once more (default 300)")
    p = sub.add_parser("early", help="advice from learning curves: gap to a control run vs how well that gap predicted the final one (never stops anything)")
    p.add_argument("runs", nargs="*", help="the run to judge")
    p.add_argument("--vs", help="the control run")
    p.add_argument("--calibrate", nargs="?", const="", metavar="ROOT", help="calibrate on the finished pairs of a runs root (default: --root)")
    p.add_argument("--calib", help="calibration file written by --calibrate --out")
    p.add_argument("--out", help="write the calibration here (json)")
    p.add_argument("--key", action="append", help="metric, e.g. val.total (repeatable for --calibrate; default: every val.* key / val.total)")
    p.add_argument("--better", choices=("higher", "lower"), help="direction of the metric: REQUIRED unless the key is named by --lower (no default: the bev_fusion val.* are losses, a score is higher)")
    p.add_argument("--lower", action="append", default=[], help="a key that is lower-is-better (repeatable; overrides --better for that key)")
    p.add_argument("--band", type=float, help="seed-noise band in the key's units: REQUIRED for --calibrate and without --calib (with --calib it is the calibration's band; a different value is refused)")
    p.add_argument("--at", type=float, help="fraction of the steps to compare at (default: as far as the run has got)")
    p.add_argument("--min-rows", type=int, default=3, help="rows of the key needed before a gap is judged (default 3)")
    p.add_argument("--min-agree", type=float, default=0.8, help="95 %% lower bound of the sign agreement needed for 'consider stopping' (default 0.8; also 15 pairs from 8 runs)")
    p.add_argument("--as-rule", action="store_true", help="also print a kill.txt line for review (never written)")
    p.add_argument("--pair", action="append", default=[], help="--calibrate: variant:control pair (repeatable; default: every two finished runs with the same name prefix)")
    p.add_argument("--group", choices=("prefix", "all"), default="prefix", help="--calibrate automatic pairing: same name prefix before the first _ (default) or all")
    p.set_defaults(fn=cmd_early)
    sub.add_parser("selftest", help="check the suite on synthetic runs").set_defaults(fn=None)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        from . import selftest
        return selftest.main()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
