"""Command line of the training suite (run from autoresearch/): python -m trainsuite <command> ...

The commands of the interface (class Training, api.py; `methods` lists them): check, run, list and Training.status / stop / resume / plot / compare. The older commands below keep
their meaning: status, health, curves, stop, resume, watch, early, compose, start, queue (the queue: show / add --id / serve / hold / resume / drop; not part of the class), selftest."""
import argparse
import json
import os
import re
import sys
import time

from . import control, modules
from .runs import Run, fmt_dur
from .view import speed_row, summary

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


def clock(t):
    """'HH:MM' of a unix time, with the day when it is not today; '-' for none"""
    if not t:
        return "-"
    try:
        lt, now = time.localtime(t), time.localtime()
        return time.strftime("%H:%M", lt) if lt[:3] == now[:3] else time.strftime("%a %H:%M", lt)
    except (OverflowError, OSError, ValueError):               # a step total of 10**30 gives a time no clock has
        return "-"


def split_ref(arg):
    """'run' or 'run:reference' -> (run, reference or None): a reference of its own for this run"""
    skip = 2 if len(arg) > 2 and arg[1] == ":" and arg[0].isalpha() and arg[2] in "/" + chr(92) else 0     # a Windows drive letter is no separator
    name, _, ref = arg[skip:].partition(":")
    return arg[:skip] + name, ref or None


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
    print(f"{'run':<24}{'state':<9}{'step/total':<13}{'ready':>5}  {'s/step':>6}  {'ETA':>7}  {'last step':>9}  {'loss':>8}  {'grad (x before)':>15}  {'skip':>4}  flags")
    for s, f, k in rows:
        flags = ",".join(sorted({x.code for x in f if x.level != "info"})) or "-"
        flags += "".join(f" KILL[{r.text}]" for r, st, _ in k if st == "hit")
        prog = "%d/%s" % (s["step"], s["total"] or "?")
        pct = "-" if s["ready_pct"] is None else "%.0f%%" % s["ready_pct"]
        sp = "-" if not s["s_per_step"] else "%.2f" % s["s_per_step"]
        eta = fmt_dur(s["eta_s"]) if s["state"] == "running" else "-"
        at = clock(s["ready_at"])
        loss = "-" if s["loss"] is None else "%.4g" % s["loss"]
        g = s["grad"]
        grad = "-" if not g else "non-finite" if g["now"] is None else "%.3g" % g["now"] + (" (x%.1f)" % g["ratio"] if g["ratio"] else "")
        print(f"{s['run']:<24}{s['state']:<9}{prog:<13}{pct:>5}  {sp:>6}  {eta:>7}  {at:>9}  {loss:>8}  {grad:>15}  {s['skipped'] or '-':>4}  {flags}")
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


def _num(x, fmt="%.3g"):
    return "-" if x is None else fmt % x


def cmd_detail(run, row):
    s, findings, kills = row
    print()
    if s["ready_pct"] is not None:
        print(f"  phase {s['phase']}, ready {s['ready_pct']:.0f} %" + (f", ETA {fmt_dur(s['eta_s'])} (last step about {clock(s['ready_at'])})" if s["ready_at"] else ""))
    else:
        print(f"  phase {s['phase']}, the number of steps is not known (no `steps` in config.json): no percentage")
    for f in findings:
        print(f"  [{f.level:<5}] {f.code}: {f.text}")
    for r, st, v in kills:
        print(f"  [kill ] {r.text}: {st}" + (f" (value {v:.4g})" if v is not None else ""))
    if run.evals:
        print("  last eval step %d: %s" % (run.evals[-1]["step"], ", ".join(
            f"{k[4:]} {v:.4g}" for k, v in run.evals[-1].items() if k.startswith("val.") and "last" not in k)))
    if run.restarts:
        print(f"  the stream was resumed {run.restarts} time(s); the later attempt of each step is used")
    if s.get("data"):
        print(f"  data: {s['data']}")
    g = s.get("grad")
    if g:
        print(f"  gradient norm: first 10 rows {_num(g['first'])}, now {_num(g['now'])} (median of the finite values in the last {modules.WINDOW} rows"
              + (f"; x{g['ratio']:.1f} against the rows before" if g["ratio"] else "") + f"), peak {_num(g['peak'])}" + ("" if g["peak_step"] is None else f" at step {g['peak_step']}")
              + (f"; {g['nonfinite_recent']} of those rows are NOT finite" if g["nonfinite_recent"] else ""))
    skipped = run.skipped_at()
    if skipped:
        n_skip, in_row = run.skipped_steps()
        shown = ", ".join(str(x) for x in skipped[:8]) + (f", ... (last {skipped[-1]})" if len(skipped) > 8 else "")
        print(f"  non-finite: {n_skip} step(s) skipped at {shown} (up to {in_row} in a row)")
    groups = [k for k in run.keys() if k.startswith("group.")]
    if groups:
        print("  loss groups (mean of the first 10 logged values -> the last 10):")
        for k in groups:
            v = [x for _, x in run.series(k)]
            print(f"    {k:<28}{_num(sum(v[:10]) / len(v[:10]))} -> {_num(sum(v[-10:]) / len(v[-10:]))}")
    mt = modules.module_table(run)
    if mt:
        print(f"  modules (median of the last {modules.WINDOW} rows; grad x = against the rows before; update = |dw| / |w| per step):")
        for m in mt:
            x = "-" if m["grad"] is None or not m["grad_base"] else "%.1f" % (m["grad"] / m["grad_base"])
            print(f"    {m['module']:<24}grad {_num(m['grad']):>9}  x{x:>5}  update {_num(m['upd']):>9}" + ("   (frozen on purpose)" if m["frozen"] else ""))
    tm = modules.timing(run)
    if tm:
        print(f"  time: {tm['t_step']:.2f} s per step, {tm['t_data']:.2f} s of it waiting for the data ({100 * tm['share']:.0f} %)")
    inc = modules.incidents(run)
    if inc:
        print(f"  incidents ({len(inc)}; the dump has the batch ids, replay it with the trainer's debug mode):")
        for i in inc[-3:]:
            print(f"    step {i['step']}: {i['kind']} ({i['file']})")
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
    try:
        print(plot(runs, keys, out, a.smooth))
    except ImportError as e:
        print(f"curves: plot needs matplotlib: {e}", file=sys.stderr)
        return 2
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
    snaps = {n: dict(state=None, find={}, kill={}, evals=0, step=None, restarts=None, mile=None, since=time.time(), initial=True, req=None, rewrote=False) for n in runs}
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
            if s["ready_pct"] is not None and s["state"] == "running":              # one line per 10 % of readiness: the progress of a run that is otherwise silent
                mile = int(s["ready_pct"] // 10) * 10
                if snap["mile"] is not None and mile > snap["mile"]:
                    _event(run, f"ready {mile} %: step {s['step']}/{s['total']}" + (f", ETA {fmt_dur(s['eta_s'])}, last step about {clock(s['ready_at'])}" if s["ready_at"] else ""), "progress")
                snap["mile"] = mile
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


def cmd_speed(a):
    """the runs side by side (the first is the baseline): seconds per step and the speed-up, the share of a step that waits for the data, the GPU busy share, the peak GPU memory
    and the loss at the last step the runs share (an optimisation that changes the loss curve is not a free speed-up)"""
    runs = [open_run(x, a.root, a.loss) for x in a.runs]
    rows = [speed_row(r, a.skip) for r in runs]
    common = min(r.step for r in runs)
    base = rows[0]["s_step"]
    print(f"{'run':<24}{'steps':>6}  {'s/step':>7}  {'speed-up':>8}  {'data wait':>9}  {'GPU busy':>8}  {'peak GB':>7}  {'loss@' + str(common):>12}  {'vs base':>8}")
    b_loss = runs[0].window_mean(runs[0].loss_key, common)
    for r, x in zip(runs, rows):
        l = r.window_mean(r.loss_key, common)
        print(f"{r.name:<24}{x['step']:>6}  {_num(x['s_step'], '%.2f'):>7}  " + (f"x{base / x['s_step']:.2f}" if base and x["s_step"] else "-").rjust(8)
              + f"  {('%.0f %%' % (100 * x['data_share'])) if x['data_share'] is not None else '-':>9}  {('%d %%' % x['gpu_busy']) if x['gpu_busy'] is not None else '-':>8}"
              + f"  {_num(x['gpu_gb'], '%.1f'):>7}  {_num(l, '%.4g'):>12}  " + (f"{100 * (l - b_loss) / abs(b_loss):+.1f} %" if (l is not None and b_loss) else "-").rjust(8))
    return 0


def cmd_queue(a):
    """who holds the GPU slots, who waits in the line (first goes first), what the RAM gate says, and which runs lie on hold: the view of run.py's queue"""
    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    sys.path.insert(0, tools)
    if a.project:
        os.environ["AR_PROJECT"] = a.project
    import project as P
    import run as R
    cfg, root = P.cfg(), P.root()
    records = P.path("run_records")
    base = R.slot_dir(root, cfg, records)
    n = int(cfg.get("gpu_slots", 2))
    settle = R.settle_s(cfg)
    loading = dict(R.GpuSlot(records, n, "queue-view", records, base=base).settling(settle))
    print(f"GPU slots ({n}, {os.path.relpath(base, root)}):")
    for i in range(n):
        s = os.path.join(base, str(i))
        o = R.record(s) if os.path.isdir(s) else {}
        if o and R.pid_alive(o.get("pid")):
            young = f", loading: {loading[o.get('id', '?')]} of {settle:g} s settle time, the next launcher waits" if o.get("id", "?") in loading else ""
            print(f"  slot {i}: {o.get('id', '?')}  (pid {o.get('pid')}, since {o.get('taken', '?')}{young})")
        else:
            print(f"  slot {i}: free" + ("  (a stale owner is reclaimed by the next launcher)" if o else ""))
    import machine_lock
    lk = machine_lock.current()
    if lk:
        what = "EXCLUSIVE: no other owner starts a run" if lk.get("exclusive") is True else "a reservation: it holds nobody"
        print(f"machine lock: {lk['owner']} until {lk['end']} ({what})" + (f", {lk['note']}" if lk.get("note") else ""))
    else:
        print("machine lock: free")
    qd = os.path.join(base, ".queue")
    line = []
    for name in sorted(os.listdir(qd)) if os.path.isdir(qd) else []:
        try:
            with open(os.path.join(qd, name), encoding="utf-8") as f:
                o = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(o, dict) and R.pid_alive(o.get("pid")):
            line.append(o)
    print("queue (the first goes first):" + ("" if line else " empty"))
    for i, o in enumerate(line, 1):
        back = f", held back by the machine lock of {lk['owner']}: it does not hold up the others" if lk and R.blocked_by_lock(o.get("owner", "")) else ""
        print(f"  {i}. {o.get('id', '?')}  (waiting since {o.get('since', '?')}, pid {o.get('pid')}{back})")
    need, avail = int(cfg.get("ram_mb", 0)), R.mem_available_mb()
    if avail is None:
        print("RAM: /proc/meminfo is not readable here: no gate")
    elif need:
        print(f"RAM: {avail} MB available, the gate asks for {need} MB on {R.RAM_CHECKS} looks in a row: " + ("a run could start" if avail >= need else "a run waits"))
    else:
        print(f"RAM: {avail} MB available, no gate (project.json ram_mb is 0)")
    from . import compose as C
    try:
        gate = C.gpu_gate(cfg.get("trainsuite") or {})
    except C.ComposeError as e:
        gate = 0
        print(f"GPU memory gate: {e}: NO queue item can start until project.json is corrected")
    if gate:
        used = R.gpu_used_mb()
        ok_ = used is not None and used < gate
        print(f"GPU memory: {'unreadable (nvidia-smi)' if used is None else str(used) + ' MB in use'}; a training starts only below {gate} MB: " + ("a training could start" if ok_ else "a training waits"))
    held = []
    for rid in sorted(os.listdir(records)) if os.path.isdir(records) else []:
        h = os.path.join(records, rid, "HOLD")
        if os.path.isfile(h):
            try:
                with open(h, encoding="utf-8") as f:
                    held.append((rid, json.load(f).get("when", "?")))
            except (OSError, ValueError):
                held.append((rid, "?"))
    if held:
        print("on hold (only the person's words clear a hold: hold.py clear ID --words ...):")
        for rid, when in held:
            print(f"  {rid}  (since {when})")
    from . import trainqueue as TQ
    TQ.show(TQ.context())
    return 0


def _queue(a):
    """(the queue module, the project's context); a QueueError becomes a message and exit code 2 in `queue_cmd`"""
    from . import trainqueue as TQ
    return TQ, TQ.context(getattr(a, "project", None))


def queue_cmd(fn):
    def run(a):
        from . import trainqueue as TQ
        try:
            return fn(a)
        except TQ.QueueError as e:
            print(f"queue: {e}", file=sys.stderr)
            return 2
    return run


@queue_cmd
def cmd_queue_add(a):
    """compose a run like `start`, run the project's checks, and put it in the queue; the runner starts it (also after the launcher of an earlier item died)"""
    import subprocess
    from . import compose as C
    TQ, c = _queue(a)
    if a.id is None:
        raise TQ.QueueError('--id is required; a job file is put in the queue with `queue-add JOB.json` (api.py)')
    if not c.block:
        raise TQ.QueueError('project.json has no "trainsuite" block')
    try:
        plan = C.plan_start(c.root, c.block, a.base, a.id, a.preset, a.set, a.hyp, a.pred, a.rule, a.no_card, a.run_arg, TQ.TOOLS, c.records)
        C.write(plan["config"], plan["preview"])
    except C.ComposeError as e:
        raise TQ.QueueError(str(e))
    print(f"{a.base} -> {os.path.relpath(plan['composed'], c.root)}: {len(plan['changes'])} change(s)")
    print("\n".join(C.describe(plan["changes"], plan["base_keys"])))
    for n in plan["notes"]:
        print(f"  note: {n}")
    if plan["held"]:
        raise TQ.QueueError(f"run id {a.id} is ON HOLD (hold.py show): `queue resume {a.id} --words ...` clears it")
    checks = ([] if a.no_check or not plan["check_preview"] else [("check", plan["check_preview"])]) + [(f"prepare {p['name']}", p["check_preview"]) for p in plan["prepare"]]
    for name, argv in checks:                                    # a missing GT store is built first (start --prepare); a queued run would only fail when its turn comes
        r = subprocess.run(argv, cwd=c.root, capture_output=True, text=True)
        print(f"{name}: {' '.join(argv)} -> exit {r.returncode}")
        if r.returncode:
            print("\n".join("  " + x for x in (r.stdout + r.stderr).strip().splitlines()[-10:]))
            raise TQ.QueueError(f"the {name} found problems; nothing queued")
    TQ.add(c, plan, a.id, [x for x in a.after.split(",") if x], a.priority, a.note)
    pid = TQ.ensure_runner(c) if not a.no_runner else 0
    print(f"queued {a.id}" + (f"; the runner is running (pid {pid})" if pid else "; no runner started (queue serve)"))
    TQ.show(c)
    return 0


@queue_cmd
def cmd_queue_serve(a):
    TQ, c = _queue(a)
    return TQ.serve(c, a.every, a.idle_exit, a.once)


@queue_cmd
def cmd_queue_hold(a):
    """stop what runs (checkpoint), keep the rest from starting; --all also pauses the runner"""
    TQ, c = _queue(a)
    if not a.ids and not a.all:
        raise TQ.QueueError("name the runs to hold, or --all")
    TQ.hold(c, a.ids, a.all, a.words, a.reason, a.wait)
    return 0


@queue_cmd
def cmd_queue_resume(a):
    """the person's words clear the holds; each run starts again from its checkpoint on the code of its first attempt"""
    TQ, c = _queue(a)
    if not a.ids and not a.all:
        raise TQ.QueueError("name the runs to resume, or --all")
    TQ.resume(c, a.ids, a.all, a.words, a.fresh_code, not a.no_runner)
    return 0


@queue_cmd
def cmd_queue_drop(a):
    TQ, c = _queue(a)
    TQ.drop(c, a.id)
    print(f"dropped {a.id} from the queue")
    return 0


def cmd_compose(a):
    from . import compose as C
    try:
        base = C.load(a.base)
        cfg, changes = C.compose(base, a.preset, a.set, a.presets_dir)
        print(f"{a.base}: {len(changes)} change(s)")
        print("\n".join(C.describe(changes, C.keys_of(base))))
        if a.out:
            C.write(cfg, a.out)
            print(f"written {a.out}")
    except C.ComposeError as e:
        print(f"compose: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_start(a):
    """compose the config, run the project's check, print the exact run.py command; with --go start it (detached). The same command again resumes the run."""
    import shlex
    import subprocess
    from . import compose as C
    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    sys.path.insert(0, tools)
    if a.project:
        os.environ["AR_PROJECT"] = a.project
    import project as P
    cfg, root = P.cfg(), P.root()
    block = cfg.get("trainsuite")
    if not block:
        print('start: project.json has no "trainsuite" block ({"presets", "out", "check", "train", "runs"}; see compose.py)', file=sys.stderr)
        return 2
    try:
        plan = C.plan_start(root, block, a.base, a.id, a.preset, a.set, a.hyp, a.pred, a.rule, a.no_card, a.run_arg, tools, os.path.join(root, cfg["run_records"]))
    except C.ComposeError as e:
        print(f"start: {e}", file=sys.stderr)
        return 2
    print(f"{a.base} -> {os.path.relpath(plan['composed'], root)}: {len(plan['changes'])} change(s)")
    print("\n".join(C.describe(plan["changes"], plan["base_keys"])))
    target, check = (plan["composed"], plan["check"]) if a.go else (plan["preview"], plan["check_preview"])      # a preview must not touch the file a running (or retrying) run reads
    if a.go and plan["held"]:                                    # run.py would refuse the training 3 s after a prepare job (hours of GT build) had started: refuse before any job
        print(f"start: run id {a.id} is ON HOLD (hold.py show); nothing started, no prepare job either")
        return 1
    if a.go:                                                     # the person's pause of the queue binds a direct start too (hard rule 6), as it does for Training.run
        from . import queue_api
        from . import trainqueue as TQ
        try:
            blocks = queue_api.queue_blocks(TQ.context(a.project), a.id)
        except TQ.QueueError as e:
            print(f"start: {e}", file=sys.stderr)
            return 2
        if blocks:
            print(f"start: {'; '.join(blocks)}; nothing started, no prepare job either")
            return 1
    try:
        C.write(plan["config"], target)
    except C.ComposeError as e:
        print(f"start: {e}", file=sys.stderr)
        return 2
    for n in plan["notes"]:
        print(f"  note: {n}")
    if check and not a.no_check:
        r = subprocess.run(check, cwd=root, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip().splitlines()
        print(f"check: {' '.join(check)} -> exit {r.returncode}")
        print("\n".join("  " + x for x in out[-15:]))
        if r.returncode:
            print("start: the check found problems; nothing started")
            return 1
    missing = []
    for pr in ([] if a.skip_prepare else plan["prepare"]):         # the data the run needs (the GT store of its version): present, or built first
        chk = pr["check"] if a.go else pr["check_preview"]
        r = subprocess.run(chk, cwd=root, capture_output=True, text=True)
        print(f"prepare {pr['name']}: {' '.join(chk)} -> exit {r.returncode} ({'present' if r.returncode == 0 else 'MISSING' if r.returncode == 1 else 'THE CHECK FAILED'})")
        print("\n".join("  " + x for x in (r.stdout + r.stderr).strip().splitlines()[-8:]))
        if r.returncode == 1:                                    # 0 present, 1 missing; anything else is the check failing: nothing is built on that
            missing.append(pr)
        elif r.returncode:
            print(f"start: the prepare check {pr['name']} failed (exit {r.returncode}, not 'missing'); nothing started")
            return 1
    cmd = lambda argv: f"cd {root} && AR_PROJECT={root} " + " ".join(shlex.quote(x) for x in argv)           # noqa: E731
    if missing and not a.prepare:
        print(f"start: {', '.join(m['name'] for m in missing)} missing. A build takes hours and belongs in the schedule: start with --prepare (the build becomes a run of its own and "
              "the training waits for it), or build it first:")
        for m in missing:
            print("  " + cmd(m["argv"]))
        return 1
    train_argv = plan["argv"]
    if missing:                                                  # the training waits for the builds (--after); a failed build blocks it instead of letting it start without its data
        i = train_argv.index("--")
        train_argv = train_argv[:i] + ["--after", ",".join(m["id"] for m in missing), "--wait-after", str(max(m["wait_min"] for m in missing))] + train_argv[i:]
    jobs = [(m["argv"], os.path.join(os.path.dirname(plan["log"]), m["id"].replace(".", "_") + ".launcher.log"), m["id"]) for m in missing] + [(train_argv, plan["log"], a.id)]
    for argv_, _, _ in jobs:
        print(cmd(argv_))
    if not a.go:
        print("(nothing started: add --go)")
        return 0
    runs = os.path.join(root, block.get("runs", "runs"))
    os.makedirs(os.path.dirname(plan["log"]), exist_ok=True)
    for argv_, log_, rid in jobs:
        with open(log_, "ab") as lf:
            proc = subprocess.Popen(argv_, cwd=root, env=dict(os.environ, AR_PROJECT=root), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=a.wait)                      # run.py refuses a held, running, finished or card-less id within a second: say so instead of "started"
        except subprocess.TimeoutExpired:
            rc = None
        if rc == 0 and rid != a.id:                             # a prepare job that is done within the wait (the data was there after all) is a success: the training goes on
            print(f"{rid} finished within {a.wait:g} s with exit 0")
            continue
        if rc is not None:
            with open(log_, encoding="utf-8", errors="replace") as lf:
                tail = lf.read().strip().splitlines()[-6:]
            print(f"the launcher of {rid} ended after {a.wait:g} s with exit {rc}: nothing runs" + (" and the later runs are not started" if rid != a.id else "") + "\n" + "\n".join("  " + x for x in tail))
            return 1
        print(f"started {rid}, pid {proc.pid}, launcher log {log_}")
    print(f"  watch it: python -m trainsuite --root {runs} watch {a.id.replace('.', '_')}")
    return 0


def main(argv=None):
    from . import api                                            # every method of the class Training is a command: `queue-add job.json`, `methods` lists them
    rc = api.CLI.dispatch(sys.argv[1:] if argv is None else list(argv))
    if rc is not None:
        return rc
    ap = argparse.ArgumentParser(prog="trainsuite", description=__doc__, epilog="The commands of the interface (check, run, queue-add, queue-list, ...): `python -m trainsuite methods`. "
                                 "They take --project and --json after the command; --root and --loss belong to the older commands above.")
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
    for name in ("curves", "plot"):                              # `plot` is the word of the interface (api.plot); `curves` the older one
        p = add(name, cmd_curves, "+", help="loss curves of one or more runs as a png")
        p.add_argument("--keys", nargs="+")
        p.add_argument("--smooth", type=int, default=10)
        p.add_argument("--out")
    for name in ("speed", "compare"):                            # `compare` is the word of the interface (api.compare); `speed` the older one
        p = add(name, cmd_speed, "+", help="runs side by side, the first is the baseline: s/step and speed-up, data wait, GPU busy, peak memory, loss at the last shared step")
        p.add_argument("--skip", type=int, default=10, help="logged rows to leave out as warm-up (default 10)")
    p = sub.add_parser("queue", help="the GPU slots, the line of waiting runs, the RAM gate, the holds and the queue items of the project; add / serve / hold / resume / drop (AR_PROJECT or --project)")
    p.add_argument("--project", help="project folder (default: AR_PROJECT)")
    p.set_defaults(fn=cmd_queue)
    qs = p.add_subparsers(dest="qcmd")

    def qsub(name, fn, **kw):
        s = qs.add_parser(name, **kw)
        s.add_argument("--project", default=argparse.SUPPRESS, help="project folder (default: AR_PROJECT)")
        s.set_defaults(fn=fn)
        return s
    s = qsub("add", cmd_queue_add, help="put a config with --id and options (like `start`) in the queue; the runner starts it when its turn comes (a job file: `queue-add JOB.json`)")
    s.add_argument("base", help="the network config")
    s.add_argument("--id", default=None, help="run id (required; the run folder is the id with . as _)")
    s.add_argument("--preset", action="append", default=[])
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--hyp"), s.add_argument("--pred"), s.add_argument("--rule")
    s.add_argument("--no-card", action="store_true", help="a diagnostic run (it never feeds a verdict)")
    s.add_argument("--no-check", action="store_true", help="skip the project's static check of the composed config")
    s.add_argument("--after", default="", help="run ids (comma separated) that must be done first; a failed or held one blocks this item")
    s.add_argument("--priority", type=int, default=5, help="1 goes first (default 5); equal priorities go in the order of adding")
    s.add_argument("--note", default="")
    s.add_argument("--run-arg", action="append", default=[], help="an argument for run.py itself (repeatable)")
    s.add_argument("--no-runner", action="store_true", help="do not start the runner")
    s = qsub("serve", cmd_queue_serve, help="the runner: start the next item, restart a run whose launcher died; ends when nothing is left (a detached `add` / `resume` starts it)")
    s.add_argument("--every", type=float, default=float(os.environ.get("TS_QUEUE_EVERY", "30")), help="seconds between looks (default 30, env TS_QUEUE_EVERY)")
    s.add_argument("--idle-exit", type=float, default=float(os.environ.get("TS_QUEUE_IDLE", "600")), help="seconds without open items before the runner ends (default 600, env TS_QUEUE_IDLE)")
    s.add_argument("--once", action="store_true", help="one look, then exit")
    s = qsub("hold", cmd_queue_hold, help="stop what runs (a checkpoint, exit 75), keep what waits from starting; --all pauses the runner. Needs the person's words")
    s.add_argument("ids", nargs="*")
    s.add_argument("--all", action="store_true", help="every run of the project that runs, waits or is queued, and the runner itself")
    s.add_argument("--words", default="", help="the person's own words that ordered it (required)")
    s.add_argument("--reason", default="")
    s.add_argument("--wait", type=float, default=180.0, help="seconds to wait for the trainers to stop")
    s = qsub("resume", cmd_queue_resume, help="the person's words clear the holds; each stopped run continues from its checkpoint on the code of its first attempt")
    s.add_argument("ids", nargs="*", help="a run of the queue, or any held / stopped / failed run (rebuilt from its record)")
    s.add_argument("--all", action="store_true", help="every held item of the queue and the pause (holds of runs that are not in the queue stay)")
    s.add_argument("--words", default="", help="the person's own words that allow it (required)")
    s.add_argument("--fresh-code", action="store_true", help="continue on the code of now, not of the first attempt (the hot-fix loop)")
    s.add_argument("--no-runner", action="store_true", help="do not start the runner")
    s = qsub("drop", cmd_queue_drop, help="remove an item that is not running (the older name of `remove`)")
    s.add_argument("id")
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
    p = sub.add_parser("compose", help="a base config + presets + --set overrides -> the composed config, with every change listed")
    p.add_argument("base")
    p.add_argument("--preset", action="append", default=[], help="name of a preset file in the presets folder (repeatable, merged in this order)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one value, a.b=c (json when it parses; repeatable)")
    p.add_argument("--presets-dir")
    p.add_argument("--out", help="write the composed config here")
    p.set_defaults(fn=cmd_compose)
    p = sub.add_parser("start", help="compose, check, and print (--go: start) the run.py command of a training; the same command again resumes it")
    p.add_argument("base")
    p.add_argument("--id", required=True, help="run id (ledger id; the run folder is the id with . as _)")
    p.add_argument("--preset", action="append", default=[])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--hyp"), p.add_argument("--pred"), p.add_argument("--rule")
    p.add_argument("--no-card", action="store_true", help="a diagnostic run: no hypothesis / prediction / rule (it never feeds a verdict)")
    p.add_argument("--no-check", action="store_true", help="skip the project's static check of the composed config")
    p.add_argument("--go", action="store_true", help="start it (detached); without it the command is only printed and the composed config goes to <id>.preview.json")
    p.add_argument("--prepare", action="store_true", help="a missing prepare item (the GT store) is built first, as a run of its own that the training waits for (--after)")
    p.add_argument("--skip-prepare", action="store_true", help="do not check the prepare items")
    p.add_argument("--wait", type=float, default=3.0, help="seconds to see whether the launcher stays alive (it refuses a held / running / finished id at once)")
    p.add_argument("--run-arg", action="append", default=[], help="an argument for run.py itself (repeatable), e.g. --run-arg=--seed --run-arg=1")
    p.add_argument("--project", help="project folder (default: AR_PROJECT)")
    p.set_defaults(fn=cmd_start)
    sub.add_parser("selftest", help="check the suite on synthetic runs").set_defaults(fn=None)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        from . import selftest
        return selftest.main()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
