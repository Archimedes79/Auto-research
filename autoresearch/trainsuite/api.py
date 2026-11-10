"""The interface of the training system: the class `Training`, plain dicts in and out. The convention for all suites is the section "The interface of the suites" of autoresearch/README.md.

    from trainsuite import Training
    t = Training("../bev_fusion")                          # the project folder (default: AR_PROJECT)
    job = "jobs/t23_lane_bce.json"                         # a job file, or the same dict (JOB_FIELDS, README section 0)
    t.check(job)                                           # compose, static check, data check: nothing is started                      -> {ok, problems, changes, command}
    t.run(job)                                             # start it now; the same call again continues a crashed run                 -> {id, started}
    t.list()                                               # the trainings of the last day and the box: slots, lock, RAM, GPU, holds   -> {runs, running, ...}
    t.status("t23.C_s1")                                   # one training: state, ETA, loss, gradient, skipped steps, data, health, kill rules, speed
    t.status()                                             # the running training
    t.stop("t23.C_s1")                                     # ask it to checkpoint and stop (exit 75; its id goes on hold)
    t.resume("t23.C_s1", words="...")                      # the person's words clear the hold; it continues from its checkpoint       -> {id, started}
    t.plot("t23.C_s1")                                     # loss curves as a png                                                      -> {png}
    t.compare("t20.base_s0", "speed3.fast")                # s/step, data wait, GPU busy, peak memory, loss at the last shared step     -> {rows}

Every method is also a command by its name: `python -m trainsuite run job.json`, `status t23_C_s1`, `resume t23_C_s1 --words "..."` (the class name in front, `Training.run`, is always valid;
`python -m trainsuite methods` lists them with their arguments; the map is in the README of trainsuite). Only `health`, `watch`, `early`, `compose` and `queue ...` are older commands that are no methods. The command prints a short text, or with `--json` exactly what the method returns. `i` is an id (`t23.C_s1`)
or a run folder name (`t23_C_s1`). Resume needs the person's own words: a rule, not a convenience (autoresearch/METHOD.md, hard rule 6). A queue is NOT part of the class (queue_api.py has it as a mixin
for whoever needs one). One process works on one project at a time: the project is passed to the modules through the environment, so the methods take a lock (two threads with two projects wait).
"""
import functools
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import suitecli

from . import compose as C
from . import control, health, modules
from . import trainqueue as TQ
from .view import detail_text, speed_row, summary
from .kill import Rule
from .runs import Run

JOB_FIELDS = ("id", "config", "preset", "set", "card", "no_card", "after", "priority", "note", "kill", "run_args", "prepare", "no_check", "project")
BAD_ENDS = ("failed", "timeout", "blocked", "gpu_busy", "stopped", "held")     # what the launcher's record can say while the run folder still looks alive


RESEND_S = 20.0                # `stop` writes the request again this often while the trainer has not stopped: one written before the trainer cleared old markers is moved aside as stale
READERS = 4                    # the run folders lie on a slow drive (about 3 ms per file access under WSL): reading many of them waits for the drive, so a few threads read at once


class TrainingError(Exception):
    """what the person has to fix; the message says what and where"""


def _map(fn, items):
    """[fn(x) for x in items], the calls on a few threads: for functions that read files of many run folders"""
    items = list(items)
    if len(items) < 2:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(min(READERS, len(items))) as pool:
        return list(pool.map(fn, items))


# ---- jobs
def _list(v, what):
    if v in (None, ""):
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
        return list(v)
    raise TrainingError(f'job "{what}": a text or a list of texts, not {v!r}')


def as_bool(d, key):
    """true or false (a text like "false" is an error: it would be true)"""
    v = d.get(key)
    if v is None:
        return False
    if not isinstance(v, bool):
        raise TrainingError(f'job "{key}": true or false, not {v!r}')
    return v


def load_job(job):
    """a job file (path of a json file) or a dict -> the normalised job: every key present, the types checked. A key that is not a job key is an error (a typo shows).
    Keys starting with `_` are comments."""
    if isinstance(job, (str, os.PathLike)):
        try:
            with open(job, encoding="utf-8-sig") as f:                  # -sig: a file written by PowerShell 5.1 starts with a BOM
                d = json.load(f)
        except (OSError, ValueError) as e:
            raise TrainingError(f"job file {job}: {e}")
    elif isinstance(job, dict):
        d = dict(job)
    else:
        raise TrainingError("a job is the path of a json file or a dict")
    if not isinstance(d, dict):
        raise TrainingError("a job file holds one json object")
    d = {k: v for k, v in d.items() if not str(k).startswith("_")}
    unknown = sorted(set(d) - set(JOB_FIELDS))
    if unknown:
        hint = " (a job of this suite is a training and has no kind; a trained network on frames is run with `python -m runner`)" if "kind" in unknown else ""
        raise TrainingError(f"unknown job key(s) {unknown}; the keys are: {', '.join(JOB_FIELDS)}{hint}")
    j = dict(d)
    j["id"] = str(d.get("id") or "")
    if not j["id"]:
        raise TrainingError('a job needs an "id" (the ledger id; the run folder is the id with . as _)')
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", j["id"]):
        raise TrainingError(f'job "id" {j["id"]!r}: starts with a letter or digit; letters, digits, . _ - only')
    j["preset"] = _list(d.get("preset"), "preset")
    s = d.get("set") or {}
    if isinstance(s, dict):
        s = [f"{k}={json.dumps(v)}" for k, v in s.items()]
    j["set"] = _list(s, "set") if isinstance(s, (str, list)) else None
    if j["set"] is None or any("=" not in x for x in j["set"]):
        raise TrainingError('job "set": {"key": value, ...} or ["key=value", ...]')
    card = d.get("card") or {}
    if not isinstance(card, dict) or set(card) - {"hyp", "pred", "rule"}:
        raise TrainingError('job "card": {"hyp": "...", "pred": "...", "rule": "..."}')
    j["card"] = dict(card)
    j["no_card"] = as_bool(d, "no_card")
    j["after"] = _list(d.get("after"), "after")
    try:
        j["priority"] = int(d.get("priority", 5))
    except (TypeError, ValueError):
        raise TrainingError(f'job "priority": a whole number (1 goes first), not {d.get("priority")!r}')
    j["note"] = str(d.get("note") or "")
    j["kill"] = _list(d.get("kill"), "kill") if isinstance(d.get("kill"), (list, str, type(None))) else None
    if j["kill"] is None:
        raise TrainingError('job "kill": a kill rule or a list of them, e.g. "group.lines@1000 > 1.48"')
    for rule in j["kill"]:
        try:
            Rule(rule)
        except ValueError as e:
            raise TrainingError(f'job "kill": {e}')
    j["run_args"] = _list(d.get("run_args"), "run_args")
    j["prepare"], j["no_check"] = as_bool(d, "prepare"), as_bool(d, "no_check")
    j["project"] = str(d.get("project") or "")
    if not d.get("config"):
        raise TrainingError('a training job needs "config" (the network file, e.g. training/configs/t20_base.json)')
    j["config"] = str(d.get("config") or "")
    return j


def _ctx(project=None):
    """the project's context. TQ.context sets AR_PROJECT, and the modules it loads read it at call time (the ledger note of a hold, the paths), so it stays set while the method runs;
    `errors` gives the old value back when the method ends"""
    try:
        return TQ.context(project or None)
    except (Exception, SystemExit) as e:  # noqa: BLE001  a missing project.json is the person's to fix (project.py ends the process with a message)
        what = e.code if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}"
        raise TrainingError(f"no project: {what} (give the project folder, set AR_PROJECT or the job's \"project\")")


_LOCK = threading.RLock()      # the project reaches the modules through the environment: one method at a time per process


def errors(fn):
    """a method of the interface: one project at a time (the lock), the environment given back, the queue's own error and a process exit of project.py become a TrainingError"""
    @functools.wraps(fn)
    def wrapped(*a, **kw):
        with _LOCK:
            old = os.environ.get("AR_PROJECT")
            try:
                return fn(*a, **kw)
            except TQ.QueueError as e:
                raise TrainingError(str(e))
            except SystemExit as e:                               # project.py ends the process with a message when it finds no project
                raise TrainingError(str(e.code) if isinstance(e.code, str) else f"the process was asked to end (exit {e.code})")
            finally:
                if old is None:
                    os.environ.pop("AR_PROJECT", None)
                else:
                    os.environ["AR_PROJECT"] = old
    return wrapped


def shell(argv):
    return " ".join(shlex.quote(x) for x in argv)


def _tail(r, n=6):
    return " | ".join((r.stdout + r.stderr).strip().splitlines()[-n:])


# ---- check and run (module helpers: they take the project context c)
def _checked_plan(c, j, final):
    """compose the config and look at everything that makes a start fail: -> (plan, problems, missing prepare items). final=False writes only the preview file (a running run
    reads the real one), final=True writes the config the run will read"""
    if not c.block:
        raise TrainingError('project.json has no "trainsuite" block')
    card = j["card"]
    try:
        plan = C.plan_start(c.root, c.block, j["config"], j["id"], j["preset"], j["set"], card.get("hyp"), card.get("pred"), card.get("rule"), j["no_card"], j["run_args"],
                            TQ.TOOLS, c.records)
    except C.ComposeError as e:
        raise TrainingError(str(e))
    problems, missing = [], []
    splits = c.cfg.get("splits")
    head = plan["argv"][:plan["argv"].index("--")] if "--" in plan["argv"] else plan["argv"]
    used = next((head[k + 1] for k in range(len(head) - 2, -1, -1) if head[k] == "--split"), None)           # the last one wins, as in run.py
    if splits and used and used not in splits:
        problems.append(f"split {used!r} is not one of this project's splits {list(splits)}: run.py would refuse the start (set run_args [\"--split\", \"<one of them>\"])")
    if plan["held"]:
        problems.append(f"run id {j['id']} is ON HOLD (hold.py show): only the person's words clear it (Training.resume)")
    if final and problems:
        return plan, problems, missing
    target, check = (plan["composed"], plan["check"]) if final else (plan["preview"], plan["check_preview"])
    try:
        C.write(plan["config"], target)
    except C.ComposeError as e:
        raise TrainingError(str(e))
    if check and not j["no_check"]:
        r = subprocess.run(check, cwd=c.root, capture_output=True, text=True)
        if r.returncode:
            problems.append("check: " + _tail(r))
    for pr in plan["prepare"]:
        r = subprocess.run(pr["check"] if final else pr["check_preview"], cwd=c.root, capture_output=True, text=True)
        if r.returncode == 1:                                           # the check's word for "not there": 0 present, 1 missing (gt_scenes.py check -> GtGenerator have), anything else is the check failing
            missing.append(pr)
        elif r.returncode:
            problems.append(f"prepare check {pr['name']} failed (exit {r.returncode}, not 'missing': nothing is built on that): " + _tail(r))
    return plan, problems, missing


def launch(c, jobs, wait):
    """start the launchers detached (builds first, the training last); a launcher that refuses within `wait` seconds (held, running, finished, no card) is an error"""
    started = []
    os.makedirs(os.path.dirname(jobs[-1][1]), exist_ok=True)
    for argv, log, rid in jobs:
        with open(log, "ab") as lf:
            proc = subprocess.Popen(argv, cwd=c.root, env=dict(os.environ, AR_PROJECT=c.root), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            rc = None
        if rc == 0:                                                     # done within the wait: a data build whose data was there after all, or a very short run (a smoke test)
            started.append(dict(id=rid, pid=proc.pid, log=log, finished=True))
            continue
        if rc is not None:
            with open(log, encoding="utf-8", errors="replace") as lf:
                tail = " | ".join(lf.read().strip().splitlines()[-6:])
            raise TrainingError(f"the launcher of {rid} ended after {wait:g} s with exit {rc}, nothing runs: {tail}")
        started.append(dict(id=rid, pid=proc.pid, log=log))
    return started


# ---- addressing: an id or a run folder name
def _run_dir(c, rid):
    return os.path.join(c.runs, TQ.folder(rid))


def _slot_ids(c):
    """the ids that hold a GPU slot now (a launcher that is alive), whoever started them"""
    base = c.R.slot_dir(c.root, c.cfg, c.records)
    out = []
    for k in range(int(c.cfg.get("gpu_slots", 2))):
        s = os.path.join(base, str(k))
        o = c.R.record(s) if os.path.isdir(s) else {}
        if o and o.get("id") and c.R.pid_alive(o.get("pid")):
            rd = o.get("run_dir")
            if rd and os.path.realpath(os.path.dirname(rd)) != os.path.realpath(c.records):      # bev_fusion and the lane project share one slot folder: a run of the other one is not ours
                continue
            out.append(o["id"])
    return out


def running(c):
    """[id] of the trainings that hold a GPU slot now (a GPU job is a training when it has a run folder of its own: an evaluation or a prediction has not)"""
    return [rid for rid in _slot_ids(c) if os.path.isdir(_run_dir(c, rid))]


def resolve(c, i):
    """-> (id, None, None). i=None is the running training. A text is an id or a run folder name (the id of the run record whose folder name it is). The triple keeps the shape
    of queue_api.resolve_item, so a subclass with a queue can reuse the methods"""
    if i is None:
        r = running(c)
        return (r[0], None, None) if r else (None, None, None)
    if not isinstance(i, str) or not i:
        raise TrainingError("i is an id or a run folder name")
    known = next((d for d in sorted(os.listdir(c.records)) if TQ.folder(d) == TQ.folder(i)), None) if os.path.isdir(c.records) else None
    return (known or i), None, None


def must_exist(c, rid):
    """a typo in an id is an error, as in the other suites: the id is a run record, a run folder or a row of the ledger (a run that is only planned is a real state)"""
    if not os.path.isdir(os.path.join(c.records, rid)) and not os.path.isdir(_run_dir(c, rid)):
        import ledger
        if not any(r.get("id") == rid for r in ledger.load()):
            raise TrainingError(f"no run record or ledger row named {rid!r}")


def _detail(run, row):
    """the detail text of a run; what is shown must never break the data of `status`"""
    try:
        return detail_text(run, row)
    except Exception as e:  # noqa: BLE001
        return f"(the detail view failed: {type(e).__name__}: {e})"


def _wait_stopped(c, rid, rd, reason, wait):
    """True when the run has stopped within `wait` seconds (STOPPED written, or its launcher's record is there and not running any more), else False. The request is written again every RESEND_S"""
    t0 = last = time.time()
    while True:
        rec = c.R.record(os.path.join(c.records, rid))
        if Run(rd, "total").stopped() is not None or (rec and rec.get("status") not in ("running", "waiting")):          # a run that has no launcher record (started by hand) is stopped when it says so
            return True
        if time.time() - t0 >= wait:
            return False
        if time.time() - last >= RESEND_S:
            control.request_stop(rd, reason)
            last = time.time()
        time.sleep(1.0)


def _run_info(c, rid):
    """one training for `list`: its state from the launcher's record, its step, the gradient, the skipped steps and the health flags"""
    d = os.path.join(c.records, rid)
    rd = _run_dir(c, rid)
    step, total = TQ.last_step(rd)
    info = dict(id=rid, state=c.R.state_of(d) or "unknown", step=step, total=total, held=c.H.read(d) is not None, grad=None, skipped=0, flags=[])
    if os.path.isdir(rd):
        try:
            run_ = Run(rd, "total")
            if run_.train:
                info["grad"], info["skipped"] = modules.grad_trend(run_), run_.skipped_steps()[0]
                info["flags"] = sorted({f.code for f in health.check(run_) if f.level != "info"})
        except Exception as e:  # noqa: BLE001  a damaged run folder must not break the view
            info["flags"] = [f"unreadable:{type(e).__name__}"]
    return info


def _machine(c):
    """the box as the launcher sees it: GPU slots, the machine lock, RAM, GPU memory against the gate, the runs on hold"""
    import machine_lock
    base = c.R.slot_dir(c.root, c.cfg, c.records)
    n = int(c.cfg.get("gpu_slots", 2))
    settle = c.R.settle_s(c.cfg)
    loading = dict(c.R.GpuSlot(c.records, n, "interface", c.records, base=base).settling(settle))
    slots = []
    for k in range(n):
        s = os.path.join(base, str(k))
        o = c.R.record(s) if os.path.isdir(s) else {}
        if o and c.R.pid_alive(o.get("pid")):
            slots.append(dict(slot=k, id=o.get("id"), pid=o.get("pid"), since=o.get("taken"), loading_s=loading.get(o.get("id"))))
        else:
            slots.append(dict(slot=k, id=None))
    lock = machine_lock.current()
    try:
        gate = C.gpu_gate(c.block)
    except C.ComposeError as e:
        gate = f"error: {e}"
    used = c.R.gpu_used_mb()
    names = sorted(os.listdir(c.records)) if os.path.isdir(c.records) else []
    found = _map(lambda rid: c.H.read(os.path.join(c.records, rid)), names)
    holds = [dict(id=rid, since=h.get("when", "?"), why=h.get("why", "")) for rid, h in zip(names, found) if h is not None]
    return dict(slots=slots, lock=lock, ram=dict(available_mb=c.R.mem_available_mb(), need_mb=int(c.cfg.get("ram_mb", 0))), gpu=dict(used_mb=used, gate_mb=gate), holds=holds)


class Training:
    """The training system as one object. `Training(project)` takes the project folder (default: AR_PROJECT); the methods are listed in `METHODS` and in the head of this file. A job is
    a json file or a dict (JOB_FIELDS). Every method is also a command: `python -m trainsuite Training.<method> ...`."""

    METHODS = ("check", "run", "status", "stop", "plot", "compare", "list", "resume")

    def __init__(self, project=None):
        self.project = os.path.abspath(str(project)) if project else ""          # once: a later chdir must not move it

    def _c(self, j=None):
        return _ctx(((j or {}).get("project") or self.project) or None)

    def _job(self, job):
        return load_job(job)

    @classmethod
    def methods(cls):
        """[{name, arguments, summary}] of the methods that are also commands: what `python -m trainsuite methods` prints"""
        return suitecli.method_list(cls)

    # ---- one training
    @errors
    def check(self, job):
        """what `run` would do, without doing it: compose the config (every change listed), the project's static check, the check of the data the run needs. Writes only a preview
        file. -> {ok, id, changes, notes, problems, will_prepare, command}"""
        j = self._job(job)
        c = self._c(j)
        plan, problems, missing = _checked_plan(c, j, final=False)
        problems = TQ.start_blocks(c, j["id"]) + problems                 # the person's pause of the queue (or a held item of it) binds a direct start (a queued one may be added)
        if missing and not j["prepare"]:
            problems.append("missing data: " + ", ".join(m["name"] for m in missing) + ' (build it first, or set "prepare": true: the build becomes a run of its own that the training waits for)')
        return dict(ok=not problems, id=j["id"], changes=C.describe(plan["changes"], plan["base_keys"]), notes=list(plan["notes"]), problems=problems,
                    will_prepare=[m["name"] for m in missing] if j["prepare"] else [], command=shell(plan["argv"]))

    @errors
    def run(self, job, wait=3.0):
        """start the training now (detached). The same call again continues a crashed run from its last checkpoint (a stopped one is on hold: `resume`). -> {id, started: [{id, pid, log}], command}"""
        j = self._job(job)
        c = self._c(j)
        plan, problems, missing = _checked_plan(c, j, final=True)
        problems = TQ.start_blocks(c, j["id"]) + problems
        if missing and not j["prepare"]:
            problems.append("missing data: " + ", ".join(m["name"] for m in missing) + ' (build it first, or set "prepare": true)')
        if problems:
            raise TrainingError("nothing started: " + "; ".join(problems))
        argv = list(plan["argv"])
        waits = j["after"] + [m["id"] for m in missing]                 # the jobs it waits for: its own `after`, and the builds of the data it needs (a failed one blocks it)
        if waits:
            i = argv.index("--")
            argv = argv[:i] + ["--after", ",".join(waits)] + (["--wait-after", str(max(m["wait_min"] for m in missing))] if missing else []) + argv[i:]
        log_dir = os.path.dirname(plan["log"])
        jobs = [(m["argv"], os.path.join(log_dir, m["id"].replace(".", "_") + ".launcher.log"), m["id"]) for m in missing] + [(argv, plan["log"], j["id"])]
        started = launch(c, jobs, wait)
        TQ.write_kill(os.path.join(c.runs, TQ.folder(j["id"])), j["kill"])     # only now: a launcher that refused (held, running, finished) leaves no folder behind
        return dict(id=j["id"], started=started, command=shell(argv))

    @errors
    def stop(self, i=None, reason="", wait=60.0):
        """ask one training to save a checkpoint and stop (it exits with 75; run.py then puts its id on hold); i=None is the running one. The request is written again every 20 s until the
        trainer has stopped or `wait` seconds are over (0: do not wait). Refused for a code snapshot without the stop check. -> {id, requested, file, verified, stopped (None: not waited)}"""
        c = self._c()
        rid, _, _ = resolve(c, i)
        if rid is None:
            raise TrainingError("nothing is running")
        rd = _run_dir(c, rid)
        if not os.path.isdir(rd):
            raise TrainingError(f"no run folder for {rid} (looked in {rd})")
        ok = control.honours_stop(rd)
        if ok is False:
            raise TrainingError(f"NOT requested: the code snapshot of {rid} (code/train.py) has no stop check, it would ignore CONTROL. Stopping it is a decision about the process, not a file")
        file = control.request_stop(rd, reason)
        return dict(id=rid, requested=True, file=file, verified=bool(ok), stopped=_wait_stopped(c, rid, rd, reason, wait) if wait > 0 else None)

    @errors
    def status(self, i=None, kill=(), ref=None, skip=10):
        """one training in full: state and phase, step, ETA, loss, the gradient norm (first / now / peak), the non-finite steps skipped, the data it trained on, the health findings,
        the kill rules (the job's and `kill`), the speed (s/step, data wait, GPU busy, peak GB). `text` is the detail a person reads (phase and ETA, findings, last evaluation, loss groups,
        modules, time split, incident dumps). i=None is the running training (state "idle" if none)."""
        c = self._c()
        rid = resolve(c, i)[0]
        if rid is None:
            return dict(id=None, state="idle")
        must_exist(c, rid)
        rd = _run_dir(c, rid)
        rules = list(kill)                                           # the job's own rules are in <run>/kill.txt, which `summary` reads itself
        if not os.path.isdir(rd):
            return dict(id=rid, state="unknown", phase="not started", step=0, total=0, pct=None, eta_s=None, ready_at=None, s_per_step=None, loss=None,
                        grad=None, skipped=0, data="", restarts=0, findings=[], kill=[], speed=None, note="no run folder yet")
        run_, refrun = Run(rd, "total"), None
        if ref:
            ref_dir = _run_dir(c, resolve(c, ref)[0])
            if not os.path.isdir(ref_dir):
                raise TrainingError(f"the reference run {ref!r} has no run folder")
            refrun = Run(ref_dir, "total")
        s, findings, kills = summary(run_, refrun, rules)
        rstate = c.R.state_of(os.path.join(c.records, rid))
        if rstate in ("done", "ok") and s["state"] != "done":     # the launcher's record is the authority: the done flag exists only for exit 0
            s.update(state="done", phase="done", ready_pct=100.0, pct=100.0, eta_s=None, ready_at=None)
        elif rstate in BAD_ENDS and s["state"] in ("running", "new", "stalled"):      # the trainer was killed (out of memory, SIGKILL): no traceback, the heartbeat is fresh for minutes
            code = c.R.record(os.path.join(c.records, rid)).get("exit")
            s.update(state=rstate, phase=rstate, eta_s=None, ready_at=None)
            findings.append(health.Finding("alert", "launcher_" + rstate, f"the launcher's record says {rstate}" + (f" (exit {code})" if code is not None else "")
                                           + " while the run folder still looks alive: the trainer is gone (killed? out of memory?)"))
        return dict(id=rid, state=s["state"], phase=s["phase"], step=s["step"], total=s["total"], pct=s["ready_pct"], eta_s=s["eta_s"], ready_at=s["ready_at"],
                    s_per_step=s["s_per_step"], loss=s["loss"], grad=s["grad"], skipped=s["skipped"], data=s["data"], restarts=s["restarts"],
                    findings=[f._asdict() for f in findings], kill=[dict(rule=r.text, status=st, value=v) for r, st, v in kills], speed=speed_row(run_, skip),
                    text=_detail(run_, (s, findings, kills)))

    @errors
    def plot(self, i, out=None, keys=None, smooth=10):
        """the loss curves of one run (or several: `a,b`) as a png (needs matplotlib); `keys`: a comma list of log keys (default: the loss and its groups). -> {png}"""
        from .curves import default_keys, plot as draw
        c = self._c()
        ids = list(i) if isinstance(i, (list, tuple)) else [x for x in str(i).split(",") if x]            # `a,b` overlays two runs
        dirs = []
        for x in ids:
            rid = resolve(c, x)[0]
            if rid is None or not os.path.isdir(_run_dir(c, rid)):
                raise TrainingError(f"no run folder for {x!r}")
            dirs.append(Run(_run_dir(c, rid), "total"))
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        try:
            return dict(png=draw(dirs, keys or default_keys(dirs[0]), out or os.path.join(dirs[0].dir, "curves.png"), smooth))
        except ImportError as e:
            raise TrainingError(f"plot needs matplotlib: {e}")

    @errors
    def compare(self, a, b, skip=10):
        """two runs side by side, a is the baseline: s/step and the speed-up, the share of a step that waits for the data, the GPU busy share, the peak GPU memory, and the loss at
        the last step both have reached (an optimisation that changes the loss curve is not a free speed-up). -> {common_step, rows: [{id, ...}]}"""
        c = self._c()
        runs = []
        for x in (a, b):
            rid = resolve(c, x)[0]
            if rid is None or not os.path.isdir(_run_dir(c, rid)):
                raise TrainingError(f"no run folder for {x!r}")
            runs.append((rid, Run(_run_dir(c, rid), "total")))
        rows = [(rid, r, speed_row(r, skip)) for rid, r in runs]
        common = min(r.step for _, r, _ in rows)
        base_s, base_loss = rows[0][2]["s_step"], rows[0][1].window_mean(rows[0][1].loss_key, common)
        out = []
        for rid, r, x in rows:
            loss = r.window_mean(r.loss_key, common)
            out.append(dict(id=rid, steps=x["step"], s_step=x["s_step"], speedup=(base_s / x["s_step"]) if (base_s and x["s_step"]) else None, data_share=x["data_share"],
                            gpu_busy=x["gpu_busy"], gpu_gb=x["gpu_gb"], loss_at_common=loss, loss_vs_base_pct=(100 * (loss - base_loss) / abs(base_loss)) if (loss is not None and base_loss) else None))
        return dict(common_step=common, rows=out)

    @errors
    def list(self, hours=24.0):
        """the trainings of the project that were active in the last `hours` hours and what the box looks like. -> {runs: [{id, state, step, total, held, grad, skipped, flags}], running: [id],
        slots, lock, ram, gpu, holds}"""
        c = self._c()
        names = sorted(os.listdir(c.records)) if os.path.isdir(c.records) else []

        def recent(rid):
            d, rd = os.path.join(c.records, rid), _run_dir(c, rid)
            if not os.path.isdir(rd) or not TQ.is_training(c, rid, c.R.record(d)):
                return False
            return time.time() - max(os.path.getmtime(rd), os.path.getmtime(d)) <= hours * 3600
        ids = [rid for rid, keep in zip(names, _map(recent, names)) if keep]
        return dict(runs=_map(lambda rid: _run_info(c, rid), ids), running=running(c), **_machine(c))

    @errors
    def resume(self, i, words, fresh_code=False, wait=3.0):
        """the person's own words clear the hold of a stopped training (`stop` puts its id on hold) and it continues from its last checkpoint on the code of its first attempt (fresh_code:
        on the code of now, the hot-fix loop). -> {id, cleared, started, command}"""
        TQ.need_words(words, "resume")
        c = self._c()
        rid = resolve(c, i)[0]
        if rid is None:
            raise TrainingError("name the training (i)")
        d = os.path.join(c.records, rid)
        rec = c.R.record(d)
        if not rec:
            raise TrainingError(f"{rid}: no run record, nothing to continue")
        if c.R.state_of(d) in ("running", "waiting"):
            raise TrainingError(f"{rid} is {c.R.state_of(d)}: stop it first")
        pin = None if fresh_code else (TQ.first_commit(c, rec) or None)         # a continued run trains on the code of its first attempt: the working tree of now does not leak in
        if pin is None and not fresh_code:
            raise TrainingError(f"{rid}: its run record names no commit of the first attempt, so there is no code to pin it to and the working tree of now would be trained on. "
                                "fresh_code=True continues on the code of now (nothing is changed yet, the hold stays)")
        h = c.H.read(d)
        if h is not None:
            c.H.clear(d, words)
            c.H.note_in_ledger(rid, f"hold cleared {dt.date.today().isoformat()} on the person's words: {words.strip()!r} (hold was: {h.get('why', '?')}, set by {h.get('who', '?')} {h.get('when', '?')})")
        argv = TQ.with_gate(c, TQ.pinned_argv({"argv": TQ.argv_from_record(c, rid, rec), "pin": pin}))
        started = launch(c, [(argv, os.path.join(c.runs, TQ.folder(rid) + ".launcher.log"), rid)], wait)
        return dict(id=rid, cleared=h is not None, started=started, command=shell(argv))


# ---- the command line: the same methods as commands (suitecli.py generates it from the signatures)
LEGACY = ("health", "queue", "watch", "early", "compose", "selftest")      # older commands: their words keep their meaning


def _check_text(r):
    print(f"{r['id']}" + (f" ({r['mode']})" if "mode" in r else "") + ": " + ("OK, nothing is started by a check" if r["ok"] else "PROBLEMS"))
    for p in r["problems"]:
        print(f"  problem: {p}")
    for n in r.get("notes", []):
        print(f"  note: {n}")
    if "changes" in r:
        print(f"  {len(r['changes'])} change(s) against the config:")
        for x in r["changes"]:
            print("    " + x)
    for n in r.get("will_prepare", []):
        print(f"  will build first: {n}")
    print(f"  command: {r['command']}")


def _run_text(r):
    for s in r["started"]:
        print(f"started {s['id']}, pid {s['pid']}, launcher log {s['log']}" + (" (finished within the wait)" if s.get("finished") else ""))
    print(f"  command: {r['command']}")


def _list_text(r):
    print(f"{'id':<26}{'state':<12}{'step/total':<14}{'grad':>9}  {'skip':>4}  flags")
    for x in r["runs"]:
        g = x["grad"]
        grad = "-" if not g else "non-finite" if g["now"] is None else "%.3g" % g["now"]
        more = ", ".join(filter(None, [",".join(x["flags"]), "HELD" if x["held"] else ""]))
        print(f"{x['id']:<26}{x['state']:<12}{'%d/%s' % (x['step'], x['total'] or '?'):<14}{grad:>9}  {x['skipped'] or '-':>4}  {more}")
    if not r["runs"]:
        print("  no training was active in that time")
    print(f"running: {', '.join(r['running']) or '-'}")
    print("slots: " + "; ".join(f"{s['slot']}: {s['id'] or 'free'}" + (f" (loading {s['loading_s']} s)" if s.get("loading_s") is not None else "") for s in r["slots"]))
    lk = r["lock"]
    print("machine lock: " + (f"{lk['owner']} until {lk['end']} ({'exclusive' if lk.get('exclusive') is True else 'reservation'})" if lk else "free"))
    ram, gpu = r["ram"], r["gpu"]
    print(f"RAM: {ram['available_mb']} MB available (gate {ram['need_mb']} MB)   GPU memory: {gpu['used_mb']} MB in use (gate {gpu['gate_mb']})")
    if r["holds"]:
        print("on hold (the person's words clear a hold): " + ", ".join(h["id"] for h in r["holds"]))


def _status_text(r):
    if r.get("id") is None:
        print("no training is running")
        return
    print(f"{r['id']}: {r['state']}, step {r['step']}/{r['total'] or '?'}" + ("" if r["loss"] is None else f", loss {r['loss']:.4g}"))
    if r.get("note"):
        print("  " + r["note"])
    if r.get("text"):
        print(r["text"])


TEXT = {"check": _check_text, "run": _run_text, "list": _list_text, "status": _status_text}


CLI = suitecli.Cli(Training, "trainsuite", TrainingError, "TRAINSUITE_TRACEBACK", legacy=LEGACY, positional=("job", "i", "a", "b"), option_names=("words",), text=TEXT)
