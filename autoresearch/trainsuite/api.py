"""The interface of the training system: the class `Training`, plain dicts in and out. The convention for all suites is autoresearch/INTERFACE.md.

    from trainsuite import Training
    t = Training("../bev_fusion")                          # the project folder (default: AR_PROJECT)
    job = "jobs/t23_lane_bce.json"                         # a job file, or the same dict (JOB_FIELDS, README section 0)
    t.check(job)                                           # compose, static check, data check: nothing is started                      -> {ok, problems, changes, command}
    t.run(job)                                             # start it now, outside the queue; the same call again resumes it           -> {id, started}
    t.queue_add(job)                                       # put it in the queue; the runner starts it when its turn comes             -> the item
    t.queue_list()                                         # the queue, the running training, slots, lock, RAM, GPU memory, holds      -> {items, running, ...}
    t.queue_get(2)                                         # one item: its job, config, command and state                              -> dict
    t.queue_remove("t23.C_s1")                             # take an item out that has not started                                     -> {removed}
    t.queue_start()                                        # make sure the runner is running                                           -> {runner}
    t.queue_hold(words="...")                              # stop all trainings (checkpoint), nothing new starts: the person's words   -> {id: outcome}
    t.queue_resume(words="...")                            # every held training continues from its checkpoint                         -> {id: what happens}
    t.status()                                             # the running training: state, ETA, loss, gradient, skipped steps, data, health, kill rules, speed
    t.status(3)                                            # item 3 of the queue (a number, an id or a run folder name do the same)
    t.stop("t23.C_s1")                                     # ask one training to checkpoint and stop (exit 75; its id goes on hold)
    t.plot("t23.C_s1")                                     # loss curves as a png                                                      -> {png}
    t.compare("t20.base_s0", "speed3.fast")                # s/step, data wait, GPU busy, peak memory, loss at the last shared step     -> {rows}

Every method is also a command with the class name in front: `python -m trainsuite Training.queue_add job.json`, `Training.queue_hold --all --words "..."`, `Training.status 3` (`python -m
trainsuite methods` lists them with their arguments; apicli.py has the map). Short forms where no older command has the word: `queue_add` and `queue-add`; `status`, `stop`, `plot` and
`compare` are older commands with their own meaning, so they always need the class name. The command prints a short text, or with `--json` exactly what the method returns. `i` is an id
(`t23.C_s1`), a run folder name (`t23_C_s1`) or the number shown by queue_list (1 = first). Hold and resume need the person's own words: a rule, not a convenience (autoresearch/METHOD.md,
hard rule 6). One process works on one project at a time: the project is passed to the modules through the environment, so the methods take a lock (two threads with two projects wait for each other).
"""
import functools
import inspect
import json
import os
import re
import shlex
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

from . import compose as C
from . import control, health, modules
from . import trainqueue as TQ
from .cli import speed_row, summary
from .kill import Rule
from .runs import Run

JOB_FIELDS = ("id", "kind", "config", "preset", "set", "card", "no_card", "after", "priority", "note", "kill", "run_args", "prepare", "no_check", "project",
              "run", "mode", "split", "n", "scene", "first", "out", "dts", "device", "cold", "frames", "code")     # the second line: the keys of a runner job (kind "run"), see runsuite
KINDS = ("train", "run")
TRAIN_ONLY = ("preset", "set", "card", "no_card", "kill", "run_args", "prepare", "no_check")     # keys that mean something for a training only
RUN_ONLY = ("run", "mode", "split", "n", "scene", "first", "out", "dts", "device", "cold", "frames", "code")     # keys that mean something for a runner job only


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
        raise TrainingError(f"unknown job key(s) {unknown}; the keys are: {', '.join(JOB_FIELDS)}")
    j = dict(d)
    j["id"] = str(d.get("id") or "")
    if not j["id"]:
        raise TrainingError('a job needs an "id" (the ledger id; the run folder is the id with . as _)')
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", j["id"]):
        raise TrainingError(f'job "id" {j["id"]!r}: starts with a letter or digit; letters, digits, . _ - only')
    j["kind"] = d.get("kind", "train")
    if j["kind"] not in KINDS:
        raise TrainingError(f'job "kind": one of {KINDS}, not {j["kind"]!r}')
    foreign = [k for k in (RUN_ONLY if j["kind"] == "train" else TRAIN_ONLY) if d.get(k) not in (None, "", [], {}, False)]
    if foreign:
        raise TrainingError(f'job key(s) {foreign} belong to a {"runner job (kind run)" if j["kind"] == "train" else "training"}, not to a job of kind "{j["kind"]}" (a typo, or the wrong kind)')
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
    if j["kind"] == "train" and not d.get("config"):
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
def queue_blocks(c, rid):
    """why the person's pause or hold forbids a start NOW of this id, outside the queue: [] when nothing does. hold(all) pauses the runner and `held` marks the items: both mean "nothing
    new starts" (hard rule 6)"""
    q = TQ.load(c)
    out = []
    if q.get("paused"):
        out.append("the queue is PAUSED by a hold (the person's words): queue_resume clears it")
    if any(x["id"] == rid and x.get("held") for x in q["items"]):
        out.append(f"the queue item {rid} is held (the person's words): queue_resume clears it")
    return out


def _checked_plan(c, j, final, direct=True):
    """compose the config and look at everything that makes a start fail: -> (plan, problems, missing prepare items). final=False writes only the preview file (a running run
    reads the real one), final=True writes the config the run will read. direct=True: it would start now, outside the queue, so a pause and a held item of the queue forbid it"""
    if not c.block:
        raise TrainingError('project.json has no "trainsuite" block')
    card = j["card"]
    try:
        plan = C.plan_start(c.root, c.block, j["config"], j["id"], j["preset"], j["set"], card.get("hyp"), card.get("pred"), card.get("rule"), j["no_card"], j["run_args"],
                            TQ.TOOLS, c.records)
    except C.ComposeError as e:
        raise TrainingError(str(e))
    problems = queue_blocks(c, j["id"]) if direct else []
    missing = []
    if plan["held"]:
        problems.append(f"run id {j['id']} is ON HOLD (hold.py show): only the person's words clear it (queue_resume)")
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
        if r.returncode:
            missing.append(pr)
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


# ---- addressing: a queue number, an id or a run folder name
def _ordered(q):
    return sorted(q["items"], key=lambda i: (i.get("priority", 5), i["seq"]))


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


def _states(c, items):
    """{id: state} of the items: state() reads files, so a call looks once"""
    return {it["id"]: TQ.state(c, it) for it in items}


def _running(c, items, states, kind=None):
    """[(id, queue item or None, number or None)] of what runs now and is of this kind (None: of any kind): the items of the queue first, then what holds a GPU slot without being in
    the queue (a run started by `run`, `start --go` or by hand; it is a training when it has a run folder)"""
    out = [(it["id"], it, n) for n, it in enumerate(items, 1) if states[it["id"]] in ("running", "orphan") and kind in (None, it.get("kind", "train"))]
    queued = {it["id"] for it in items}
    for rid in _slot_ids(c):
        if rid not in queued and kind in (None, "train" if os.path.isdir(_run_dir(c, rid)) else "run"):
            out.append((rid, None, None))
    return out


def resolve(c, i, kind="train"):
    """-> (id, queue item or None, number or None). i=None is the running one of this kind. A text is an id or a run folder name first; a number (an int, or a text of digits that is no
    id) is the place in `queue_list`; what is neither in the queue is a run of the records (no item, no number)"""
    items = _ordered(TQ.load(c))
    if i is None:
        running = _running(c, items, _states(c, items), kind)
        return running[0] if running else (None, None, None)
    if isinstance(i, bool):
        raise TrainingError("i is a queue number, an id or a run folder name")
    if isinstance(i, str):
        for n, it in enumerate(items, 1):
            if it["id"] == i or TQ.folder(it["id"]) == TQ.folder(i):
                return it["id"], it, n
        known = next((d for d in sorted(os.listdir(c.records)) if TQ.folder(d) == TQ.folder(i)), None) if os.path.isdir(c.records) else None
        if known:
            return known, None, None
        if not i.isdigit():
            return i, None, None
        i = int(i)
    if isinstance(i, int):
        if not 1 <= i <= len(items):
            raise TrainingError(f"no queue item number {i}: the queue has {len(items)} item(s)")
        return items[i - 1]["id"], items[i - 1], i
    raise TrainingError("i is a queue number, an id or a run folder name")


def _run_dir(c, rid):
    return os.path.join(c.runs, TQ.folder(rid))


def get_item(c, i):
    """one item of the queue of the project c, in full (queue_get)"""
    rid, it, n = resolve(c, i)
    if it is None:
        raise TrainingError(f"{i!r} is not in the queue")
    info = _item_info(c, it, n, _states(c, [it]))
    return dict(info, job=it.get("job"), kill=it.get("kill") or [], config=it.get("config"), composed=it.get("composed", ""), added=it.get("added"), by=it.get("by", ""),
                command=shell(TQ.with_gate(c, TQ.pinned_argv(it))))


def _item_info(c, it, n, states):
    rid = it["id"]
    rd = _run_dir(c, rid)
    step, total = TQ.last_step(rd)
    info = dict(number=n, id=rid, kind=it.get("kind", "train"), state=states[rid], step=step, total=total, after=list(it.get("after") or []), priority=it.get("priority", 5),
                note=it.get("note", ""), held=bool(it.get("held")), pin=(it.get("pin") or "")[:10], refused=it.get("refused", ""), gave_up=bool(it.get("gave_up")),
                grad=None, skipped=0, flags=[])
    if os.path.isdir(rd):
        try:
            run_ = Run(rd, "total")
            if run_.train:
                info["grad"], info["skipped"] = modules.grad_trend(run_), run_.skipped_steps()[0]
                info["flags"] = sorted({f.code for f in health.check(run_) if f.level != "info"})
        except Exception as e:  # noqa: BLE001  a damaged run folder must not break the view of the line
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

    METHODS = ("check", "run", "status", "stop", "plot", "compare", "queue_add", "queue_list", "queue_get", "queue_remove", "queue_start", "queue_hold", "queue_resume")
    KIND = "train"

    def __init__(self, project=None):
        self.project = os.path.abspath(str(project)) if project else ""          # once: a later chdir must not move it

    def _c(self, j=None):
        return _ctx(((j or {}).get("project") or self.project) or None)

    def _job(self, job):
        j = load_job(job)
        if j["kind"] != self.KIND:
            raise TrainingError(f'this is a job of kind "{j["kind"]}": use {"RunJobs" if self.KIND == "train" else "Training"} for it')
        return j

    @classmethod
    def methods(cls):
        """[{name, arguments, summary}] of the methods that are also commands: what `python -m trainsuite methods` prints"""
        out = []
        for name in cls.METHODS:
            fn = getattr(cls, name)
            args = ", ".join(str(p) for k, p in inspect.signature(fn).parameters.items() if k != "self")
            out.append(dict(name=name, arguments=args, summary=" ".join((inspect.getdoc(fn) or "").split()).split(". ")[0].rstrip(".") + "."))
        return out

    # ---- one training
    @errors
    def check(self, job):
        """what `run` would do, without doing it: compose the config (every change listed), the project's static check, the check of the data the run needs. Writes only a preview
        file. -> {ok, id, changes, notes, problems, will_prepare, command}"""
        j = self._job(job)
        c = self._c(j)
        plan, problems, missing = _checked_plan(c, j, final=False)
        if missing and not j["prepare"]:
            problems.append("missing data: " + ", ".join(m["name"] for m in missing) + ' (build it first, or set "prepare": true: the build becomes a run of its own that the training waits for)')
        return dict(ok=not problems, id=j["id"], changes=C.describe(plan["changes"], plan["base_keys"]), notes=list(plan["notes"]), problems=problems,
                    will_prepare=[m["name"] for m in missing] if j["prepare"] else [], command=shell(plan["argv"]))

    @errors
    def run(self, job, wait=3.0):
        """start the training now (detached), outside the queue. The same call again resumes a stopped or crashed run from its last checkpoint. -> {id, started: [{id, pid, log}], command}"""
        j = self._job(job)
        c = self._c(j)
        plan, problems, missing = _checked_plan(c, j, final=True)
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
    def stop(self, i=None, reason=""):
        """ask one training to save a checkpoint and stop (it exits with 75; run.py then puts its id on hold); i=None is the running one. Refused for a code snapshot without the stop
        check. -> {id, requested, file, verified}"""
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
        return dict(id=rid, requested=True, file=control.request_stop(rd, reason), verified=bool(ok))

    @errors
    def status(self, i=None, kill=(), ref=None, skip=10):
        """one training in full: state and phase, step, ETA, loss, the gradient norm (first / now / peak), the non-finite steps skipped, the data it trained on, the health findings,
        the kill rules (the job's and `kill`), the speed (s/step, data wait, GPU busy, peak GB) and its place in the queue. i=None is the running training (state "idle" if none)."""
        c = self._c()
        rid, it, n = resolve(c, i)
        if rid is None:
            return dict(id=None, state="idle", queue=None, number=None)
        rd = _run_dir(c, rid)
        qstate = TQ.state(c, it) if it else None
        rules = list(kill)                                           # the job's own rules are in <run>/kill.txt, which `summary` reads itself
        if not os.path.isdir(rd):
            return dict(id=rid, number=n, queue=qstate, state=qstate or "unknown", phase="not started", step=0, total=0, pct=None, eta_s=None, ready_at=None, s_per_step=None, loss=None,
                        grad=None, skipped=0, data="", restarts=0, findings=[], kill=[], speed=None, note="no run folder yet")
        run_, refrun = Run(rd, "total"), None
        if ref:
            ref_dir = _run_dir(c, resolve(c, ref)[0])
            if not os.path.isdir(ref_dir):
                raise TrainingError(f"the reference run {ref!r} has no run folder")
            refrun = Run(ref_dir, "total")
        s, findings, kills = summary(run_, refrun, rules)
        if c.R.state_of(os.path.join(c.records, rid)) in ("done", "ok") and s["state"] != "done":     # the launcher's record is the authority: the done flag exists only for exit 0
            s.update(state="done", phase="done", ready_pct=100.0, pct=100.0, eta_s=None)
        return dict(id=rid, number=n, queue=qstate, state=s["state"], phase=s["phase"], step=s["step"], total=s["total"], pct=s["ready_pct"], eta_s=s["eta_s"], ready_at=s["ready_at"],
                    s_per_step=s["s_per_step"], loss=s["loss"], grad=s["grad"], skipped=s["skipped"], data=s["data"], restarts=s["restarts"],
                    findings=[f._asdict() for f in findings], kill=[dict(rule=r.text, status=st, value=v) for r, st, v in kills], speed=speed_row(run_, skip))

    @errors
    def plot(self, i, out=None, keys=None, smooth=10):
        """the loss curves of one run as a png (needs matplotlib); `keys`: a comma list of log keys (default: the loss and its groups). -> {png}"""
        from .curves import default_keys, plot as draw
        c = self._c()
        ids = [i] if not isinstance(i, (list, tuple)) else list(i)
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
        base_s, base_loss = rows[0][2]["s_step"], rows[0][1].window_mean("total", common)
        out = []
        for rid, r, x in rows:
            loss = r.window_mean("total", common)
            out.append(dict(id=rid, steps=x["step"], s_step=x["s_step"], speedup=(base_s / x["s_step"]) if (base_s and x["s_step"]) else None, data_share=x["data_share"],
                            gpu_busy=x["gpu_busy"], gpu_gb=x["gpu_gb"], loss_at_common=loss, loss_vs_base_pct=(100 * (loss - base_loss) / abs(base_loss)) if (loss is not None and base_loss) else None))
        return dict(common_step=common, rows=out)

    # ---- the queue (trainings and runner jobs share it)
    @errors
    def queue_list(self):
        """the queue in the order it runs, with what the box looks like. -> {items: [{number, id, kind, state, step, total, after, priority, held, grad, skipped, flags, ...}], running,
        runner (pid or 0), paused, slots, lock, ram, gpu, holds}"""
        c = self._c()
        q = TQ.load(c)
        items = _ordered(q)
        states = _states(c, items)
        infos = _map(lambda nx: _item_info(c, nx[1], nx[0], states), enumerate(items, 1))
        running = next(iter(_running(c, items, states)), (None,))[0]
        return dict(items=infos, running=running, runner=TQ.runner_alive(c), paused=bool(q.get("paused")), **_machine(c))

    @errors
    def queue_get(self, i):
        """one item: its place and state, the job it was made from, the composed config, and the exact command (the memory gate and the code pin included). -> dict"""
        return get_item(self._c(), i)

    @errors
    def queue_add(self, job, start=True):
        """check the job like `check`, put it in the queue and make sure the runner runs (start=False: not). A missing data build is refused here (a queued run would only fail when its
        turn comes). -> the item as queue_get shows it"""
        j = self._job(job)
        c = self._c(j)
        plan, problems, missing = _checked_plan(c, j, final=False, direct=False)
        if missing:
            problems.append("missing data: " + ", ".join(m["name"] for m in missing) + " (build it first)")
        if problems:
            raise TrainingError("nothing queued: " + "; ".join(problems))
        TQ.add(c, plan, j["id"], j["after"], j["priority"], j["note"], extra=dict(kind="train", job={k: v for k, v in j.items() if k != "project"}, kill=j["kill"]))
        if start:
            TQ.ensure_runner(c)
        return get_item(c, j["id"])                                        # the context of the job (its "project" may differ from this object's)

    @errors
    def queue_remove(self, i):
        """take an item out of the queue that has not started (or has ended); a running one is refused (it has to be stopped first). -> {removed}"""
        c = self._c()
        rid, it, _ = resolve(c, i)
        if it is None:
            raise TrainingError(f"{i!r} is not in the queue")
        TQ.drop(c, rid)
        return dict(removed=rid)

    @errors
    def queue_start(self):
        """make sure the runner runs: it starts the first startable item every 30 s, restarts a run whose launcher died with the machine, and ends after 10 idle minutes. -> {runner: pid}"""
        return dict(runner=TQ.ensure_runner(self._c()))

    @errors
    def queue_hold(self, words, i=None, reason="", wait=180.0):
        """stop the trainings (a checkpoint each, exit 75) and keep what waits from starting; i=None holds ALL and pauses the runner, i holds one (the command line wants `--all` for
        all). `words` are the person's own words that ordered it (required). -> {id: outcome}"""
        c = self._c()
        ids = () if i is None else (resolve(c, i)[0],)
        return TQ.hold(c, ids, i is None, words, reason, wait, out=lambda *a, **k: None)

    @errors
    def queue_resume(self, words, i=None, fresh_code=False):
        """the person's words clear the holds: every held training (i=None) or one continues from its last checkpoint on the code of its first attempt (fresh_code: on the code of now,
        the hot-fix loop). -> {id: what happens}"""
        c = self._c()
        ids = () if i is None else (resolve(c, i)[0],)
        return TQ.resume(c, ids, i is None, words, fresh_code, True, out=lambda *a, **k: None)


def delegate(cls, name):
    """the method `name` of `cls` as a function of a default object: `trainsuite.check(job)` is `Training().check(job)`"""
    def f(*a, **kw):
        return getattr(cls(), name)(*a, **kw)
    f.__name__, f.__doc__ = name, getattr(cls, name).__doc__
    sig = inspect.signature(getattr(cls, name))
    f.__signature__ = sig.replace(parameters=list(sig.parameters.values())[1:])        # help(trainsuite.queue_hold) shows the arguments, not (*a, **kw)
    return f


for _n in Training.METHODS:
    globals()[_n] = delegate(Training, _n)
