"""The training queue that outlives its launchers: a persistent list of runs, a runner that starts the next one, hold and resume that continue where a run stopped.
Standard library only; the launcher's own files (tools/run.py, hold.py, project.py) are imported when a command needs them.

  add      compose a run like `compose` and put it in the list (the config is stored with the item: what is queued is what runs)
  serve    the runner: every `every` seconds start the first startable item (its --after runs are done, nothing else waits for a slot); a run whose launcher died
           because the machine went down (WSL shutdown) is restarted from its last checkpoint; a held, failed, killed or finished item is never started; it ends when
           nothing is left to do
  hold     stop what trains (a checkpoint, exit 75), keep what waits from starting, remember how each run was started; `--all` also pauses the runner
  resume   the person's words clear the holds; each stopped run starts again from its checkpoint, on the code of its first attempt (run.py --pin), in its old place
  drop     remove an item that has not started (or has ended)

The list is `queue.<project>.json` in the GPU slot folder (not in git; one file per problem, the slot folder is shared by problems that share the GPU): items, a
`paused` mark, a sequence number. A run's STATE is never stored: it is read from the run record and the run folder (run.py's state_of), so the list cannot go stale.
The runner starts a run through run.py (the slots, the RAM gate and the holds are run.py's); one item waits for a slot at a time, so the line keeps its order.
A launcher that somebody killed by hand is not restarted by the runner (the person stopped it: `resume` continues it); only a launcher that died in an earlier boot of
the machine is (a WSL shutdown, a reboot).
"""
import contextlib
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
MAX_RESTARTS = 3                                   # restarts in a row without a new logged step; then the runner leaves the run alone (look at it)
ERROR_LIMIT = 10                                   # the runner ends after this many identical errors of its look in a row (it would only repeat them)
LAUNCH_RETRIES = 3                                 # a launcher that exits with 1 (not a refusal: run.py refuses with 2) is tried this many times
WAITING_WATCH_S = 60.0                             # `hold` watches a launcher that waited for its slot this long: it may start the trainer after its last look at the hold
HOLD_RESEND_S = 20.0                               # the runner asks a held item that runs anyway to stop at most this often
KILL_RESEND_S = 60.0                               # ... and an item whose kill rule hit again after this long, at most 3 times
LAUNCH_GRACE_S = 90.0                              # a launcher started this long ago that has no run record yet is still starting: not started a second time
LAUNCHABLE = ("queued", "restart", "interrupted")
OPEN = ("queued", "restart", "interrupted", "waiting", "running", "orphan", "held")      # the runner stays while one of these exists (held: the person's word is awaited)
_CHILDREN = []                                     # the launchers this process started: polled every look, so a killed one does not stay a zombie


class QueueError(Exception):
    pass


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def boot_id():
    """identity of this boot of the machine ('' where /proc does not say): a launcher that died in another boot died with the machine"""
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def context(project=None):
    """the project's paths and the launcher's modules"""
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    if project:
        os.environ["AR_PROJECT"] = project
    import hold as H
    import project as P
    import run as R
    cfg, root = P.cfg(), P.root()
    records = P.path("run_records")
    block = cfg.get("training") or {}
    slots = R.slot_dir(root, cfg, records)
    name = os.path.basename(root)
    return types.SimpleNamespace(P=P, R=R, H=H, cfg=cfg, root=root, records=records, block=block, slots=slots, qpath=os.path.join(slots, f"queue.{name}.json"),
                                 runs=os.path.join(root, block.get("runs", "runs")), runner=os.path.join(slots, f"queue.runner.{name}"))


def folder(rid):
    return rid.replace(".", "_")


def load(c):
    """the list; a file that cannot be read is moved aside (`.bad.<time>`) and refused: an empty list written over it would lose the holds and the pause"""
    try:
        with open(c.qpath, encoding="utf-8") as f:
            q = json.load(f)
        if not isinstance(q, dict):
            raise ValueError("not an object")
    except FileNotFoundError:
        q = {}
    except (OSError, ValueError) as e:
        bad = f"{c.qpath}.bad.{int(time.time())}"
        os.replace(c.qpath, bad)
        raise QueueError(f"{os.path.basename(c.qpath)} could not be read ({e}); moved to {os.path.basename(bad)}: the holds and the pause in it are not in force, look at it first")
    q.setdefault("paused", None)
    q.setdefault("seq", 0)
    q.setdefault("items", [])
    for it in q["items"]:
        if not isinstance(it, dict) or not it.get("id"):
            raise QueueError(f"{os.path.basename(c.qpath)} holds an item without an id ({str(it)[:80]}): fix or remove it first (the file is left as it is)")
        if not isinstance(it.get("seq"), int):                  # a hand edit or an older writer: the place in the line is the next free number
            q["seq"] += 1
            it["seq"] = q["seq"]
    return q


@contextlib.contextmanager
def edit(c):
    """the list under its lock; written back (atomically) when the block ends without an error"""
    os.makedirs(c.slots, exist_ok=True)
    with c.P.locked(c.qpath):
        q = load(c)
        yield q
        c.P.atomic_write(c.qpath, json.dumps(q, indent=1, ensure_ascii=False) + "\n")


def order(it):
    """the place of an item in the line: priority first (1 goes first), then the order it was added in"""
    return (it.get("priority", 5), it.get("seq", 0))


# ---- reading a run
def proc_argv(pid):
    """the command line of a process as a list (None where /proc does not exist or the process is gone; [] for a zombie)"""
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            raw = f.read().split(b"\0")
    except (OSError, ValueError, TypeError):
        return None
    if raw and raw[-1] == b"":
        raw.pop()
    return [x.decode("utf-8", "replace") for x in raw]


def launcher_alive(c, rid, rec):
    """is the launcher process of this run record alive, and really a run.py of this id (a reused pid is not)?"""
    pid = rec.get("launcher_pid")
    if not c.R.pid_alive(pid):
        return False
    argv = proc_argv(pid)
    return True if argv is None else (any(x.endswith("run.py") for x in argv) and rid in argv)


RUNNER_COMMAND = re.compile(r"(^|\s)-m\s+runner(\s|$)")        # `python -m runner ...`: the Runner of a trained network (bev_fusion/runner), no training


def is_training(c, rid, rec):
    """a run of this project that trains: its run folder exists, or its command names its own folder (the config is named after the id). An evaluation, a data build, a runner job
    (its command starts the runner; its output folder may carry the id) or a timing job is not one: hold leaves it alone"""
    cmd = rec.get("command") or ""
    if RUNNER_COMMAND.search(cmd):
        return False
    return os.path.isdir(os.path.join(c.runs, folder(rid))) or folder(rid) in cmd


def orphans(it):
    """pids of a trainer of this item that outlived its launcher (the launcher was killed, the trainer runs in a session of its own): they carry the command's last
    argument (the config path) in their command line. [] where /proc does not exist or the command has no such argument"""
    argv = it["argv"]
    key = argv[-1] if "--" in argv and argv[-1] != "--" and not argv[-1].startswith("-") else None
    if not key or not os.path.isdir("/proc"):
        return []
    out = []
    for name in os.listdir("/proc"):
        if name.isdigit() and int(name) != os.getpid():
            a = proc_argv(name)
            if a and key in a and not any(x.endswith("run.py") for x in a):
                out.append(int(name))
    return out


def state(c, it):
    """queued | waiting | running | orphan | restart | interrupted | held | done | failed | blocked, from the run record and the item
    waiting = the launcher stands in the line for a slot; restart = stopped, hold cleared; interrupted = the launcher died without ending the record and
    nothing trains any more; orphan = the launcher died but its trainer still runs (it is neither started again nor ignored)"""
    d = os.path.join(c.records, it["id"])
    s, rec = c.R.state_of(d), c.R.record(d)
    if s in ("done", "ok"):
        return "done"
    held = bool(it.get("held")) or c.H.read(d) is not None
    if s == "running":
        return "waiting" if rec.get("status") == "waiting" else "running"
    if s == "stalled":
        if launcher_alive(c, it["id"], rec):                # a suspended machine: the heartbeat is old, the launcher is not dead
            return "waiting" if rec.get("status") == "waiting" else "running"
        if orphans(it):
            return "orphan"
        return "held" if held else "interrupted"
    if held:
        return "held"
    if s in ("stopped", "held") or (it.get("retry") and s in ("failed", "timeout", "gpu_busy", "blocked")):      # `held`: hold cleared on a run that was held while it waited
        return "restart"
    if s in ("", None):
        return "queued"
    return "blocked" if s == "blocked" else "failed"


def restartable(it):
    """may the runner start an interrupted item by itself? Only when its launcher died in an earlier boot (the machine went down) and nobody saw it die in the boot it was started in
    (`person_stopped`: killed by hand), or the person asked (`resume`)"""
    return bool(it.get("retry")) or (bool(it.get("boot")) and it["boot"] != boot_id() and not it.get("person_stopped"))


def last_step(run_dir):
    """(step, total) of a run folder: the last logged step and the config's steps (0 where unknown)"""
    step = total = 0
    try:
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
            total = int(json.load(f).get("steps") or 0)
    except (OSError, ValueError, TypeError):
        pass
    try:
        with open(os.path.join(run_dir, "log.jsonl"), "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 20000))
            for line in f.read().decode("utf-8", "replace").splitlines()[::-1]:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and isinstance(r.get("step"), (int, float)):
                    step = int(r["step"])
                    break
    except OSError:
        pass
    return step, total


def first_commit(c, rec):
    """the commit of attempt 1 (the tag run.py made at the first start), '' when the run had no snapshot"""
    tag = rec.get("tag") or ""
    if not tag:
        return ""
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", c.root, "rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def pinned_argv(it):
    """the run.py command of an item, with `--pin <commit>` before the `--` when the item is pinned (an older --pin is replaced)"""
    argv = list(it["argv"])
    if "--" not in argv:
        return argv
    while "--pin" in argv[:argv.index("--")]:
        i = argv.index("--pin")
        del argv[i:i + 2]
    if it.get("pin"):
        argv[argv.index("--"):argv.index("--")] = ["--pin", it["pin"]]
    return argv


def with_gate(c, argv):
    """the run.py command with the project's memory gate (`gpu_max_mb` of the training block) before the `--`, unless it has one or asks for no GPU: an item queued before the gate
    existed, one rebuilt from its run record and one added by hand start under the same rule as `run` and `add`"""
    from . import compose as C
    try:
        gate = C.gpu_gate(c.block)
    except C.ComposeError as e:
        raise QueueError(str(e))
    head = argv[:argv.index("--")] if "--" in argv else list(argv)
    if not gate or "--gpu" not in head or any(x == "--gpu-max-mb" or x.startswith("--gpu-max-mb=") for x in head):
        return list(argv)
    return head + ["--gpu-max-mb", str(gate)] + argv[len(head):]


def argv_from_record(c, rid, rec):
    """the run.py command of a run that is not in the list, rebuilt from its record (the launcher is gone: nothing else is left to read it from)"""
    if not rec.get("command"):
        raise QueueError(f"{rid}: its run record has no command: cannot rebuild how it was started")
    argv = ["python3", os.path.join(TOOLS, "run.py"), "--id", rid, "--gpu", "--split", rec.get("split") or "train"]
    after = rec.get("after")
    if after:                                                   # the runs it waited for (they were done when it started: the wait is over at once)
        argv += ["--after", ",".join(after) if isinstance(after, (list, tuple)) else str(after)]
    if rec.get("timeout_min"):
        argv += ["--timeout", str(rec["timeout_min"])]
    if rec.get("cwd") and os.path.realpath(rec["cwd"]) != os.path.realpath(c.root):
        argv += ["--cwd", rec["cwd"]]
    card = [(f, rec.get(k)) for f, k in (("--hyp", "hypothesis"), ("--pred", "prediction"), ("--rule", "rule")) if rec.get(k)]
    if len(card) == 3:
        argv += [x for f, v in card for x in (f, v)]
    else:
        argv.append("--no-card")
    for flag, key in (("--basis", "basis"), ("--seed", "seed")):
        if rec.get(key) not in (None, ""):
            argv += [flag, str(rec[key])]
    words = rec.get("argv")                                     # run.py stores the list it ran: a word with a space, a quote or a $ survives; `command` is only that list joined with spaces
    if isinstance(words, list) and words and all(isinstance(x, str) for x in words):
        return argv + ["--"] + list(words)
    try:                                                        # a record from before `argv`: the joined text, split as well as it can be
        return argv + ["--"] + shlex.split(rec["command"])
    except ValueError as e:
        raise QueueError(f"{rid}: the command in its run record cannot be split back into words ({e})")


def new_item(q, rid, argv, **kw):
    q["seq"] += 1
    it = dict(id=rid, argv=list(argv), seq=q["seq"], priority=5, added=now(), after=[], held=None, pin=None, restarts=0, note="", by="")
    it.update(kw)
    q["items"].append(it)
    return it


# ---- add
def add(c, plan, rid, after=(), priority=5, note="", extra=None):
    """put a composed plan in the list; -> the item. `extra`: more fields of the item (the job, its kill rules). A plan without a config of its own (only `argv`) is stored without one"""
    with edit(c) as q:
        twin = next((i["id"] for i in q["items"] if folder(i["id"]) == folder(rid)), None)
        if twin == rid:
            raise QueueError(f"{rid} is already in the queue (queue drop {rid} first)")
        if twin:
            raise QueueError(f"{rid} has the run folder of {twin} (a . becomes _): two items must not share one folder")
        if c.R.state_of(os.path.join(c.records, rid)) in ("done", "ok"):
            raise QueueError(f"{rid} finished with exit 0: run.py refuses the id (use a new id)")
        composed = os.path.relpath(plan["composed"], c.root).replace(os.sep, "/") if plan.get("composed") else ""
        return new_item(q, rid, plan["argv"], composed=composed, config=plan.get("config"), after=list(after), priority=int(priority), note=note, **(extra or {}))


def write_config(c, it):
    """the item's composed config on disk, where the training command reads it (only when it differs: a resume needs the same file)"""
    if not it.get("config") or not it.get("composed"):
        return
    from . import compose as C
    p = os.path.join(c.root, it["composed"])
    try:
        if C.load(p) == it["config"]:
            return
    except C.ComposeError:
        pass
    os.makedirs(os.path.dirname(p), exist_ok=True)
    C.write(it["config"], p)


def write_kill(run_dir, rules):
    """append the kill rules that <run_dir>/kill.txt does not hold yet (an existing file is kept, a UTF-16 one is left alone); creates the run folder: call it only for a run that
    has started (a stray folder with only a kill.txt would be listed as a run). -> the number of rules added"""
    if not rules:
        return 0
    os.makedirs(run_dir, exist_ok=True)
    kp = os.path.join(run_dir, "kill.txt")
    txt = ""
    if os.path.exists(kp):
        with open(kp, "rb") as f:
            if f.read(2) in (b"\xff\xfe", b"\xfe\xff"):
                return 0
        with open(kp, encoding="utf-8-sig", errors="replace") as f:
            txt = f.read()
    norm = lambda s: " ".join(s.split())                              # noqa: E731  spacing is no difference between two rules
    have = [norm(h.split("#", 1)[0]) for h in txt.splitlines()]
    new = [r for r in rules if norm(r) not in have]
    if new:
        with open(kp, "a", encoding="utf-8") as f:
            f.write(("\n" if txt and not txt.endswith("\n") else "") + "".join(r + "\n" for r in new))
    return len(new)


# ---- the runner
def ran_command(c, it, pid):
    """True when the launcher with this pid ran the command and wrote its end to the run record (status ok / failed / timeout / stopped, launcher_pid = pid): its exit code is then the
    command's. run.py also exits non-zero for itself (refused, a hold, a card missing, a lock): nothing is recorded by that launcher then"""
    rec = c.R.record(os.path.join(c.records, it["id"]))
    return rec.get("launcher_pid") == pid and rec.get("status") in ("ok", "failed", "timeout", "stopped")


def launch(c, it, wait=3.0):
    """start the item's run.py detached; a launcher that refuses within `wait` seconds (held id, finished, no card ...) marks the item `refused`"""
    write_config(c, it)
    try:
        argv = with_gate(c, pinned_argv(it))
    except QueueError as e:                             # a project.json whose gate is no number is the project's fault, not the item's: nothing starts (the view says why), the item stays queued
        return False, f"not started: {str(e)[:300]}"
    log = os.path.join(c.runs, folder(it["id"]) + ".launcher.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "ab") as lf:
        proc = subprocess.Popen(argv, cwd=c.root, env=dict(os.environ, AR_PROJECT=c.root), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        rc = None
    ran = rc is not None and rc > 0 and ran_command(c, it, proc.pid)
    if rc is not None and rc > 0 and not ran:           # a launcher killed by a signal (rc < 0) did not refuse: its record tells
        with open(log, encoding="utf-8", errors="replace") as lf:
            tail = [x for x in lf.read().strip().splitlines()[-2:] if x]
        text = " | ".join(tail)[:300] or f"exit {rc}"
        if rc == 1 and it.get("launch_failures", 0) < LAUNCH_RETRIES - 1:          # run.py refuses with 2; exit 1 is a launcher that failed (a file of tools/ being edited, a hiccup): again
            it["launch_failures"] = it.get("launch_failures", 0) + 1
            it["launched_t"] = time.time()                                          # tick waits LAUNCH_GRACE_S before the next try
            return False, f"launcher failed (exit 1, try {it['launch_failures']} of {LAUNCH_RETRIES}): {text}"
        it["refused"] = text
        return False, f"launcher refused (exit {rc}): {it['refused']}"
    write_kill(os.path.join(c.runs, folder(it["id"])), it.get("kill"))     # the launcher survived its first seconds: the job's kill rules go where `watch` and `status` read them
    _CHILDREN.append(proc)
    it["launched"], it["launched_t"], it["boot"] = now(), time.time(), boot_id()
    it.pop("retry", None)
    it.pop("launch_failures", None)
    if ran:                                             # the trainer failed within the wait (a bad config, an import): a failed run with its record, not a launcher to try again
        return True, f"started, launcher pid {proc.pid}; the command ended with exit {rc} within {wait:g} s: the run failed (its log tells why)"
    return True, f"started, launcher pid {proc.pid}"


def dependency(c, it, states):
    """'' when every --after run of the item is done, 'wait' while one is still to come, else 'blocked by <id> (<state>)'"""
    wait = False
    for dep in it.get("after") or []:
        s = states.get(dep) or c.R.state_of(os.path.join(c.records, dep)) or "unknown (no record, not in the queue)"
        if s in ("done", "ok"):
            continue
        if s in ("queued", "waiting", "running", "restart", "interrupted"):
            wait = True
        else:
            return f"blocked by {dep} ({s})"
    return "wait" if wait else ""


def tick(c, q, wait=3.0):
    """one look: start the first startable item (at most one); -> the lines to log. Changes the items (launched, refused, restarts, gave_up, pin)"""
    acts = []
    if q.get("paused"):
        return acts
    items = sorted(q["items"], key=order)
    states = {i["id"]: state(c, i) for i in items}
    if any(s == "waiting" for s in states.values()):                   # one item stands in the line for its slot at a time: the order stays
        return acts
    if sum(s in ("running", "orphan") for s in states.values()) >= int(c.cfg.get("gpu_slots", 2)):
        return acts
    for it in items:
        st = states[it["id"]]
        if st not in LAUNCHABLE or it.get("refused") or it.get("gave_up"):
            continue
        if st == "queued" and time.time() - it.get("launched_t", 0) < LAUNCH_GRACE_S:     # started a moment ago, the launcher has not written its record yet
            continue
        if dependency(c, it, states):
            continue
        rec = c.R.record(os.path.join(c.records, it["id"]))
        if st == "interrupted":
            if not restartable(it):
                continue                                                # killed in this boot: the person's call (`resume`), not the runner's
            step = last_step(os.path.join(c.runs, folder(it["id"])))[0]
            if step > it.get("restart_step", -1):                       # it got further since the last restart: not "in a row"
                it["restarts"], it["restart_step"] = 0, step
            it["restarts"] = it.get("restarts", 0) + 1
            if it["restarts"] > MAX_RESTARTS:
                it["gave_up"] = True
                acts.append(f"{it['id']}: its launcher died {MAX_RESTARTS} times in a row without a new step: not started again (look at it, then `queue resume {it['id']} --words ...`)")
                continue
        if st in ("interrupted", "restart") and not it.get("pin") and not it.get("fresh"):
            it["pin"] = first_commit(c, rec) or None                    # a restart continues the experiment: the code of its first attempt, not the working tree of now
        ok, text = launch(c, it, wait)
        acts.append(f"{it['id']} ({st}): {text}")
        break
    return acts


def enforce(c, q):
    """what the runner does besides starting items. (1) A held item that runs anyway (its launcher was past its last look at the hold when `hold` came) is asked to stop again.
    (2) A running item whose job has kill rules: a rule that hits asks it to stop (run.py then puts its id on hold; the person's `resume` waives that rule). -> the lines to log"""
    from .kill import Rule
    from .runs import Run
    acts = []
    for it in q["items"]:
        rid = it["id"]
        if state(c, it) == "interrupted" and it.get("boot") == boot_id() and not it.get("person_stopped"):
            # a launcher whose heartbeat is 5 minutes old in the boot it was started in: the machine going down takes this runner with it, so somebody killed it. A reboot later must not
            # restart it against that decision (only `resume` does)
            it["person_stopped"] = now()
            acts.append(f"{rid}: its launcher died in this boot (killed by hand): it is not restarted by the runner, not even after a reboot; `resume {rid} --words ...` continues it")
        if state(c, it) not in ("running", "orphan"):
            continue
        if it.get("held"):
            if time.time() - it.get("hold_stop_t", 0) > HOLD_RESEND_S:
                err = _request_stop(c, rid, "queue hold")
                it["hold_stop_t"] = time.time()
                acts.append(f"{rid}: on hold but running: " + ("stop requested" if not err else f"NOT stopped: {err}"))
            continue
        hit = it.get("kill_hit")
        if not it.get("kill") or (hit and (hit["sent"] >= 3 or time.time() - hit["t"] < KILL_RESEND_S)):
            continue
        run = Run(os.path.join(c.runs, folder(rid)), "total")
        for text in it["kill"]:
            if text in (it.get("kill_waived") or []):
                continue
            try:
                status, value = Rule(text).evaluate(run)
            except ValueError:                                                  # a rule that does not parse was refused when the job was read
                continue
            if status == "hit":
                err = _request_stop(c, rid, f"kill rule {text!r} hit (value {value})")
                hit = it.setdefault("kill_hit", {"rule": text, "value": value, "sent": 0})
                hit.update(t=time.time(), sent=hit["sent"] + 1)
                acts.append(f"{rid}: kill rule {text!r} hit (value {value}): " + ("stop requested" if not err else f"NOT stopped: {err}"))
                break
    return acts


def runner_alive(c):
    """pid of the live runner of this project, else 0"""
    try:
        with open(c.runner, encoding="utf-8") as f:
            pid = int(json.load(f)["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0
    argv = proc_argv(pid)
    return pid if c.R.pid_alive(pid) and (argv is None or "serve" in argv) else 0


def ensure_runner(c):
    """start the runner detached when none is alive; -> its pid (a new one or the old one). A second runner that is started at the same moment ends itself (serve)"""
    pid = runner_alive(c)
    if pid:
        return pid
    os.makedirs(c.slots, exist_ok=True)
    with open(c.runner + ".log", "ab") as lf:
        p = subprocess.Popen([sys.executable, "-m", "Training", "queue", "serve", "--project", c.root], cwd=os.path.dirname(HERE), env=dict(os.environ, AR_PROJECT=c.root),
                             stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    return p.pid


def serve(c, every=30.0, idle_exit=600.0, once=False, out=print):
    """the runner loop. -> exit code"""
    me = json.dumps({"pid": os.getpid(), "project": c.root, "since": now()})
    os.makedirs(c.slots, exist_ok=True)
    with c.P.locked(c.runner + ".start"):                               # check and claim in one step: two runners started at once cannot both pass
        other = runner_alive(c)
        if other not in (0, os.getpid()):
            out(f"a runner is already running (pid {other}): nothing to do")
            return 1
        c.P.atomic_write(c.runner, me)
    idle_since = None
    try:
        same_error = ("", 0)                                        # the last error of a look and how often in a row
        while True:
            c.P.atomic_write(c.runner, me)
            _CHILDREN[:] = [p for p in _CHILDREN if p.poll() is None]
            try:
                with edit(c) as q:
                    acts = tick(c, q) + enforce(c, q)
                    states = [state(c, i) for i in q["items"]]
                    paused = q.get("paused")
                same_error = ("", 0)
            except QueueError as e:                                     # an unreadable list: say so, never write over it
                out(f"[{now()}] {e}")
                if once:
                    return 2
                time.sleep(every)
                continue
            except Exception as e:                                      # noqa: BLE001  one bad item must not end the runner
                text = f"{type(e).__name__}: {e}"
                same_error = (text, same_error[1] + 1 if same_error[0] == text else 1)
                out(f"[{now()}] runner error ({text}); looking again in {every:g} s")
                if once:
                    return 2
                if same_error[1] >= ERROR_LIMIT:
                    out(f"[{now()}] the same error {same_error[1]} times in a row: the runner ends (look at the queue, then `queue add` / `queue resume` start it again)")
                    return 2
                time.sleep(every)
                continue
            for a in acts:
                out(f"[{now()}] {a}")
            if once:
                return 0
            busy = paused or any(s in OPEN and not (s == "interrupted" and not restartable(i)) for i, s in zip(q["items"], states))
            idle_since = None if busy else (idle_since or time.time())
            if idle_since and time.time() - idle_since >= idle_exit:
                out(f"[{now()}] nothing left to do for {idle_exit:g} s: the runner ends (`queue add` / `queue resume` start it again)")
                return 0
            time.sleep(every)
    finally:
        try:
            with open(c.runner, encoding="utf-8") as f:
                if json.load(f).get("pid") == os.getpid():
                    os.remove(c.runner)
        except (OSError, ValueError):
            pass


# ---- hold and resume
def need_words(words, what):
    if not words or not words.strip():
        raise QueueError(f"{what} needs --words: the person's own words (the order is theirs, hard rule 6)")


def _request_stop(c, rid, why):
    """CONTROL in the run folder (made when the trainer has not created it yet); -> '' or the reason it failed"""
    from . import control
    rd = os.path.join(c.runs, folder(rid))
    try:
        if control.honours_stop(rd) is False:
            return "its code snapshot has no stop check (stopping it is a decision about the process)"
        os.makedirs(rd, exist_ok=True)
        control.request_stop(rd, why)
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    return ""


def hold(c, ids=(), all_=False, words="", reason="", wait=180.0, out=print):
    """stop what trains (a checkpoint, exit 75), keep what waits from starting; remember how each run was started. `all_` also pauses the runner. An evaluation, a data
    build or any run that is not a training is not touched. -> {id: outcome}"""
    need_words(words, "hold")
    why = reason or "queue hold"
    w = words.strip()
    result, pending, stop_sent = {}, [], {}
    with edit(c) as q:                                                  # the runner cannot start anything while this block holds the lock
        active = {}                                                     # id -> (record, launcher argv): runs whose launcher is alive
        for rid in sorted(os.listdir(c.records)) if os.path.isdir(c.records) else []:
            rec = c.R.record(os.path.join(c.records, rid))
            if rec.get("status") in ("running", "waiting") and launcher_alive(c, rid, rec):
                active[rid] = (rec, proc_argv(rec["launcher_pid"]))
        known = {i["id"]: i for i in q["items"]}
        live = {r for r, i in known.items() if state(c, i) in LAUNCHABLE + ("orphan", "running", "waiting")}
        for rid in list(active):
            if rid not in known and not is_training(c, rid, active[rid][0]):
                if all_ or rid in ids:
                    result[rid] = "not touched: not a training (no run folder of its own, its command does not name one)"
                del active[rid]
        wanted = sorted((set(active) | live) if all_ else set(ids))
        for rid in ids:
            if rid not in active and rid not in known and rid not in result:
                raise QueueError(f"{rid}: no such run in the queue and no live launcher of it")
        wanted = [r for r in wanted if r not in result]
        base = min([i["seq"] for i in q["items"]] + [0]) - 1000
        for n, rid in enumerate(sorted((r for r in wanted if r in active and r not in known), key=lambda r: active[r][0].get("created", ""))):
            rec, argv = active[rid]
            known[rid] = new_item(q, rid, argv or argv_from_record(c, rid, rec), by="captured by hold")
            known[rid]["seq"] = base + n                                # a run that was going keeps its place in front of what was queued
        for rid in wanted:
            known[rid]["held"] = {"since": now(), "words": w, "why": why}
            d = os.path.join(c.records, rid)
            if rid in active:
                c.H.write(d, "queue hold", why, w)                      # run.py reads the hold while it waits and before it starts the command
        if all_:
            q["paused"] = {"since": now(), "words": w, "why": why}
        for rid in wanted:
            it, s = known[rid], state(c, known[rid])
            rec = c.R.record(os.path.join(c.records, rid))
            if rec.get("status") == "running" and (rid in active or s in ("orphan", "running")):
                if s == "running" and rid not in active and not orphans(it):
                    result[rid] = "its launcher is gone and no trainer is running: held in the queue"
                    continue
                err = _request_stop(c, rid, why)
                result[rid] = "stop requested" if not err else f"RUNNING and NOT stopped: {err}"
                if not err:
                    stop_sent[rid] = time.time()
                    pending.append(rid)
            elif rid in active:
                result[rid] = "was waiting for a slot: on hold"
            else:
                result[rid] = f"not started ({s}): held in the queue"
    watching = [r for r in wanted if result[r].startswith("was waiting for a slot")]
    deadline = time.time() + wait
    watch_until = time.time() + WAITING_WATCH_S
    while (pending or watching) and time.time() < max(deadline, watch_until if watching else 0):
        for r in list(watching):                                        # a launcher past its last look at the hold starts the trainer anyway: stop it when its record turns running
            status = c.R.record(os.path.join(c.records, r)).get("status")
            if status == "running":
                err = _request_stop(c, r, why)
                result[r] = "stop requested" if not err else f"RUNNING and NOT stopped: {err}"
                if not err:
                    stop_sent[r] = time.time()
                    pending.append(r)
                watching.remove(r)
            elif status != "waiting" or time.time() > watch_until:
                watching.remove(r)
        pending = [r for r in pending if state(c, known[r]) in ("running", "waiting", "orphan")]
        for r in pending:
            if time.time() - stop_sent[r] > 20:                         # a CONTROL written before the trainer cleared stale ones is lost: ask again
                _request_stop(c, r, why)
                stop_sent[r] = time.time()
        if pending or watching:
            time.sleep(2)
    for rid in wanted:
        if result[rid].startswith("stop requested"):
            step, total = last_step(os.path.join(c.runs, folder(rid)))
            s = state(c, known[rid])
            if rid in pending:
                result[rid] = f"stop requested, still running after {wait:g} s (the trainer stops at its next step boundary)"
            elif s == "done":
                result[rid] = f"FINISHED (step {step} of {total}) before it could be stopped; its hold is cleared"
                with edit(c) as q:
                    for i in q["items"]:
                        if i["id"] == rid:
                            i["held"] = None
                d = os.path.join(c.records, rid)
                if c.H.read(d) is not None:
                    c.H.clear(d, w)
            else:
                result[rid] = f"stopped at step {step} of {total} (checkpoint saved, on hold)"
    for rid in sorted(result):
        out(f"  {rid}: {result[rid]}")
    if all_:
        out("the runner is paused: nothing new starts until `queue resume --all --words ...`")
    return result


def resume(c, ids=(), all_=False, words="", fresh_code=False, start=True, out=print):
    """clear the holds of the given runs (all = every held item and the pause), each stopped run starts again from its checkpoint on the code of its first attempt
    (`fresh_code`: on the code of now: the hot-fix loop); a run that is not in the queue is rebuilt from its record. -> {id: what will happen}"""
    need_words(words, "resume")
    result = {}
    with edit(c) as q:
        known = {i["id"]: i for i in q["items"]}
        wanted = [i["id"] for i in sorted(q["items"], key=lambda i: i["seq"]) if state(c, i) == "held" or i.get("held")] if all_ else list(ids)
        for rid in wanted:
            d = os.path.join(c.records, rid)
            rec = c.R.record(d)
            if rid not in known:
                if not rec:
                    raise QueueError(f"{rid}: neither in the queue nor a run with a record")
                known[rid] = new_item(q, rid, argv_from_record(c, rid, rec), by="rebuilt from the run record by resume")
                known[rid]["seq"] = min([i["seq"] for i in q["items"]] + [0]) - 1
            it = known[rid]
            h = c.H.read(d)
            if h is not None:
                c.H.clear(d, words)
                c.H.note_in_ledger(rid, f"hold cleared {dt.date.today().isoformat()} on the person's words: {words.strip()!r} (hold was: {h.get('why', '?')}, set by {h.get('who', '?')} {h.get('when', '?')})")
            it["held"] = None
            it.pop("person_stopped", None)
            if it.get("kill_hit"):                                      # the person's words continue a run that a kill rule stopped: that rule is waived for this item
                it.setdefault("kill_waived", []).append(it.pop("kill_hit")["rule"])
            it.pop("refused", None)
            it.pop("gave_up", None)
            it["restarts"], it["retry"] = 0, True
            it["pin"] = None if fresh_code else (first_commit(c, rec) or None)
            it["fresh"] = bool(fresh_code)                              # the person asked for the code of now: no pin, also when the runner restarts it later
            step, total = last_step(os.path.join(c.runs, folder(rid)))
            result[rid] = (f"starts again from its last checkpoint (step {step} of {total}), " + ("on the code of now" if fresh_code or not it["pin"] else f"on the code of its first attempt ({it['pin'][:10]})")
                           if step else "starts from the beginning (no checkpoint)")
        if all_:
            q["paused"] = None
        elif q.get("paused") and wanted:
            out("note: the runner is still PAUSED (hold --all): `queue resume --all --words ...` ends the pause")
    for rid, text in result.items():
        out(f"  {rid}: {text}")
    if start:
        pid = ensure_runner(c)
        out(f"the runner is running (pid {pid}); `queue` shows the line")
    return result


def start_blocks(c, rid):
    """why the person's pause or hold forbids a start NOW of this id (also one that does not go through the queue): [] when nothing does. hold(all) pauses the runner and `held` marks the
    items: both mean "nothing new starts" (hard rule 6). Reads the queue file only"""
    q = load(c)
    out = []
    if q.get("paused"):
        out.append("the queue is PAUSED by a hold (the person's words): queue_resume clears it")
    if any(x["id"] == rid and x.get("held") for x in q["items"]):
        out.append(f"the queue item {rid} is held (the person's words): queue_resume clears it")
    return out


def drop(c, rid):
    with edit(c) as q:
        it = next((i for i in q["items"] if i["id"] == rid), None)
        if it is None:
            raise QueueError(f"{rid} is not in the queue")
        s = state(c, it)
        if s in ("waiting", "running"):
            raise QueueError(f"{rid} is {s}: `queue hold {rid} --words ...` stops it first")
        q["items"].remove(it)


def run_brief(c, rid):
    """what a person wants to see of a run in the line: the gradient norm now (against before), the skipped non-finite steps and the health flags that are no info, in one
    text ('grad 19.1 (x1.4); 3 skipped; flags: grad_norm,skipped'); '' for a run without metrics or one that cannot be read"""
    from . import health, modules
    from .runs import Run
    try:
        run = Run(os.path.join(c.runs, folder(rid)))
        if not run.train:
            return ""
        parts, g, n_skip = [], modules.grad_trend(run), run.skipped_steps()[0]
        if g:
            parts.append("grad NOT finite" if g["now"] is None else f"grad {g['now']:.3g}" + (f" (x{g['ratio']:.1f})" if g["ratio"] else ""))
        if n_skip:
            parts.append(f"{n_skip} skipped")
        flags = sorted({f.code for f in health.check(run) if f.level != "info"})
        if flags:
            parts.append("flags: " + ",".join(flags))
        return "; ".join(parts)
    except Exception as e:  # noqa: BLE001  a damaged run folder must not break the view of the line
        return f"(its metrics could not be read: {type(e).__name__})"


def show(c, out=print):
    """the runner, the pause and the items with their derived state"""
    try:
        q = load(c)
    except QueueError as e:
        out(f"queue: {e}")
        return
    pid = runner_alive(c)
    out(f"queue runner: " + (f"running (pid {pid})" if pid else "not running (`queue add` / `queue resume` start it, or `queue serve`)"))
    if q.get("paused"):
        out(f"PAUSED since {q['paused'].get('since')}: nothing new starts. The person's words: {q['paused'].get('words')!r}")
    if not q["items"]:
        out("queue items: none")
        return
    out("queue items (the runner starts the first startable one when its --after runs are done and nothing else waits for a slot):")
    states = {i["id"]: state(c, i) for i in q["items"]}
    for it in sorted(q["items"], key=order):
        s = states[it["id"]]
        step, total = last_step(os.path.join(c.runs, folder(it["id"])))
        extra = [f"step {step}/{total}"] if step or total else []
        brief = run_brief(c, it["id"])
        if brief:
            extra.append(brief)
        if s == "interrupted" and not restartable(it):
            extra.append("its launcher was killed in this boot: `resume` continues it")
        if it.get("after"):
            dep = dependency(c, it, states)
            extra.append("after " + ",".join(it["after"]) + (f" ({dep})" if dep else ""))
        if it.get("pin"):
            extra.append("pinned to " + it["pin"][:10])
        for key in ("refused", "gave_up"):
            if it.get(key):
                extra.append(f"{key}: {it[key]}")
        if it.get("held"):
            extra.append("held: " + repr(it["held"].get("words", ""))[:70])
        out(f"  {it['id']:<22} {s:<12} " + "; ".join(extra))
