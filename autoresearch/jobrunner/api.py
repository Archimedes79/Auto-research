"""JobRunner - one way to say "run this method of that suite on these data", and one place to see what was asked and what became of it.

    JobRunner(project).check(job)     # looks, starts nothing: problems, the machine it needs, the data it reads and writes
    JobRunner(project).submit(job)    # writes the job and starts it under run.py (which waits in its own line for slots, RAM, the lock and --after)
    JobRunner(project).list()         # every job and its state;  .status(id)  one job with its log tail and result

A job is a dict or a json file (README.md has the fields). It names ONE call (`Class.method` of a suite) and the arguments of that method; the data it works on are
arguments (`network`, `frames`, `data_root`...), and the catalogue turns them into typed references (`run:t23.C_s1`, `set:large.val`) that are checked and give the
dependencies. Nothing here schedules: run.py is the queue (its FIFO line, GPU slots, RAM gate, machine lock, --after, ledger row, card and TEST gates). The job lives in
the folder `<run_records>/.jobs/<id>/`; its state is read with run.py's own functions from run.py's record `<run_records>/<id>/` - there is no second store.
Standard library only; it needs Linux (/proc), i.e. WSL on this machine."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from . import catalogue as C

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.dirname(HERE)                                                 # autoresearch/
RUN_PY = os.path.join(AR, "tools", "run.py")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
KEYS = ("id", "call", "args", "needs", "reads", "writes", "after", "priority", "card", "no_card", "split", "python", "run_args", "retries", "timeout", "note")
LAUNCH_REFUSES = ("after", "timeout", "split", "retries", "run_args", "card", "no_card", "python")  # run.py options: a `launch` method starts its own run and takes them in its arguments
LIVE = ("queued", "running", "held")                                       # not started again; `stalled` counts as live while a process of it lives; failed, blocked, stopped, planned
                                                                           # and a dead `stalled` can be started (run.py resumes those ids)
RESERVED_RUN_ARGS = ("--", "--id", "--after", "--gpu", "--gpu-max-mb", "--ram-mb", "--no-card", "--require-card", "--hyp", "--pred", "--rule", "--basis", "--split", "--retries", "--timeout",
                     "--no-git", "--again", "--change-card", "--cwd", "--result")           # the job form owns these; in run_args they would contradict it
RUN_NAMES = ("state_of", "record", "command_alive", "pid_alive", "twin_refusal", "is_holdout", "run_folder", "HOLD", "ledger_row", "test_log_refusal")   # what jobrunner uses of run.py
SPLIT_PARAMS = ("split", "frames", "set", "sets", "subset")                # the arguments scanned for the holdout split unless the catalogue entry names its own (`splits`)
LOCK_WAIT_S = float(os.environ.get("JOBRUNNER_LOCK_WAIT_S", "10"))
_MODULES: Dict[str, Any] = {}


class JobError(Exception):
    """what to fix, in a sentence"""


def _tool(name: str):
    """a module of autoresearch/tools/, loaded by path once (run.py gives the state of a run, project.py the problem folder)"""
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location("ar_" + name, os.path.join(AR, "tools", name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
    return _MODULES[name]


def _stat(pid: Any) -> Optional[List[str]]:
    """the fields after the name in /proc/<pid>/stat (state first, start time at index 19), None when there is no such process. Needs /proc: os.kill(pid, 0) would terminate it on Windows"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if not os.path.isdir("/proc"):
        raise JobError("the job runner needs Linux /proc: run it under WSL")
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None


def _alive(pid: Any, start: Optional[str] = None) -> bool:
    """is a process with this pid alive (a zombie is not; with `start` it must also be the process that began at that time, not a later one that got the pid)"""
    st = _stat(pid)
    return st is not None and st[0] != "Z" and (start is None or len(st) > 19 and st[19] == start)


def _spawn(argv: List[str], cwd: str, env: Dict[str, str], log: str) -> int:
    """the default launcher: run.py in the background in its own session, its output to the launch log"""
    with open(log, "ab") as f:
        return subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True).pid


def _write(path: str, doc: Any) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=str)
    os.replace(tmp, path)


def _read(path: str) -> Optional[Dict[str, Any]]:
    for k in range(3):                                                     # the files are replaced atomically; /mnt/c can still show a half-written one for a moment
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            return None
        except ValueError:
            time.sleep(0.05 * (k + 1))
    return None


def _tail(path: str, n: int = 15) -> List[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def _strings(obj: Any):
    """every text inside args (values, not keys)"""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


class JobRunner:
    """see the module docstring; every method is also a command (`python -m jobrunner JobRunner.submit job.json`)"""
    METHODS = ("catalogue", "check", "submit", "start", "run", "list", "status")

    def __init__(self, project: Optional[str] = None, launcher: Optional[Callable[[List[str], str, Dict[str, str], str], int]] = None, python: Optional[str] = None):
        self.python = python or os.environ.get("JOBRUNNER_PYTHON", "")          # the interpreter of the child processes, as an option; nothing needs to be set in project.json
        try:
            tool = _tool("project")
            if project:
                self.root = os.path.abspath(project)
                if not os.path.isfile(os.path.join(self.root, "project.json")):
                    raise JobError(f"no project.json in {self.root}: that folder is not a problem folder")
            else:
                self.root = tool.root()
            self.cfg = tool.cfg_of(self.root)
        except SystemExit as ex:
            raise JobError(str(ex.code)) from None
        self.launcher = launcher or _spawn
        self.records = os.path.normpath(os.path.join(self.root, self.cfg["run_records"]))
        self.jobs = os.path.join(self.records, ".jobs")
        self.dirs = (self.root, AR)                                        # the order of sys.path in the child (call.py): the project's own packages first
        self._table: Optional[Dict[str, Dict[str, Any]]] = None
        self._run = None
        self._sigs: Dict[Tuple[str, str, str, str], Any] = {}

    @property
    def R(self):
        """run.py as a module: its functions say what a run record means (state_of, command_alive, twin_refusal, is_holdout, the TEST log). A name that run.py no longer has is a JobError"""
        if self._run is None:
            mod = _tool("run")
            missing = [n for n in RUN_NAMES if not hasattr(mod, n)] + ([] if hasattr(mod, "grant_ids") or hasattr(mod, "decided_ids") else ["grant_ids"])
            if missing:
                raise JobError("run.py has changed: it no longer has " + ", ".join(missing) + " (jobrunner/api.py RUN_NAMES)")
            self._run = mod
        return self._run

    def _sig(self, entry: Dict[str, Any], python: str):
        """the signature of a catalogue method, asked once per runner (importing a heavy suite in a child interpreter takes a while)"""
        key = (entry["suite"], entry["cls"], entry["method"], python)
        if key not in self._sigs:
            self._sigs[key] = C.signature(entry, self.dirs, python)
        return self._sigs[key]

    # ------------------------------------------------------------------------------------------------------------ the catalogue
    def table(self) -> Dict[str, Dict[str, Any]]:
        if self._table is None:
            self._table, problems = C.entries(self.cfg, self.dirs)
            if problems:
                raise JobError("the catalogue is wrong: " + "; ".join(problems))
        return self._table

    def catalogue(self) -> List[Dict[str, Any]]:
        """the calls that are jobs: how each runs, what it takes, which data it reads and writes, and its parameters (read from the class, here or by the interpreter of the job)"""
        out = []
        for call, e in sorted(self.table().items()):
            params, why = self._sig(e, self._python("", e))
            shown = ", ".join(p["name"] + ("" if p["required"] else "=") for p in params) if params is not None else f"(not readable here: {why})"
            out.append(dict(call=call, how=e["how"], may_feed_verdict=bool(e.get("may_feed_verdict")), needs=e.get("needs", {}), reads=e.get("reads", []), writes=e.get("writes", []), parameters=shown))
        return out

    # ------------------------------------------------------------------------------------------------------------ one job: normalise and check
    def _jd(self, jid: str, name: str = "") -> str:
        return os.path.join(self.jobs, jid, name) if name else os.path.join(self.jobs, jid)

    def _load(self, job: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(job, (str, os.PathLike)):
            doc = _read(os.fspath(job))
            if doc is None:
                raise JobError(f"cannot read the job file {job}")
        elif isinstance(job, dict):
            doc = dict(job)
        else:
            raise JobError("a job is a dict or the path of a json file")
        doc = {k: v for k, v in doc.items() if not str(k).startswith("_")}
        if "call" not in doc and "config" in doc:                          # a training job file of trainsuite, as it is
            doc.pop("kind", None)
            doc = dict(id=doc.get("id", ""), call="Training.run", args=dict(job=doc))
        return doc

    def _plan(self, job: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        d = self._load(job)
        problems: List[str] = [f"unknown key {k!r} (keys: {', '.join(KEYS)})" for k in d if k not in KEYS]
        warnings: List[str] = []
        table = self.table()
        call, args = d.get("call", ""), d.get("args", {})
        entry = table.get(call) if isinstance(call, str) else None
        if not entry:
            problems.append(f"call {call!r} is not in the catalogue (`catalogue` lists them)")
        if not isinstance(args, dict):
            problems.append("args must be a dict")
            args = {}
        try:
            args = json.loads(json.dumps(args))                            # what the child will read: tuples are lists
        except (TypeError, ValueError) as ex:
            problems.append(f"args must be plain json ({ex})")
            args = {}
        jid = d.get("id") or (C.id_from(entry["id_from"], args) if entry and entry.get("id_from") else "")
        if not isinstance(jid, str) or not ID_RE.match(jid):
            problems.append(f"id {jid!r} is missing or not a plain name (letters, digits, . _ -)")
            jid = jid if isinstance(jid, str) else ""
        if entry and entry.get("id_from") and C.id_from(entry["id_from"], args) not in ("", jid):
            problems.append(f"id {jid!r} differs from {entry['id_from']} of the arguments")
        launch = bool(entry) and entry["how"] == "launch"
        if launch:
            problems += [f"{k} has no effect on a `launch` call ({call} starts its own run; give it in the arguments of the method)" for k in LAUNCH_REFUSES if d.get(k) not in (None, False, [], "", 0)]
        needs = dict(entry.get("needs", {})) if entry else {}
        extra = d.get("needs", {})
        if not isinstance(extra, dict) or any(k not in C.NEEDS_KEYS for k in extra):
            problems.append(f"needs must be a dict with keys from {C.NEEDS_KEYS}")
        else:
            needs.update(extra)
        problems += C.check_needs(needs)
        if needs.get("gpu") is False and needs.get("gpu_max_used_mb"):
            problems.append("needs.gpu is false but gpu_max_used_mb is set")
        for k in ("after", "run_args", "reads", "writes"):
            if k in d and not (isinstance(d[k], list) and all(isinstance(x, str) for x in d[k])):
                problems.append(f"{k} must be a list of texts")
        if isinstance(d.get("run_args"), list):
            problems += [f"run_args {t!r}: the job form owns this option (use the field of the job)" for t in d["run_args"] if isinstance(t, str) and t.split("=", 1)[0] in RESERVED_RUN_ARGS]
        card, no_card = d.get("card"), bool(d.get("no_card"))
        if card is not None and not (isinstance(card, dict) and all(isinstance(card.get(k), str) and card.get(k) for k in ("hyp", "pred", "rule")) and set(card) <= {"hyp", "pred", "rule"}):
            problems.append("card must be {hyp, pred, rule}, three non-empty texts")
            card = None
        if card and no_card:
            problems.append("card and no_card together")
        if card and entry and not entry.get("may_feed_verdict"):
            problems.append(f"{call} never feeds a verdict: remove the card")
        if "retries" in d and (isinstance(d["retries"], bool) or not isinstance(d["retries"], int) or d["retries"] < 0):
            problems.append("retries must be a whole number, 0 or more")
        if "priority" in d and (isinstance(d["priority"], bool) or not isinstance(d["priority"], int)):
            problems.append("priority must be a whole number")
        if "timeout" in d and (isinstance(d["timeout"], bool) or not isinstance(d["timeout"], (int, float)) or d["timeout"] < 0):
            problems.append("timeout must be a number of minutes")
        for k in ("split", "python", "note"):
            if k in d and not isinstance(d[k], str):
                problems.append(f"{k} must be a text")
        py = self._python(d.get("python", ""), entry or {})
        if entry:
            params, why = self._sig(entry, py)
            if params is None:
                warnings.append(f"the arguments were not checked against the method: {why}")
            else:
                problems += C.check_args(params, args)
        values = dict(args, id=jid)
        reads, writes = d.get("reads"), d.get("writes")
        for name, given in (("reads", reads), ("writes", writes)):
            if given is None and entry:
                filled = [(t, C.fill(t, values)) for t in entry.get(name, [])]
                warnings += [f"{name} {t!r} could not be filled from the arguments" for t, v in filled if v is None]
                given = [v for _, v in filled if v is not None]
            for ref in given or []:
                if ":" not in ref:
                    problems.append(f"{name} {ref!r} is not a reference 'kind:name'")
            if name == "reads":
                reads = [r for r in given or [] if ":" in r]
            else:
                writes = [r for r in given or [] if ":" in r]
        after = []
        for x in (d.get("after", []) if isinstance(d.get("after", []), list) else []):
            if not isinstance(x, str) or not ID_RE.match(x):
                problems.append(f"after {x!r} is not a plain run or job id")
            elif x == jid:
                problems.append("after names the job itself")
            elif self._find_run(x) is None:
                problems.append(f"after {x}: no such run and no job with that id (run.py would wait for it and then block)")
            else:
                after.append(x)
        live = self._live_writes(exclude=jid)
        inferred, more = self._inputs(jid, reads, live)
        problems += more
        after += [x for x in inferred if x not in after and x != jid]
        problems += [f"writes {w}: job {live[w]} is writing that now" for w in writes if w in live]
        split = d.get("split", "") if isinstance(d.get("split", ""), str) else ""
        problems += self._gates(jid, args, split, card, launch, entry or {})
        existing = _read(self._jd(jid, "job.json")) if jid else None
        same = bool(existing) and existing["spec"].get("call") == call and existing["spec"].get("args") == args
        if existing and not same:
            problems.append(f"the id {jid!r} was submitted with another call or other arguments; use a new id")
        if jid and not existing:
            problems += self._collisions(jid)
        spec = dict(id=jid, call=call, args=args, needs=needs, reads=reads or [], writes=writes or [], after=after, priority=d.get("priority", 0), card=card, no_card=(no_card or card is None) and not launch,
                    split=split, python=d.get("python", ""), run_args=list(d.get("run_args", [])) if isinstance(d.get("run_args", []), list) else [], retries=d.get("retries"),
                    timeout=d.get("timeout", 0), note=d.get("note", ""))
        argv = self._argv(spec, entry) if entry and not problems else None
        return dict(spec=spec, entry=entry, problems=problems, warnings=warnings, argv=argv, existing=existing if same else None)

    def _gates(self, jid: str, args: Dict[str, Any], split: str, card: Any, launch: bool, entry: Dict[str, Any]) -> List[str]:
        """what run.py would refuse and a job could slip past: the TEST split named in the arguments that name a split or set (run.py reads only its own command line, and the arguments
        are in a file; free text such as a card or a network name is not scanned), a ledger row with a card for a `process` job that says it is a diagnostic, a HOLD marker, a twin run
        folder. A `launch` call (Training.run) starts a run.py run of its own, which has all these gates itself."""
        if not jid:
            return []
        out = []
        R = self.R
        holdout = str(self.cfg.get("holdout_split", "test")).lower()
        aliases = tuple(str(x).strip().lower() for x in self.cfg.get("holdout_aliases", []))
        names = tuple(entry.get("splits") or SPLIT_PARAMS)
        scanned = [s for k, v in args.items() if k in names for s in _strings(v)]
        if not launch and not R.is_holdout(split, holdout, aliases) and any(R.is_holdout(tok, holdout, aliases) for s in scanned for part in s.split(",") for tok in re.split(r"[.:/@\s]+", part) if tok):
            out.append(f"the arguments {', '.join(names)} name the holdout split ({holdout}): give the job split={holdout!r} so that run.py's TEST gate (a logged row naming this id) applies, or use another name")
        if R.is_holdout(split, holdout, aliases):
            log_path = os.path.join(self.root, self.cfg["test_log"])
            log = open(log_path, encoding="utf-8").read() if os.path.exists(log_path) else ""
            grant = getattr(R, "grant_ids", None) or R.decided_ids
            why = R.test_log_refusal(log, jid, grant(self.root, self.cfg))
            if why:
                out.append(f"the holdout split is refused by run.py: {why}")
        row = R.ledger_row(os.path.join(self.root, self.cfg["ledger"]), jid)
        if row and (row.get("prediction") or row.get("rule")) and card is None and not launch:
            out.append("the ledger has a row with a card (prediction / rule) for this id: give the card or use another id (a diagnostic must not carry a rule)")
        rd = self._rdir(jid)
        if R.HOLD.read(rd) is not None:
            out.append("the id has a HOLD marker (hold.py): it starts only after the person's words clear it")
        twin = R.twin_refusal(self.records, jid)
        if twin:
            out.append(twin)
        return out

    def _collisions(self, jid: str) -> List[str]:
        """a run that this runner did not make, or a name that differs only in upper / lower case (/mnt/c does not tell them apart)"""
        out = []
        if os.path.exists(os.path.join(self._rdir(jid), "run.json")):
            out.append(f"a run with the id {jid!r} exists that this runner did not make: use another id")
        for base in (self.jobs, self.records):
            try:
                names = os.listdir(base)
            except OSError:
                continue
            out += [f"the id {jid!r} differs from {n!r} only in case: they share a folder on this disk" for n in names if n != jid and n.lower() == jid.lower() and not n.startswith(".")]
        return out

    def _rdir(self, jid: str) -> str:
        return os.path.join(self.records, jid)

    def _find_run(self, name: str) -> Optional[str]:
        """the record id for a way of naming a run: `t23.C_s1`, `t23_C_s1` (the run folder), `runs/t23_C_s1/`"""
        name = name.strip().rstrip("/")
        prefix = self.cfg["runs"].rstrip("/") + "/"
        for p in (prefix, "./" + prefix):
            if name.startswith(p):
                name = name[len(p):]
        fold = lambda n: self.R.run_folder(n).lower()                      # noqa: E731
        names = set()
        for base in (self.records, self.jobs):
            try:
                names.update(n for n in os.listdir(base) if not n.startswith("."))
            except OSError:
                pass
        if name in names:
            return name
        same = sorted(n for n in names if fold(n) == fold(name))
        return same[0] if same else None

    def _inputs(self, jid: str, reads: List[str], live_writes: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """-> (ids the job must wait for, problems) from the references it reads: `run:X` must be a done run, a run in progress or a job, a failed one is a problem; `path:P` must
        exist; any other reference that a live job writes (`live_writes`) makes the job wait for that job"""
        after, problems = [], []
        for ref in reads:
            kind, _, name = ref.partition(":")
            if kind == "path":
                if not os.path.exists(os.path.join(self.root, os.path.expanduser(name))):
                    problems.append(f"reads {ref}: not found")
            elif kind == "run":
                rid = self._find_run(name)
                if rid == jid or (jid and self.R.run_folder(name.strip().rstrip("/")).lower() == self.R.run_folder(jid).lower()):
                    problems.append(f"reads {ref}: a job cannot wait for itself")
                elif rid is None:
                    if not any(os.path.isdir(os.path.join(self.root, self.cfg["runs"], n)) for n in (name, name.replace(".", "_"))):
                        problems.append(f"reads {ref}: no such run and no job with that id")
                else:
                    st, _ = self._state(rid)
                    if st in LIVE or st in ("planned", "stalled"):
                        after.append(rid)
                    elif st in ("failed", "blocked", "stopped"):
                        problems.append(f"reads {ref}: that run is {st}")
            elif ref in live_writes and live_writes[ref] not in after:
                after.append(live_writes[ref])
        return after, problems

    def _live_writes(self, exclude: str = "") -> Dict[str, str]:
        """{reference: job id} of the jobs that are planned or in progress"""
        out: Dict[str, str] = {}
        for row in self._all():
            if row["id"] != exclude and (row["state"] in LIVE or row["state"] == "planned" or (row["state"] == "stalled" and self._stalled_alive(row["id"]))):
                for w in row["writes"]:
                    out.setdefault(w, row["id"])
        return out

    def _python(self, given: Any, entry: Dict[str, Any]) -> str:
        """the interpreter of the child process: the job's `python`, the entry's, the option (`JobRunner(python=...)`, `--python`, the environment variable JOBRUNNER_PYTHON), project.json
        jobrunner.python, else the one of this process"""
        py = given if isinstance(given, str) else ""
        return os.path.expanduser(py or entry.get("python") or self.python or (self.cfg.get("jobrunner") or {}).get("python") or sys.executable)

    def _argv(self, spec: Dict[str, Any], entry: Dict[str, Any]) -> Optional[List[str]]:
        """the run.py command line of a `process` job (None for `launch`, which calls the method itself)"""
        if entry["how"] != "process":
            return None
        n = spec["needs"]
        gpu = bool(n.get("gpu") or n.get("gpu_max_used_mb"))
        summary = (spec["call"] + " " + json.dumps(spec["args"], sort_keys=True, default=str))[:300]
        argv = [sys.executable, RUN_PY, "--id", spec["id"]]
        argv += ["--gpu"] if gpu else []
        argv += ["--gpu-max-mb", str(int(n["gpu_max_used_mb"]))] if n.get("gpu_max_used_mb") else []
        argv += ["--ram-mb", str(int(n["ram_gb"] * 1024))] if gpu and n.get("ram_gb") else []
        argv += ["--after", ",".join(spec["after"])] if spec["after"] else []
        if spec["card"]:
            argv += [f"--hyp={spec['card']['hyp']}", f"--pred={spec['card']['pred']}", f"--rule={spec['card']['rule']}", f"--basis={summary}", "--require-card"]
        else:
            argv += ["--no-card", "--no-git", f"--hyp={spec['note'] or summary}"]          # a diagnostic: no card, no code tag, never a verdict; the ledger says what was called
        argv += ["--retries", str(spec["retries"])] if spec["retries"] is not None else []
        argv += ["--timeout", str(spec["timeout"])] if spec["timeout"] else []
        argv += ["--split", spec["split"]] if spec["split"] else []
        py = self._python(spec["python"], entry)
        return argv + spec["run_args"] + ["--", py, "-m", "jobrunner", "call", "--suite", entry["suite"], "--cls", entry["cls"], "--method", entry["method"],
                                         "--args-file", self._jd(spec["id"], "args.json"), "--out", self._jd(spec["id"], "result.json"), "--project", self.root]

    # ------------------------------------------------------------------------------------------------------------ the interface
    def check(self, job: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Look at a job and start nothing: its problems, what it takes from the machine, the data it reads and writes, what it waits for, and the run.py command it would be."""
        p = self._plan(job)
        s = p["spec"]
        state = self._state(s["id"])[0] if p["existing"] else "new"
        launch = bool(p["entry"]) and p["entry"]["how"] == "launch"
        return dict(ok=not p["problems"], id=s["id"], call=s["call"], how=(p["entry"] or {}).get("how", ""), state=state, problems=p["problems"], warnings=p["warnings"], needs=s["needs"],
                    reads=s["reads"], writes=s["writes"], after=s["after"], diagnostic=None if launch else s["no_card"], command=p["argv"])

    def submit(self, job: Union[str, Dict[str, Any]], start: bool = True) -> Dict[str, Any]:
        """Put a job in: check it, write the job and (start) run it under run.py, which waits for its turn. The same job again is not started twice; a failed, blocked,
        stopped or dead one is started anew."""
        p = self._plan(job)
        s = p["spec"]
        skip = lambda: self._skip(s["id"])                                 # noqa: E731
        if p["problems"]:
            if p["existing"] and skip():                                   # the same job that is already in progress or done: what has changed around it is no news
                return skip()
            raise JobError(f"job {s['id'] or '?'}: " + "; ".join(p["problems"]))
        with self._lock(s["id"]):
            cur = _read(self._jd(s["id"], "job.json"))                     # read again under the lock: another submit may have finished while this one waited
            if cur and "spec" in cur and (cur["spec"].get("call") != s["call"] or cur["spec"].get("args") != s["args"]):
                raise JobError(f"job {s['id']}: the id was submitted with another call or other arguments while this one waited; use a new id")
            if cur and skip():
                return skip()
            _write(self._jd(s["id"], "args.json"), s["args"])
            _write(self._jd(s["id"], "job.json"), dict(spec=s, how=p["entry"]["how"], submitted=(cur or {}).get("submitted") or time.strftime("%Y-%m-%dT%H:%M:%S"), started=False,
                                                       launcher_pid=None, command=p["argv"]))
            return self._start(s["id"]) if start else dict(ok=True, id=s["id"], state="planned", started=False)

    def _skip(self, jid: str) -> Optional[Dict[str, Any]]:
        """the answer for a job that is already done or in progress (it is not started again), else None"""
        if _read(self._jd(jid, "job.json")) is None:
            return None
        st = self._state(jid)[0]
        if st == "done" or st in LIVE or (st == "stalled" and self._stalled_alive(jid)):
            return dict(ok=True, id=jid, state=st, started=False, note=f"already submitted ({st})")
        return None

    def start(self, i: str) -> Dict[str, Any]:
        """Start a job that was submitted with start=False (or ended failed, blocked, stopped or dead): run.py is launched in the background (`process`) or the method is called (`launch`)."""
        self._job(i)
        with self._lock(i):
            return self._start(i)

    def _start(self, i: str) -> Dict[str, Any]:
        rec = self._job(i)
        s, how = rec["spec"], rec["how"]
        st = self._state(i)[0]
        if st == "done" or st in LIVE or (st == "stalled" and self._stalled_alive(i)):
            raise JobError(f"job {i} is {st}: not started again")
        entry = self.table().get(s["call"])
        if not entry:
            raise JobError(f"job {i}: {s['call']} is no longer in the catalogue")
        result = self._jd(i, "result.json")
        if os.path.exists(result):                                         # the result of the earlier attempt is kept: result.1.json, result.2.json ...
            k = 1
            while os.path.exists(self._jd(i, f"result.{k}.json")):
                k += 1
            os.replace(result, self._jd(i, f"result.{k}.json"))
        rec.update(started=True, launcher_pid=os.getpid() if how == "launch" else None, error="", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   prev_created=(self.R.record(self._rdir(i)) or {}).get("created", "") if os.path.isdir(self._rdir(i)) else "")
        _write(self._jd(i, "job.json"), rec)                               # a `launch` method runs in this process: while it runs the job is queued (this pid), not a failed launcher
        try:
            if how == "process":
                env = dict(os.environ, AR_PROJECT=self.root, PYTHONPATH=os.pathsep.join(filter(None, [AR, os.environ.get("PYTHONPATH", "")])))
                rec["launcher_pid"] = self.launcher(rec["command"], self.root, env, self._jd(i, "launch.log"))
            else:
                with C.import_paths(*self.dirs):
                    inst = getattr(importlib.import_module(entry["suite"]), entry["cls"])(self.root)
                    out = getattr(inst, entry["method"])(**s["args"])
                _write(result, out)
                rec["launcher_pid"] = None
        except (Exception, SystemExit) as ex:                              # noqa: BLE001
            rec["error"] = f"{type(ex).__name__}: {ex}"
            rec["launcher_pid"] = None
            _write(self._jd(i, "job.json"), rec)
            raise JobError(f"job {i}: could not be started: {rec['error']}") from None
        _write(self._jd(i, "job.json"), rec)
        return dict(ok=True, id=i, state=self._state(i)[0], started=True, pid=rec["launcher_pid"])

    def run(self, job: Union[str, Dict[str, Any]], wait: float = 0.0) -> Dict[str, Any]:
        """Submit a job and wait for its end (wait = seconds at most, 0 = until it ends or is held); -> status, with ok = the job is done. The job still waits in run.py's line."""
        jid = self.submit(job)["id"]
        t0 = time.time()
        while self._state(jid)[0] in ("queued", "running", "planned") and not (wait and time.time() - t0 > wait):
            time.sleep(0.5)
        st = self.status(jid)
        st["ok"] = st["state"] == "done"
        return st

    def list(self, state: str = "") -> List[Dict[str, Any]]:
        """Every job with its state (planned, queued, running, held, stalled, blocked, stopped, done, failed), oldest first; `state` keeps one state."""
        return [{k: v for k, v in r.items() if k != "spec"} for r in self._all() if not state or r["state"] == state]

    def _all(self) -> List[Dict[str, Any]]:
        rows = []
        for n in sorted(os.listdir(self.jobs)) if os.path.isdir(self.jobs) else []:
            rec = _read(os.path.join(self.jobs, n, "job.json"))
            if rec and "spec" in rec:
                st, why = self._state(n)
                rows.append(dict(id=n, call=rec["spec"]["call"], state=st, submitted=rec.get("submitted", ""), writes=rec["spec"].get("writes", []), why=why))
        rows.sort(key=lambda r: r["submitted"])
        return rows

    def status(self, i: str) -> Dict[str, Any]:
        """One job: its state and why, the data it reads and writes, what it waits for, the last lines of its log and its result."""
        rec = self._job(i)
        st, why = self._state(i)
        run = self.R.record(self._rdir(i)) if os.path.isdir(self._rdir(i)) else {}
        return dict(id=i, call=rec["spec"]["call"], state=st, why=why, record_status=run.get("status", ""), exit_code=run.get("exit"), args=rec["spec"]["args"],
                    reads=rec["spec"]["reads"], writes=rec["spec"]["writes"], after=rec["spec"]["after"], diagnostic=rec["spec"]["no_card"] if rec["how"] == "process" else None,
                    result=_read(self._jd(i, "result.json")), log=_tail(os.path.join(self._rdir(i), "run.log")) or _tail(self._jd(i, "launch.log")))

    # ------------------------------------------------------------------------------------------------------------ the state of a job, from run.py's record
    def _job(self, i: str) -> Dict[str, Any]:
        rec = _read(self._jd(i, "job.json")) if isinstance(i, str) and ID_RE.match(i) else None
        if rec is None or "spec" not in rec:
            raise JobError(f"no job {i!r} (`list` shows them)")
        return rec

    @contextmanager
    def _lock(self, jid: str):
        """one submit / start of an id at a time (the file holds the pid of its owner; a dead owner's file is taken over)"""
        os.makedirs(self._jd(jid), exist_ok=True)
        path = self._jd(jid, "submit.lock")
        me = threading.get_ident()
        st = _stat(os.getpid())
        t0 = time.time()
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {st[19] if st and len(st) > 19 else ''} {me}".encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    pid, start, tid = (open(path).read().split() + ["", "", ""])[:3]
                    owner = int(pid or 0)
                except (OSError, ValueError):
                    owner, start, tid = 0, "", ""
                if owner and not (owner == os.getpid() and tid == str(me)) and _alive(owner, start or None):          # a live owner (another process or thread): wait for it, then look again
                    if time.time() - t0 >= LOCK_WAIT_S:
                        raise JobError(f"job {jid}: another submit or start of this id is running (pid {owner}) and did not finish in {LOCK_WAIT_S:g} s") from None
                    time.sleep(0.1)
                    continue
                try:
                    os.remove(path)                                        # its owner is gone
                except OSError:
                    pass
        try:
            yield
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _stalled_alive(self, i: str) -> bool:
        """a run in state stalled whose command or launcher still lives is waiting or working, only its heartbeat is late"""
        rec = self.R.record(self._rdir(i)) or {}
        return bool(self.R.command_alive({**rec, "status": "running"}) or _alive(rec.get("launcher_pid")))      # _alive: a launcher that is a zombie is gone

    @staticmethod
    def _age(job: Dict[str, Any]) -> float:
        """seconds since the job was started"""
        try:
            return time.time() - time.mktime(time.strptime(job.get("started_at", ""), "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            return 1e9

    def _why(self, rec: Dict[str, Any]) -> str:
        text = str(rec.get("notes") or "")
        if not text:
            last = rec.get("last_lines")
            text = (last[-1] if isinstance(last, list) and last else str(last or ""))
        return text.strip()[:200]

    def _state(self, i: str) -> Tuple[str, str]:
        """-> (state, why): planned, queued, running, held, stalled, blocked, stopped, done, failed, unknown. run.py's state_of decides (done flag first, heartbeat age, HOLD marker);
        a job without a record is queued while its launcher lives and failed when it died before it wrote one"""
        rd = self._rdir(i)
        job = _read(self._jd(i, "job.json"))
        rec = self.R.record(rd) if os.path.isdir(rd) else {}
        if rec or os.path.exists(os.path.join(rd, "done")):
            st = self.R.state_of(rd)
            if st not in ("done", "ok") and job and job.get("started"):                    # the record may still be the one of the attempt before this start
                old = job.get("prev_created", "") and rec.get("created", "") == job["prev_created"] and _alive(job.get("launcher_pid"))
                age = self._age(job)
                if old or (job.get("how") == "launch" and rec.get("created", "") and rec["created"] < job.get("started_at", "") and age < 120):
                    return "queued", "started again; run.py has not written the new record yet"
            if st in ("done", "ok"):
                return "done", ""
            if st == "running":
                return ("queued", "") if rec.get("status") == "waiting" else ("running", "")
            if st == "stalled":
                return "stalled", ("late heartbeat, but its process lives" if self._stalled_alive(i) else "no heartbeat and no process: it may be started again")
            if st in ("held", "blocked", "stopped"):
                return st, self._why(rec)
            if st in ("failed", "timeout", "gpu_busy"):
                return "failed", f"{st}: {self._why(rec)}".rstrip(": ")
            return ("failed", f"run.py status {st!r}") if st else ("unknown", "the record has no status")
        if job is None:
            return "unknown", "no job and no run record"
        if job.get("error"):
            return "failed", job["error"]
        if not job.get("started"):
            return "planned", ""
        if _alive(job.get("launcher_pid")):
            return "queued", ""
        if job.get("how") == "launch" and os.path.exists(self._jd(i, "result.json")):                 # the method has returned; the record of the run it started comes soon
            return ("queued", "launched, no run record yet") if self._age(job) < 120 else ("failed", "launched, but no run record appeared in 120 s")
        return "failed", "the launcher ended without a run record (see the launch log in status)"
