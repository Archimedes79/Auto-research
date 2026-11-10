"""The interface of the runner (see __init__.py): the class `RunJobs`, a Training whose jobs are of kind "run". check, run, status and queue_add are its own; the rest of the queue is the
one of Training (one queue for trainings and runner jobs). Jobs are the jobs of trainsuite.api.load_job with kind "run"."""
import os
import shlex

import suitecli
from trainsuite import api as T
from trainsuite import compose as C
from trainsuite import trainqueue as TQ

MODES = ("check", "precompute", "network", "full")
POSITIONAL = {"check": ("config",), "precompute": ("config",), "network": ("run", "frames"), "full": ("run",)}      # mode -> the job keys that are arguments, not flags
FLAGS = {                                                       # mode -> the runner flags a job may set (job key -> flag)
    "check": ("split", "n", "scene", "first"),
    "precompute": ("split", "n", "scene", "first", "out"),
    "network": ("out", "dts", "cold", "device", "code"),
    "full": ("config", "split", "n", "scene", "first", "out", "dts", "device", "code"),
}


def _block(c):
    block = c.cfg.get("runsuite")
    if not block or not block.get("command"):
        raise T.TrainingError('project.json has no "runsuite" block ({"command": "python3 runner/bev_runner.py", "runs": "runs"})')
    return block


def _path(c, block, name):
    """a run named by id, folder name or path -> its folder relative to the project root"""
    if os.path.isdir(os.path.join(c.root, name)):
        return name
    return os.path.join(block.get("runs", "runs"), TQ.folder(name)).replace(os.sep, "/")


def _embed(command, args):
    """the project's runner command with the arguments in its {args} slot (or after it): -> argv. Inside a double-quoted bash -c "..." the quotes of the arguments must survive two readings
    (this one, then bash's), so the backslash and the double quote are escaped for the first"""
    try:
        if "{args}" not in command:
            return shlex.split(command) + args
        quoted = " ".join(shlex.quote(x) for x in args)
        if command[:command.index("{args}")].count('"') % 2 == 1:
            quoted = quoted.replace("\\", "\\\\").replace('"', '\\"')
        return shlex.split(command.replace("{args}", quoted))
    except ValueError as e:
        raise T.TrainingError(f"project.json runsuite command or an argument cannot be quoted: {e}")


def _plan(c, j):
    """-> (command argv, run.py argv, problems, out folder)"""
    block = _block(c)
    mode, problems = j["mode"], []
    args = [mode]
    if mode in ("check", "precompute"):
        if not j["config"]:
            problems.append(f'mode {mode} needs "config" (the network config)')
        else:
            args.append(j["config"])
            if not os.path.isfile(os.path.join(c.root, j["config"])):
                problems.append(f"config {j['config']} does not exist")
    else:
        run_dir = _path(c, block, j["run"]) if j["run"] else ""
        if not run_dir:
            problems.append(f'mode {mode} needs "run" (the trained run: id, folder name or path)')
        else:
            args.append(run_dir)
            if not os.path.isdir(os.path.join(c.root, run_dir)):
                problems.append(f"run folder {run_dir} does not exist")
        if mode == "network":
            if not j["frames"]:
                problems.append('mode network needs "frames" (the folder a precompute wrote)')
            else:
                args.append(j["frames"])
    for key in FLAGS[mode]:
        v = j.get(key)
        if key == "dts":
            if v:
                args += ["--dts"] + [str(x) for x in v]
        elif key == "cold":
            if v:
                args.append("--cold")
        elif v not in (None, "", []):
            args += [f"--{key}", str(v)]
    given = [k for k in ("config", "run", "frames", "split", "n", "scene", "first", "out", "dts", "device", "cold", "code") if j.get(k) not in (None, "", [], False)]
    stray = [k for k in given if k not in FLAGS[mode] and k not in POSITIONAL[mode]]
    if stray:
        problems.append(f"mode {mode} does not take {', '.join(stray)}")
    cmd = _embed(block["command"], args)      # a command with a shell around it: bash -c "$HOME/.../python runner/bev_runner.py {args}"
    out = j["out"] or (f"{run_dir}/runner/{mode}" if mode in ("network", "full") and j["run"] else "")      # where the runner writes when no --out is given (run/runner/<mode>)
    pre = ["--id", j["id"], "--no-card"]
    if j["split"]:
        pre += ["--split", j["split"]]                          # the ledger row says which frames it ran on
    if j["device"] != "cpu":                                    # a job on the CPU takes no GPU slot and waits for no memory gate
        pre.append("--gpu")
        try:
            gate = C.gpu_gate(c.block)
        except C.ComposeError as e:
            raise T.TrainingError(str(e))
        if gate:
            pre += ["--gpu-max-mb", str(gate)]
    run_argv = ["python3", os.path.join(TQ.TOOLS, "run.py")] + pre + ["--"] + cmd
    return cmd, run_argv, problems, out


class RunJobs(T.Training):
    """The runner as one object, like Training: `RunJobs(project)`. Its jobs have "kind": "run" and a "mode" (check, precompute, network, full). Every method is also a command:
    `python -m runsuite RunJobs.run job.json`."""

    METHODS = ("check", "run", "status", "queue_add", "queue_list", "queue_get", "queue_remove", "queue_start", "queue_hold", "queue_resume")
    KIND = "run"

    def _job(self, job):
        j = T.Training._job(self, job)
        j["mode"] = str(j.get("mode") or "full")
        if j["mode"] not in MODES:
            raise T.TrainingError(f'job "mode": one of {MODES}, not {j["mode"]!r}')
        for k in ("run", "split", "scene", "out", "device", "frames", "code"):
            j[k] = str(j.get(k) or "")
        for k in ("n", "first"):
            try:
                j[k] = int(j[k]) if j.get(k) not in (None, "") else None
            except (TypeError, ValueError):
                raise T.TrainingError(f'job "{k}": a whole number, not {j.get(k)!r}')
        if j["n"] is not None and j["scene"]:
            raise T.TrainingError('job: "n" (the first n frames) or "scene", not both')
        dts = j.get("dts") or []
        if not isinstance(dts, (list, tuple)) or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in dts):
            raise T.TrainingError('job "dts": a list of seconds, e.g. [0, 0.5]')
        j["dts"], j["cold"] = list(dts), T.as_bool(j, "cold")
        return j

    @T.errors
    def check(self, job):
        """are the fields and paths of the job right, and which command would run? Starts nothing. -> {ok, id, mode, problems, command, out}"""
        j = self._job(job)
        c = self._c(j)
        cmd, run_argv, problems, out = _plan(c, j)
        problems = T.queue_blocks(c, j["id"]) + problems
        return dict(ok=not problems, id=j["id"], mode=j["mode"], problems=problems, command=T.shell(run_argv), out=out or "(the runner's default)")

    @T.errors
    def run(self, job, wait=3.0):
        """start the job now (detached, through run.py: record, ledger row without card, GPU slot). -> {id, started, out, command}"""
        j = self._job(job)
        c = self._c(j)
        cmd, run_argv, problems, out = _plan(c, j)
        problems = T.queue_blocks(c, j["id"]) + problems          # the person's pause and a held item forbid a start (hard rule 6)
        if problems:
            raise T.TrainingError("nothing started: " + "; ".join(problems))
        if j["after"]:                                          # wait for those runs (run.py --after: a failed or stopped one blocks this job)
            i = run_argv.index("--")
            run_argv = run_argv[:i] + ["--after", ",".join(j["after"])] + run_argv[i:]
        log = os.path.join(c.runs, TQ.folder(j["id"]) + ".launcher.log")
        return dict(id=j["id"], started=T.launch(c, [(run_argv, log, j["id"])], wait), out=out, command=T.shell(run_argv))

    @T.errors
    def queue_add(self, job, start=True):
        """put the job in the queue of the trainings (kind "run"): it runs when its turn comes, `after` and `priority` work as for a training. -> the item as queue_get shows it"""
        j = self._job(job)
        c = self._c(j)
        cmd, run_argv, problems, out = _plan(c, j)
        if problems:
            raise T.TrainingError("nothing queued: " + "; ".join(problems))
        j["out"] = out                                          # the default folder is written into the stored job: `status` lists its files
        TQ.add(c, {"argv": run_argv}, j["id"], j["after"], j["priority"], j["note"], extra=dict(kind="run", job={k: v for k, v in j.items() if k != "project"}))
        if start:
            TQ.ensure_runner(c)
        return T.get_item(c, j["id"])                               # the context of the job (its "project" may differ from this object's)

    @T.errors
    def status(self, i=None):
        """one runner job: the state from its run record, the exit code, the duration, the files in its output folder and the tail of its log. i=None is the running job"""
        c = self._c()
        rid, it, n = T.resolve(c, i, "run")
        if rid is None:
            return dict(id=None, state="idle")
        T.must_exist(c, rid, it)
        d = os.path.join(c.records, rid)
        rec = c.R.record(d)
        job = (it or {}).get("job") or {}
        out = job.get("out") or ""
        if not out:                                             # a job started by `run` is not in the queue: its --out is in the command of its record
            try:
                words = shlex.split(rec.get("command") or "")
                out = words[words.index("--out") + 1] if "--out" in words else ""
            except (ValueError, IndexError):
                out = ""
        files = sorted(os.listdir(os.path.join(c.root, out)))[:30] if out and os.path.isdir(os.path.join(c.root, out)) else []
        return dict(id=rid, number=n, queue=TQ.state(c, it) if it else None, state=c.R.state_of(d) or "unknown", exit=rec.get("exit"), duration_s=rec.get("duration_s"),
                    started=rec.get("started"), ended=rec.get("ended"), out=out, files=files, log_tail=(rec.get("last_lines") or [])[-6:])


CLI = suitecli.Cli(RunJobs, "runsuite", T.TrainingError, "TRAINSUITE_TRACEBACK", positional=T.CLI.positional, option_names=T.CLI.option_names, need_all=T.CLI.need_all, text=T.CLI.text)
