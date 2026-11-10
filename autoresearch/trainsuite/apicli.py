"""The command line of the interface (api.py): every method of the class is a command with the same name. `python -m trainsuite Training.queue_add job.json`, `Training.queue_hold
--words "..."`, `Training.status 3`; a method that does not clash with an older command also works without the class name (`queue_add job.json`). The arguments are the method's: the
ones called job, i, a, b are positional, every other one is an option (`--words`, `--wait`, `--no-start`, `--kill` repeatable). `--json` prints exactly what the method returns, without
it a short text. `python -m trainsuite methods` lists the methods. Exit code: 0 ok, 1 a check that found problems, 2 an error (the message goes to stderr)."""
import argparse
import inspect
import json
import os
import sys

from . import api

POSITIONAL = ("job", "i", "a", "b")


def _emit(obj, as_json, human):
    if as_json:
        print(json.dumps(obj, indent=1, default=str))
    else:
        human(obj)
    return 0


def is_job_file(path):
    """does this file look like a job ({"id": ..., "config": ...}) and not like a network config? (a config has no top-level `id` and `config`)"""
    try:
        with open(path, encoding="utf-8-sig") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and "id" in d and ("config" in d or d.get("kind") == "run")


def _flat(obj, indent=""):
    if not isinstance(obj, dict):
        print(f"{indent}{obj}")
        return
    for k, v in obj.items():
        if isinstance(v, (dict, list)) and v:
            print(f"{indent}{k}:")
            if isinstance(v, dict):
                _flat(v, indent + "  ")
            else:
                for x in v:
                    print(f"{indent}  - {x}" if not isinstance(x, dict) else f"{indent}  - " + ", ".join(f"{a}={b}" for a, b in x.items()))
        else:
            print(f"{indent}{k}: {v}")


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
    print(f"{'#':>2}  {'id':<24}{'state':<12}{'step/total':<13}{'grad':>9}  {'skip':>4}  flags / notes")
    for x in r["items"]:
        g = x["grad"]
        grad = "-" if not g else "non-finite" if g["now"] is None else "%.3g" % g["now"]
        more = ", ".join(filter(None, [",".join(x["flags"]), ("after " + ",".join(x["after"])) if x["after"] else "", "held" if x["held"] else "", x["refused"], x["note"]]))
        print(f"{x['number']:>2}  {x['id']:<24}{x['state']:<12}{'%d/%s' % (x['step'], x['total'] or '?'):<13}{grad:>9}  {x['skipped'] or '-':>4}  {more}")
    if not r["items"]:
        print("  the queue is empty")
    print(f"running: {r['running'] or '-'}   runner: {('pid %s' % r['runner']) if r['runner'] else 'not running'}   {'PAUSED' if r['paused'] else ''}")
    print("slots: " + "; ".join(f"{s['slot']}: {s['id'] or 'free'}" + (f" (loading {s['loading_s']} s)" if s.get("loading_s") is not None else "") for s in r["slots"]))
    lk = r["lock"]
    print("machine lock: " + (f"{lk['owner']} until {lk['end']} ({'exclusive' if lk.get('exclusive') is True else 'reservation'})" if lk else "free"))
    ram, gpu = r["ram"], r["gpu"]
    print(f"RAM: {ram['available_mb']} MB available (gate {ram['need_mb']} MB)   GPU memory: {gpu['used_mb']} MB in use (gate {gpu['gate_mb']})")
    if r["holds"]:
        print("on hold (the person's words clear a hold): " + ", ".join(h["id"] for h in r["holds"]))


TEXT = {"check": _check_text, "run": _run_text, "queue_list": _list_text,
        "queue_remove": lambda r: print(f"removed {r['removed']} from the queue"), "queue_start": lambda r: print(f"the runner runs (pid {r['runner']})")}


def methods_text(cls):
    for m in cls.methods():
        print(f"{cls.__name__}.{m['name']}({m['arguments']})")
        print(f"    {m['summary']}")


def call_cli(cls, argv, prog):
    """run the method named by argv[0] (`Training.queue_add` or `queue_add`) of `cls(project)` with the arguments of the rest of argv; -> exit code"""
    name = argv[0].split(".", 1)[-1]
    fn = getattr(cls, name)
    params = [p for k, p in inspect.signature(fn).parameters.items() if k != "self"]
    ap = argparse.ArgumentParser(prog=f"{prog} {cls.__name__}.{name}", description=inspect.getdoc(fn), formatter_class=argparse.RawDescriptionHelpFormatter)
    for p in params:
        flag = "--" + p.name.replace("_", "-")
        required = p.default is inspect.Parameter.empty
        if p.name in POSITIONAL:
            ap.add_argument(p.name) if required else ap.add_argument(p.name, nargs="?", default=p.default)
        elif isinstance(p.default, bool):
            if p.default:
                ap.add_argument("--no-" + p.name.replace("_", "-"), dest=p.name, action="store_false", help=f"switch {p.name} off")
            else:
                ap.add_argument(flag, dest=p.name, action="store_true")
        elif isinstance(p.default, (tuple, list)):
            ap.add_argument(flag, dest=p.name, action="append", default=[], help="repeatable")
        elif isinstance(p.default, (int, float)):
            ap.add_argument(flag, dest=p.name, type=type(p.default), default=p.default)
        else:
            ap.add_argument(flag, dest=p.name, required=required, default=None if required else p.default)
    ap.add_argument("--project", default=None, help="project folder (default: AR_PROJECT)")
    ap.add_argument("--json", action="store_true", help="print what the method returns")
    a = ap.parse_args(argv[1:])
    kw = {p.name: getattr(a, p.name) for p in params}
    # a text of digits stays a text: an id comes first, only what is no id is a queue number (api._resolve)
    try:
        result = getattr(cls(a.project), name)(**kw)
    except api.TrainingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _emit(result, a.json, TEXT.get(name, _flat))
    return 1 if name == "check" and not result.get("ok") else 0


def dispatch(cls, argv, prog, legacy=()):
    """-> the exit code when argv names a method of `cls` (as `Class.method`, or as `method` if it is not one of the older commands `legacy`), else None. `methods` lists them"""
    if not argv:
        return None
    if argv[0] == "methods":
        methods_text(cls)
        return 0
    head = argv[0]
    prefixed = head.startswith(cls.__name__ + ".")
    name = head.split(".", 1)[-1] if prefixed else head
    if name in cls.METHODS and (prefixed or name not in legacy):
        return call_cli(cls, [head] + argv[1:], prog)
    return None


# ---- the nested older forms, kept: `queue list`, `queue get I`, `queue remove I`, `queue start`, `queue status [I]`, `queue add JOB`
def _project(a):
    if getattr(a, "project", None):
        os.environ["AR_PROJECT"] = a.project


def guarded(fn):
    def run(a):
        _project(a)
        try:
            return fn(a)
        except api.TrainingError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    return run


@guarded
def cmd_queue_list(a):
    return _emit(api.queue_list(), a.json, _list_text)


@guarded
def cmd_queue_get(a):
    return _emit(api.queue_get(a.i), a.json, _flat)


@guarded
def cmd_queue_remove(a):
    return _emit(api.queue_remove(a.i), a.json, TEXT["queue_remove"])


@guarded
def cmd_queue_start(a):
    return _emit(api.queue_start(), a.json, TEXT["queue_start"])


@guarded
def cmd_queue_status(a):
    return _emit(api.status(a.i, a.kill, a.vs), a.json, _flat)


@guarded
def cmd_queue_add_job(a):
    return _emit(api.Training(getattr(a, "project", None)).queue_add(a.base, not a.no_runner), a.json, _flat)
