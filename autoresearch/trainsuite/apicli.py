"""The command line of the interface (api.py): every method of the class is ONE command, the same call as in Python (the convention: autoresearch/INTERFACE.md).

    python -m trainsuite <command> [arguments] [--json] [--project DIR]

The command is the method name with hyphens (autogt and evalsuite do the same: `generate-bev`, `selftest-adapter`); `queue_add` and `Training.queue_add` work too. The arguments of the method
are the arguments of the command: the ones called job, i, a, b are positional, every other one is an option (`--words`, `--wait`, `--kill` repeatable, `--no-start` for a switch that is on
by default). `--json` prints exactly what the method returns, without it a short text. The map (`python -m trainsuite methods` prints it with the arguments):

    check JOB              Training.check          what run would do; nothing is started (exit 1 when it found problems)
    run JOB                Training.run            start now, outside the queue (the same call again resumes)
    queue-add JOB          Training.queue_add      check, then put it in the queue
    queue-list             Training.queue_list     the queue, the running training, slots, lock, RAM, GPU memory, holds
    queue-get I            Training.queue_get      one item: job, config, command, state
    queue-remove I         Training.queue_remove   take out an item that has not started
    queue-start            Training.queue_start    make sure the runner runs
    queue-hold I|--all --words W   Training.queue_hold   stop the trainings with a checkpoint, nothing new starts (the person's words)
    queue-resume I|--all --words W Training.queue_resume the holds are cleared, the runs continue from their checkpoint
    Training.status [I]    Training.status         one training in full: state, ETA, loss, gradient, skipped steps, data, health, kill rules, speed
    Training.stop [I]      Training.stop           checkpoint and stop one training
    Training.plot I        Training.plot           the loss curves as a png
    Training.compare A B   Training.compare        two runs side by side

`status`, `stop`, `plot` and `compare` are older commands with their own output, so the new ones need the class name. Exit codes: 0 done, 1 `check` found problems, 2 an error (a message on
stderr, no traceback; TRAINSUITE_TRACEBACK=1 shows it)."""
import argparse
import inspect
import json
import math
import os
import sys

from . import api

POSITIONAL = ("job", "i", "a", "b")
NEED_ALL = ("queue_hold", "queue_resume")      # no `i` means every item in Python; the command line asks for it by name (--all), so a forgotten id does not hold everything


def command(method):
    """the command of a method: queue_add -> queue-add"""
    return method.replace("_", "-")


def _clean(obj):
    """json has no NaN or Infinity: the loss of a diverged run is null"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _emit(obj, as_json, human):
    if as_json:
        print(json.dumps(_clean(obj), indent=1, default=str, allow_nan=False))
    else:
        human(obj)


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


def methods_text(cls, legacy=()):
    """what `methods` prints: the command, the arguments of the method and its first sentence (a word of `legacy` is an older command: the new one needs the class name)"""
    for m in cls.methods():
        word = f"{cls.__name__}.{m['name']}" if m["name"] in legacy else command(m["name"])
        print(f"{word}  ({cls.__name__}.{m['name']}({m['arguments']}))")
        print(f"    {m['summary']}")


def call_cli(cls, name, args, prog):
    """run method `name` of `cls(project)` with the command line `args`; -> the exit code"""
    fn = getattr(cls, name)
    params = [p for k, p in inspect.signature(fn).parameters.items() if k != "self"]
    ap = argparse.ArgumentParser(prog=f"python -m {prog} {cls.__name__}.{name}", description=inspect.getdoc(fn), formatter_class=argparse.RawDescriptionHelpFormatter)
    for p in params:
        flag = "--" + p.name.replace("_", "-")
        required = p.default is inspect.Parameter.empty
        if p.name in POSITIONAL:
            if required:
                ap.add_argument(p.name)
            else:
                ap.add_argument(p.name, nargs="?", default=p.default)
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
    if name in NEED_ALL:
        ap.add_argument("--all", action="store_true", help="every item (instead of naming one)")
    a = ap.parse_args(args)
    if name in NEED_ALL and a.i is None and not a.all:
        ap.error("name the item (I) or give --all: without an item the call means every item")
    try:
        result = getattr(cls(a.project), name)(**{p.name: getattr(a, p.name) for p in params})
    except (api.TrainingError, ValueError, KeyError, OSError) as e:         # a damaged file is a message too, not a traceback
        if os.environ.get("TRAINSUITE_TRACEBACK"):
            raise
        print(f"error: {e if isinstance(e, api.TrainingError) else type(e).__name__ + ': ' + str(e)}", file=sys.stderr)
        return 2
    _emit(result, a.json, TEXT.get(name, _flat))
    return 1 if name == "check" and not result.get("ok") else 0


def dispatch(cls, argv, prog, legacy=()):
    """-> the exit code when argv names a method of `cls` (`queue-add`, `queue_add` or `Class.queue_add`; a word of `legacy` needs the class name), else None. `methods` lists them"""
    if not argv:
        return None
    if argv[0] == "methods":
        methods_text(cls, legacy)
        return 0
    prefixed = argv[0].startswith(cls.__name__ + ".")
    name = (argv[0].split(".", 1)[1] if prefixed else argv[0]).replace("-", "_")
    if name in cls.METHODS and (prefixed or name not in legacy):
        return call_cli(cls, name, argv[1:], prog)
    return None
