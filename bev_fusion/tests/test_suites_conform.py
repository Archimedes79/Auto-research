"""Are the three classes one piece? The command lines of Training (autoresearch/Training), Evaluation (autoresearch/Evaluation) and Runner (bev_fusion/runner) against the convention they share
(autoresearch/suitecli.py), on the BEV project, CPU, about 40 s, no data and no network run. Asserted: (1) every command line lists its commands with `methods`, each with a description; (2) `--json`
of a read-only call prints exactly one json document and nothing else; (3) a bad argument is exit code 2 with `error: ...` on stderr and no traceback, an id that does not exist is the same error in
every one, a `check` / `plan` that finds problems is exit 1; (4) every method name starts with a verb; (5) layering: Evaluation's core imports no torch / nuScenes and not the runner, suitecli.py only
the standard library, bev_free.py only Evaluation / evaluation / torch / the run's `interface` module / the standard library (nothing of the trainer), runner/runner.py only the standard library,
numpy and Evaluation, runner/__main__.py only the runner and suitecli, and importing the Runner loads neither torch nor Training. Printed: the commands per class.
Run (from the bev_fusion folder): ~/miniconda3/envs/gcar/bin/python tests/test_suites_conform.py"""
import ast
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AR = os.path.abspath(os.path.join(ROOT, "..", "autoresearch"))
ENV = {**os.environ, "AR_PROJECT": ROOT}
SUITES = {"Training": ("Training", AR), "Evaluation": ("Evaluation", AR), "runner": ("Runner", ROOT)}      # command line -> (its class, the folder it is started from)
VERBS = {"check", "run", "status", "stop", "plot", "compare", "list", "get", "remove", "add", "start", "hold", "resume", "score", "tap", "time", "probe", "queue", "selftest",
         "predict", "plan", "show", "split", "find", "make", "draw", "report"}


def call(*argv, cwd=AR):
    p = subprocess.run([sys.executable, *argv], cwd=cwd, env=ENV, capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout, p.stderr


def commands(suite):
    """{(class, method): (command word, description)} from `python -m <suite> methods`"""
    rc, out, err = call("-m", suite, "methods", cwd=SUITES[suite][1])
    assert rc == 0, (suite, rc, err[-300:])
    found, last = {}, None
    for line in out.splitlines():
        m = re.match(r"^(\S+)\s{2}\((\w+)\.(\w+)\(", line)
        if m:
            last = (m.group(2), m.group(3))
            found[last] = [m.group(1), ""]
        elif last and line.startswith("    "):
            found[last][1] += line.strip() + " "
    return found


def imports_of(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split(".")[0])
    return mods


def main():
    table, every = {}, {}
    for suite in SUITES:
        cmds = commands(suite)
        assert cmds, f"{suite} lists no commands"
        for (cls, method), (_, doc) in cmds.items():
            assert doc.strip(), f"{suite} {cls}.{method} has no description"
        table[suite] = cmds
        for (cls, method) in cmds:
            every.setdefault(cls, []).append(method)
    assert all({cls for (cls, _) in table[suite]} == {SUITES[suite][0]} for suite in SUITES), "each command line offers exactly its class"
    nouns = sorted({m for ms in every.values() for m in ms if m.split("_")[0] not in VERBS})
    assert not nouns, f"method names that do not start with a verb: {nouns}"

    # (2) --json prints one document and nothing else
    probes = [(["-m", "Training", "Training.list", "--json"], AR), (["-m", "Evaluation", "Evaluation.list_sets", "--json"], AR)]
    runs = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "config.json")))
    if runs:                                                                # the Runner's read-only call needs one run folder (an export of the code has none)
        probes.append((["-m", "runner", "Runner.list_results", os.path.relpath(os.path.dirname(runs[0]), ROOT), "--json"], ROOT))
    else:
        print("SKIP: the --json probe of Runner.list_results needs a run folder with a config.json under runs/ (there is none here)")
    for argv, cwd in probes:
        rc, out, err = call(*argv, cwd=cwd)
        assert rc == 0, (argv, rc, err[-300:])
        json.loads(out)                                                     # raises when there is anything else than one json document

    # (3) a bad argument is exit 2 with a message and no traceback, in every one; a check / plan that finds problems is exit 1
    for argv, cwd in ((["-m", "Evaluation", "Evaluation.show_numbers", "nope@e123"], AR), (["-m", "runner", "Runner.predict", "runs/nope-xyz", "val"], ROOT),
                      (["-m", "Training", "Training.check", "/nonexistent.json"], AR), (["-m", "Training", "Training.status", "nope-xyz"], AR)):
        rc, out, err = call(*argv, cwd=cwd)
        assert rc == 2 and err.startswith("error:") and "Traceback" not in err, (argv, rc, err[-300:])
    for argv, cwd in ((["-m", "runner", "Runner.plan", "runs/nope-xyz", "val"], ROOT), (["-m", "Evaluation", "Evaluation.check", "runs/nope-xyz", "val"], AR)):
        rc, out, err = call(*argv, cwd=cwd)
        assert rc == 1 and "ok: False" in out, (argv, rc, out[-300:], err[-300:])

    # (5) layering
    rc, out, err = call("-c", "import sys, Evaluation, Evaluation.api; bad = [m for m in ('torch', 'nuscenes', 'cv2', 'runner') if m in sys.modules]; print(bad)")
    assert rc == 0 and out.strip() == "[]", ("Evaluation core imports", out, err[-200:])
    std = set(sys.stdlib_module_names)
    assert imports_of(os.path.join(AR, "suitecli.py")) <= std, imports_of(os.path.join(AR, "suitecli.py")) - std
    free = imports_of(os.path.join(ROOT, "evaluation", "bev_free.py"))             # the BEV side: torch and the run's own `interface` module (the contract of the network), nothing of the trainer
    assert free <= std | {"Evaluation", "evaluation", "torch", "interface"}, free - std
    runner = imports_of(os.path.join(ROOT, "runner", "runner.py"))                 # the class: the evaluation's pieces and numpy, no training, no torch, not even the command-line generator
    assert runner <= std | {"numpy", "Evaluation"}, runner - std
    command_line = imports_of(os.path.join(ROOT, "runner", "__main__.py"))        # the command line above it: the class and the generator
    assert command_line <= std | {"runner", "suitecli"}, command_line - std
    rc, out, err = call("-c", f"import sys; sys.path.insert(0, {ROOT!r}); import runner.runner; print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'Training', 'nuscenes', 'cv2')))", cwd=ROOT)
    assert rc == 0 and out.strip() == "[]", ("the Runner loads", out, err[-200:])

    print("commands per class:")
    for cls in ("Training", "Runner", "Evaluation"):
        print(f"  {cls:11s} {' '.join(every[cls])}")
    print(f"test_suites_conform passed: {sum(len(v) for v in table.values())} commands in 3 classes, descriptions, verbs first, --json, exit codes 0/1/2 (also for an unknown id), layering")


if __name__ == "__main__":
    main()
