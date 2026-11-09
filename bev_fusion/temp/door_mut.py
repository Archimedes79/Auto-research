"""throw-away: plant one bug at a time into a copy of evaluate.py and run the plan / executor / refusal part of tests/test_evaluate_door.py on it"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
src = open(os.path.join(ROOT, "evaluation", "evaluate.py")).read()
CONST = "= 30, 500, 8, 3, 6, 21"
MUTANTS = [
    ("prior 500 -> 400", CONST, "= 30, 400, 8, 3, 6, 21"),
    ("timing n 30 -> 20", CONST, "= 20, 500, 8, 3, 6, 21"),
    ("showcase seed 21 -> 22", CONST, "= 30, 500, 8, 3, 6, 22"),
    ("soft off", "G, soft = global_options(a, a.out), a.all", "G, soft = global_options(a, a.out), False"),
    ("timing not run for --time", "    if a.time or a.all:  ", "    if a.all:  "),
    ("time step dropped", "    if a.time:\n        steps.append(step(\"time\"", "    if False:\n        steps.append(step(\"time\""),
    ("no second try", "tries=2,", "tries=1,"),
    ("store deleted not moved", "os.replace(store, dest)", "shutil.rmtree(store)"),
    ("fatal ignored", "            if st[\"fatal\"]:\n                stopped = True\n                break", "            if False:\n                stopped = True\n                break"),
    ("door.json overwritten", "\"calls\": earlier + [call]", "\"calls\": [call]"),
    ("test split offered", 'choices=["val", "train"]', 'choices=["val", "train", "test"]'),
    ("showcase out dropped", "global_options(a, show) + [", "global_options(a) + ["),
    ("qual fatal in --all", 'fatal=not soft) if isinstance(q, int)', 'fatal=True) if isinstance(q, int)'),
    ("curves always on", 'if a.all or getattr(a, f)]', 'if True]'),
    ("seed ignored", '"--seed", str(a.seed)]', '"--seed", "1"]'),
    ("frames not all named", 'step("frames", frames=q, fatal', 'step("frames", frames=q[:1], fatal'),
    ("report not last", 'steps.append(step("report"', 'steps.insert(0, step("report"'),
    ("report tables dropped", '"--val-table", os.path.join(by_n, f"table_n{a.n}.txt"),', '"--val-table", "x",'),
    ("fatal numbers soft", 'steps.append(step("numbers", G + more', 'steps.append(step("numbers", fatal=False, argv=G + more'),
    # the critic's round
    ("timing gets the folder as --out", 'global_options(a, into)', 'global_options(a, a.out)'),
    ("n = 0 accepted", "    if n < 1:", "    if n < 0:"),
    ("exit 1 whatever the step gave", "code = first if 1 <= first <= 255 else 1 if failed else 0", "code = 1 if failed else 0"),
    ("soft failure exit 0", "code = first if 1 <= first <= 255 else 1 if failed else 0", "code = (first if 1 <= first <= 255 else 1) if stopped else 0"),
    ("odd code passed on", "code = first if 1 <= first <= 255 else 1 if failed else 0", "code = first if failed else 0"),
    ("summary names nothing", '"FAILED " + ", ".join(f"{name} (exit {c})" for name, c in failed)', '"FAILED"'),
    ("overwrite check off", "risks = [] if a.replace else outputs_at_risk(a)", "risks = []"),
    ("--replace ignored", "risks = [] if a.replace else outputs_at_risk(a)", "risks = outputs_at_risk(a)"),
    ("quant check on card.json missing", 'checks.append((os.path.join(ev, f"card{tag}.json")', 'checks.append((os.path.join(ev, f"metrics{tag}.json")'),
    ("train tag ignored", 'tag = os.path.join(a.run, "eval"), "_train" if a.split == "train" else ""', 'tag = os.path.join(a.run, "eval"), ""'),
    ("timing check ignores --out", 'timing = os.path.join(a.out, "timing_modules.json") if a.out else os.path.join(ev, "timing_modules.json")', 'timing = os.path.join(ev, "timing_modules.json")'),
    ("latency not checked", '    if a.time:\n        checks.append((os.path.join(ev, "latency.json")', '    if False:\n        checks.append((os.path.join(ev, "latency.json")'),
    ("all checks latency", '    if a.time:\n        checks.append((os.path.join(ev, "latency.json")', '    if a.time or a.all:\n        checks.append((os.path.join(ev, "latency.json")'),
    ("ckpt-dir not checked", 'if a.ckpt_dir and not os.path.isdir(a.ckpt_dir):', 'if False:'),
    ("checkpoint not checked", 'if not glob.glob(os.path.join(ckpt, "ckpt_*.pt")):', 'if False:'),
    ("ckpt looked for in the run only", 'ckpt = a.ckpt_dir or a.run', 'ckpt = a.run'),
    ("names not checked", "    return name_refusal(a, lister)", "    return None"),
    ("frame index not checked", "if not idx.isdigit() or int(idx) >= have[scene]:", "if not idx.isdigit():"),
    ("scene of a frame not checked", "        if scene not in have:\n            return f\"--qual", "        if False:\n            return f\"--qual"),
    ("video scenes not checked", "        if s not in have:", "        if False:"),
    ("list on the GPU", '"device": "cpu", "workers": 0', '"device": a.device, "workers": 0'),
    ("random picks list the scenes", "    if not (frames or scenes):\n        return None", "    if False:\n        return None"),
    ("refusal exit 1", 'print(f"evaluate: refused: {why}", file=sys.stderr)\n        return 2', 'print(f"evaluate: refused: {why}", file=sys.stderr)\n        return 1'),
]
for name, old, new in MUTANTS:
    assert src.count(old) >= 1, name
    mutated = src.replace(old, new, 1).replace("import subprocess", "import shutil\nimport subprocess", 1)
    tree = os.path.join(ROOT, "temp", "door_mut_tree")
    shutil.rmtree(tree, ignore_errors=True)
    os.makedirs(os.path.join(tree, "evaluation"))
    os.makedirs(os.path.join(tree, "tests"))
    open(os.path.join(tree, "evaluation", "evaluate.py"), "w").write(mutated)
    shutil.copy(os.path.join(ROOT, "tests", "test_evaluate_door.py"), os.path.join(tree, "tests"))
    r = subprocess.run([sys.executable, os.path.join(tree, "tests", "test_evaluate_door.py")], capture_output=True, text=True)
    last = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or [""])[-1][:100]
    print(("CAUGHT " if r.returncode else "ESCAPED") + f" {name:36s} {last}")
shutil.rmtree(os.path.join(ROOT, "temp", "door_mut_tree"), ignore_errors=True)
