"""throw-away: plant one bug at a time into a copy of evaluate.py and run the plan / executor part of tests/test_evaluate_door.py on it"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
src = open(os.path.join(ROOT, "evaluation", "evaluate.py")).read()
MUTANTS = [
    ("prior 500 -> 400", "= 30, 500, 8, 3, 6, 21", "= 30, 400, 8, 3, 6, 21"),
    ("timing n 30 -> 20", "= 30, 500, 8, 3, 6, 21", "= 20, 500, 8, 3, 6, 21"),
    ("showcase seed 21 -> 22", "= 30, 500, 8, 3, 6, 21", "= 30, 500, 8, 3, 6, 22"),
    ("soft off", "G, soft = global_options(), a.all", "G, soft = global_options(), False"),
    ("timing not run for --time", 'if a.time or a.all:', 'if a.all:'),
    ("time step dropped", 'if a.time:\n        steps.append(step("time", G + ["time"]))', 'if False:\n        steps.append(step("time", G + ["time"]))'),
    ("no second try", "tries=2,", "tries=1,"),
    ("store deleted not moved", "os.replace(store, dest)", "shutil.rmtree(store)"),
    ("fatal ignored", "            if st[\"fatal\"]:\n                code = 1\n                break", "            if False:\n                code = 1\n                break"),
    ("door.json overwritten", "\"calls\": earlier + [call]", "\"calls\": [call]"),
    ("test split offered", 'choices=["val", "train"]', 'choices=["val", "train", "test"]'),
    ("showcase out dropped", "global_options(show) + [", "global_options() + ["),
    ("qual fatal in --all", 'fatal=not soft) if isinstance(q, int)', 'fatal=True) if isinstance(q, int)'),
    ("curves always on", 'if a.all or getattr(a, f)]', 'if True]'),
    ("seed ignored", '"--seed", str(a.seed)]', '"--seed", "1"]'),
    ("frames not all named", 'step("frames", frames=q, fatal', 'step("frames", frames=q[:1], fatal'),
    ("report not last", "steps.append(step(\"report\"", "steps.insert(0, step(\"report\""),
    ("report tables dropped", "\"--val-table\", os.path.join(by_n, f\"table_n{a.n}.txt\"),", "\"--val-table\", \"x\","),
    ("fatal numbers soft", 'steps.append(step("numbers", G + more', 'steps.append(step("numbers", fatal=False, argv=G + more'),
]
for name, old, new in MUTANTS:
    assert old in src, name
    mutated = src.replace(old, new, 1).replace("import subprocess", "import shutil\nimport subprocess", 1)
    tree = os.path.join(ROOT, "temp", "door_mut_tree")
    shutil.rmtree(tree, ignore_errors=True)
    os.makedirs(os.path.join(tree, "evaluation"))
    os.makedirs(os.path.join(tree, "tests"))
    open(os.path.join(tree, "evaluation", "evaluate.py"), "w").write(mutated)
    shutil.copy(os.path.join(ROOT, "tests", "test_evaluate_door.py"), os.path.join(tree, "tests"))
    r = subprocess.run([sys.executable, os.path.join(tree, "tests", "test_evaluate_door.py")], capture_output=True, text=True)
    last = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or [""])[-1][:110]
    print(("CAUGHT " if r.returncode else "ESCAPED") + f" {name:28s} {last}")
shutil.rmtree(os.path.join(ROOT, "temp", "door_mut_tree"), ignore_errors=True)
