"""throw-away: plant one bug at a time into a copy of the evaluation folder and run bev_timing.py --selftest on it"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
src = open(os.path.join(ROOT, "evaluation", "bev_timing.py")).read()
MUTANTS = [
    ("own pid not dropped", '!= str(os.getpid())]', '!= "x"]'),
    ("parent dropped too", '!= str(os.getpid())]', 'not in (str(os.getpid()), str(os.getppid()))]'),
    ("settled without is_available", 'and torch.cuda.is_available():', ':'),
    ("settled without sync", '        torch.cuda.synchronize()\n        time.sleep(1.5)', '        time.sleep(1.5)'),
    ("settled without wait", '        time.sleep(1.5)', '        time.sleep(0)'),
    ("settled never looks", 'return gpu_state(samples)', 'return {}'),
    ("second look always idle", 'idle2 = is_idle(state2, device, check_memory=False)', 'idle2 = True'),
    ("second look direct", 'state2 = settled_state(device)', 'state2 = gpu_state(samples=2)'),
    ("second look refusal ignores force", 'if not idle2 and not force:', 'if not idle2:'),
    ("first look refusal off", 'if not idle and not force:', 'if False:'),
    ("first look ignores force", 'if not idle and not force:', 'if not idle:'),
]
for name, old, new in MUTANTS:
    assert old in src, name
    tree = os.path.join(ROOT, "temp", "timing_mut_tree")
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "evaluation"), os.path.join(tree, "evaluation"), ignore=shutil.ignore_patterns("__pycache__", "golden"))
    open(os.path.join(tree, "evaluation", "bev_timing.py"), "w").write(src.replace(old, new, 1))
    r = subprocess.run([sys.executable, os.path.join(tree, "evaluation", "bev_timing.py"), "--selftest"], capture_output=True, text=True)
    last = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or [""])[-1][:100]
    print(("CAUGHT " if r.returncode else "ESCAPED") + f" {name:36s} {last}")
shutil.rmtree(os.path.join(ROOT, "temp", "timing_mut_tree"), ignore_errors=True)
