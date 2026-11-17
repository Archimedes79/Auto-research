"""Scratch: which command-line flags of the live trainers / scorers are named by a recent card recipe, a chain script or a live caller.
usage: python3 temp/_session_scratch_s292/flag_audit.py"""
import glob
import re
import os

FILES = ["training/train/jends_train.py", "training/train/jmaps_train.py", "training/train/qgraph_train.py", "evaluation/jends_eval.py", "evaluation/qends_eval.py",
         "evaluation/jmaps_eval.py", "evaluation/qgraph_eval.py", "training/train/jwire_train.py", "evaluation/jwire_data.py", "evaluation/jmaneuver.py"]
RECENT = sorted(glob.glob("experiments/E3[0-3]*/experiment.md")) + sorted(glob.glob("experiments/E29[0-9]*/experiment.md")) + sorted(glob.glob("experiments/E28[5-9]*/experiment.md"))
scripts = glob.glob("temp/**/*.sh", recursive=True) + glob.glob("temp/**/*.py", recursive=True)
scripts = [s for s in scripts if "_session_scratch_" not in s or "s292" in s]
cards = {p: open(p, encoding="utf-8", errors="replace").read() for p in RECENT}
sh = {}
for p in scripts:
    try:
        sh[p] = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        pass
live = {}
for p in glob.glob("training/**/*.py", recursive=True) + glob.glob("evaluation/**/*.py", recursive=True):
    live[p] = open(p, encoding="utf-8", errors="replace").read()
print(f"{len(RECENT)} cards, {len(sh)} scripts, {len(live)} live files")
for f in FILES:
    if not os.path.exists(f):
        print("missing", f)
        continue
    src = open(f, encoding="utf-8").read()
    flags = re.findall(r'add_argument\(\s*"(--[a-z0-9\-]+)"', src)
    print(f"\n== {f}: {len(src.splitlines())} lines, {len(flags)} flags")
    for fl in flags:
        pat = re.compile(re.escape(fl) + r"(?![a-z0-9\-])")
        nc = sum(1 for t in cards.values() if pat.search(t))
        ns = sum(1 for t in sh.values() if pat.search(t))
        nl = sum(len(pat.findall(t)) for p, t in live.items() if p != f)
        nself = len(pat.findall(src))
        mark = "   <-- DEAD?" if nc == 0 and ns == 0 and nl == 0 else ""
        print(f"  {fl:24s} cards {nc:2d} scripts {ns:3d} otherlive {nl:2d} self {nself}{mark}")
