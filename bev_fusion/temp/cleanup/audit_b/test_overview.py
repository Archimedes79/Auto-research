"""Read-only: per test file: lines, first docstring line, local modules it imports (names from bev_fusion roots), whether run_gates.sh and tests/README.md name it,
and the git last-commit date. Usage: python test_overview.py"""
import ast, os, subprocess, glob

REPO = "/mnt/c/Auto-research/bev_fusion"
gates = open(REPO + "/tests/run_gates.sh").read()
readme = open(REPO + "/tests/README.md", encoding="utf-8").read()
LOCAL = {os.path.basename(p)[:-3] for d in ("training", "evaluation", "runner", "tests/support") for p in glob.glob(f"{REPO}/{d}/*.py")}
for f in sorted(glob.glob(REPO + "/tests/test_*.py")):
    src = open(f, encoding="utf-8").read()
    tree = ast.parse(src)
    doc = (ast.get_docstring(tree) or "").split("\n")[0][:110]
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in LOCAL:
                    mods.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in LOCAL:
            mods.add(n.module.split(".")[0])
    b = os.path.basename(f)
    date = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short", "--", f], cwd=REPO, capture_output=True, text=True).stdout.strip()
    print(f"{b:34s} {len(src.splitlines()):4d}L gates={'Y' if b in gates else '-'} readme={'Y' if b in readme else '-'} last={date}  mods={','.join(sorted(mods))}\n      {doc}")
