"""Scratch: which python files of training/ evaluation/ tools/ are reachable (imports, flat module names) from the live roots? One-off audit for the restructuring."""
import ast, os, sys, collections
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIRS = ["training", "evaluation", "tools", "reports/build"]
files = []
for d in DIRS:
    for dp, dn, fn in os.walk(os.path.join(R, d)):
        if "__pycache__" in dp: continue
        for f in fn:
            if f.endswith(".py"): files.append(os.path.join(dp, f))
mod = collections.defaultdict(list)
for f in files:
    mod[os.path.splitext(os.path.basename(f))[0]].append(f)
def imports(f):
    try: t = ast.parse(open(f, encoding="utf-8", errors="ignore").read())
    except SyntaxError: return set()
    s = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names: s.add(a.name.split(".")[0]); s.add(a.name.split(".")[-1])
        elif isinstance(n, ast.ImportFrom) and n.module:
            s.add(n.module.split(".")[0]); s.add(n.module.split(".")[-1])
            for a in n.names: s.add(a.name)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".py"):
            s.add(os.path.splitext(os.path.basename(n.value))[0])   # script names launched by string
    return s
rel = lambda f: os.path.relpath(f, R).replace("\\", "/")
roots = set()
def add(pattern_prefix):
    for f in files:
        if rel(f).startswith(pattern_prefix): roots.add(f)
for p in ["evaluation/evaluate.py", "evaluation/eval_parts.py", "evaluation/lane_suite.py", "evaluation/tests/", "evaluation/sparse_lanes/", "evaluation/lane_grid/", "evaluation/lane_task/",
          "evaluation/drivers/p16_fast_check", "evaluation/drivers/p18_asm_check", "evaluation/drivers/e2e_gradient_report", "evaluation/drivers/p16_timing", "evaluation/drivers/p16_final_eval",
          "evaluation/drivers/p16_pipeline_eval", "evaluation/drivers/p18_latency", "training/trainer", "training/train/p16_train", "training/train/p16_ctrl", "training/grid/", "training/pipeline/p16_",
          "tools/", "reports/build/", "training/bflc_paths.py", "training/models/p16_", "training/models/p18_regis", "training/data/p16_", "training/losses/"]:
    add(p)
seen, q = set(roots), list(roots)
while q:
    f = q.pop()
    for m in imports(f):
        for g in mod.get(m, []):
            if g not in seen: seen.add(g); q.append(g)
dead = sorted(set(files) - seen)
tot = collections.Counter()
for f in dead: tot[rel(f).split("/")[0] + "/" + (rel(f).split("/")[1] if rel(f).count("/") > 1 else "")] += os.path.getsize(f)
print(f"{len(files)} python files, {len(seen)} reachable, {len(dead)} unreachable ({sum(os.path.getsize(f) for f in dead)//1024} KB)")
for f in dead: print("  DEAD", rel(f), os.path.getsize(f)//1024, "KB")
