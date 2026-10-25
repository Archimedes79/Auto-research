"""Consistency check of the migrated tree: (A) scripts named inside kept wrappers/scripts exist in a live folder, (B) every import resolves, (C) absolute paths written as plain strings exist.
Run in WSL: _run.sh mig_check.py"""
import ast
import os
import re
import sys

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
LIVE = [os.path.join(dp) for base in ("training", "evaluation", "tools", "reports/build", "temp") for dp, dn, fn in os.walk(f"{R}/{base}") if "__pycache__" not in dp]
names = {}
for d in LIVE:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if os.path.isfile(f"{d}/{f}"):
                names[f] = d
mods = {f[:-3] for f in names if f.endswith(".py")}
stdlib = set(sys.stdlib_module_names)
where = {}
for dp, dn, fn in os.walk(R):
    if "/_to_delete" in dp or "/runs" in dp or "/literature" in dp or "/data" in dp or "/downloads" in dp:
        continue
    for f in fn:
        where.setdefault(f, []).append(dp.replace(R, "R"))
A, B, C = [], [], []
files = [f"{d}/{f}" for f, d in names.items() if f.endswith((".py", ".sh", ".ps1"))]
for p in files:
    try:
        s = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    base = os.path.basename(p)
    # A: referenced script names
    for m in set(re.findall(r"\b([A-Za-z0-9_\-]+\.(?:py|sh|ps1))\b", s)):
        if m not in names and m != base and m in where and not any(w.startswith("R/scripts") or w.startswith("R/evaluation") or w.startswith("R/tools") for w in where[m]):
            A.append((base, m, where[m][0]))
        elif m not in names and m not in where and not m.startswith(("wandb", "setup.py", "main.py", "train.py")) and re.match(r"^(_?(p\d+|run|make|fig|build|e2e|check|eval)[A-Za-z0-9_]*)", m):
            A.append((base, m, "MISSING"))
    # B: imports
    if p.endswith(".py"):
        try:
            tree = ast.parse(s)
        except SyntaxError as e:
            B.append((base, f"SYNTAX {e}"))
            continue
        for nd in ast.walk(tree):
            mm = []
            if isinstance(nd, ast.Import):
                mm = [a.name.split(".")[0] for a in nd.names]
            elif isinstance(nd, ast.ImportFrom) and nd.module and nd.level == 0:
                mm = [nd.module.split(".")[0]]
            for x in mm:
                if x in stdlib or x in mods:
                    continue
                # third party: only report names that exist as project files somewhere (moved)
                if x in {f[:-3] for f in where if f.endswith(".py")}:
                    B.append((base, x, where.get(x + ".py", ["?"])[0]))
    # C: plain absolute paths
    for m in set(re.findall(r"/mnt/c/ML-Experiment/bev_fusion_lane_center/[A-Za-z0-9_./\-+]+", s)):
        q = m.rstrip(".,)")
        if "{" in q:
            continue
        if not os.path.exists(q):
            C.append((base, q.replace(R, "R")))
print("A scripts named but not in a live folder:", len(A))
for x in sorted(set(A))[:60]:
    print("   ", x)
print("B imports of project modules that are not importable:", len(B))
for x in sorted(set(B))[:60]:
    print("   ", x)
print("C literal paths that do not exist:", len(C))
for x in sorted(set(C))[:80]:
    print("   ", x)
