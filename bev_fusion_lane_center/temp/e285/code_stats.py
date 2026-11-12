"""Code-health numbers of the lane project (scratch): size, function length, branching, switches, environment variables, path hacks, duplicated function names, tests per module."""
import ast
import collections
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIRS = ["training", "evaluation", "tools"]
SKIP = ("__pycache__", "/legacy/", "/data/prep/")

files = []
for d in DIRS:
    for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
        if any(s in dp.replace("\\", "/") + "/" for s in SKIP):
            continue
        for f in fn:
            if f.endswith(".py"):
                files.append(os.path.join(dp, f))

rows = []
funcs_all = collections.defaultdict(list)
tot_lines = tot_code = 0
for p in files:
    src = open(p, encoding="utf-8", errors="replace").read()
    lines = src.split("\n")
    code = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    longest, longest_name, branches, nfun = 0, "", 0, 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nfun += 1
            n = node.end_lineno - node.lineno + 1
            funcs_all[node.name].append(os.path.relpath(p, ROOT))
            if n > longest:
                longest, longest_name = n, node.name
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.IfExp)):
            branches += 1
    flags = len(re.findall(r"add_argument\(", src))
    env = len(re.findall(r"os\.environ|getenv", src))
    syspath = len(re.findall(r"sys\.path\.(insert|append)", src))
    longl = sum(1 for ln in lines if len(ln) > 200)
    rows.append(dict(f=os.path.relpath(p, ROOT).replace("\\", "/"), lines=len(lines), code=len(code), funs=nfun, longest=longest, longest_name=longest_name, br=branches, flags=flags, env=env,
                     syspath=syspath, longl=longl, test=("test" in os.path.basename(p) or "selftest" in src)))
    tot_lines += len(lines)
    tot_code += len(code)

print(f"files {len(rows)}  lines {tot_lines}  code lines {tot_code}")
for key, label in (("lines", "largest files"), ("longest", "longest functions"), ("flags", "most command-line flags"), ("env", "most environment reads"), ("longl", "most lines over 200 characters")):
    print(f"\n{label}:")
    for r in sorted(rows, key=lambda r: -r[key])[:8]:
        extra = f" ({r['longest_name']})" if key == "longest" else ""
        print(f"  {r[key]:6d}  {r['f']}{extra}")
print("\nsys.path hacks total:", sum(r["syspath"] for r in rows), " files with one:", sum(1 for r in rows if r["syspath"]))
print("environment reads total:", sum(r["env"] for r in rows), " in files:", [r["f"] for r in rows if r["env"]][:12])
print("files without a test or selftest in themselves:", sum(1 for r in rows if not r["test"]), "of", len(rows))
dups = {k: v for k, v in funcs_all.items() if len(set(v)) > 1 and not k.startswith("_") and k not in ("main", "selftest", "run", "load", "forward", "__init__", "check")}
print(f"\nfunction names defined in several files (public, {len(dups)}):")
for k, v in sorted(dups.items(), key=lambda kv: -len(set(kv[1])))[:12]:
    print(f"  {k}: {sorted(set(v))[:4]}")
big = [r for r in rows if r["longest"] > 120]
print(f"\nfunctions over 120 lines: {len(big)} files:", [(r['f'].split('/')[-1], r['longest_name'], r['longest']) for r in big][:10])
