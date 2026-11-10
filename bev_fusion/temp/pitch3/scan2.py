"""second dead-code scan (scratch): a def/class/method of my area that no ast Name / Attribute / import / string-key of any tracked .py outside tests refers to.
python scan2.py   (repository root, WSL)"""
import ast
import os
import re
import subprocess

files = [f for f in subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True).stdout.split() if os.path.isfile(f)]
mine = [f for f in files if f.startswith(("autoresearch/evalsuite/", "bev_fusion/evaluation/"))]


def is_test(p):
    b = os.path.basename(p)
    return "selftest" in b or b.startswith("test_") or "/tests/" in p or b in ("bev_mutants.py", "bev_gtcheck.py", "bev_response.py", "check.py")


refs_prod, refs_test, strs = {}, {}, set()
for f in files:
    if "/runs/" in f or "/temp/" in f:
        continue
    try:
        t = ast.parse(open(f, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        continue
    bucket = refs_test if is_test(f) else refs_prod
    for n in ast.walk(t):
        names = []
        if isinstance(n, ast.Name):
            names.append(n.id)
        elif isinstance(n, ast.Attribute):
            names.append(n.attr)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names += [a.name.split(".")[-1] for a in n.names]
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            strs.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", n.value))
        elif isinstance(n, ast.keyword) and n.arg:
            names.append(n.arg)
        for nm in names:
            bucket.setdefault(nm, set()).add(f)

for f in mine:
    if is_test(f):
        continue
    t = ast.parse(open(f, encoding="utf-8").read())
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            nm = n.name
            if nm.startswith("__") or nm in ("main", "build", "name"):
                continue
            outside = {x for x in refs_prod.get(nm, set())}
            # uses inside the same file count only if they are not the def itself (Name/Attribute nodes are uses, a def is not a Name)
            if not outside:
                kind = "tests only" if nm in refs_test else ("string only" if nm in strs else "NO REFERENCE")
                print(f"{f}:{n.lineno} {nm}: {kind}")
