"""Scratch: unused imports, unused local variables and module-level names never referenced in the repo (a work list, not a proof). Usage: python unused_scan.py file.py ..."""
import ast
import sys


def scan(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                name = (a.asname or a.name).split(".")[0]
                if name not in used and "noqa" not in src.splitlines()[n.lineno - 1]:
                    out.append(f"{path}:{n.lineno}: unused import {name}")
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                name = a.asname or a.name
                if name not in used and "noqa" not in src.splitlines()[n.lineno - 1]:
                    out.append(f"{path}:{n.lineno}: unused import {name}")
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stores, loads = {}, set()
            for m in ast.walk(n):
                if isinstance(m, ast.Name):
                    (stores.setdefault(m.id, m.lineno) if isinstance(m.ctx, ast.Store) else loads.add(m.id))
            for k, ln in stores.items():
                if k not in loads and not k.startswith("_"):
                    out.append(f"{path}:{ln}: variable {k} assigned in {n.name} and never read")
            args = [a.arg for a in n.args.args + n.args.kwonlyargs if a.arg not in ("self", "cls")]
            for a in args:
                if a not in loads and not n.name.startswith("__"):
                    out.append(f"{path}:{n.lineno}: parameter {a} of {n.name} never used")
    return out


for p in sys.argv[1:]:
    for line in scan(p):
        print(line)
print("scan done")
