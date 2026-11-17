"""E358 scratch: unused imports / names in the changed files (ast, no dependencies)"""
import ast, sys
for p in sys.argv[1:]:
    t = ast.parse(open(p, encoding="utf-8").read())
    for fn in ast.walk(t):
        if isinstance(fn, (ast.FunctionDef, ast.Module)):
            imported = {}
            for n in (fn.body if isinstance(fn, (ast.FunctionDef, ast.Module)) else []):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    for a in n.names:
                        imported[(a.asname or a.name).split(".")[0]] = n.lineno
            if imported:
                names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {n.value.id for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
                for k, ln in imported.items():
                    if k not in names:
                        print(f"{p}:{ln}: unused import {k} in {getattr(fn, 'name', 'module')}")
