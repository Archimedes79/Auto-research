"""Read-only: unused imports (module- or function-level) and unused local variables of simple form in the given files. Imports re-exported
(another tracked file does `from <this module> import name`) are marked. Usage: python unused_imports.py file ..."""
import ast, os, pickle, re, sys

REPO = "/mnt/c/Auto-research"
idx = pickle.load(open("/tmp/audit_b_index.pkl", "rb"))


def reexported(modname, name):
    rx = re.compile(r"from\s+[\w.]*\b" + re.escape(modname) + r"\s+import\s+[^\n]*\b" + re.escape(name) + r"\b")
    for f, t in idx.items():
        if f.endswith(".py") and rx.search(t):
            return f
    return None


for f in sys.argv[1:]:
    p = os.path.join(REPO, f)
    if not os.path.exists(p):
        continue
    src = open(p, encoding="utf-8").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    mod = os.path.basename(f)[:-3]
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute):
            pass
    # names mentioned in strings of __all__ or docstring-free text are ignored
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            if isinstance(n, ast.ImportFrom) and n.module == "__future__":
                continue
            for a in n.names:
                if a.name == "*":
                    continue
                bound = (a.asname or a.name).split(".")[0]
                if bound not in used:
                    line = lines[n.lineno - 1]
                    re_ex = reexported(mod, bound)
                    tag = f"  [re-exported to {re_ex}]" if re_ex else ""
                    print(f"{f}:{n.lineno}: unused import {a.name}{(' as ' + a.asname) if a.asname else ''}{'  (noqa)' if 'noqa' in line else ''}{tag}")
    # unused simple locals: assigned Name in a function never loaded in that function
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stores, loads = {}, set()
            for n in ast.walk(fn):
                if isinstance(n, ast.Name):
                    if isinstance(n.ctx, ast.Store):
                        stores.setdefault(n.id, n.lineno)
                    else:
                        loads.add(n.id)
            for name, ln in stores.items():
                if name not in loads and not name.startswith("_") and name not in ("self",):
                    # skip tuple-unpacking targets (common _ idiom) by checking the source line for '=' directly after the name
                    line = lines[ln - 1]
                    if re.search(r"(^|[\s,(])" + re.escape(name) + r"\s*=[^=]", line) and "for " not in line.split(name)[0]:
                        print(f"{f}:{ln}: local {name} assigned in {fn.name}() but never read")
