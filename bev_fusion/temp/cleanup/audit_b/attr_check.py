"""Read-only: for each given .py file, for `import M` / `import M as A` of a LOCAL module (bev_fusion roots), check that every `A.attr` used in the file
is a top-level name of M (or a submodule). Detects tests / scripts that use a function that is gone. Usage: python attr_check.py file ..."""
import ast, os, sys

REPO = "/mnt/c/Auto-research"
ROOTS = [REPO + "/bev_fusion/training", REPO + "/bev_fusion/evaluation", REPO + "/bev_fusion/tests/support", REPO + "/bev_fusion/runner",
         REPO + "/bev_fusion/studies/early", REPO + "/bev_fusion/training/legacy", REPO + "/bev_fusion/tests", REPO + "/bev_fusion",
         REPO + "/autoresearch", REPO + "/autoresearch/tools", REPO]


def find(mod):
    parts = mod.split(".")
    for r in ROOTS:
        p = os.path.join(r, *parts)
        if os.path.isfile(p + ".py"):
            return p + ".py"
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "__init__.py")):
            return os.path.join(p, "__init__.py")
    return None


def names(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    s = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            s.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            s.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                s.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name):
            s.add(n.attr)
    return s


cache = {}
for f in sys.argv[1:]:
    if not os.path.exists(f):
        continue
    try:
        tree = ast.parse(open(f, encoding="utf-8").read())
    except SyntaxError as e:
        print(f"{f}: SYNTAX {e}")
        continue
    alias = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                p = find(a.name)
                if p and a.name not in ("os", "sys"):
                    alias[(a.asname or a.name).split(".")[0]] = (a.name, p)
    bad = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias:
            mod, p = alias[n.value.id]
            if p not in cache:
                cache[p] = names(p)
            if n.attr not in cache[p] and not (os.path.basename(p) == "__init__.py" and find(mod + "." + n.attr)):
                bad.setdefault((n.value.id, mod, n.attr), n.lineno)
    for (a, mod, attr), ln in sorted(bad.items(), key=lambda x: x[1]):
        print(f"{f}:{ln}: {a}.{attr}  (module {mod}) has no top-level name {attr}")
