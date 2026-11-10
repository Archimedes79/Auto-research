"""Static import check (read-only): for every given .py file, resolve each import to a local module (several roots)
or to an installed/stdlib module, and for 'from X import Y' of a local module check that Y is a top-level name of X.
Usage: python import_check.py file...  (run from /mnt/c/Auto-research)"""
import ast, sys, os, importlib.util

REPO = "/mnt/c/Auto-research"
ROOTS = [REPO, REPO + "/bev_fusion", REPO + "/bev_fusion/training", REPO + "/bev_fusion/evaluation",
         REPO + "/bev_fusion/tests/support", REPO + "/bev_fusion/runner", REPO + "/autoresearch",
         REPO + "/bev_fusion/studies/early", REPO + "/bev_fusion/training/legacy", REPO + "/bev_fusion/tests",
         REPO + "/bev_fusion/tools"]


def find_local(modname):
    parts = modname.split(".")
    for r in ROOTS:
        p = os.path.join(r, *parts)
        if os.path.isfile(p + ".py"):
            return p + ".py"
        if os.path.isdir(p):
            init = os.path.join(p, "__init__.py")
            return init if os.path.isfile(init) else p
    return None


def top_names(path):
    if path is None or not path.endswith(".py"):
        return None
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        return None
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        names.add(x.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, (ast.If, ast.Try)):
            for x in ast.walk(n):
                if isinstance(x, (ast.FunctionDef, ast.ClassDef)):
                    names.add(x.name)
                elif isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    names.add(x.id)
                elif isinstance(x, (ast.Import, ast.ImportFrom)):
                    for a in x.names:
                        names.add((a.asname or a.name).split(".")[0])
    return names


def main():
    for f in sys.argv[1:]:
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except SyntaxError as e:
            print(f"{f}: SYNTAX ERROR {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    top = a.name.split(".")[0]
                    if find_local(a.name) is None and importlib.util.find_spec(top) is None:
                        print(f"{f}:{node.lineno}: import {a.name} UNRESOLVED")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                m = node.module or ""
                top = m.split(".")[0]
                loc = find_local(m)
                if loc is None:
                    try:
                        if importlib.util.find_spec(top) is None:
                            print(f"{f}:{node.lineno}: from {m} UNRESOLVED")
                    except Exception:
                        print(f"{f}:{node.lineno}: from {m} UNRESOLVED(exc)")
                    continue
                names = top_names(loc)
                if names is None:
                    # package dir: check submodule or name in __init__
                    for a in node.names:
                        if a.name == "*":
                            continue
                        if find_local(m + "." + a.name) is None:
                            print(f"{f}:{node.lineno}: from {m} import {a.name}: package, not found as submodule (check __init__)")
                    continue
                for a in node.names:
                    if a.name == "*":
                        continue
                    if a.name not in names and find_local(m + "." + a.name) is None:
                        print(f"{f}:{node.lineno}: from {m} import {a.name}: NAME NOT IN {os.path.relpath(loc, REPO)}")


main()
