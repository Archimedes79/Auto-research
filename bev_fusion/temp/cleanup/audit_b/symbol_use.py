"""Read-only: for each module file given, list its top-level symbols (functions, classes, methods, constants) and the tracked files
that mention each symbol (word match over an in-memory index of the tracked text files), split into: tests, other code (.py/.sh), docs.
Index is cached in /tmp/audit_b_index.pkl (scratch outside the repo; delete to rebuild).
Usage: python symbol_use.py module.py [module.py ...]   (run from /mnt/c/Auto-research)"""
import ast, subprocess, sys, os, re, pickle

REPO = "/mnt/c/Auto-research"
CACHE = "/tmp/audit_b_index.pkl"
SKIP = ("bev_fusion/temp/cleanup/",)
EXT = (".py", ".sh", ".md", ".json", ".txt", ".csv", ".ps1", ".yaml", ".yml", ".cfg", ".toml", ".svg")


def build():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split("\n")
    idx = {}
    for f in out:
        if not f or f.startswith(SKIP) or not f.endswith(EXT):
            continue
        p = os.path.join(REPO, f)
        try:
            if os.path.getsize(p) > 3_000_000:
                continue
            idx[f] = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
    pickle.dump(idx, open(CACHE, "wb"))
    return idx


def load():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    return build()


def kind(f):
    if f.endswith((".py", ".sh", ".ps1")):
        b = os.path.basename(f)
        if "/tests/" in f or b.startswith("test_") or "selftest" in b:
            return "test"
        return "code"
    return "doc"


def main():
    idx = load()
    for m in sys.argv[1:]:
        tree = ast.parse(open(os.path.join(REPO, m), encoding="utf-8").read())
        syms = []
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                syms.append((n.name, n.lineno, "def"))
            elif isinstance(n, ast.ClassDef):
                syms.append((n.name, n.lineno, "class"))
                for b in n.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and not b.name.startswith("__"):
                        syms.append((b.name, b.lineno, "  meth " + n.name))
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        syms.append((t.id, n.lineno, "const"))
        print(f"=== {m}")
        for name, ln, k in syms:
            rx = re.compile(r"\b" + re.escape(name) + r"\b")
            groups = {"code": [], "test": [], "doc": []}
            for f, txt in idx.items():
                if f == m:
                    continue
                if rx.search(txt):
                    groups[kind(f)].append(f)
            own = len(rx.findall(idx.get(m, "")))
            flag = ""
            if not groups["code"]:
                flag = "  <-- NO OTHER CODE"
                if not groups["test"] and not groups["doc"]:
                    flag = "  <-- NOWHERE ELSE"

            def short(lst):
                return ",".join(os.path.basename(x) for x in lst[:4]) + ("+%d" % (len(lst) - 4) if len(lst) > 4 else "")
            print(f"  {ln:4d} {k:14s} {name:30s} own{own:2d} code[{len(groups['code'])}]:{short(groups['code'])} test[{len(groups['test'])}]:{short(groups['test'])} doc[{len(groups['doc'])}]{flag}")


main()
