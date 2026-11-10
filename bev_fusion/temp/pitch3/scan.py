"""dead-code scan of my area (scratch): unused imports, unused locals, definitions that nothing but tests mention, commented-out code, if False.
python scan.py  (run from the repository root, in WSL)"""
import ast
import os
import re
import subprocess
import sys

ROOT = os.getcwd()
AREAS = ("autoresearch/evalsuite", "bev_fusion/evaluation")
files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout.split()
mine = [f for f in files if f.endswith(".py") and any(f.startswith(a + "/") for a in AREAS)]
text_files = [f for f in files if re.search(r"\.(py|md|sh|json|csv|txt)$", f) and "/temp/" not in f and "_to_delete" not in f and not f.startswith("literature")]
BLOB = {}
for f in text_files:
    try:
        BLOB[f] = open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
    except OSError:
        pass


def is_test(path):
    b = os.path.basename(path)
    return "selftest" in b or b.startswith("test_") or "/tests/" in path or b.startswith("bev_mutants") or b.startswith("bev_gtcheck") or b.startswith("bev_response")


WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
COUNTS = {f: __import__("collections").Counter(WORD.findall(t)) for f, t in BLOB.items()}


def uses(name, own_file, own_line_span):
    out = {}
    for f, c in COUNTS.items():
        n = c.get(name, 0) - (1 if f == own_file else 0)
        if n > 0:
            out[f] = n
    return out


def scan(path):
    src = BLOB.get(path) or open(os.path.join(ROOT, path), encoding="utf-8").read()
    tree = ast.parse(src)
    res = []
    # unused imports
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported[a.asname or a.name] = node.lineno
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    alls = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                alls = set(ast.literal_eval(node.value))
            except Exception:
                pass
    for nm, ln in imported.items():
        if nm not in names and nm not in alls and nm != "annotations" and nm != "*":
            res.append((path, ln, "unused import " + nm))
    # unused locals
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        stored, loaded = {}, set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Store):
                    stored.setdefault(n.id, n.lineno)
                else:
                    loaded.add(n.id)
        for nm, ln in stored.items():
            if nm not in loaded and not nm.startswith("_"):
                res.append((path, ln, f"local '{nm}' assigned, never read (in {fn.name})"))
    # if False / if 0
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Constant) and not n.test.value:
            res.append((path, n.lineno, "if False/0 block"))
    # definitions
    defs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            defs.append((node.name, node.lineno))
    for nm, ln in defs:
        if nm.startswith("__") or nm in ("main", "selftest", "run", "setUp"):
            continue
        u = uses(nm, path, None)
        prod = {f: n for f, n in u.items() if not is_test(f) and not f.endswith(".md")}
        if not prod:
            kind = "tests/docs only" if u else "NO USE ANYWHERE"
            res.append((path, ln, f"def {nm}: {kind}: " + ", ".join(sorted(u))[:160]))
    # commented-out code: runs of >= 3 comment lines that parse as python
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"\s*#\s*(def |class |import |from \S+ import|if .*:$|for .*:$|return |[a-z_]+\s*=\s*\S|print\()", lines[i]):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("#"):
                j += 1
            if j - i >= 2:
                res.append((path, i + 1, f"commented-out code? {j - i} lines: {lines[i].strip()[:80]}"))
            i = j
        else:
            i += 1
    return res


allres = []
for p in mine:
    try:
        allres += scan(p)
    except SyntaxError as e:
        allres.append((p, 0, "SYNTAX " + str(e)))
for p, ln, msg in sorted(allres):
    print(f"{p}:{ln} {msg}")
print(len(allres), "findings in", len(mine), "files")
