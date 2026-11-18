"""Throw-away static scan of the lane project code (reviewer pass 2026-11-17): unused imports,
unreferenced top-level names, duplicated function bodies, very long functions.
Run: wsl python3 temp/_session_scratch_lead/scan_code.py   (from bev_fusion_lane_center/)"""
import ast, hashlib, os, re, sys, collections

ROOT = "."
SKIP = ("legacy", "__pycache__", "_to_delete", "/temp/", "\\temp\\", "runs/", "node_modules")
CODE_DIRS = ["training", "evaluation", "tools"]


def py_files():
    for d in CODE_DIRS:
        for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
            if any(s in dp.replace("\\", "/") + "/" for s in SKIP):
                continue
            for f in fn:
                if f.endswith(".py"):
                    yield os.path.join(dp, f)


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


files = sorted(py_files())
src = {p: read(p) for p in files}
trees = {}
for p, s in src.items():
    try:
        trees[p] = ast.parse(s)
    except SyntaxError as e:
        print("SYNTAX", p, e)

# corpus for name references: every py / sh / md / json text of the project (not runs, not temp)
corpus = {}
for dp, dn, fn in os.walk(ROOT):
    d = dp.replace("\\", "/") + "/"
    if any(x in d for x in ("/runs/", "/temp/", "/_to_delete/", "/__pycache__/", "/.git/", "/results/")):
        continue
    for f in fn:
        if f.endswith((".py", ".sh", ".md", ".json", ".csv")):
            p = os.path.join(dp, f)
            try:
                if os.path.getsize(p) < 3_000_000:
                    corpus[p] = read(p)
            except OSError:
                pass
alltext = "\n".join(corpus.values())
words = collections.Counter(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", alltext))

# 1. unused imports
unused_imp = []
for p, t in trees.items():
    if os.path.basename(p) == "__init__.py":
        continue
    imported = {}
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                imported[(a.asname or a.name).split(".")[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name != "*":
                    imported[a.asname or a.name] = n.lineno
    used = {n.id for n in ast.walk(t) if isinstance(n, ast.Name)} | {
        n.value.id for n in ast.walk(t) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    txt = src[p]
    for name, ln in imported.items():
        if name not in used and not re.search(r"__all__.*['\"]%s['\"]" % re.escape(name), txt):
            unused_imp.append((p, ln, name))

# 2. top-level names referenced nowhere else (count of word occurrences == definitions in own file only)
dead = []
for p, t in trees.items():
    own = collections.Counter(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", src[p]))
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            nm = n.name
            if nm.startswith(("test_", "_test", "main", "selftest", "Test")) or nm in ("run", "cli", "build"):
                continue
            total = words[nm]
            if total <= 1:  # defined, never mentioned anywhere (even as a string, a doc or a call)
                dead.append((p, n.lineno, nm, n.end_lineno - n.lineno + 1))

# 3. duplicated function bodies
def norm(fn):
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False, include_attributes=False)

groups = collections.defaultdict(list)
for p, t in trees.items():
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nstm = n.end_lineno - n.lineno + 1
            if nstm >= 7:
                h = hashlib.md5(norm(n).encode()).hexdigest()
                groups[h].append((p, n.lineno, n.name, nstm))
dups = [g for g in groups.values() if len({(x[0]) for x in g}) > 1 or len(g) > 1]

# 4. long functions
longf = []
for p, t in trees.items():
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            L = n.end_lineno - n.lineno + 1
            if L >= 160 and not n.name.startswith("test"):
                longf.append((L, p, n.lineno, n.name))

out = []
out.append("== files %d, lines %d" % (len(files), sum(s.count("\n") for s in src.values())))
out.append("== UNUSED IMPORTS (%d)" % len(unused_imp))
for p, ln, nm in sorted(unused_imp):
    out.append("  %s:%d %s" % (p, ln, nm))
out.append("== NEVER-MENTIONED TOP-LEVEL DEFS (%d)" % len(dead))
for p, ln, nm, L in sorted(dead):
    out.append("  %s:%d %s (%d lines)" % (p, ln, nm, L))
out.append("== DUPLICATED FUNCTION BODIES (%d groups)" % len(dups))
for g in sorted(dups, key=lambda g: -g[0][3] * len(g)):
    out.append("  " + " | ".join("%s:%d %s(%d)" % x for x in g))
out.append("== LONG FUNCTIONS >= 160 lines (%d)" % len(longf))
for L, p, ln, nm in sorted(longf, reverse=True):
    out.append("  %4d %s:%d %s" % (L, p, ln, nm))
text = "\n".join(out)
with open("temp/_session_scratch_lead/scan_code.out", "w", encoding="utf-8") as f:
    f.write(text)
print(text[:12000])
