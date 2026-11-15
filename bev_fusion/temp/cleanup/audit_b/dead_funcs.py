"""Read-only, fast version: per file, top-level defs / classes / methods / assigned constants whose NAME is used nowhere else: not in the file
itself (beyond its definition) and not in any other tracked code/test file (identifier-token match over an in-memory index built once).
Prints also names referenced only by tests. Usage: python dead_funcs.py file.py ... (run from /mnt/c/Auto-research)"""
import ast, os, pickle, re, sys, collections

REPO = "/mnt/c/Auto-research"
idx = pickle.load(open("/tmp/audit_b_index.pkl", "rb"))
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def is_test(f):
    b = os.path.basename(f)
    return "/tests/" in f or b.startswith("test_") or "selftest" in b


# token -> list of files (set per file)
where = collections.defaultdict(set)
count_in = {}
for f, txt in idx.items():
    c = collections.Counter(TOK.findall(txt))
    count_in[f] = c
    for t in c:
        where[t].add(f)


def main():
    for m in sys.argv[1:]:
        src = open(os.path.join(REPO, m), encoding="utf-8").read()
        tree = ast.parse(src)
        items = []
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                items.append((n.name, n.lineno, type(n).__name__))
                if isinstance(n, ast.ClassDef):
                    for b in n.body:
                        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and not (b.name.startswith("__") and b.name.endswith("__")):
                            items.append((b.name, b.lineno, "method of " + n.name))
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        items.append((t.id, n.lineno, "const"))
        own_c = collections.Counter(TOK.findall(src))
        out = []
        for name, ln, k in items:
            own = own_c[name] - 1
            others = [f for f in where.get(name, ()) if f != m]
            code = [f for f in others if f.endswith((".py", ".sh", ".ps1")) and not is_test(f)]
            test = [f for f in others if f.endswith((".py", ".sh", ".ps1")) and is_test(f)]
            doc = [f for f in others if not f.endswith((".py", ".sh", ".ps1"))]
            if own <= 0 and not code and not test:
                out.append(f"  {ln:4d} {k:22s} {name:32s} UNUSED (docs: {len(doc)})")
            elif own <= 0 and not code:
                out.append(f"  {ln:4d} {k:22s} {name:32s} only tests: {','.join(os.path.basename(x) for x in test[:3])} (docs: {len(doc)})")
        print(f"=== {m}")
        print("\n".join(out) if out else "  (nothing)")
        sys.stdout.flush()


main()
