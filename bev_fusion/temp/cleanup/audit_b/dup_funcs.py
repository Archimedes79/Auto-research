"""Read-only: functions (>= 5 statements/lines) whose normalised AST body is identical in 2+ files of the given list. Usage: python dup_funcs.py file ..."""
import ast, hashlib, os, sys, collections

REPO = "/mnt/c/Auto-research"
by_hash = collections.defaultdict(list)
for f in sys.argv[1:]:
    p = os.path.join(REPO, f)
    if not os.path.exists(p):
        continue
    try:
        tree = ast.parse(open(p, encoding="utf-8").read())
    except SyntaxError:
        continue
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nlines = (n.end_lineno or n.lineno) - n.lineno + 1
            if nlines < 5:
                continue
            body = ast.dump(ast.Module(body=[ast.Expr(value=ast.Constant(0))] + [b for b in n.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant))], type_ignores=[]), annotate_fields=False)
            h = hashlib.md5((n.name + body).encode()).hexdigest()
            by_hash[h].append((f, n.lineno, n.name, nlines))
for h, v in by_hash.items():
    files = {x[0] for x in v}
    if len(files) >= 2:
        print(f"{v[0][2]} ({v[0][3]} lines) identical in: " + "; ".join(f"{os.path.relpath(x[0], 'bev_fusion')}:{x[1]}" for x in v))
# same-name functions, different body, long (possible near duplicates)
names = collections.defaultdict(set)
sizes = {}
for h, v in by_hash.items():
    for f, ln, name, nl in v:
        names[name].add(f)
print("--- same function name in 3+ files of scope (bodies may differ):")
for name, fs in sorted(names.items()):
    if len(fs) >= 3 and not name.startswith("__") and name not in ("main", "check", "_selftest", "selftest", "forward", "run", "test", "setUp"):
        print(f"  {name}: " + ", ".join(sorted(os.path.basename(x) for x in fs)))
