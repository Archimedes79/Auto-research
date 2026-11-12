import ast, os, collections
p = os.path.join(os.environ["BFLC_ROOT"], "tools", "viz", "qgraph_view.py")
src = open(p, encoding="utf-8").read()
tree = ast.parse(src)
names = collections.Counter(n.id for n in ast.walk(tree) if isinstance(n, ast.Name)) + collections.Counter(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute))
defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
imps = [(a.asname or a.name).split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
print("defined, never used:", [d for d in defs if names[d] == 0 and d not in ("main",)])
print("imported, never used:", [i for i in imps if names[i] == 0])
for f in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
    used = {n.id for n in ast.walk(f) if isinstance(n, ast.Name)}
    un = [a.arg for a in f.args.args if a.arg not in used]
    if un: print("unused parameter in", f.name, un)
    assigned = collections.Counter(n.id for n in ast.walk(f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    loads = collections.Counter(n.id for n in ast.walk(f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    dead = [k for k in assigned if loads[k] == 0 and not k.startswith("_")]
    if dead: print("assigned, never read in", f.name, dead)
