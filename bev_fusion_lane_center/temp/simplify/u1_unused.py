"""U1 scratch probe: top-level names of a module that nothing in it uses (a name another module imports is listed too: grep before deleting),
and names loaded inside it that are defined nowhere (module level, builtins, function locals / params, comprehension targets)."""
import ast
import builtins
import sys

src = open(sys.argv[1], encoding="utf-8").read()
tree = ast.parse(src)
top = {}
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        top[n.name] = n
    elif isinstance(n, ast.Assign):
        for t in n.targets:
            for x in ast.walk(t):
                if isinstance(x, ast.Name):
                    top[x.id] = n
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
            top[(a.asname or a.name).split(".")[0]] = n
used = {}
for n in ast.walk(tree):
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
        used[n.id] = used.get(n.id, 0) + 1
print("unused top-level:", sorted(k for k in top if k not in used and not k.startswith("__")))
# undefined loads: every Name load must be bound somewhere in the file or be a builtin
bound = set(top) | set(dir(builtins))
for n in ast.walk(tree):
    if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
        bound.add(n.id)
    elif isinstance(n, ast.arg):
        bound.add(n.arg)
    elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        bound.add(n.name)
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
            bound.add((a.asname or a.name).split(".")[0])
    elif isinstance(n, ast.ExceptHandler) and n.name:
        bound.add(n.name)
print("loaded but never bound:", sorted(k for k in used if k not in bound))
