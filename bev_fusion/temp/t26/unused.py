import ast, sys
for p in sys.argv[1:]:
    t = ast.parse(open(p).read())
    imported = {}
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                imported[(a.asname or a.name).split(".")[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                imported[a.asname or a.name] = n.lineno
    used = {n.id for n in ast.walk(t) if isinstance(n, ast.Name)} | {n.value.id for n in ast.walk(t) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    print(p, "unused imports:", sorted(k for k in imported if k not in used))
