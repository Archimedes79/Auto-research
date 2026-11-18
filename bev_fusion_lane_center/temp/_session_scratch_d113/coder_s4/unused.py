import ast, sys
for path in sys.argv[1:]:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    imported = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imported[(a.asname or a.name).split(".")[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                imported[a.asname or a.name] = n.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    print(path, "unused imports:", [(k, v) for k, v in imported.items() if k not in used])
    # assigned-but-unused locals (simple): names stored in a function and never loaded in it
    for f in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        stores = {n.id: n.lineno for n in ast.walk(f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        loads = {n.id for n in ast.walk(f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        un = [(k, v) for k, v in stores.items() if k not in loads and k != "_"]
        if un:
            print("  ", f.name, "stored, never loaded:", un)
