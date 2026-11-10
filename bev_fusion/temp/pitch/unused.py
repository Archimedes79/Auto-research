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
    text_used = lambda name: src.count(name) > 1
    dead = [f"{k}@{v}" for k, v in imported.items() if k not in used and k != "annotations" and not (k in src.split("import", 1)[-1] and False)]
    # functions / methods defined and never referenced by name anywhere in the repo files given
    defs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("__")]
    print(path.split("/")[-1], "unused imports:", dead)
    all_src = "".join(open(p, encoding="utf-8").read() for p in sys.argv[1:])
    print("   defined, referenced once only:", [d for d in defs if all_src.count(d) == 1])
