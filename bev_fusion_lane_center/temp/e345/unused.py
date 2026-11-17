"""poor man's unused-name check: names imported / defined at module level and never used again"""
import ast, sys
for path in sys.argv[1:]:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    imported = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: imported[(a.asname or a.name).split(".")[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom):
            for a in n.names: imported[a.asname or a.name] = n.lineno
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    print(path, "unused imports:", sorted(k for k in imported if k not in names))
    defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("test_")]
    uses = src
    print("  defined, used once only:", [d for d in defs if src.count(d) == 1])
