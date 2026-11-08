import ast, builtins, sys
for fn in sys.argv[1:]:
    t = ast.parse(open(fn, encoding="utf-8").read())
    defined = set(dir(builtins))
    for n in ast.walk(t):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            defined.add(n.name)
            if not isinstance(n, ast.ClassDef):
                for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs: defined.add(a.arg)
                if n.args.vararg: defined.add(n.args.vararg.arg)
                if n.args.kwarg: defined.add(n.args.kwarg.arg)
        elif isinstance(n, ast.Lambda):
            for a in n.args.args: defined.add(a.arg)
            if n.args.vararg: defined.add(n.args.vararg.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): defined.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name: defined.add(n.name)
        elif isinstance(n, ast.arg): defined.add(n.arg)
    undef = sorted({n.id for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined})
    print(fn, "undefined:", undef)
