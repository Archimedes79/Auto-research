"""names loaded in _selftest / _selftest_fusion_area / _demo_scene that are neither assigned there, nor module globals, nor builtins"""
import ast, builtins, os, sys
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "losses.py")
tree = ast.parse(open(path, encoding="utf-8").read())
glob = {n.id for node in tree.body for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
glob |= {a.asname or a.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for a in node.names}
glob |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
bad = 0
for fn in tree.body:
    if isinstance(fn, ast.FunctionDef) and fn.name in ("_selftest", "_selftest_fusion_area", "_demo_scene"):
        local = {a.arg for a in fn.args.args}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                local.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.Lambda)):
                if isinstance(n, ast.FunctionDef):
                    local.add(n.name)
                local |= {a.arg for a in n.args.args}
            elif isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                local |= {a.asname or a.name.split(".")[0] for a in n.names}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in local | glob and not hasattr(builtins, n.id):
                print(fn.name, "undefined:", n.id, "line", n.lineno); bad += 1
print("undefined names:", bad)
