"""List imported names never used in the module body (scratch, AST based). usage: python unused_imports.py file.py ..."""
import ast
import sys

for path in sys.argv[1:]:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.setdefault((a.asname or a.name).split(".")[0], node.lineno)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.setdefault(a.asname or a.name, node.lineno)
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            pass
    # names used only as attribute base are ast.Name too; strings in __all__ ignored
    for name, ln in sorted(imported.items(), key=lambda kv: kv[1]):
        if name not in used:
            print(f"{path}:{ln}: unused import {name}")
