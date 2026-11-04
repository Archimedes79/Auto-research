"""scratch: imported names and top-level functions of a module that nothing in the module (or in the given other files) uses"""
import ast
import re
import sys

path = sys.argv[1]
others = "".join(open(p, encoding="utf-8", errors="replace").read() for p in sys.argv[2:])
src = open(path, encoding="utf-8").read()
tree = ast.parse(src)
imported = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for a in node.names:
            imported[(a.asname or a.name).split(".")[0]] = node.lineno
    elif isinstance(node, ast.ImportFrom):
        for a in node.names:
            imported[a.asname or a.name] = node.lineno
used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
lines = src.splitlines()
for name, ln in sorted(imported.items(), key=lambda x: x[1]):
    if name not in used:
        tag = "noqa F401" if "F401" in lines[ln - 1] else ""
        print(f"unused import {name} (line {ln}) {tag}")
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        n = node.name
        if len(re.findall(r"\b" + re.escape(n) + r"\b", src)) == 1 and not re.search(r"\b" + re.escape(n) + r"\b", others):
            print(f"defined, never used anywhere: {n} (line {node.lineno})")
