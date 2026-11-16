"""scratch: imported names that are not used again, and module-level functions / classes that nothing uses"""
import ast
import re
import sys

for p in sys.argv[1:]:
    src = open(p, encoding="utf-8").read()
    tree = ast.parse(src)
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [(a.asname or a.name).split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            names += [a.asname or a.name for a in n.names]
    for nm in sorted(set(names)):
        if len(re.findall(rf"\b{re.escape(nm)}\b", src)) < 2:
            print(p, "unused import", nm)
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            if len(re.findall(rf"\b{re.escape(n.name)}\b", src)) < 2:
                print(p, "defined, never used in the file:", n.name)
