import ast, re, sys
mod = open("evaluation/jtest_eval.py", encoding="utf-8").read()
tst = open("evaluation/tests/test_jtest_eval.py", encoding="utf-8").read()
tree = ast.parse(mod)
names = []
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        names.append(n.name)
    elif isinstance(n, ast.Assign):
        names += [t.id for t in n.targets if isinstance(t, ast.Name)]
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        names += [(a.asname or a.name).split(".")[0] for a in n.names]
for nm in names:
    c = len(re.findall(r"\b" + re.escape(nm) + r"\b", mod)) + len(re.findall(r"\bJ\." + re.escape(nm) + r"\b", tst))
    if c <= 1:
        print("used once:", nm)
# local unused imports/variables inside functions: names imported in a function and never used afterwards
for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
    imp = [(a.asname or a.name).split(".")[0] for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
    src = ast.get_source_segment(mod, fn)
    for i in imp:
        if len(re.findall(r"\b" + re.escape(i) + r"\b", src)) <= 1:
            print("unused local import:", fn.name, i)
