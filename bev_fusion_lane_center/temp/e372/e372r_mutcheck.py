"""throw-away: every mutant's text is found exactly once in its function (no run)"""
import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "e372r_jtoken_eval.py")).read()
tree = ast.parse(src)
funcs = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
mut = open(os.path.join(HERE, "e372r_mutants.py")).read()
for m in re.finditer(r'\("([^"]+)", E\.(\w+), (".*?"), (".*?")\),\n', mut):
    old = ast.literal_eval(m.group(3))
    print(funcs[m.group(2)].count(old), m.group(1))
