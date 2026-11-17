import importlib.util, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location("m", os.path.join(ROOT, "temp/e353/e352_mutants.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
src = open(os.path.join(ROOT, m.F), encoding="utf-8").read()
for name, old, new, tests in m.MUTANTS:
    c = src.count(old)
    if c != 1:
        print("PROBLEM", name, c)
print(len(m.MUTANTS), "mutants checked")
