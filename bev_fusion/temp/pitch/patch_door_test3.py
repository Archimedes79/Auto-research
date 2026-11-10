p = "tests/test_evaluate_door.py"
s = open(p, encoding="utf-8").read()
old = '''os.path.isdir(os.path.join(ne, "suite", "quant", f"eval{tag}_n{n}"))'''
new = '''os.path.isdir(os.path.join(ROOT, jobs[named][0], "suite", "quant", f"eval{tag}_n{n}"))'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
