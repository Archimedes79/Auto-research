p = "tests/test_evaluate_door.py"
s = open(p, encoding="utf-8").read()
old = ''' and steps_of("--time", "--split", "xl.val")'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, ""))
print("ok")
