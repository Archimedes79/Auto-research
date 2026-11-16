p = "temp/e291/mutate.py"
s = open(p, encoding="utf-8").read()
old = 'if want and mid not in want and not (want == ["cheap"] and test == JS):'
new = 'if want and mid not in want and not ("cheap" in want and test == JS):'
assert old in s
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("ok")
