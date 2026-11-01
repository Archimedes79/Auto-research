import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = "!/CLAUDE.md\n"
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, old + "!/README.md\n"))
print("ok")
