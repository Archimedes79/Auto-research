p = "experiments/E375_champion-per-class-decode/experiment.md"
s = open(p, encoding="utf-8").read()
a = "(result follows)"
assert s.count(a) == 1
s = s.replace(a, open("temp/e375/result.txt", encoding="utf-8").read().rstrip("\n"))
open(p, "w", encoding="utf-8").write(s)
print("ok")
