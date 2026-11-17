p = "experiments/E365_e2e-token-champion-s1/experiment.md"
s = open(p, encoding="utf-8").read()
a = "## 3. Summary\n(empty until the runs exist)"
assert a in s
s = s.replace(a, open("temp/e365/result3.txt", encoding="utf-8").read().rstrip("\n"))
open(p, "w", encoding="utf-8").write(s)
print("ok")
