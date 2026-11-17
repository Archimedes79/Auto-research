p = "experiments/E373_b-bal5-confirmation-fresh-head-seeds/experiment.md"
s = open(p, encoding="utf-8").read()
a = "## 3. Summary\n(empty until the runs exist)"
assert a in s
s = s.replace(a, open("temp/e373/result.txt", encoding="utf-8").read().rstrip("\n"))
open(p, "w", encoding="utf-8").write(s)
print("ok")
