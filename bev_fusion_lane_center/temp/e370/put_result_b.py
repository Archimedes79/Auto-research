p = "experiments/E370_e365-rule-distilled-init/experiment.md"
s = open(p, encoding="utf-8").read()
a = "(stage B follows)"
assert s.count(a) == 1
s = s.replace(a, open("temp/e370/result_b.txt", encoding="utf-8").read().rstrip("\n"))
open(p, "w", encoding="utf-8").write(s)
print("ok")
