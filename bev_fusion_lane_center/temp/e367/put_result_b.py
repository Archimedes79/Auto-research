p = "experiments/E367_e365-capacity-balance-schedule-arms/experiment.md"
s = open(p, encoding="utf-8").read()
a = "\n\n(results follow)"
assert s.count(a) == 1
s = s.replace(a, "\n\n" + open("temp/e367/result_b.txt", encoding="utf-8").read().rstrip("\n") + "\n\n(a_wide and c_long follow)")
open(p, "w", encoding="utf-8").write(s)
print("ok")
