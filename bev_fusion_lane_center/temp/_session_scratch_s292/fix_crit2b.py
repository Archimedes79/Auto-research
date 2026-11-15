import io
p = "experiments/E343_test-e324s/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
a = "wired 6262 against 6259, 6443 against 6441, 6415 against 6412), the maneuver counts inside; **arm k"
assert s.count(a) == 1
s = s.replace(a, "wired 6262 against 6259, 6443 against 6441, 6415 against 6412; E324 s has no stored maneuver card to compare); **arm k")
io.open(p, "w", encoding="utf-8", newline="").write(s)
