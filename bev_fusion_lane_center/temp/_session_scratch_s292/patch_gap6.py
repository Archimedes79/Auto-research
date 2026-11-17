import io
p = "experiments/E2E_gap_table.md"
s = io.open(p, encoding="utf-8", newline="").read()
a = "vertices per 100 m 18.99 / 20.83 against 18.94 / 19.84 (+0.05 / +0.99)"
assert s.count(a) == 1
s = s.replace(a, "vert100 (not the DP basis of the guard) 18.99 / 20.83 against 18.94 / 19.84 (+0.05 / +0.99); on the guard's basis vert100_dp the layer is LOWER: 23.13 / 23.90 against c2 plain 29.23 / 31.44 (-6.1 / -7.5; label corrected 2026-11-16 on the E360 critic's remark)")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
