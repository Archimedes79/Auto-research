p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
old = 'for run, label in (("e308_t_s0", "E308 seed 0"), ("e305_e_s0", "E305 seed 0")):'
assert old in s
s = s.replace(old, 'for run, label in (("e308_t_s0", "E308 seed 0"), ("e305_e_s0", "E305 seed 0"), ("e312_p_s0", "E312 seed 0"), ("e312_p_s1", "E312 seed 1")):', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
