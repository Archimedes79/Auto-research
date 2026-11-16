p = "evaluation/tests/test_jprune_b.py"
s = open(p, encoding="utf-8").read()
old = 'assert 0 < s1["macs"] < s2["macs"] and s1["macs_maps"] > 0 and s1["macs_pyr"] > 0 and'
new = 'assert 0 < s1["macs"] < s2["macs"] and s1["macs_maps"] == 0 and s2["macs_maps"] > 0 and s1["macs_pyr"] > 0 and'
assert old in s
s = s.replace(old, new)
s = s.replace('"a random draw of the same counts costs the same"', '"T1 = e2, d2, e3 are not read by the maps head (its d1 / d0 inputs are in T2); a random draw of the same counts costs the same"')
open(p, "w", encoding="utf-8").write(s)
