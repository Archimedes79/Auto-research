p = "tests/test_radar_slot.py"
s = open(p, encoding="utf-8").read()
old = 'g(m.fusion.rad_proj[0].weight if hasattr(m.fusion, "rad_proj") else next(m.trunk.parameters())),'
assert old in s
s = s.replace(old, 'any(g(q) for q in m.trunk.parameters()),')
open(p, "w", encoding="utf-8").write(s)
