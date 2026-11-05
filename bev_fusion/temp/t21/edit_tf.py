import re
p = "training/temporal_fusion.py"
s = open(p, encoding="utf-8").read()
a = '        pose = state["pose"] if step is None else advance_pose(state["pose"], *step)\n'
b = '        pose = advance_chain(state["pose"], step)\n'
assert s.count(a) == 1
s = s.replace(a, b)
a = '            dt = _col(step[0], cur).abs().expand(B) if step is not None else cur.new_zeros(B)       # scalar or (B,)\n'
b = '            dt = sum((_col(s[0], cur).abs() for s in as_chain(step)), cur.new_zeros(B))              # scalar or (B,) per step, summed over a chain\n'
assert s.count(a) == 1
s = s.replace(a, b)
a = '''        [(dt, v, omega), ...] to the current time (as splat_window); state: the dict of the previous call or None; step:
        (dt, v, omega) from the previous current frame to this one (moves the memory pose); obs_mass (B, 1, X, Y): where
        the sensor observed.'''
b = '''        [(dt, v, omega), ...] to the current time (as splat_window); state: the dict of the previous call or None; step:
        (dt, v, omega) from the previous current frame to this one (moves the memory pose), or the chain of such triples
        (advanced one after the other, the decay time is their sum); obs_mass (B, 1, X, Y): where the sensor observed.'''
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="").write(s)
