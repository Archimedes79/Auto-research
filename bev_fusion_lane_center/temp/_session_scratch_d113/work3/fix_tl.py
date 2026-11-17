import os
p = os.path.join(os.environ["BFLC_ROOT"], "training/grid/test_lane_map.py")
s = open(p, encoding="utf-8").read()
old = '''    ("a duplicate is near any node", "dup=weak & near.any(1) & ~(near & miss[None]).any(1)", "dup=weak & near.any(1)", "potential"),'''
new = '''    ("a duplicate is near any node", "dup=weak & near.any(1) & ~nov,", "dup=weak & near.any(1),", "potential"),
    ("the delivered threshold is exclusive", "deliv = valid & (ep >= 0.5)", "deliv = valid & (ep > 0.5)"),
    ("the heading cosine is exclusive", "(dirv @ gdir.T >= COS)", "(dirv @ gdir.T > COS)"),'''
assert old in s
s = s.replace(old, new)
old2 = '''    ("the near radius is exclusive", "(np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD)", "(np.linalg.norm(pos[:, None] - gpos[None], axis=2) < RAD)", "potential"),'''
new2 = '''    ("the near radius is exclusive", "(np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD)", "(np.linalg.norm(pos[:, None] - gpos[None], axis=2) < RAD)"),'''
assert old2 in s
s = s.replace(old2, new2)
open(p, "w", encoding="utf-8").write(s)
print("ok")
