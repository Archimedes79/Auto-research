import io
p = "training/grid/test_lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''    (PT, "ylong is the class of the GT link", "yl = (y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT)", "yl = (y > 0) & ylong_gt"),'''
b = '''    (PT, "ylong marks every pair", "yl = (y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT)", "yl = (y > 0)"),
    (PT, "ylong from the transposed chord matrix", "np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT)", "np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1).T > QT.SHORT)"),'''
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
