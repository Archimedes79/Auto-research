import io
p = "training/data/lane_ptok_data.py"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep("import lane_tokcache as LC  # noqa: E402\n", "import lane_tokcache as LC  # noqa: E402\nimport qtargets as QT  # noqa: E402\n")
rep('''        y, yl, ys = densify(np.array(f["ypair"][i]), np.array(f["ypair_long"][i]), np.array(f["ypair_scale"][i]), self.K, keep if self.drop > 0 else None)''',
    '''        y, _, ys = densify(np.array(f["ypair"][i]), np.array(f["ypair_long"][i]), np.array(f["ypair_scale"][i]), self.K, keep if self.drop > 0 else None)
        pos = np.array(f["pos"][i], np.float64)
        yl = (y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT)                  # the class of a pair = the chord of its TOKENS (> 8 m); the class of the GT link is in the cache (`ypair_long`)''')
rep('''["ylong"] (B, K, K) bool, ["yscale"] (B, K, K, 2) NaN elsewhere''', '''["ylong"] (B, K, K) bool = y and the chord |pos_j - pos_i| of the TOKENS above 8 m (not the chord of the GT link), ["yscale"] (B, K, K, 2) NaN outside the long GT links with a fit''')
# toy cache test: positions with controlled chords, GT class opposite to the token class
rep('''        arr["ypair"][:, :3] = [[0, 2], [0, 3], [4, 1]]
        arr["ypair_long"][:, 1] = True''', '''        arr["ypair"][:, :3] = [[0, 2], [0, 3], [4, 1]]
        arr["pos"][:] = 0.0
        arr["pos"][:, 2], arr["pos"][:, 3], arr["pos"][:, 1] = [0.0, 5.0], [0.0, 20.0], [0.0, 9.0]              # token chords: (0, 2) 5 m, (0, 3) 20 m, (4, 1) 9 m
        arr["ypair_long"][:, 0] = True                                                                            # the GT class is the opposite of the token class for (0, 2) and (0, 3)''')
rep('''int(b["y"][0].sum()) == 3 and b["y"][0, 4, 1] == 1 and bool(b["ylong"][0, 0, 3]) and not bool(b["valid"][1, 31])''', '''int(b["y"][0].sum()) == 3 and b["y"][0, 4, 1] == 1 and not bool(b["valid"][1, 31])''')
rep('''        assert sorted(b["tokens_np"][0])''', '''        assert bool(b["ylong"][0, 0, 3]) and bool(b["ylong"][0, 4, 1]) and not bool(b["ylong"][0, 0, 2]) and int(b["ylong"][0].sum()) == 2, "ylong = the chord of the tokens"
        assert not b["ylong"][0, 3, 0] and np.allclose(b["yscale"][0, 0, 3], [0.3, 0.4]) and np.isnan(b["yscale"][0, 0, 2]).all()
        assert sorted(b["tokens_np"][0])''')
io.open(p, "w", encoding="utf-8").write(s)

p = "training/grid/test_lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''    (PT, "a GT field in the read-out input",'''
b = '''    (PT, "ylong is the class of the GT link", "yl = (y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT)", "yl = (y > 0) & ylong_gt"),
    (PT, "ylong from the wrong token distance", "> QT.SHORT)", "> 2 * QT.SHORT)"),
    (PT, "a GT field in the read-out input",'''
assert s.count(a) == 1
s = s.replace(a, b)
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
