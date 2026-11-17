src = open("temp/_session_scratch_s292/rank/patch_jgt.py", encoding="utf-8").read()
# idempotent: do nothing when the loop oracle is already in the selftest
a = "s = rd(p)\n"
assert src.count(a) == 1
src = src.replace(a, 's = rd(p)\nif "def groups_loop(E):" in s:\n    print("already installed")\n    raise SystemExit(0)\n', 1)
extra = '''    # the edge semantics (the critic's mutants M2, M3, M9, M10 survived the first selftest): the boundary is <=, a zero-heading row is its own group, float32 headings with a self-dot above 1 still link
    cases = [([[0, 0, 1, 0], [6, 0, 1, 0]], [0, 0]), ([[0, 0, 1, 0], [0, 6, 1, 0]], [0, 0]), ([[0, 0, 1, 0], [3, 0, 0, 0], [6, 0, 1, 0]], [0, 1, 0])]
    for rows, want in cases:
        Ec = np.array(rows, float)
        assert groups_loop(Ec).tolist() == want == _groups(Ec).tolist(), (rows, want)
    h = (0.9431353807449341, 0.33240893483161926)
    E32 = np.array([[0, 0, *h], [2, 0, *h], [4, 0, *h]], np.float32)
    assert groups_loop(E32).tolist() == [0, 0, 0] == _groups(E32).tolist()
    # a shuffled path of 32 ends 5.9 m apart is one group (the squaring count: 5 rounds are needed)
    perm = rng.permutation(32)
    Ep = np.array([[5.9 * i, 0.0, 1.0, 0.0] for i in perm])
    assert groups_loop(Ep).tolist() == _groups(Ep).tolist() == [0] * 32
    print("jgt selftest ok")'''
needle = '    print("jgt selftest ok")'
assert src.count(needle) == 2, src.count(needle)          # once as the anchor in jgt.py, once at the end of the inserted selftest: the second one is extended
i = src.rindex(needle)
src = src[:i] + extra + src[i + len(needle):]
old_write = 'open(p, "w", encoding="utf-8", newline="\\n").write(s)'
assert src.count(old_write) == 1, src.count(old_write)
new_write = 'import os\ntmp = os.path.join(os.path.dirname(p), ".jgt.py.new")\nwith open(tmp, "w", encoding="utf-8", newline="\\n") as f:\n    f.write(s)\nos.replace(tmp, p)'
src = src.replace(old_write, new_write)
open("temp/_session_scratch_s292/rank/install_jgt.py", "w", encoding="utf-8").write(src)
print("written")
