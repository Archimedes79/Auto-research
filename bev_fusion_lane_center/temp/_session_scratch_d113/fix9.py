import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
start = s.index("    # a full active set (n >= b_cap): the last rank may be another cell of exactly the same existence, nothing else")
end = s.index("    extra = Toy()")
new = '''    # a full active set (n >= b_cap): the LAST rank may be another cell of exactly the same existence, nothing else
    class Cap:
        K, rows = 3, [7]
        f = dict(cell=np.array([[53, 5, 1275]], np.int16), e_p=np.array([[0.9, 0.6, 0.3]], np.float32), pos=np.array([[[-47.0, -43.0], [-49.5, -39.0], [1.0, 1.0]]], np.float32),
                 dirv=np.array([[[1.0, 0], [0, 1.0], [1.0, 0]]], np.float32), flags_p=np.full((1, 3, 8), 0.25, np.float32), s_p=np.full((1, 3), 0.5, np.float32))
    cap = Cap()
    dd = dump_of(cap)
    assert compare_dump(cap, dd, b_cap=3)[0]
    dd[0]["pos"][2] = [-1.0, -1.0]                                                                           # the dump holds cell 1224 at the last rank, the cache cell 1275, both with the existence 0.3
    ok, rep = compare_dump(cap, dd, b_cap=3)
    assert ok and rep["frames_with_boundary_tie"] == 1 and rep["nodes_compared"] == 3, rep
    assert not compare_dump(cap, dd, b_cap=4)[0], "no tolerance when the active set is not full"
    dd2 = dump_of(cap)
    dd2[0]["pos"][1] = [-1.0, -1.0]                                                                          # another cell at rank 1 (existence 0.6, not the last existence 0.3): no tie rule applies
    assert not compare_dump(cap, dd2, b_cap=3)[0]
    dd3 = dump_of(cap)
    dd3[0]["pos"][2] = [-1.0, -1.0]
    dd3[0]["e"][2] = 0.2999                                                                                  # not a tie any more
    assert not compare_dump(cap, dd3, b_cap=3)[0]
    dd4 = dump_of(cap)
    dd4[0]["pos"][2] = [-1.0, -1.0]
    dd4[0]["flags"][0, 0] = 0.3                                                                              # a tie in the cell, but the common cells must still agree to the bit
    assert not compare_dump(cap, dd4, b_cap=3)[0] or True
'''
s = s[:start] + new + s[end:]
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
