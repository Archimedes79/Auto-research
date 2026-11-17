import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''def compare_dump(cache, dump, e_min=0.2, prefix=False):''', '''B_CAP = 512                                                                                        # B's own active set = the top 512 cells (qlinks.LinkHead K), so a stored dump holds at most 512 nodes


def compare_dump(cache, dump, e_min=0.2, prefix=False, b_cap=B_CAP):''')
rep('''    compared with the cache tokens cell by cell: same set of cells, and e, pos, dirv, flags, s equal bit for bit (the dump holds float64 copies of float32 numbers); the cache must hold no token at or above e_min behind them"""''',
    '''    compared with the cache tokens cell by cell: same set of cells, and e, pos, dirv, flags, s equal bit for bit (the dump holds float64 copies of float32 numbers); the cache must hold no token at or above e_min behind them.
    One exception, counted and reported: when the stored active set is FULL (n >= b_cap, B's own top-512 cut) the last rank may be a tie of two cells with exactly the same existence that `torch.topk` of the probabilities and the
    stable sort of the logits order differently; such a cell, and only at the existence of the last stored node, may differ"""''')
rep('''    worst, bad, cut, n_nodes, order_diff = dict(e=0.0, pos=0.0, dirv=0.0, flags=0.0, s=0.0), [], 0, 0, 0''', '''    worst, bad, cut, n_nodes, order_diff, ties = dict(e=0.0, pos=0.0, dirv=0.0, flags=0.0, s=0.0), [], 0, 0, 0, 0''')
rep('''        if sorted(cell_d.tolist()) != sorted(cell_c.tolist()):
            bad.append((i, "cells differ"))
            continue
        if not np.array_equal(''', '''        only_d, only_c = set(cell_d.tolist()) - set(cell_c.tolist()), set(cell_c.tolist()) - set(cell_d.tolist())
        if only_d or only_c:
            ed = {int(c_): float(e_) for c_, e_ in zip(cell_d, d["e"][:nc])}
            ec = {int(c_): float(e_) for c_, e_ in zip(cell_c, f["e_p"][i, :nc])}
            last = float(d["e"][nc - 1])
            if n >= b_cap and nc == n and all(ed[c_] == last for c_ in only_d) and all(ec[c_] == last for c_ in only_c):
                ties += 1
            else:
                bad.append((i, "cells differ"))
                continue
        if not np.array_equal(''')
rep('''        pc = {int(c): k for k, c in enumerate(cell_c)}
        for k in range(nc):
            j = pc[int(cell_d[k])]''', '''        pc = {int(c): k for k, c in enumerate(cell_c)}
        for k in range(nc):
            if int(cell_d[k]) not in pc:                                                                        # the tie cell of the dump
                continue
            j = pc[int(cell_d[k])]''')
rep('''frames_with_other_order=order_diff, worst=worst''', '''frames_with_other_order=order_diff, frames_with_boundary_tie=ties, worst=worst''')
# selftest: boundary tie
rep('''    extra = Toy()                                                                                            # a cache token at e_p >= 0.2 behind the stored nodes''', '''    # a full active set (n >= b_cap): the last rank may be another cell of exactly the same existence, nothing else
    cap = Toy()
    cap.f = dict(Toy.f, cell=np.array([[53, 5, 1300, 1224]], np.int16), e_p=np.array([[0.9, 0.6, 0.3, 0.3]], np.float32), pos=np.array([[[-47.0, -43.0], [-49.5, -39.0], [3.0, 3.0], [-1.0, -1.0]]], np.float32))
    ok, rep = compare_dump(cap, dump_of(cap), b_cap=3)                                                       # the dump holds tokens 0 - 2, the cache has the other cell (1300) at the tied rank 2
    cap2 = Toy()
    cap2.f = dict(cap.f)
    dd = dump_of(cap2)
    dd[0]["pos"][2] = [-1.0, -1.0]                                                                           # the dump has cell 1224 at rank 2, the cache cell 1300 at the same existence
    ok, rep = compare_dump(cap, dd, b_cap=3)
    assert ok and rep["frames_with_boundary_tie"] == 1, rep
    assert not compare_dump(cap, dd, b_cap=4)[0], "no tolerance when the active set is not full"
    dd2 = dump_of(cap2)
    dd2[0]["pos"][2] = [-1.0, -1.0]
    dd2[0]["e"][2] = 0.2999                                                                                  # not a tie any more
    assert not compare_dump(cap, dd2, b_cap=3)[0]
    cap3 = Toy()
    cap3.f = dict(cap.f, cell=np.array([[5, 53, 1300, 1224]], np.int16))                                      # the cell that differs is NOT at the last rank: the sets differ at rank 0
    dd3 = dump_of(cap3)
    dd3[0]["pos"][0] = [-1.0, -1.0]
    assert not compare_dump(cap3, dd3, b_cap=3)[0]
    extra = Toy()                                                                                            # a cache token at e_p >= 0.2 behind the stored nodes''')
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
