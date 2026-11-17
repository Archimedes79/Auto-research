p = "temp/e365/patch_s.py"
s = open(p, encoding="utf-8").read()
a = s.index("    assert _links_set(rec[\"rec\"]) == set() or True")
b = s.index("    must_fail(E.decode_variant, f, valid, 0.04)")
new = '''    got = lambda **k: _links_set(E.decode_variant(f, valid, **k)["rec"])  # noqa: E731
    assert got() == {(0, 1)} and got(link_thr=0.2) == {(0, 1)}, "token 2 is no node at p_none 0.55: no link to or from it"
    assert got(keep_thr=0.6) == {(0, 1), (0, 2)} and got(link_thr=0.2, keep_thr=0.6) == {(0, 1), (0, 2)}, "a higher keep threshold adds the node"
    assert got(keep_thr=0.6, none_gate=False) == {(0, 1), (0, 2), (2, 3)}, "the none gate off adds 2 -> 3 (0.35 below p(none) 0.65)"
    assert got(link_thr=0.2, keep_thr=0.6, none_gate=False) == {(0, 1), (0, 2), (1, 2), (2, 3)} and got(link_thr=0.09, keep_thr=0.6, none_gate=False) == {(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)}
    for kt in (0.5, 0.6):
        for gate in (True, False):
            chain = [got(link_thr=t, keep_thr=kt, none_gate=gate) for t in (0.3, 0.2, 0.09, 0.05)]
            assert all(x <= y for x, y in zip(chain, chain[1:])), "a lower link threshold only adds links"
        assert got(link_thr=0.2, keep_thr=kt) <= got(link_thr=0.2, keep_thr=kt, none_gate=False), "no none gate only adds links"
'''
s = s[:a] + new + s[b:]
a = s.index("    assert np.allclose(E.decode_variant(old, valid, link_thr=0.1")
b = s.index("    # a real head (random weights)")
new2 = '''    ex = E.decode_variant(old, valid, link_thr=0.09, keep_thr=0.6, none_gate=False)["rec"]
    added = [k for k, l in enumerate(zip(ex["src"].tolist(), ex["dst"].tolist())) if l not in _links_set(f["rec"])]
    assert len(added) == 4 and np.allclose(ex["a"][added], 1 / 3) and np.allclose(ex["a"][[k for k in range(len(ex["a"])) if k not in added]], 0.25), "links that are not the record's get 1/3"
'''
s = s[:a] + new2 + s[b:]
open(p, "w", encoding="utf-8").write(s)
