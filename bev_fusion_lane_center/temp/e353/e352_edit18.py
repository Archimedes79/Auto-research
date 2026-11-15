p = "evaluation/tests/test_jprune_b.py"
s = open(p, encoding="utf-8").read()
old = '''    assert r["nodes_only_in_ref"] == 1 and r["nodes_only_in_new"] == 1 and r["links"]["p"] >= 0.1 - 1e-9, r
'''
new = '''    assert r["nodes_only_in_ref"] == 1 and r["nodes_only_in_new"] == 1 and r["links"]["p"] >= 0.1 - 1e-9, r
    gone = copy.deepcopy(ref)
    for k in ("link_src", "link_dst", "link_p", "link_long", "link_a", "link_b"):
        gone[0][k] = gone[0][k][:-1]                                                                     # the last link of frame 0 is missing in the new dump
    r = PB.compare_dumps(ref, gone)
    assert r["links_only_in_ref"] == 1 and r["links_only_in_new"] == 0 and r["nodes_only_in_ref"] == 0, r
'''
assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
