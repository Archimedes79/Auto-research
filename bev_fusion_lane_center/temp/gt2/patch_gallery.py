p = "temp/jr/fail_gallery.py"
s = open(p, encoding="utf-8").read()
a = '''    e2 = dict(e)
    e2["logit"] = e["logit"]
    _, hp0, _, cg0 = cover(base_raw, ref)[0:4] if False else (None,) + cover(base_raw, ref)[1:]
    P, hp, G, cg = cover(raw, ref)
    deliv, missed, gtp, cand = pairs_of(dict(e, logit=e["logit"][:, :], kp_pos=e["kp_pos"], kp_dir=e["kp_dir"]), jg)
'''
b = '''    cg0 = cover(base_raw, ref)[3]
    P, hp, G, cg = cover(raw, ref)
    deliv, missed, gtp, cand = pairs_of(e, jg)
'''
assert a in s
s = s.replace(a, b)
s = s.replace('    K = e["logit"].shape[0] // 1\n', '    K = e["logit"].shape[0]\n')
open(p, "w", encoding="utf-8").write(s)
