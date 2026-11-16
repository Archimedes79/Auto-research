p = "training/grid/jprior.py"
s = open(p, encoding="utf-8").read()
def sub(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:70])
    s = s.replace(old, new)
sub('The read-outs (`Spec.readout`, names `clean`, `noise10`, `noise30`, `pm1`, `u`, `s`) and the two training specs (`Spec.train`) are in one place; S = the table of another frame: the first frame with a junction GT at or after position (index + N // 2) of the split (`partners`).',
    'The read-outs (`Spec.readout`, names `clean`, `noise10`, `noise30`, `pm1`, `u`, `s`, `sc`) and the two training specs (`Spec.train`) are in one place; S = the table of another frame: the first frame with a junction GT at or after position (index + N // 2) of the split (`partners`);\nSC = the frame\'s OWN arm centres and axes with the counts of that partner frame (arm i of the frame gets the count of arm i modulo the number of arms of the partner, in the row order of the clean tables; a partner without arms: unknown): the geometry is right and only the counts are another frame\'s (S puts the partner\'s arms in the partner\'s ego frame, so most keypoints see no arm).')
sub('MODES = ("clean", "noise", "pm1", "U", "S", "mix")', 'MODES = ("clean", "noise", "pm1", "U", "S", "SC", "mix")')
sub('U (every count unknown, geometry kept) | S (the table of ANOTHER frame) | mix', 'U (every count unknown, geometry kept) | S (the table of ANOTHER frame) | SC (own arms, the counts of another frame) | mix')
sub('assert name in ("clean", "pm1", "u", "s"), f"read-out {name!r}: clean, noise<percent>, pm1, u or s"\n        return Spec.readout({"u": "U", "s": "S"}.get(name, name))',
    'assert name in ("clean", "pm1", "u", "s", "sc"), f"read-out {name!r}: clean, noise<percent>, pm1, u, s or sc"\n        return Spec.readout({"u": "U", "s": "S", "sc": "SC"}.get(name, name))')
sub('"""the read-out of a name: clean, noise<percent>, pm1, u, s"""', '"""the read-out of a name: clean, noise<percent>, pm1, u, s, sc"""')
sub('a comma list of clean, noise10 .., pm1, u, s (case free)', 'a comma list of clean, noise10 .., pm1, u, s, sc (case free)')
sub('''    if spec.mode == "S":
        jg = partner(token)
''', '''    if spec.mode == "SC":                                                                      # own arms, the counts of the partner frame (i modulo the partner's number of arms, unknown without arms)
        own, par = table(jg, token, Spec("clean")), table(partner(token), token, Spec("clean"))
        k = len(par["n_in"])
        for key in ("n_in", "n_out"):
            own[key] = par[key][np.arange(len(own[key])) % k] if k else np.full(len(own[key]), UNK, np.int64)
        return own
    if spec.mode == "S":
        jg = partner(token)
''')
sub('''    assert [s.name for s in readouts("clean,Noise,U,s,pm1", 0.1)] == ["clean", "noise10", "u", "s", "pm1"]''',
    '''    assert [s.name for s in readouts("clean,Noise,U,s,pm1,SC", 0.1)] == ["clean", "noise10", "u", "s", "pm1", "sc"] and Spec.parse("sc").mode == "SC"''')
sub('''    order = list(partners("b", ["a", "b", "c", "d", "e"]))''', '''    p2 = dict(arms=[fake(0, (4, 5)), fake(0, (6, 7))])                                                # SC: the frame's own arms (rows 4 here), the counts of the partner's arms i modulo 2
    own = table(jg, "tok", Spec("clean"))
    sc = table(jg, "tok", Spec.readout("SC"), partner=lambda tk: p2)
    assert sc["n_in"].tolist() == [4, 6, 4, 6] and sc["n_out"].tolist() == [5, 7, 5, 7] and np.array_equal(sc["c"], own["c"]) and np.array_equal(sc["t"], own["t"]) and own["n_in"].tolist() == [2, 1, 2, 4], (sc, own)
    assert all(same_table(table(jg, "tok", Spec.readout("SC"), partner=lambda tk: p2), sc) for _ in range(2)) and not same_table(sc, own)
    for none in (None, dict(arms=[])):
        nc = table(jg, "tok", Spec.readout("SC"), partner=lambda tk: none)
        assert (nc["n_in"] == UNK).all() and (nc["n_out"] == UNK).all() and np.array_equal(nc["c"], own["c"]), "a partner without arms: every count unknown, the geometry kept"
    assert len(table(None, "tok", Spec.readout("SC"), partner=lambda tk: p2)["n_in"]) == 0
    order = list(partners("b", ["a", "b", "c", "d", "e"]))''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
