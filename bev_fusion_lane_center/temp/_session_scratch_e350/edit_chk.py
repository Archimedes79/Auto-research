p = "evaluation/jwire_check.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
# 1. the two run / seed conditions are separate
rep('''    for what, c in (("stored", stored), ("twin", twin)):
        if c["run"] != run or c["seed"] != seed:
            diff.append(f"the {what} card is of run {c['run']} seed {c['seed']}, not {run} seed {seed}")''', '''    for what, c in (("stored", stored), ("twin", twin)):
        if c["run"] != run:
            diff.append(f"the {what} card is of run {c['run']}, not {run}")
        if c["seed"] != seed:
            diff.append(f"the {what} card is of seed {c['seed']}, not {seed}")''')
# 2. live comparison: one field alone
rep('''    L = live()
    L[0]["attrs"] = L[0]["attrs"] + 0.1
    ok, _, d = compare_records(L, ref)
    assert not ok and d == {0: ["attrs"]}, d                                                          # another field''', '''    for field, plant in (("attrs", lambda x: x + 0.1), ("ends", lambda x: x + 0.1), ("kp_p", lambda x: x * 0.5), ("gts", lambda x: x + 0.1), ("pairs", lambda x: x + 0.1)):
        L = live()
        L[0][field] = plant(L[0][field])
        ok, _, d = compare_records(L, ref)
        assert not ok and d == {0: [field]}, (field, d)                                                   # each field alone refuses''')
# 3. twin: more planted cases
rep('''            put(card(), twin, counts(), counts(1.5))
            assert twin_check("rx", 0, "twin")[1] == ["counts head"]                                   # one per-frame count''', '''            put(card(), twin, counts(), counts(1.5))
            assert twin_check("rx", 0, "twin")[1] == ["counts head"]                                   # one per-frame count
            put(card(), twin, counts(), dict(counts(), base=[dict(tp=1.0, n=3)] * 3))
            assert twin_check("rx", 0, "twin")[1] == ["counts base"]
            put(card(), twin, counts(), dict(counts(), scenes=[1, 2, 4]))
            assert twin_check("rx", 0, "twin")[1] == ["counts scenes"]
            put(card(), dict(twin, tag="other"), counts(), counts())
            assert "not the twin's" in twin_check("rx", 0, "twin")[1][0]                               # the tag asked for is the card's
            put(card(wire="head"), twin, counts(), counts())
            assert twin_check("rx", 0, "twin")[1] == ["the stored card is not the rule's (wire 'head')"]   # the stored card is the rule's (`wire` is ignored by the field comparison)
            for over, what in ((dict(run="ry"), "run"), (dict(seed=1), "seed")):                       # both cards of another run / seed: equal to each other, not the ones asked for
                put(card(**over), dict(twin, **over), counts(), counts())
                ok, d = twin_check("rx", 0, "twin")
                assert not ok and len(d) == 2 and all(f"card is of {what}" in x for x in d), (what, d)''')
open(p, "w", encoding="utf-8").write(s)
