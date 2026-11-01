import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = '''    assert "vert100" in DC.lower_names(T), "no key, the lane problem: the legacy list until its key is written"
    json.dump(pj, open(f"{T}/project.json", "w"))'''
new = old + '''
    # the evaluator declares directions in its cards ("lower_is_better", card contract): decide.py reads them there, not from names
    def fcard(q, err, lib):
        return dict(full, name="x", metrics=dict(Q=q, err=err), **({"lower_is_better": lib} if lib is not None else {}))
    json.dump(dict(a=fcard(0.76, 0.20, ["err"]), b=fcard(0.75, 0.21, ["err"]), ca=fcard(0.73, 0.10, ["err"]), cb=fcard(0.72, 0.11, ["err"]),
                   na=fcard(0.76, 0.20, None), nb=fcard(0.75, 0.21, None), nca=fcard(0.73, 0.10, None), ncb=fcard(0.72, 0.11, None),
                   xa=fcard(0.76, 0.20, []), xb=fcard(0.75, 0.21, [])), open(f"{T}/dir.json", "w"))
    arm = lambda *ks: dict(cards=[f"dir.json#{k}" for k in ks])  # noqa: E731
    errguard = [dict(metric="Q", min_effect=0.01), dict(metric="err", max_loss=0.02)]
    assert verdict(arms=dict(a=arm("a", "b"), c=arm("ca", "cb")), rule=errguard) == "CUT", "err declared lower by the cards: +0.10 breaks the guard"
    assert verdict(arms=dict(a=arm("na", "nb"), c=arm("nca", "ncb")), rule=errguard) == "KEEP", "no declaration, err not a known name: read higher"
    try:
        verdict(arms=dict(a=arm("xa", "xb"), c=arm("ca", "cb")), rule=errguard)
        raise AssertionError("cards that disagree about a direction must be refused")
    except ValueError as e:
        assert "disagree" in str(e)
    contra = dict(base, arms=dict(a=arm("a", "b"), c=arm("ca", "cb")), metric="err", direction="higher", rule=[dict(metric="err", min_effect=0.01)])
    assert any("contradicts the cards" in b for b in X.check(contra, f"{T}/experiments/E1_x/experiment.json")), X.check(contra, f"{T}/experiments/E1_x/experiment.json")
    assert not any("contradicts" in b for b in X.check(dict(contra, direction="lower"), f"{T}/experiments/E1_x/experiment.json"))
    import card as CARD
    assert any("lower_is_better" in b for b in CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), lower_is_better="err")))
    assert any("lower_is_better" in b for b in CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), lower_is_better=[3])))
    assert CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), lower_is_better=["err"])) == []'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
