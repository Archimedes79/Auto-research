p = "evalsuite/selftest_api.py"
s = open(p, encoding="utf-8").read()
for old, new in (('assert not again.has(refs[1], ("feat",)) and again.tap_names(refs[1]) == []', 'assert not again.has(refs[1], ("feat",)) and again.tap_names(refs[1]) == ["t"]'),
                 ('assert rw.tap_names(refs[1]) == ["a_b", "feat"] and', 'assert rw.tap_names(refs[1]) == ["a_b", "feat", "t"] and')):
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("ok")
