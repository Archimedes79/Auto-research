import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = '''    assert verdict(arms=dict(a=arm("qa", "b"), c=arm("ca", "cb")), rule=errguard) == "CUT", "a card that does not declare err is undeclared, not 'higher'"
    for arms_, why in ((dict(a=arm("xa", "xb"), c=arm("ca", "cb")), "disagree"), (dict(a=arm("a", "b"), c=arm("nca", "ncb")), "other way")):'''
new = '''    # a card that holds err without declaring it is undeclared (read by the fallback), not "higher": next to cards declaring it lower that is refused
    for arms_, why in ((dict(a=arm("xa", "xb"), c=arm("ca", "cb")), "disagree"), (dict(a=arm("a", "b"), c=arm("nca", "ncb")), "other way"),
                       (dict(a=arm("qa", "b"), c=arm("ca", "cb")), "other way")):'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
print("ok")
