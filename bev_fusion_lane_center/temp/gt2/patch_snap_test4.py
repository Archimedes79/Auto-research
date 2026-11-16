p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()
old = "    assert len(rb[\"pts\"]) == 3 - 1 + 0 or len(rb[\"pts\"]) == 3, len(rb[\"pts\"])            # the unwired far pair adds no curve (the covered-link rule is unchanged)"
assert old in s
s = s.replace(old, "    assert len(rb[\"pts\"]) == 3 and nb == 1 and wb == 0, (len(rb[\"pts\"]), nb, wb)                    # the unwired far pair adds no curve (the delivered long link stays)", 1)
open(p, "w", encoding="utf-8").write(s)
