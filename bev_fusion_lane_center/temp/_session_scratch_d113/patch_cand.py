import os
p = os.path.join(os.environ["BFLC_ROOT"], "experiments", "E379_next_candidates.md")
s = open(p, encoding="utf-8").read()
reps = [
 ("(all four pairs, VAL and TRAIN, intervals above 0 on 7 of 8)", "(all eight paired comparisons, VAL and TRAIN, two seeds, intervals above 0 on all eight)"),
 ("(mutual top-1: 0.189 / 0.200;", "(mutual top-1: 0.189 / 0.203;"),
 ("**78 % of the non-true delivered long links share their DESTINATION token with another delivered link** (41 - 42 % share their source)", "**76 - 80 % of the non-true delivered long links share their DESTINATION token with another delivered link** (41 - 42 % share their source)"),
 ("The gain is as large on TRAIN as on VAL: the teachers' soft targets carry information on the frames a student trains on.", "The gain is as large on TRAIN as on VAL: the teachers' soft targets carry information on the frames a student trains on. (Seed 1 on VAL and on TRAIN gives the same differences to four decimals, +0.0065 / +0.0097 / +0.0032 with a coinciding E - B interval; checked at six decimals: VAL 0.006549 / 0.009714, TRAIN 0.006515 / 0.009739, two different computations, `temp/e379/ens_digits.py`.)"),
]
for a, b in reps:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
