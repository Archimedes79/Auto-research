p = "evaluation/README.md"
s = open(p, encoding="utf-8").read()
old = "`--check` proves it against the card (with every detection counted, tp / fn / fp per class are the card's numbers: identical on `t22_B_s0`)"
new = ("`--check` proves it against the card (with every detection counted, tp / fn / fp per class are the card's numbers: identical on `t22_B_s0`; at 2.0 m only); `--class X` filters the rows but classifies every class, `--store NAME` picks one of several stores, `--n all`; the table counts the GT boxes with a lidar / radar point (the `det` numbers) while the pictures use the D38 `detf` rule, so an `isolated` ghost can be a real box without a point; the redraw command carries `--history none` (the table is a cold evaluation)")
assert s.count(old) == 1
s = s.replace(old, new)
old2 = "are tested by `bev_views.selftest()` inside `bev_selftest.py` |"
assert s.count(old2) == 1
s = s.replace(old2, "are tested by `bev_views.selftest()` inside `bev_selftest.py` (`--only \"bev_views helpers\"` runs just that, in `tests/run_gates.sh`) |")
open(p, "w", encoding="utf-8").write(s)
