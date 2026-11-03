import re
def edit(path, pairs):
    t = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert t.count(a) == 1, (path, a[:60], t.count(a))
        t = t.replace(a, b)
    open(path, "w", encoding="utf-8", newline="\n").write(t)

edit("evaluation/bev_curves.py", [(
    'assert np.isnan(one["ap"]) and np.isnan(one["roc_auc"]) and abs(one["ece"] - (abs(1 - 0.2) + abs(1 - 0.9)) / 2 * 0 - one["ece"]) < 1e-9 and one["brier"] > 0',
    'assert np.isnan(one["ap"]) and np.isnan(one["roc_auc"]) and np.isfinite(one["ece"]) and np.isfinite(one["brier"])')])

edit("evaluation/bev_suite.py", [
    # better_of: directionless numbers are left out of the card
    ('''        if f is not None and m in by and last != "n" and not last.startswith("n_"):
            out[f] = "lower" if by[m].direction(rest) == "low" else "higher"''',
     '''        if f is not None and m in by and last != "n" and not last.startswith("n_"):
            d = by[m].direction(rest)
            if d != "none":                                              # prevalence, thr_f1: not quality numbers
                out[f] = "lower" if d == "low" else "higher"'''),
    # curves in their own store
    ('''    name = f"eval{tag}_n{e.n or 'all'}"''', '''    name = f"eval{tag}{'_curves' if getattr(a, 'curves', False) else ''}_n{e.n or 'all'}"      # curve numbers need their own store: an eval made without them is not resumed'''),
])
print("ok")
