import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)
rep('''    rng = np.random.default_rng(seed)
    lines = [''', '''    from evalsuite.stats import percentile_interval, scene_picks    # the one scene bootstrap of the research system (D24)
    lines = [''')
rep('''        for _ in range(n_boot):
            ix = rng.integers(0, len(sc), len(sc))
            ds.append(_seg_value(Bc[ix].sum(0), metric) - _seg_value(A[ix].sum(0), metric))
        lo, hi = np.percentile(ds, [2.5, 97.5])''', '''        for ix in scene_picks(len(sc), n_boot, seed):
            ds.append(_seg_value(Bc[ix].sum(0), metric) - _seg_value(A[ix].sum(0), metric))
        lo, hi, _ = percentile_interval(ds)''')
rep('''            for _ in range(n_boot):
                ix = rng.integers(0, len(sc), len(sc))
                s_ix = [sc[i] for i in ix]
                ds.append(ap_from_flags(fl_b, s_ix) - ap_from_flags(fl_a, s_ix))
            lo, hi = np.percentile(ds, [2.5, 97.5])''', '''            for ix in scene_picks(len(sc), n_boot, seed):
                s_ix = [sc[i] for i in ix]
                ds.append(ap_from_flags(fl_b, s_ix) - ap_from_flags(fl_a, s_ix))
            lo, hi, _ = percentile_interval(ds)''')
rep('''    """paired scene bootstrap of the headline numbers: b - a, mean and 95 % interval.''', '''    """paired scene bootstrap of the headline numbers: b - a, mean and 95 % interval (draws: evalsuite.stats.scene_picks, the same stream for
    every number; since 2026-10-31, D24 - intervals of earlier compares used numpy default_rng draws and differ in the random part only).''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
