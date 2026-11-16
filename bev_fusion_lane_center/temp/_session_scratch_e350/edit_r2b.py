p = "evaluation/jwire_data.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
rep('''        pr = a.pair_run
        path = a.out or cache_path(a.split, a.seed, a.max, 0.1, pr)''', '''        pr = a.pair_run
        resolve_pair_run(a.seed, pr)                                                                       # an empty name is refused here, not read as the default
        path = a.out or cache_path(a.split, a.seed, a.max, 0.1, pr)''')
rep('''        ps = pilot_stats(recs)
        print(f"keypoint error of the matched ends: mean {ps['err_mean']:.2f} / median {ps['err_median']:.2f} / p90 {ps['err_p90']:.2f} m ({ps['matched_ends']} ends)")''', '''    ps = pilot_stats(recs)
    print(f"keypoint error of the matched ends: mean {ps['err_mean']:.2f} / median {ps['err_median']:.2f} / p90 {ps['err_p90']:.2f} m ({ps['matched_ends']} ends)")''')
open(p, "w", encoding="utf-8").write(s)
