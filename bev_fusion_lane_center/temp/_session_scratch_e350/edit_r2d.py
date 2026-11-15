p = "evaluation/jwire_data.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
rep('''        globals()["val_records"] = lambda seed, run, thr=0.1, nmax=0: calls.append(("val", seed, run, thr, nmax)) or [toy_record(1, 2), toy_record(2, 3)]''',
'''        none = dict(row=5, scene=1, ends=np.zeros((0, 8)), attrs=np.zeros((0, 6), np.float32), pairs=np.zeros((0, 8)), kp_p=np.zeros((0, 2)), gts=np.zeros((0, 2, 4)))   # a subset with no pair and no matched end
        assert np.isnan(pilot_stats([none])["err_median"]), "pilot_stats of a subset without a matched end is NaN, not a crash"
        globals()["val_records"] = lambda seed, run, thr=0.1, nmax=0: calls.append(("val", seed, run, thr, nmax)) or ([none] if nmax == 1 else [toy_record(1, 2), toy_record(2, 3)])''')
rep('''                main(["live", "--seed", "0", "--split", "val_clean", "--max", "24", "--pair-run", "e324_s_s0"])
            assert calls == [("val", 0, "e324_s_s0", 0.1, 0), ("val", 1, "e312_p_s1", 0.1, 0), ("live", 0, "val_clean", 24, "e324_s_s0")], calls''',
'''                main(["live", "--seed", "0", "--split", "val_clean", "--max", "24", "--pair-run", "e324_s_s0"])
                main(["val", "--seed", "1", "--max", "1"])                                                    # no matched end: the statistics print NaN
            assert calls == [("val", 0, "e324_s_s0", 0.1, 0), ("val", 1, "e312_p_s1", 0.1, 0), ("live", 0, "val_clean", 24, "e324_s_s0"), ("val", 1, "e312_p_s1", 0.1, 1)], calls
            for argv in (["val", "--seed", "0", "--pair-run", ""], ["val", "--seed", "0", "--run", ""], ["live", "--seed", "0", "--pair-run", ""], ["live", "--seed", "0", "--pair-run", "", "--out", os.path.join(d, "x.pkl")]):
                assert "run folder name" in (refused(main, argv) or ""), argv                                   # an empty pair run is refused, not read as the default
            assert len(calls) == 4 and not os.path.exists(os.path.join(d, "x.pkl")), "a refused call computed or wrote something"''')
open(p, "w", encoding="utf-8").write(s)
