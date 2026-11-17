p = "evaluation/jwire_data.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
# section 1: the empty name and resolve
rep('''        for bad in ("../x", "a/b", ""):
            try:
                cache_path("val", 0, pair_run=bad)
                raise AssertionError(f"pair run {bad!r} must be refused")
            except AssertionError as e:
                assert "run folder name" in str(e), e''', '''        for bad in ("../x", "a/b", "", "a b"):
            for f in (lambda b: cache_path("val", 0, pair_run=b), lambda b: meta_of(0, b), lambda b: resolve_pair_run(0, b)):                    # an empty name is NOT the default
                try:
                    f(bad)
                    raise AssertionError(f"pair run {bad!r} must be refused")
                except AssertionError as e:
                    assert "run folder name" in str(e), e
        assert resolve_pair_run(1) == "e312_p_s1" and resolve_pair_run(0, "e324_s_s0") == "e324_s_s0"''')
# section 2: save race + no leftover
rep('''        assert refused(save_records, p, recs, 0), "an existing cache must not be overwritten"''', '''        assert refused(save_records, p, recs, 0), "an existing cache must not be overwritten"
        assert os.listdir(os.path.dirname(p)) == [os.path.basename(p)], "no temp file is left behind"
        race = cache_path("val", 0, 7)

        class Racer:                                                                                          # pickling it creates the cache: another builder finishes between the check and the link
            def __reduce__(self):
                open(race, "wb").write(b"the other builder's cache")
                return (list, ())
        assert refused(save_records, race, [Racer()], 0) and open(race, "rb").read() == b"the other builder's cache" and sorted(os.listdir(os.path.dirname(p))) == sorted([os.path.basename(p), os.path.basename(race)]), "two builders of one cache: the first one stays"''')
# section 4: empty name for live_records
rep('''        assert refused(live_records, 0, "test"), "TEST is never read"''', '''        assert refused(live_records, 0, "test"), "TEST is never read"
        n_got = len(got)
        assert refused(live_records, 0, "val_clean", 0.1, 4, device="cpu", pair_run="") and len(got) == n_got, "an empty pair run is not the default"''')
open(p, "w", encoding="utf-8").write(s)
