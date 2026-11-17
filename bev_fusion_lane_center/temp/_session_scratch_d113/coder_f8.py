p = "training/data/lane_past_data.py"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep('''def loader_fixture_checks(tokens, hows=LOADER_HOWS, tol=TOL):
    """the rule of the loader fixture on the token stats per variant (dicts of `stats`) -> name -> bool: the loader's pose gives p90 <= 1.5 m, every wrong pose / shifted entry FAILS that"""
    ok = {"loader path within tolerance": tokens["loader"]["n"] > 0 and tokens["loader"]["p90"] <= tol["tok_p90"]}
    for how in hows:
        if how != "loader":
            ok[f"{how} fails"] = not (tokens[how]["n"] > 0 and tokens[how]["p90"] <= tol["tok_p90"])
    return ok
''', '''POSE_TOL = 1e-4                                                                  # the loader's pose entry equals a fresh `rel_pose` of the pair from the poses file within this (float32 storage: about 1e-6)


def loader_fixture_checks(tokens, pose_err, hows=LOADER_HOWS, tol=TOL):
    """the rule of the loader fixture on the token stats per variant (dicts of `stats`) and the largest deviation of the pose entry from a fresh `rel_pose` of the pair (`pose_err`: variant -> number) -> name -> bool: the loader's variant has
    p90 <= 1.5 m AND the right pose entry; every other variant (identity pose, the entry of the previous item, a shifted previous frame) FAILS at least one of the two (consecutive ego steps are alike: the entry of the neighbour alone passes the token test)"""
    good = lambda h: tokens[h]["n"] > 0 and tokens[h]["p90"] <= tol["tok_p90"] and pose_err[h] <= POSE_TOL  # noqa: E731
    ok = {"loader path within tolerance": good("loader")}
    for how in hows:
        if how != "loader":
            ok[f"{how} fails"] = not good(how)
    return ok
''')
rep('''    err = {h: [] for h in hows}
    for how in hows:
        for ip, ic in pairs(ds.prev, n_pairs, seed, shift=int(how == "frame_shifted")):''', '''    import lane_seq_data as LS
    pose = dict(np.load(pose_path(split)))
    rows = ds.tok.rows                                                                  # only the rows of the cache are read from the poses file (the VAL file also holds TEST rows)
    err, perr = {h: [] for h in hows}, {h: 0.0 for h in hows}
    for how in hows:
        for ip, ic in pairs(ds.prev, n_pairs, seed, shift=int(how == "frame_shifted")):''')
rep('''            past = JP.make_past(dict(x=torch.zeros(1, K, JP.D), w=torch.zeros(1, K)), fp, fc, R, t, False, bc["has_prev"][:, 0])''', '''            Rf, tf = LS.rel_pose(pose["R"][rows[ic]], pose["t"][rows[ic]], pose["R"][rows[ip]], pose["t"][rows[ip]])      # the pose of THIS pair, fresh from the poses file
            perr[how] = max(perr[how], float(np.abs(R[0].numpy() - Rf).max()), float(np.abs(t[0].numpy() - tf).max()))
            past = JP.make_past(dict(x=torch.zeros(1, K, JP.D), w=torch.zeros(1, K)), fp, fc, R, t, False, bc["has_prev"][:, 0])''')
rep('''tokens={h: stats(np.concatenate(e) if e else np.zeros(0)) for h, e in err.items()})
    ok = res["checks"] = loader_fixture_checks(res["tokens"], hows)''', '''tokens={h: stats(np.concatenate(e) if e else np.zeros(0)) for h, e in err.items()}, pose_err=perr)
    ok = res["checks"] = loader_fixture_checks(res["tokens"], perr, hows)''')
rep('''        log(f"  {how:<14} n {k['n']:5d} median {k['median']:.2f} p90 {k['p90']:.2f} m")''', '''        log(f"  {how:<14} n {k['n']:5d} median {k['median']:.2f} p90 {k['p90']:.2f} m | pose entry deviation from the pair's fresh pose {perr[how]:.2g}")''')
rep('''    lt = dict(loader=gd, identity=bd, entry_shifted=bd, frame_shifted=bd)
    assert all(loader_fixture_checks(lt).values()) and len(loader_fixture_checks(lt)) == 4
    for name_, t2 in dict(loader=dict(lt, loader=dict(gd, p90=1.6)), identity=dict(lt, identity=gd), entry=dict(lt, entry_shifted=gd), frame=dict(lt, frame_shifted=gd), empty=dict(lt, loader=dict(gd, n=0))).items():
        assert not all(loader_fixture_checks(t2).values()), f"the loader fixture accepted bad {name_}"''', '''    lt, pe = dict(loader=gd, identity=bd, entry_shifted=gd, frame_shifted=bd), dict(loader=1e-6, identity=0.5, entry_shifted=0.2, frame_shifted=0.3)       # the neighbour's pose entry passes the token test, not the pose check
    assert all(loader_fixture_checks(lt, pe).values()) and len(loader_fixture_checks(lt, pe)) == 4
    for name_, (t2, p2) in dict(loader=(dict(lt, loader=dict(gd, p90=1.6)), pe), pose=(lt, dict(pe, loader=2e-4)), identity=(dict(lt, identity=gd), dict(pe, identity=1e-6)), entry=(lt, dict(pe, entry_shifted=1e-5)), frame=(dict(lt, frame_shifted=gd), dict(pe, frame_shifted=0.0)),
                                empty=(dict(lt, loader=dict(gd, n=0)), pe)).items():
        assert not all(loader_fixture_checks(t2, p2).values()), f"the loader fixture accepted bad {name_}"''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
