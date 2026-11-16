p = "evaluation/jwire_data.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
rep('''def default_pair_run(seed):
    """the pair run of E325's records and heads (E312)"""
    return f"e312_p_s{seed}"
''', '''def default_pair_run(seed):
    """the pair run of E325's records and heads (E312)"""
    return f"e312_p_s{seed}"


def resolve_pair_run(seed, pair_run=None):
    """the pair run asked for: None = E312's (the default); anything else must be a run folder name (an empty string is refused, not read as the default)"""
    if pair_run is None:
        return default_pair_run(seed)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", pair_run), f"pair run {pair_run!r}: a run folder name"
    return pair_run
''')
rep("    mb, mp = load_models(seed, pair_run or default_pair_run(seed), dev)", "    mb, mp = load_models(seed, resolve_pair_run(seed, pair_run), dev)")
rep('''    assert pair_run is None or re.fullmatch(r"[A-Za-z0-9_-]+", pair_run), f"pair run {pair_run!r}: a run folder name"
    other = pair_run is not None and pair_run != default_pair_run(seed)
    return os.path.join(ROOT, "temp", "jwire", f"{kind}_records_v{RECORD_VERSION}_s{seed}{'_max' + str(nmax) if nmax else ''}{'_thr' + str(thr) if thr != 0.1 else ''}{'_pair_' + pair_run if other else ''}.pkl")''',
'''    pair_run = resolve_pair_run(seed, pair_run)
    other = pair_run != default_pair_run(seed)
    return os.path.join(ROOT, "temp", "jwire", f"{kind}_records_v{RECORD_VERSION}_s{seed}{'_max' + str(nmax) if nmax else ''}{'_thr' + str(thr) if thr != 0.1 else ''}{'_pair_' + pair_run if other else ''}.pkl")''')
rep("    return dict(pair_run=pair_run or default_pair_run(seed), seed=seed, record_version=RECORD_VERSION)", "    return dict(pair_run=resolve_pair_run(seed, pair_run), seed=seed, record_version=RECORD_VERSION)")
rep('''    """writes `dict(meta, records)`; a cache file that exists is never overwritten (move it away to rebuild it)"""
    assert not os.path.exists(path), f"{path} exists: a cache is never overwritten"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(dict(meta=meta_of(seed, pair_run), records=records), f)
    os.replace(path + ".tmp", path)''', '''    """writes `dict(meta, records)`; a cache file that exists is never overwritten (move it away to rebuild it): the file is written under a name of its own and linked to `path`, which fails when `path` exists, so
    two builders of the same cache cannot overwrite each other"""
    assert not os.path.exists(path), f"{path} exists: a cache is never overwritten"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(dict(meta=meta_of(seed, pair_run), records=records), f)
        os.link(tmp, path)
    except FileExistsError:
        raise AssertionError(f"{path} exists: a cache is never overwritten (another builder was faster)") from None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)''')
rep('''    pr = a.pair_run or a.run
    path = cache_path("val", a.seed, a.max, a.thr, pr)''', '''    pr = a.pair_run if a.pair_run is not None else a.run
    path = cache_path("val", a.seed, a.max, a.thr, pr)''')
rep("lambda: val_records(a.seed, pr or default_pair_run(a.seed), a.thr, a.max))", "lambda: val_records(a.seed, resolve_pair_run(a.seed, pr), a.thr, a.max))")
open(p, "w", encoding="utf-8").write(s)
