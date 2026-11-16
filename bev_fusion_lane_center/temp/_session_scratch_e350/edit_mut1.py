p = "temp/_session_scratch_e350/mutate_pair_run.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
rep('''    ("M8 --name not mandatory with --pair-run", PT, "    assert name, f\\"--name is mandatory", "    name = name or f\\"e325_w_s{seed}\\"\n    assert name, f\\"--name is mandatory", "T"),''',
    '''    ("M8 --name not mandatory with --pair-run", PT, "    assert name is not None or not given, f\\"--name is mandatory", "    assert True or not given, f\\"--name is mandatory", "T"),''')
rep('''    ("M11 default run name changed", PT, 'return name or f"e325_w_s{seed}", JD.default_pair_run(seed)', 'return name or f"e350_w_s{seed}", JD.default_pair_run(seed)', "T"),''',
    '''    ("M11 default run name changed", PT, 'name = f"e325_w_s{seed}" if name is None else name', 'name = f"e350_w_s{seed}" if name is None else name', "T"),''')
rep('''"load_models(seed, pair_run or default_pair_run(seed), dev)", "load_models(seed, default_pair_run(seed), dev)"''', '''"load_models(seed, resolve_pair_run(seed, pair_run), dev)", "load_models(seed, default_pair_run(seed), dev)"''')
rep('''"pr = a.pair_run or a.run", "pr = a.pair_run", "D"),''', '''"pr = a.pair_run if a.pair_run is not None else a.run", "pr = a.pair_run", "D"),''')
rep('''    ("M26 default pair run named differently",''', '''    ("N1 the exists check only with --pair-run (round 1)", PT, 'for f in ("wire_agreement.json", "model.pt"):', 'for f in (("wire_agreement.json", "model.pt") if given else ()):', "T"),
    ("N2 run name not validated", PT, 'assert re.fullmatch(r"[A-Za-z0-9_-]+", name), f"--name {name!r}: a run folder name"', "pass", "T"),
    ("N3 only wire_agreement.json checked", PT, 'for f in ("wire_agreement.json", "model.pt"):', 'for f in ("wire_agreement.json",):', "T"),
    ("N4 only model.pt checked", PT, 'for f in ("wire_agreement.json", "model.pt"):', 'for f in ("model.pt",):', "T"),
    ("N5 an empty --name is the default name", PT, 'name = f"e325_w_s{seed}" if name is None else name', 'name = name or f"e325_w_s{seed}"', "T"),
    ("N6 an empty pair run is the default", PD, "    if pair_run is None:\n        return default_pair_run(seed)", "    if not pair_run:\n        return default_pair_run(seed)", "D"),
    ("N7 save_records overwrites (no exclusive link)", PD, "        os.link(tmp, path)", "        os.replace(tmp, path)", "D"),
    ("N8 pilot_stats of no matched end crashes", PD, 'q = lambda x, f: float(f(x)) if np.size(x) else float("nan")', "q = lambda x, f: float(f(x))", "D"),
    ("N9 the temp file is left behind", PD, "        if os.path.exists(tmp):\n            os.remove(tmp)", "        pass", "D"),
    ("M26 default pair run named differently",''')
open(p, "w", encoding="utf-8").write(s)
