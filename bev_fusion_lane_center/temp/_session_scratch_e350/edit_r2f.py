p = "training/train/jwire_train.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
rep("jwire_train.py run --seed 0 [--pair-run e312_p_s0 --name e325_w_s0] [--heads 3]", "jwire_train.py run --seed 0 [--name e325_w_s0 | --pair-run e324_s_s0 --name e350_w_s0] [--heads 3]")
rep('''With `--pair-run` the run needs its own `--name` and a run folder that already holds a `wire_agreement.json` or a
`model.pt` is refused (the default name e325_w_s<seed> would overwrite E325's run).''', '''A run folder that already holds a `wire_agreement.json` or a
`model.pt` is refused in EVERY case (`trainer.Run` has no overwrite guard; the default name e325_w_s<seed> is E325's run); with `--pair-run` the run needs its own `--name`; a name is a folder name.''')
rep("import argparse\nimport os\nimport sys\n", "import argparse\nimport os\nimport re\nimport sys\n")
rep('''def run_target(seed, name, pair_run=None):
    """-> (run name, pair run) of a training run. Default (pair_run None): E325's run on E312's records, name e325_w_s<seed>. With a pair run: it needs its own `name`, and a run folder that already holds a
    `wire_agreement.json` or a `model.pt` is refused (`trainer.Run` has no overwrite guard: the default name would overwrite E325's run)"""
    if pair_run is None:
        return name or f"e325_w_s{seed}", JD.default_pair_run(seed)
    assert name, f"--name is mandatory with --pair-run (the default name e325_w_s{seed} would overwrite E325's run)"
    for f in ("wire_agreement.json", "model.pt"):
        assert not os.path.exists(os.path.join(ROOT, "runs", name, f)), f"runs/{name}/{f} exists: a finished run is never overwritten (give a new --name)"
    return name, pair_run''', '''def run_target(seed, name, pair_run=None):
    """-> (run name, pair run) of a training run. Default (pair_run None): E312's records and the name e325_w_s<seed> (E325); with a pair run `name` is mandatory. A name is a folder name, and a run folder that already
    holds a `wire_agreement.json` or a `model.pt` is refused in EVERY case (`trainer.Run` has no overwrite guard: a second call with the default name would overwrite E325's run)"""
    given = pair_run is not None
    pair_run = JD.resolve_pair_run(seed, pair_run)
    assert name is not None or not given, f"--name is mandatory with --pair-run (the default name e325_w_s{seed} would overwrite E325's run)"
    name = f"e325_w_s{seed}" if name is None else name
    assert re.fullmatch(r"[A-Za-z0-9_-]+", name), f"--name {name!r}: a run folder name"
    for f in ("wire_agreement.json", "model.pt"):
        assert not os.path.exists(os.path.join(ROOT, "runs", name, f)), f"runs/{name}/{f} exists: a finished run is never overwritten (give a new --name)"
    return name, pair_run''')
open(p, "w", encoding="utf-8").write(s)
