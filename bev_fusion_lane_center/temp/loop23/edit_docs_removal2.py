def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:70], s.count(a))
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8").write(s)


sub("evaluation/README.md", [
    ("Removed 2026-11-03 (no importer, restore from `lcr/pre-restructure-2026-11-03`): `loops/loop16/p16_eval.py`, `p16_ensemble.py`, `p16_hyst.py`, `loops/loop17/p17_compare.py`, `p17_dense.py`, `drivers/p16_fast_val.py`, `p16_pareto.py`, `p16_route_check.py`. ",
     "Removed 2026-11-03 (no importer; restore from the git tag lcr/pre-restructure-2026-11-03): three loop-16 scripts (eval, ensemble, hysteresis), two loop-17 scripts (compare, dense), three drivers (fast_val, pareto, route_check). "),
    ("(the example run of `examples/` was removed 2026-11-03: `evaluate.py --compare` and `lane_suite.py` do the same)", "(the example run of the old examples folder was removed 2026-11-03: `evaluate.py --compare` and `lane_suite.py` do the same)"),
])
sub("training/README.md", [
    ("| ~~`legacy/`~~ | removed 2026-11-03 (28 modules of loops 2-8; no live code, test or document reached them); restore a file with `git checkout lcr/pre-restructure-2026-11-03 -- training/legacy/<file>` | - |",
     "| (legacy folder) | removed 2026-11-03: 28 modules of loops 2-8 that no live code, test or document reached; restore a file from the git tag lcr/pre-restructure-2026-11-03 | - |"),
])
print("ok")
