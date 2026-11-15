import re


def sub(path, pairs, regex=()):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:70], s.count(a))
        s = s.replace(a, b)
    for pat, b in regex:
        s, n = re.subn(pat, b, s, flags=re.M)
        assert n == 1, (path, pat, n)
    open(path, "w", encoding="utf-8").write(s)


TAG = "`lcr/pre-restructure-2026-11-03`"
# architecture spec: the legacy folder is gone, probe.py is a helper of the trainer
sub("training/architecture.json", [(
    '    "training/legacy": "old loop 2-8 modules that live code or old experiments still import"\n',
    '    "training/probe.py": "gradient and update size per module for the trainer (trainsuite contract), copied from the BEV system"\n')])
# training README: the legacy row
s = open("training/README.md", encoding="utf-8").read().split("\n")
i = [k for k, ln in enumerate(s) if ln.startswith("| `legacy/` |")]
assert len(i) == 1
s[i[0]] = ("| ~~`legacy/`~~ | removed 2026-11-03 (28 modules of loops 2-8; no live code, test or document reached them); restore a file with `git checkout " + TAG[1:-1] + " -- training/legacy/<file>` | - |")
open("training/README.md", "w", encoding="utf-8").write("\n".join(s))
# evaluation README
sub("evaluation/README.md", [
    ("`p16_fast_val.py`, `p16_fast_check.py`", "`p16_fast_check.py`"),
    ("Moved 2026-10-31 (D24): `loops/loop16/p16_eval.py` (from `drivers/`), `loops/loop16/p16_ensemble.py`, `loops/loop16/p16_hyst.py` (from `training/pipeline/`, no importer), ",
     "Removed 2026-11-03 (no importer, restore from " + TAG + "): `loops/loop16/p16_eval.py`, `p16_ensemble.py`, `p16_hyst.py`, `loops/loop17/p17_compare.py`, `p17_dense.py`, `drivers/p16_fast_val.py`, `p16_pareto.py`, `p16_route_check.py`. Moved 2026-10-31 (D24): "),
    ("Details: `METRIC_RESPONSE.md`, `examples/RESULT.md`.", "Details: `METRIC_RESPONSE.md` (the example run of `examples/` was removed 2026-11-03: `evaluate.py --compare` and `lane_suite.py` do the same)."),
])
s = open("evaluation/README.md", encoding="utf-8").read().split("\n")
i = [k for k, ln in enumerate(s) if ln.startswith("| `examples/` |")]
assert len(i) == 1
del s[i[0]]
open("evaluation/README.md", "w", encoding="utf-8").write("\n".join(s))
sub("evaluation/sparse_lanes/README.md", [
    ("`p10_route_eval2.py`, `p16_route_check.py`, `p14_chain_consumer.py`", "`p10_route_eval2.py`, `eval_parts.py`, `p14_chain_consumer.py`"),
    ("old `stability.py` in `training/legacy/`", "old `stability.py` (removed 2026-11-03, in git history)"),
])
print("ok")
