"""D24 doc edits, part 2 (README rows of the tests)"""
import os
R = os.environ["BFLC_ROOT"]
def sub(path, old, new):
    p = os.path.join(R, path)
    t = open(p, encoding="utf-8").read()
    assert t.count(old) == 1, (path, old[:60], t.count(old))
    open(p, "w", encoding="utf-8", newline="").write(t.replace(old, new))
    print("ok", path)
sub("evaluation/README.md", "`test_dp_simplify.py` (9 spec tests of the output stage `simplify_dp`, toy graphs) |",
    "`test_dp_simplify.py` (9 spec tests of the output stage `simplify_dp`, toy graphs); `test_evaluate_guard.py` (the TEST guard of `evaluate.py` accepts whole logged run ids only, review bug L2) |")
sub("evaluation/sparse_lanes/README.md", "| `selftest.py` | perfect graph scores 1,",
    "| `selftest.py` | spec of `graph_from_vertices` (fork, merge, cycles, isolated chain; review bug L1 fixed 2026-10-31, D24: a split vertex with one predecessor now starts its branches), perfect graph scores 1,")
sub("evaluation/lane_grid/README.md", "| `selftest.py` | re-scores the full validation maps",
    "| `selftest.py` | spec of `skeletons` with hysteresis and range threshold together (CPU, review bug L3 fixed 2026-10-31: far seeds survive), then re-scores the full validation maps")
sub("tools/maintenance/CLEANUP_LIST.md", "| inference variants | `pipeline/p16_{ensemble,hyst}`, `line_tracer.py`, `data/p14_chain_data.py` | nothing imports them (`p14_chain_data` is superseded by `p16_chain_data`) |",
    "| inference variants | ~~`pipeline/p16_{ensemble,hyst}`~~ (moved to `evaluation/loops/loop16/` 2026-10-31, D24), `data/p14_chain_data.py`; `line_tracer.py` is live (imported by `graph_from_dense.py`) | `p14_chain_data` is superseded by `p16_chain_data` but `data/rebuild.json` names it as the builder of the loop-14 chain data |")
