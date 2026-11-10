"""Scratch: README corrections after the critic (which networks run, list_results keys, memory_checked, no 'cold mode', no 'frozen')"""
B = "/mnt/c/Auto-research/bev_fusion/"


def edit(path, pairs):
    s = open(B + path, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80])
        s = s.replace(old, new)
    open(B + path, "w", encoding="utf-8", newline="").write(s)


s = open(B + "runner/README.md", encoding="utf-8").read()
i0 = s.index("Which networks run:")
i1 = s.index("```", i0)
new_par = ("Which networks run: a run whose code snapshot (or, without a snapshot, the live `training/`) has what the free subject calls before the model, `train.amp_context`, `batch_ego(gap_steps, …)`,\n"
           "`supports_ego` and `to_device` (`NEEDS` in `evaluation/bev_free.py`), and whose dataset makes `gap_steps` from the poses: `t16_full_s0`, `t20_base_s0`, `t23_C_s1` and everything after them (the three of\n"
           "them were run through `tests/test_runner_real.py`). A run from before that (`t8_*`, `t10_*`, `t11_*`, which are the three networks of `records/keep3.json`) cannot run without ground truth: its dataset\n"
           "takes the ego steps from the ground truth and its `train.py` has none of these. `plan` says so, and `predict` refuses with `ValueError: … too old for the run without ground truth` before any frame;\n"
           "such a run is evaluated by `evaluation/bev_suite.py` (the live path with the ground truth).\n\n")
s = s[:i0] + new_par + s[i1:]
open(B + "runner/README.md", "w", encoding="utf-8", newline="").write(s)

edit("runner/README.md", [
    ("the health of the memory is checked per scene (a warning, never an error) |",
     "the health of the memory is checked per scene (a warning, never an error; `memory_checked` says whether it could be judged); a tap the network does not produce is an error before anything is stored |"),
    ("| `list_results(network)` | the stored predictions of a network: id, frames, megabytes, created |", "| `list_results(network)` | the stored predictions of a network: id, kind, frames, megabytes, created |"),
    ("`--cold` restarts the memory at every frame (the suite's cold mode).", "`--cold` restarts the memory at every frame (for the planner's smoke tests only: D70 allows no cold run for a number that is compared)."),
])
edit("README.md", [("(`bev_runner.py`, frozen)", "(`bev_runner.py`)")])
print("ok")
