import re
# configs README
p = "training/configs/README.md"
s = open(p, encoding="utf-8").read()
a = "| `presets/smoke.json` | 40 steps, 24 evaluation frames: the end-to-end smoke of a new version | |"
b = ("| `e273_head.json` | the lane head of E273 = baseline B (design `qgraph_head`: phases 2 - 7 of `qgraph_train.py`, a gate after each; later arms add the flags of their card, key = flag) | `e273_p_s0` / `e273_p_s1` ran by the command line of `qgraph_train.py` (`temp/e273/start_p.sh`); the config reproduces that command |\n"
     "| `jobs/e273_head_example.json` | the same as a job file of the Training class (`Training.run`), a smoke as a diagnostic | `Training.check` ok on 2026-11-10; not run (GPU window) |\n"
     "| `presets/smoke.json` | 40 steps, 24 evaluation frames: the end-to-end smoke of a new `grid_head` version | |\n"
     "| `presets/smoke_qgraph.json` | the smoke of a `qgraph_head` run (the trainer's own few steps per phase; `steps` of that design is a list) | |")
assert a in s
s = s.replace(a, b)
a2 = "New designs (a hold, an assignment, a delivered graph) add a row to `DESIGNS`"
b2 = ("**The Training class (since 2026-11-10)**: the same run as a job file, `python -m trainsuite Training.run job.json --project ../bev_fusion_lane_center` from `autoresearch/` (`Training.check` first: it composes, checks and starts nothing; `status`, `stop`, `resume`, `plot` work on the run folder; a `qgraph_head` run writes one folder per phase, `runs/<name>_p<k>`, the final model in `runs/<name>`). The job needs `\"run_args\": [\"--split\", \"val\"]` (the split argument above), and either `card` {hyp, pred, rule} / the arm's run id in the `run_ids` of its card, or `no_card: true` (a diagnostic never feeds a verdict). The shared job runner (`python -m jobrunner`) is not wired to this project: it lists the suites in a `jobrunner` block of `project.json`, which this project does not have, and the head has no `JOBS` table yet.\n"
      "The GT the head reads is the lane GT of `GtGenerator` (`python -m GtGenerator key lane_center --set full`: release v12, lane store `0702a065c6` = l4, complete on 2026-11-10); `startable.py` notes when the store of this checkout is missing.\n\n"
      "New designs (a hold, an assignment, a delivered graph) add a row to `DESIGNS`")
assert a2 in s
s = s.replace(a2, b2)
open(p, "w", encoding="utf-8").write(s)
# training README
p = "training/README.md"
s = open(p, encoding="utf-8").read()
s = s.replace("(`configs/presets/smoke.json`, `configs/presets/seed1.json`)", "(`configs/presets/`), job files of the Training class (`configs/jobs/`)")
s = s.replace("| `configs/e271_grid.json` |", "| `configs/e271_grid.json`, `configs/e273_head.json` |")
s = s.replace("(design `grid_head` -> `grid_train.py`)", "(design `grid_head` -> `grid_train.py`, `qgraph_head` -> `qgraph_train.py`)")
open(p, "w", encoding="utf-8").write(s)
# gates
p = "tools/maintenance/run_gates.sh"
s = open(p, encoding="utf-8").read()
a = 'run "grid metric reproduces stored numbers" lane_grid/selftest.py\n'
b = a + '''# the lane head of E273 and later (CPU, synthetic frames; the tests on stored lane GT frames are in run_all_tests.sh)
run "head: targets, nodes, links, Hermite oracle"        grid/qtargets.py
run "head: prediction contract"                          grid/qformat.py
run "head: map simulator"                                grid/map_sim.py
run "head: all phases, gradients, E274 switches"         grid/qgraph.py
run "head: pixel targets"                                grid/qraster.py
run "head: tier loss switches (defaults = E273 loss)"    grid/test_tier_loss.py
run "head: close-node read-out (rule cases, 300 frames)" test_qgraph_readout.py
run "head: scorer glue (toy lane)"                       test_qgraph_score_glue.py
run "head: tier metrics"                                 test_tier_metrics.py
run "head: pictures selftest"                            viz/qgraph_view.py --selftest
'''
assert a in s
open(p, "w", encoding="utf-8").write(s.replace(a, b))
