p = "experiments/E342_pair-head-host-time/experiment.md"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


rep("run with workers 3 twice, 6 and 0 (every field compared, NaN padding equal to NaN).", "run with workers 3 twice, 6, 0 and (D101) 4 (every field compared, NaN padding equal to NaN).")
rep("and they differ even between two runs with 3 workers: `make_map(..., rng=np.random.RandomState())` in `map_mode=\"train\"` draws from OS entropy (the 5 % map errors of the training), so they are not reproducible with ANY worker count and the pair stage never reads them.",
    "and they differ even between two runs with 3 workers: `make_map(..., rng=np.random.RandomState())` in `map_mode=\"train\"` is UNSEEDED (OS entropy; the 5 % map errors of the training), so they are not reproducible with ANY worker count. **Which stages read them:** B's training and its gates (`training/grid/qassign.py`, `qgraph.py`, `training/train/qgates.py`, `evaluation/qgraph_timing.py`: the assignment head and the map pieces), so every B training arm carries this non-reproducibility regardless of workers or seeds; the pair stage and the maps stage (`jends_train.py`, `jmaps_train.py`) read none of them.")
rep("**Speed under the load of the box (two trainings running):** per batch of 8 about 2.4 - 2.5 s with 3 workers, 1.4 - 1.8 s with 6, 4.8 - 5.5 s with 0 workers: the loader is the step. **Consequence:** `--workers` is not a recipe parameter; new chain stages use 6 for a single job and 4 per job when two run (D100); the not yet running start scripts of E319 (`temp/e319/start_maps.sh`, `start_pairs.sh`) were changed from 3 to 4",
    "**Workers 4 (the number Mirko chose, D101):** against workers 3 every field the stage reads is bit-identical, only the simulated-map fields differ (the same set as the repeat with 3 workers; `temp/loader/identity_w4.txt`). **Speed under the load of the box (two trainings running):** per batch of 8 about 2.4 - 2.5 s with 3 workers, 1.8 s with 4, 1.4 - 1.8 s with 6, 4.8 - 5.5 s with 0 workers: the loader is the step. **Consequence (D101):** `--workers` is not a recipe parameter; every NEW chain stage and run start uses `--workers 4` (a single job and two jobs alike, never 6); the not yet running start scripts of E319 (`temp/e319/start_maps.sh`, `start_pairs.sh`) were changed from 3 to 4")
open(p, "w", encoding="utf-8").write(s)
print("ok")
