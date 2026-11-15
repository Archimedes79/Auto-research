p = "experiments/E348_e324s-recipe-without-paint-branch/experiment.md"
s = open(p, encoding="utf-8").read()
a = "(2) a CPU smoke of the exact command (`--smoke`, `--no-card`, as `runs/e320_smoke_cpu_s0` did): attach + nopaint + scales + e2e + geometry loss has no test together; the step-100 `geom_n` / attach log lines must appear, and `jends_eval.py dump` and `jmaneuver.py` must run on the resulting model;"
assert s.count(a) == 1
b = "(2) a smoke of the exact configuration: attach + nopaint + scales + e2e + geometry loss has no test together. The trainer needs CUDA, so the exact command cannot run on the CPU; **done 2026-11-15 01:3x on the CPU with `temp/e348/smoke_loss.py`** (the model, the loader and the loss of the command, one real TRAIN batch of 2, forward + loss + backward): the pair MLP has 150 inputs, the maps head has no 0.5 m branch (ctx `dil+nopaint`, `base` falls back to B), the loss is finite (4.59) with `geom_n` 34 of `pos_pairs` 34 and `geom_end_d` 1.02 m, the gradient reaches the pair head (315), the attachment (34.8) and the maps head (805) and not the frozen pyramid or B's node head (0); the step-100 `geom_n` log line of the real run is checked once more, and `jends_eval.py dump` and `jmaneuver.py` on the resulting model are checked by the chain's first post-processing;"
s = s.replace(a, b)
a2 = "the start script `temp/e348/start_pairs.sh <seed>`"
assert s.count(a2) == 1
s = s.replace(a2, "the start script `temp/e348/start_pairs.sh <seed>` (with `--ram-mb 6000`: the launcher starts a training only when 6 GB of RAM are available: the box had ENOMEM crashes under D100, E318.u_s1 at step 69)")
a3 = "a NEW chain file (never `day14.sh`"
assert s.count(a3) == 1
s = s.replace(a3, "a NEW chain file `temp/day24/day24.sh` (never `day14.sh`")
open(p, "w", encoding="utf-8").write(s)
print("ok")
