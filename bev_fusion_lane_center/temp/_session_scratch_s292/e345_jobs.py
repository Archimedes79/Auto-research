p = "experiments/E345_pruning-sensitivity-step0/experiment.md"
s = open(p, encoding="utf-8").read()
a = "four GPU jobs `E345.a_e312_s0`, `E345.a_e312_s1`, `E345.a_e324s_s0`, `E345.a_e324s_s1` (the sweep passes, the dumps of the scored points, the random / reverse / null / plumbing controls; about 170 passes per model-seed at about 25 - 40 s, unmeasured: the selftest measures one pass under load; about 2 h each, the slot wait is not counted) and one CPU job `E345.s` (`--after` the four, no `--gpu`: the scorings and the bootstraps, two at a time)."
assert s.count(a) == 1
b = ("**the job flow per model (E312, E324 s), seed 0 first: (1) GPU `E345.a_<model>_s<S>` (the sweep passes, the nulls, the plumbing, random and reverse controls; about 170 passes at about 25 - 40 s, unmeasured: `jprune_probe.py time` measures one pass first; about 2 h each, the slot wait is not counted); "
     "(2) CPU `E345.p_<model>` (`jprune_table.py plan` on seed 0's sweep: the point list `score_points.json`; for seed 1 the same file is used, its family comes from the `sweep.json` next to it); "
     "(3) GPU `E345.d_<model>_s<S>` (`sweep --score-points`: the dumps of the planned points on BOTH seeds; each GPU invocation rebuilds the 2.7 GB VAL cache and re-runs the plumbing, about 15 minutes); "
     "(4) CPU `E345.s_<model>_s<S>` (`score`: the stored scorer on the point folders, `--after` the dumps; the null points need all read-outs, every other point only the judged ones; then the handle-scale points are scored); "
     "(5) CPU `plan` again (v needs the scored handle-scale points); (6) GPU `E345.v_<model>_s<S>` (the dump of v and of the second null, about 5 minutes); (7) CPU `score` of v, `boot` (the paired scene bootstrap of every scored point against the null, loss = null minus ablated, the opposite sign of `jdiag.py boot`'s A - B), `table`.** "
     "That is four sweep jobs, four dump jobs and four v jobs on the GPU (the last two kinds short) and the CPU chain after them (about 12 - 28 h of scoring over the weekend); the GPU jobs wait with a long timeout for room (`GPU_GATE_TIMEOUT_S`, 12 h) and then exit with code 75 instead of going on into a full GPU (the code critic: the shared gate would otherwise go on after 4 h at 10.4 GB in use); the probe prints and records the peak GPU memory (`torch.cuda.max_memory_allocated`) of every pass, so that the gate's 9000 MiB limit can be re-set from a measurement.")
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
