import re
p = "bev_fusion_lane_center/results/E2E_Final_Report_2026-11-18/parts/07_speed.md"
s = open(p, encoding="utf-8").read()
a = s.index("| token champion after the speed path and the simplification")
b = s.index("\n", a)
s = s[:a] + "| token champion, K 384 (diagnostic only, not taken) | 15.47 (against 16.24 at K 512 in the same two timings) | | 6.07 (against 6.40) | | flagged not quiet by the sampling artefact only | E383 |" + s[b:]
i = s.index("**Training step (the speed path, card E383).**")
j = s.index("**What was not sped up, on purpose.**")
new = """**Training step: a speed lever was tried, measured and not adopted (card E383).** The plain training step of the head was profiled on the real TRAIN cache (bs 8, K 512, one RTX 3060): it is GPU-bound (a whole-step CUDA graph gains only 13 %), and about 60 % of it is dense K x K pair work. A speed path kept the function and the parameters and changed how they are computed: the handle scales only at the pairs the loss reads, the four pair MLPs on one shared first matmul, a loss without host syncs, and a GPU-resident loader with the plain loader's data order. The card wrote a launch gate before any run (a step-time ratio of at least 1.3, peak memory below 4 GB, GPU tests, a twin calibration of the early curve). It failed on the ratio, so no arm ran and the benchmark is the result.

Table: The launch gate of the speed path (E383, 2026-11-18 05:42 - 06:13, quiet gap, host load 1.09; ABBA blocks in one process; the numbers are the card's).

| quantity | plain trainer | speed path | gate | outcome |
|---|---|---|---|---|
| ms per training step with the loader (60-step ABBA blocks) | 98.8 | 92.3 | | |
| ratio plain / fast | | 1.071 | at least 1.3 | **MISS** |
| peak GPU memory | | 3.09 GB | below 4 GB | met |
| GPU tests (bf16, no host sync) | | ok | must pass | met |
| early curve, loss difference at steps 100 / 300 / 500 / 600 (twin runs, same arguments) | | +0.0000 / +0.0001 / +0.0000 / +0.0001; data order equal at all 41 logged steps | within the half band | met |

What it says: the path is correct (same loss, same data order) and it buys 7 % (the fast head with the CPU loader is slower than the plain step at 113.8 ms; the lever is the loader, and the dense pair work stays). The prediction of the card (40 - 60 ms per step) was far too low. The code is removed behind the archive tag `archive/token-speed-path-2026-11-18`; the inference path is unchanged at 16.21 ms (quiet, K 512).

**Read-out time (scoring) is the speed gain that was kept.** After the training an arm spends most of its wall time on the CPU scorer: the TRAIN sweep (36 cells, about 35 s each, about 22 min) and the VAL scores (about 100 s each). The cells are independent: `jtoken_eval.py sweep --jobs N` scores them in a fork pool (`evaluation/pmap.py`), three processes give 2.5 times the speed with identical numbers (measured on a loaded host, load 11; its test compares `--jobs 3` with the serial table to the last digit). With a training running keep N small; a fork after a torch thread pool can hang, then use `--jobs 1`.

**K = 384 (diagnostics only, no verdict).** The pair work at K = 384 is 0.5625 of K = 512; measured: the training step 0.634 times K 512 (57.6 against 90.9 ms, ABBA), the whole inference path 15.47 against 16.24 ms (head 6.07 against 6.40 ms). The GT-node coverage on VAL is 0.9878 / 0.9847 against the written 0.99 of the E365 rule (GT links with both ends covered 0.9425 against 0.9541): not met, K 384 is not taken. Whether a head trained at K 384 keeps the champion's quality belongs to the combined optimisation card (E384), which was cut by hand at step 0.

"""
s = s[:i] + new + s[j:]
s = s.replace("The whole-step CUDA graph (13 %), stacked runs and K = 384 training are not part of the speed path; they are reconsidered when the step is below about 20 ms.",
              "The whole-step CUDA graph (13 %), stacked runs and K = 384 training were not taken; the dense pair work is what costs, and the gate showed that rewriting around it gives 7 %.")
s = s.replace("**sources** E383, E365, E324 cards; the profile files are scratch (named as such in the table).", "**sources** E383, E365, E324 cards, `experiments/E2E_cost_table.md`; the profile numbers quoted from scratch files are labelled in the text.")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("PENDING left:", s.count("PENDING"))
