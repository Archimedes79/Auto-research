# 7. Speed: what is measured, on which host

**In short.** Two kinds of time: the time of a frame at inference (ms, the cost gate of the project, measured on a quiet host) and the time of an experiment (training step, read-out, scoring). Both are written with the host state, because a loaded host moves a number by a factor of two. A number that was not measured on a quiet host is labelled "loaded".

Table: Inference time per frame of the champion path (batch 1, bf16, cached features; CUDA events; median).

| system | whole path ms | B' ms | head ms | parameters | host | card |
|---|---|---|---|---|---|---|
| E324 s (previous E2E system) | 19.7 | | | 2.741 M | quiet | E324 s |
| token champion (S1 head, the first recipe; the champion has the identical architecture) | 16.83 / 16.57 (s0 / s1) | 11.41 / 11.37 | 6.70 / 6.64 (0.86 GMACs) | 2.32 M | flagged not quiet by the tool (GPU up to 10 % before the run); the quiet re-measurement is 16.21 ms | E371, E2E cost table |
| token champion, the quiet measurement of the night of 2026-11-18 (before the landing of the clean-up and the speed path) | 16.21 (p90 19.89) | 11.20 | 5.94 | 2.323 M (B' 2.196 M + head 0.127 M) | quiet (load 1.1, GPU idle) | E2E cost table (E379 timing run) |
| token champion, K 384 (diagnostic only, not taken) | 15.47 (against 16.24 at K 512 in the same two timings) | | 6.07 (against 6.40) | | flagged not quiet by the sampling artefact only | E383 |

Table: What the pieces cost at inference (the same protocol: batch 1, bf16, RTX 3060, 24 frames x 5 repeats, the median of the timed frames, the host read QUIET before and after; `experiments/E2E_cost_table.md`, every row carries its quiet verdict).

| piece | median ms | parameters | against the champion path |
|---|---|---|---|
| champion path (B' + one head) | 16.21 | 2.323 M | - |
| e2: two heads in sequence | 21.84 | 2.450 M | +5.63 ms (+35 %); above the 19.7 ms cost gate of E379 |
| e2: two heads stacked (`vmap`) | 25.19 | 2.450 M | +8.98 ms; slower than the sequence, because the head is latency-bound at batch 1 |
| S2 block (cross-attention to the past), head alone | 7.88 | 144,547 (+17,416) | +1.23 ms over the head without it (6.66) |
| S3 block (cross-attention to the SD map), head alone | 7.95 | 147,395 (+20,264) | +1.70 ms over the head without it (6.26) |
| S4 (the node head trained in the loop) | not timed | the same modules as the champion path | the same network with other weights; expected equal, not measured |

The first timings of the same night were flagged not quiet because the tool took the GPU after-sample without the 2.5 s wait that the timing protocol needs (nvidia-smi averages its utilisation over a window); the repeat with the wait reads quiet and agrees within 0.2 ms (except the S3 block: 0.65 ms); the quiet rows are the ones above.

**Training step: a speed lever was tried, measured and not adopted (card E383).** The plain training step of the head was profiled on the real TRAIN cache (bs 8, K 512, one RTX 3060): it is GPU-bound (a whole-step CUDA graph gains only 13 %), and about 60 % of it is dense K x K pair work (a scratch estimate of the card; its own profile, stubbing the pair MLPs and the scale branch, saved 21 of 68.1 ms, about 31 %, and the 13 % of the CUDA graph is from the same scratch work). A speed path kept the function and the parameters and changed how they are computed: the handle scales only at the pairs the loss reads, the four pair MLPs on one shared first matmul, a loss without host syncs, and a GPU-resident loader with the plain loader's data order. The card wrote a launch gate before any run (a step-time ratio of at least 1.3, peak memory below 4 GB, GPU tests, a twin calibration of the early curve). It failed on the ratio, so no arm ran and the benchmark is the result.

Table: The launch gate of the speed path (E383, 2026-11-18 05:42 - 06:13, quiet gap, host load 1.09; ABBA blocks in one process; the numbers are the card's).

| quantity | plain trainer | speed path | gate | outcome |
|---|---|---|---|---|
| ms per training step with the loader (60-step ABBA blocks) | 98.8 | 92.3 | | |
| ratio plain / fast | | 1.071 | at least 1.3 | **MISS** |
| peak GPU memory | | 3.09 GB | below 4 GB | met |
| GPU tests (bf16, no host sync) | | ok | must pass | met |
| early curve, loss difference at steps 100 / 300 / 500 / 600 (twin runs, same arguments) | | +0.0000 / +0.0001 / +0.0000 / +0.0001; data order equal at all 41 logged steps | within the half band | met |

What it says: the path is correct (same loss, same data order) and it buys 7 % (the fast head with the CPU loader is slower than the plain step at 113.8 ms; the lever is the loader, and the dense pair work stays). The prediction of the card (40 - 60 ms per step) was far too low. The code is removed behind the archive tag `archive/token-speed-path-2026-11-18`; the inference path is unchanged at 16.21 ms (quiet, K 512).

**Read-out time (scoring) is the speed gain that was kept.** After the training an arm spends most of its wall time on the CPU scorer: the TRAIN sweep (36 cells, about 35 s each, about 22 min) and the VAL scores (about 100 s each). The cells are independent: `jtoken_eval.py sweep --jobs N` scores them in a fork pool (`evaluation/pmap.py`), three processes give 2.5 times the speed with identical numbers (measured with three processes on a loaded host, on the scoring function `pmap`, not on the whole sweep; its test compares `--jobs 3` with the serial table to the last digit). With a training running keep N small; a fork after a torch thread pool can hang, then use `--jobs 1`.

**K = 384 (diagnostics only, no verdict).** The pair work at K = 384 is 0.5625 of K = 512; measured: the training step 0.634 times K 512 (57.6 against 90.9 ms, ABBA), the whole inference path 15.47 against 16.24 ms (head 6.07 against 6.40 ms). The GT-node coverage on VAL is 0.9878 / 0.9847 against the written 0.99 of the E365 rule (GT links with both ends covered 0.9425 against 0.9541): not met, K 384 is not taken. Whether a head trained at K 384 keeps the champion's quality has no card yet (E384 is the recursive node-state card and has no K 384 content; E385 says that a change of K needs its own card): training at K 384 is open.

**What was not sped up, on purpose.** The whole-step CUDA graph (13 %), stacked runs and K = 384 training were not taken; the dense pair work is what costs, and the gate showed that rewriting around it gives 7 %.

**sources** E383, E365, E324 cards, `experiments/E2E_cost_table.md`; the profile numbers quoted from scratch files are labelled in the text.
