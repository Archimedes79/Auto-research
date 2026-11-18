# 7. Speed: what is measured, on which host

**In short.** Two kinds of time: the time of a frame at inference (ms, the cost gate of the project, measured on a quiet host) and the time of an experiment (training step, read-out, scoring). Both are written with the host state, because a loaded host moves a number by a factor of two. A number that was not measured on a quiet host is labelled "loaded".

Table: Inference time per frame of the champion path (batch 1, bf16, cached features; CUDA events; median).

| system | whole path ms | B' ms | head ms | parameters | host | card |
|---|---|---|---|---|---|---|
| E324 s (previous E2E system) | 19.7 | | | 2.741 M | quiet | E324 s |
| token champion (S1 head, the first recipe; the champion has the identical architecture) | 16.83 / 16.57 (s0 / s1) | 11.41 / 11.37 | 6.70 / 6.64 (0.86 GMACs) | 2.32 M | quiet | E365 |
| token champion, the quiet measurement of the night of 2026-11-18 (before the landing of the clean-up and the speed path) | 16.21 (p90 19.89) | 11.20 | 5.94 | 2.323 M (B' 2.196 M + head 0.127 M) | quiet (load 1.1, GPU idle) | E2E cost table (E379 timing run) |
| token champion after the speed path and the simplification | PENDING: re-measured after the landing | PENDING | PENDING | PENDING | PENDING | E383 |

Table: What the pieces cost at inference (the same protocol: batch 1, bf16, RTX 3060, 24 frames x 5 repeats, the median of the timed frames, the host read QUIET before and after; `experiments/E2E_cost_table.md`, every row carries its quiet verdict).

| piece | median ms | parameters | against the champion path |
|---|---|---|---|
| champion path (B' + one head) | 16.21 | 2.323 M | - |
| e2: two heads in sequence | 21.84 | 2.450 M | +5.63 ms (+35 %); above the 19.7 ms cost gate of E379 |
| e2: two heads stacked (`vmap`) | 25.19 | 2.450 M | +8.98 ms; slower than the sequence, because the head is latency-bound at batch 1 |
| S2 block (cross-attention to the past), head alone | 7.88 | 144,547 (+17,416) | +1.23 ms over the head without it (6.66) |
| S3 block (cross-attention to the SD map), head alone | 7.95 | 147,395 (+20,264) | +1.70 ms over the head without it (6.26) |
| S4 (the node head trained in the loop) | not timed | the same modules as the champion path | the same network with other weights; expected equal, not measured |

The first timings of the same night were flagged not quiet because the tool took the GPU after-sample without the 2.5 s wait that the timing protocol needs (nvidia-smi averages its utilisation over a window); the repeat with the wait reads quiet and agrees within 0.2 ms; the quiet rows are the ones above.

**Training step (the speed path, card E383).** The plain training step of the head was profiled on the real TRAIN cache (bs 8, K 512, one RTX 3060): it is GPU-bound (a whole-step CUDA graph gains only 13 %), and about 60 % of it is dense K x K pair work. The speed path keeps the function and the parameters and changes how they are computed: the handle scales only at the pairs the loss reads, the four pair MLPs on one shared first matmul, a loss without host syncs, and a GPU-resident loader with the plain loader's data order. Parity is the question of the card (rule written before the run, both seeds):

Table: The speed path against the plain trainer. PENDING cells are filled from the card's read-out log; the profile numbers are from a scratch profile (labelled, not on a card).

| quantity | plain trainer | speed path | host / source |
|---|---|---|---|
| ms per training step, fixed GPU batch (profile) | 83.2 | PENDING | loaded host (about 3), scratch profile |
| ms per training step with the loader, ABBA in one process (the gate) | PENDING | PENDING | quiet gap, E383 `gate_abba` |
| ratio plain / fast (gate: at least 1.3) | | PENDING | E383 |
| peak GPU memory (gate: below 4 GB) | | PENDING | E383 |
| topo F1 f - ref at the D115 cell, s0 / s1 | | PENDING | E383 paired cards |
| PARITY verdict | | PENDING | by the rule of the card |

**Read-out time (scoring).** After the training an arm spends most of its wall time on the CPU scorer: the TRAIN sweep (36 cells, about 35 s each, about 22 min) and the VAL scores (about 100 s each). The cells are independent: scored in a fork pool, three processes give 2.5 times the speed with identical numbers (measured on a loaded host, load 11; `evaluation/pmap.py`). PENDING: the `--jobs` option in the evaluator and its quiet-host timing.

**K = 384 (diagnostics only, no verdict).** The pair work at K = 384 is 0.5625 of K = 512; the cache facts (GT-node coverage VAL 0.9878 / 0.9847 against the written 0.99 at K 384; GT links with both ends covered 0.9425 against 0.9541) and the measured step-time ratio are diagnostics of E383; whether a head trained at K 384 keeps the champion's quality belongs to the combined optimisation card (E384). PENDING: the ABBA step-time ratio and the inference ms at K 384.

**What was not sped up, on purpose.** The whole-step CUDA graph (13 %), stacked runs and K = 384 training are not part of the speed path; they are reconsidered when the step is below about 20 ms.

**sources** E383, E365, E324 cards; the profile files are scratch (named as such in the table).
