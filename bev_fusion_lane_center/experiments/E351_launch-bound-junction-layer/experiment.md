# E351 - the launch-bound junction layer: how much of the forward's ms are kernel launches and host syncs, and what a CUDA graph of the captured parts gives back (inference path only, outputs unchanged)

Card of 2026-11-15 09:20, version 1 (a fresh critic pass comes before anything is built). Written by the lane experiments chat on the lane lead's order after E342 part B (`experiments/E342_pair-head-host-time/experiment.md` section 5: the forward is LAUNCH bound, 582 / 550 / 785 / 753 CUDA launches and 11 syncs per forward, no hot section; D92's ask "gucke, wie man sonst Rechenzeit sparen kann": no narrowing). It is the compute card: **every ms counts toward "little compute" (D86), the card is E2E-neutral (training untouched, the inference path only; D87 / D88 are not touched: no rule step is replaced, no output changes).** **Frozen files:** the TEST use E343 pins a code digest of 71 lane files (among them `jends.py`, `jmaps.py`, `qends_eval.py`, `jends_eval.py`); until it ends nothing of them is edited: every lever below is built in NEW files (wrappers and copies); a copy of a frozen forward is guarded by an equality test against the original, so a later change of the original makes the test fail loudly. Priority after E345 step 1 (the lead); the GPU part needs idle windows; nothing is started.

## 0. The card (what the tools run)
```json card
{
 "id": "E351",
 "card": "",
 "title": "CUDA-graph capture of the launch-bound parts of the junction forward (pyramid + maps head + B's node head as one graph at batch 1, the pair head in graph segments around its host rank matrix), on E312 and on E324 s, outputs equal to the eager forward on all 873 VAL frames, ms per frame paired on an idle GPU",
 "kind": "variant",
 "deterministic": true,
 "metric": "ms_per_frame",
 "direction": "lower",
 "band": 0.2,
 "prediction": {"value": -5.5, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the eager forward as in HEAD (`JEndModel.forward`, bf16 autocast, batch 1, `no_grad`): E312 17.30 ms, E324 s 24.58 ms (E342 section 5, loaded host)"},
  "g1": {"recipe": "lever 1: ONE CUDA graph of pyramid + maps head + (E324 s) B's node head: static input buffer (the 1 x C x H x W feature tensor), the maps outputs and the node outputs are static buffers the eager pair head reads; capture once per checkpoint after warm-up, replay per frame (`evaluation/jgraph.py`, new file)", "run_ids": ["E351.g1"]},
  "g2": {"recipe": "lever 2 on top of g1: the pair head's sections before the rank matrix (peaks, soft-argmax, tokens, geometry, evidence: 231 launches) as graph 2 and the sections after it (MLP, scales, attachment: 14 / 97 launches) as graph 3, the host rank matrix (9 syncs) eager between them; the pair head code is a graph-safe COPY of `JEnds.forward` in a new file (`training/grid/jends_graph.py`: constants created on the device, no host-to-device copy inside a captured region)", "run_ids": ["E351.g2"]},
  "t": {"recipe": "lever 1b, only if cheap: `torch.compile` (mode reduce-overhead, i.e. inductor + CUDA graphs) of the same captured parts instead of the manual graph; run only if triton / inductor work in the env `gcar` (checked in the build step); outputs to a tolerance, not bit for bit", "run_ids": ["E351.t"]}
 },
 "control": "c",
 "variant": "g1",
 "pairing": "position",
 "seeds": [0],
 "rule": [
  {"metric": "ms_per_frame", "vs": "c", "min_effect": 2.6},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.0001}
 ],
 "budget_gpu_h": 1.0,
 "kill": "an arm's outputs differ from the eager forward beyond the tolerance of section 1 on any of the 873 VAL frames of a checkpoint after one repair attempt, or the capture fails (a host sync or a host-to-device copy in the captured region that cannot be removed without changing the numbers): that lever is dropped, the others are read",
 "evaluate": {"split": "val"}
}
```
**The rule in words (written before the run).** A lever is judged per SYSTEM (E312 and E324 s separately; E324 s is the system of record, D94). **ADOPT (for that system's inference path)** when (1) the outputs equal the eager forward on ALL 873 VAL frames of BOTH seeds' checkpoints (`e312_p_s0 / s1`, `e324_s_s0 / s1`) to the tolerance of section 1, and the delivered-graph cards scored from the lever's dump (snapped, and PRIMARY for E324 s) reproduce the stored cards: topo F1 within 1e-4, every other number within the tolerance of the E343 card style (3e-3); and (2) the ms per frame (batch 1, bf16, idle GPU, alternating blocks A B | B A, `evaluation/jtiming.py`, seed-0 checkpoint) falls by at least 15 % of the eager ms (E312: 2.6 ms, E324 s: 3.7 ms) with a paired 95 % interval that excludes 0. **TIE (the simpler one, eager, stays):** outputs identical but the drop is below 15 % or its interval contains 0; **CUT** a lever whose outputs differ (the kill). Among adoptable levers the one with the lowest ms wins; g2 is adopted over g1 only if it is at least 5 % of the eager ms faster with a paired interval that excludes 0 (it costs a graph-safe copy of the pair head). Nothing is trained, TEST untouched, `decide.py` is not used (`by hand: measured` with the rule above).

## 1. Card text
- **Question:** is the forward's time kernel launches and host syncs (so that a graph capture returns most of it with unchanged outputs), and how far can the pair head's host steps go?
- **Step 1, a count (measured 2026-11-15 09:04, `E351.k2_kernels`, `evaluation/jtiming.py kernels`, `runs/eval/jtiming/e351_kernels_20261115.json`, one forward of the middle VAL frame; launches / memcopies / syncs; the prediction "the pair head and attachment hold about 40 % of the launches, the syncs sit in the rank matrix" HIT):**

| stage / section | E312 | E341 | E324 s | E348 |
|---|---|---|---|---|
| pyramid | 186 / 0 / 0 | 186 / 0 / 0 | 186 / 0 / 0 | 186 / 0 / 0 |
| maps head | 138 / 0 / 0 | 114 / 0 / 0 | 138 / 0 / 0 | 114 / 0 / 0 |
| B's node head (attach models) | - | - | 120 / 17 / 0 | 120 / 17 / 0 |
| pair head: peaks + top-K | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |
| soft-argmax and gathers | 50 / 3 / 1 | 50 / 3 / 1 | 50 / 3 / 1 | 50 / 3 / 1 |
| kind tensor | 0 / 1 / 1 | 0 / 1 / 1 | 0 / 1 / 1 | 0 / 1 / 1 |
| token linear + 2 SetBlocks | 35 / 0 / 0 | 35 / 0 / 0 | 35 / 0 / 0 | 35 / 0 / 0 |
| pair geometry | 50 / 0 / 0 | 50 / 0 / 0 | 50 / 0 / 0 | 50 / 0 / 0 |
| evidence | 56 / 0 / 0 | 48 / 0 / 0 | 56 / 0 / 0 | 48 / 0 / 0 |
| rank matrix (host) | 13 / 9 / 9 | 13 / 9 / 9 | 13 / 9 / 9 | 13 / 9 / 9 |
| pair MLP | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 |
| outputs, scales, attachment | 5 / 0 / 0 | 5 / 0 / 0 | 88 / 1 / 0 | 88 / 1 / 0 |
| **total** | **582 / 13 / 11** | 550 / 13 / 11 | **785 / 31 / 11** | 753 / 31 / 11 |

  **Reading:** the pair head and the attachment hold 258 of 582 launches (44 %) on E312 and 341 of 785 (43 %) on E324 s; pyramid + maps head + node head (the part with NO sync: 0 syncs in the table) hold 324 (56 %) and 444 (57 %): that part is capturable as it is. Of the 11 syncs, 9 are the rank matrix (3 device-to-host `.cpu()` and 6 host-to-device uploads of its numpy results), 1 the constants of the soft-argmax (`torch.tensor(..., device=)`) and 1 the kind tensor (`torch.zeros(...).to(dev)`): the last two become device-side constants in a graph-safe copy, the rank matrix is the real host step. E342 part B: about 28 us per launch, GPU timeline equal to host time in every section: the forward is launch bound.
- **Mechanism / what is expected (a guess, not derived):** the GPU executes the 8 GMACs in a few ms (about 3 - 5 ms at a realistic 4 TFLOPs); the other 12 - 20 ms are the host issuing 582 - 785 small launches and waiting at 11 syncs. A graph replay issues the captured launches as one call. **Prediction (numbers, written before the run; a hit is within +-30 % of the ms drop):** g1 on E312 **-5.5 ms** (17.3 -> about 11.8; the pyramid and maps head are 8.3 ms eager), g1 on E324 s **-8 ms** (24.6 -> about 16.6; with the node head 12 ms eager), g2 a further **-3 ms** on both (the pair head 10 ms -> about 4 ms: the rank matrix 2.3 ms stays); the outputs bit-identical for the graph arms (a replay runs the same kernels in the same order), within 1e-4 for arm t.
- **Tolerance of the equality (section 0 rule):** graph arms: the max abs difference of every output tensor of the dump (keypoint positions, directions, probabilities, pair logits, scales, the attachment outputs of E324 s) against the eager forward is **0** on every frame (bit identity; a non-zero difference is looked at, 1e-6 is the largest accepted without a named cause: cuDNN algorithm choice at capture time); arm t: 1e-4. The stored cards are reproduced as stated in the rule.
- **Build (a coder step, new files only, CPU tests with a toy model; then GPU):** (a) `evaluation/jgraph.py`: capture helper (static buffers, warm-up on a side stream, `torch.cuda.graph`, autocast with `cache_enabled=False` inside the capture), the equality dump (eager and graph on all 873 frames, per-frame max abs differences, the delivered-graph cards through the existing scorer `jends_eval.py score`), the timing arms for `jtiming.py` (the same alternating blocks); (b) `training/grid/jends_graph.py`: a graph-safe copy of `JEnds.forward` split at the rank matrix (device constants, no `.to(dev)` of a CPU tensor in the captured parts), with an equality test against `JEnds.forward` on random inputs and on real frames (the guard against a later change of `jends.py`); (c) tests (CPU: the split reproduces the forward; mutation: a missing constant, a changed section boundary); (d) `ar-critic` mode code before any GPU run; the GPU steps through `run.py` (equality dump: 4 checkpoints x 873 frames x 2 forwards, about 0.4 GPU h; timing: one idle window of about 10 minutes, the host load recorded).
- **Risks:** (1) a CUDA graph is a per-process, static-shape object (batch 1, 100 x 100 grids, K = 32: fixed): a different batch needs its own capture; memory of the graph pool (small); (2) the pair head's host rank matrix stays eager (2.3 ms, 9 syncs): graph 2 / 3 do not remove it, a device-side rank matrix (E342 candidate 1, a card of its own) would; so even g2 leaves about 2.3 ms plus the sync latency; (3) the node head has 17 memcopies in the captured region: if they are host-to-device uploads of constants the capture fails and they are made device-side in a copy; (4) `torch.compile` may not work in `gcar` (arm t then not run, named); (5) idle-GPU timing needs a quiet host to resolve a 15 % drop (E342: the host-bound blocks scatter by 25 % on a loaded host): the arms are timed in alternating blocks and the host load is recorded; the rule asks for the paired interval.
- **What it decides:** whether the inference path of the system of record gets about a third of its time back without any change of the outputs, and which graph segments are worth their copy of code. It does not change a number of any card; TEST is not used. **Not claimed:** anything about training speed (the training step is bound by the loader: E342 section 4), about batch > 1, about the pyramid's own design.
- **Price line (for the report to Mirko):** E324 s costs +7.5 ms over E312 for +0.0156 / +0.0146 topo F1 (E324 card section 6): this card asks what part of the ms is launch overhead that can be given back.

## 2. Referee
Pending: a fresh `ar-critic` (mode cards) before anything is built.
