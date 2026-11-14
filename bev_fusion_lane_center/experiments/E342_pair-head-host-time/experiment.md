# E342 - where the junction forward spends its milliseconds: the pair head's host bookkeeping, vectorised (part A, CPU, bit for bit) and an idle-GPU profile of the pair head (part B)

Card of 2026-11-14 16:10, written by the lane chat on Mirko's order of 2026-11-14 (D92: do not narrow the heads, "gucke, wie man sonst Rechenzeit sparen kann"). A measurement that already exists changes the question. `temp/jr/timing_junction.json` (2026-11-13 04:16, RTX 3060, batch 1, bf16, idle GPU, median of 60 frames of E312 seed 0): pyramid 4.55 ms, **junction maps head 2.82 ms (3.256 GMACs), pair head 9.72 ms (0.022 GMACs)**, the whole forward 17.09 ms. The width of the maps head is not where the time is: halving it would save about 1.4 ms, removing its 0.5 m branch (E341) about 1.3 ms, and the pair head, which has next to no multiply-adds, costs more than three times the whole maps head. What it costs is host work and kernel launches, not MACs.

## 0. The card (what the tools run)
```json card
{
 "id": "E342",
 "card": "",
 "title": "the host-side group bookkeeping of the pair head (`jgt._groups`, called by `JEnds.rank_matrix` for every frame and by the loader for every GT frame) vectorised: the same groups, less time; part B profiles `JEnds.forward` section by section on an idle GPU",
 "kind": "variant",
 "deterministic": true,
 "metric": "ms_per_frame",
 "direction": "lower",
 "band": 0.2,
 "prediction": {"value": -1.4, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = `jgt._groups` as in HEAD (python double loop over the pairs of ends)"},
  "v": {"recipe": "the same function, all pairs at once in numpy (float32 / float64 as the input), the components by repeated squaring of the adjacency matrix"}
 },
 "control": "c",
 "variant": "v",
 "pairing": "position",
 "seeds": [0],
 "rule": [
  {"metric": "ms_per_frame", "vs": "c", "min_effect": 0.5}
 ],
 "budget_gpu_h": 0.0,
 "kill": "one set of ends gets other group labels than the loop gives (tested on the 3 300 real sets of the E312 VAL dump and on random sets), or the new function is not faster on the real sets",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: measured"): part A is a deterministic change judged by equality (every label of every set equal to the loop's) and by the CPU time of the function on the real sets; part B (a run on the idle GPU, Sunday night) decides what comes next and has its own rule below. Nothing here is trained; TEST untouched.

## 1. Part A (CPU, measured 2026-11-14 16:10, installed by commit 27fee60b at 21:48 after two critic passes of the patch): `jgt._groups` vectorised
`rank_matrix` (`JEnds.forward`, order rule on) copies the keypoints to the host (a full GPU sync), groups the valid ends of each kind by single linkage (`jgt._groups`: heading within 25 degrees, 6 m along / across) and ranks them inside a group (`jgt._ranks`), then moves five small tensors back; named hard bookkeeping, no gradient (D82). On the 3 300 (frame, kind) sets of the E312 seed 0 and 1 VAL dumps (`runs/e312_p_s<S>/ends_val.pkl`, 13.0 valid ends per set on average, at most 32): **the python loop 0.80 ms per set, the vectorised version 0.095 ms (8.4 times faster)**, groups plus ranks 1.09 ms against 0.37 ms per set (ranks alone 0.30 ms). **Labels (the critic's run on the installed text):** 0 mismatches on 3 322 non-empty real sets of the E312 VAL dumps (3 300 with n >= 2 and 22 with n = 1; the dump holds 3 492 sets, 170 with n = 0), `JEnds.rank_matrix` bitwise equal on 1 746 frames, 0 mismatches on 20 000 random clustered float32 sets and on 3 304 real float64 GT sets of the loader path, and on an adversarial suite (exact-boundary grids, degenerate rows, chains and shuffled paths up to 32 ends, f32 and f64). **Limit of 'bit for bit':** in float64 the labels are identical on every tested input, but the arithmetic is not bit-identical at inputs that sit exactly on a threshold (OpenBLAS ddot uses FMA: 1 ulp differences in the dot product and the norm; on 40 000 constructed pairs at exactly 25.0 degrees +-3 ulp, 498 / 529 differ in the angle test; real data sits there with a probability of about 1e-15); the old float64 result is itself CPU- and BLAS-dependent at that knife edge. The selftest in `jgt.py` compares with the loop kept as the oracle and pins the edge semantics (boundary `<=`, a zero heading, float32 headings with self-dot above 1, a shuffled path of 32 ends): it is an equivalence test, so it passes with the old code too. **Speed by set size** (the critic's loaded box, old -> new per set): 2 ends 15 -> 53 microseconds (slower), 3 ends 56 -> 79, 4 ends 89 -> 61, 13 ends 1 488 -> 88, 32 ends 10 700 -> 141; groups 1.17 -> 0.105 ms, groups plus ranks 1.91 -> 0.63 ms, one `rank_matrix` call 5.5 -> 2.6 ms per frame. Per frame (both kinds): about 2.2 ms before, 0.75 ms after = **-1.4 ms of the 17.09 ms** (8 %); a bigger gain than any single MAC saving of the maps head. The loader calls the same function for every GT frame: a little less host time in training, nothing in the numbers.
**Prediction (a number):** -1.4 ms per frame on the CPU part of the forward (the measured time of `rank_matrix`'s groups); on the idle GPU the same, because the work is the host's. Code: `training/grid/jgt.py` (`_groups`, selftest), README row; no other file (committed 27fee60b).
**Mixed versions (stated, as the critic asked):** the E317.g_s1 training and the evaluation jobs that were running at 21:48 keep the old code in memory; later chain stages (post_pair, E318.u, E319, the reads) and every run that starts after 21:48 use the new one; the outputs are identical (above), and each run's `lcr/<id>` tag shows the code version it started from. The 17.09 ms of `temp/jr/timing_junction.json` (2026-11-13) predates the patch: the first idle-GPU timing after the install is the new baseline, expected about 15.7 ms, and the two are not comparable across the install.

## 2. Part B (idle GPU, Sunday night, a measurement): what the other 7 ms of the pair head are
After part A the pair head still costs about 8 ms (9.72 - 1.4) with 0.022 GMACs: kernel launches (the forward is about 150 small tensor operations: peak search, top-K, unfold / gather for the soft-argmax, the K x K x 12 Hermite samples of the evidence, the pair MLP) and the host sync of `rank_matrix`. A profile (CUDA events around each section plus the number of kernels per section, `torch.profiler`) on the idle GPU after the E334 / E325 / E341 chains tells which section it is, one number per section. Candidates after the profile, each its own card with the written rule "outputs equal (max abs difference of the pair logits 0 or the stored cards reproduce), ms per frame lower by a paired interval that excludes 0": (1) the group and rank bookkeeping on the GPU (no sync: the same boolean-matrix squaring in torch, the ranks by a masked sort); (2) CUDA-graph capture of the junction forward at batch 1 (the same kernels, so the same numbers; the graph needs fixed shapes: K = 32 and 100 x 100 already are). Not now: anything that changes the network.
Part B also times the maps head of the E341 checkpoint (`runs/e341_p_s0/model.pt`, no 0.5 m branch) against E312's (`runs/e312_p_s0/model.pt`), in the same process, blocks A B | B A: the ms that E341 saves, which E332 and E341 only estimate.
**GPU start rule:** a measurement of an idle GPU is a GPU start: only nights 22:00 - 05:00 and Mon - Fri 08:30 - 15:30; the chains of tonight (E334, E325, E341) must be done (`temp/day14/DONE_e334`, `temp/day16/DAY16_DONE`, `temp/day18/DAY18_DONE`) so that the timing is on an idle GPU.

## 3. Referee
Part A: a fresh `ar-critic` (code) before the commit of `jgt.py`, with the equality test run by the critic. Part B: a card per candidate and a critic pass before any code.
