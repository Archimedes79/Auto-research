# E330 - inventory of the E312 system, what it costs, which options are dead, and the plan to simplify it (D84 item 4, D85)

Card of 2026-11-12 23:40, written by the lane chat (fork D) on Mirko's order of 2026-11-12 ("Versuche das System zu vereinfachen und die Komplexitaet zu reduzieren", D84) with the priority of D85 (E320 first: the simplification arms E331 - E335 run AFTER the E320 result). CPU analysis of files and code, no training. Numbers: `evaluation/jmodel_cost.py` (parameters and MACs, torch FlopCounterMode, batch 1, CPU), `runs/e273/e273_p_s0/timing.json` (B on an idle GPU), the logs of the runs (`runs/<run>/log.txt`), the cards of the runs. The ms of the junction heads are measured by `temp/jr/timing_junction.py` on an idle GPU in the night chain (pending).

## 0. The card (what the tools run)
```json card
{
 "id": "E330",
 "card": "",
 "title": "inventory of the E312 system: parameters, MACs, step times, GT layers, options, hard steps; the removal arms E331 - E335 and their schedule",
 "kind": "variant",
 "deterministic": true,
 "metric": "gmacs_per_frame",
 "direction": "lower",
 "band": 0.05,
 "prediction": {"value": 5.5, "vs": "c"},
 "arms": {
  "c": {"recipe": "the E312 system as it stands: 7.955 GMACs per frame (pyramid 4.677 + junction maps head 3.256 + pair head 0.022), 2.632 M parameters"},
  "s": {"recipe": "the system with the removals that the arms E332 (no paint branch) and E333 (width 40) can adopt: 5.464 GMACs (-31 %), 2.353 M parameters; the other arms save code, GT layers and one training stage, not MACs"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0],
 "rule": [{"metric": "gmacs_per_frame", "vs": "c", "min_effect": 0.0}],
 "budget_gpu_h": 0.0,
 "kill": "the MACs of the blocks do not add up to the whole forward within 2 % (the counter misses a block)",
 "evaluate": {"split": "val"}
}
```
By hand (a measurement and a plan, label "by hand: analysis"): sections 1 - 7 are the result; the arms of section 6 each have their own card with a rule written before the run.

## 1. Blocks, parameters, MACs, time (E312, batch 1)
| block | params M | GMACs | ms (idle GPU, batch 1, bf16) | differentiable | remark |
|---|---|---|---|---|---|
| pyramid (B's, frozen in the pair stages) | 2.097 | 4.677 | 4.3 (B timing) | yes | shared with B |
| junction maps head | 0.444 | 3.256 | pending (`timing_junction.py`) | yes | 41 % of the system's MACs |
| - context (p2 + 4 dilated residual blocks at 2 m) | 0.305 | 0.760 | | yes | |
| - 1 m branch (m1 + head1: zone, core, entry / exit heat, headings) | 0.102 | 1.019 | | yes | |
| - 0.5 m paint branch (m05 + head05: stop lines, crossings) | 0.037 | **1.477** | | yes | 45 % of the head, 18.6 % of the system; the only reader is the evidence along the curve |
| keypoints (3 x 3 peaks, soft-argmax) | 0 | ~0 | | soft-argmax yes; **top-K = 32 is an index choice (hard, named, D7)** | the gradient flows through the values |
| rank / group bookkeeping (`jgt._groups`, `_ranks`) | 0 | 0 | numpy loops per batch | **hard bookkeeping, no gradient needed (named, D82)** | a function of the keypoint positions |
| pair head | 0.090 | 0.022 | pending | yes | keypoint token + 2 set blocks 0.071 M / 0.002; pair MLP 0.014 / 0.014; handle-scale head 0.005 / 0.005 |
| connectors (Hermite curve, learned handle scales) | 0 | ~0 | | yes | |
| **whole junction system** | **2.632** | **7.955** | | | the heads add 0.534 M (+22 %) and 3.278 GMACs (+70 %) to the pyramid B already runs |
| B: node head / links / assignment (map encoder + assignment) | 0.099 / 0.117 / 0.123 | not counted | 2.9 / 5.4 / 8.3 | soft (absorbing walk for the sections, 2.8 ms) | B total 24.2 ms with the pyramid 4.3 ms |
| read-out: p >= 0.1 threshold, snap, close-node step 1.5 m (E275), Douglas-Peucker 0.3 m | 0 | 0 | CPU python, not timed yet | **hard (named)** | section 5 |

## 2. Training steps (measured, `runs/<run>/log.txt`, seconds per step at step 3000; the machine load differed between the runs)
| stage | run | s / step | what it contains |
|---|---|---|---|
| maps (E307 recipe, 3000 steps) | e307_n_s0 | 1.01 - 1.10 | maps head only, frozen pyramid, host-bound |
| pair head, maps frozen, no scales | e287_p2_s0 | 0.82 | |
| pair + maps jointly (`--e2e`), no scales | e305_e_s0 | 1.07 | |
| pair + maps jointly + handle scales (the E312 recipe) | e312_p_s0 / e298_w_s0 | 1.38 / 1.42 | |
| pair head, frozen maps, scales | e303_s_s0 | 1.8 - 1.9 | measured while other jobs ran: not comparable |
The step is host-bound (GPU about 33 % busy, `training_speedup` notes): the loader workers build the lane GT targets per frame. **Where the extra time of the scales goes:** the fitted handle scales are computed in the loader (`qjunction.fit_pair_scales`, a grid search of 8 x 8 curves per true pair, about 0.15 s per frame, `lane_gt_data.QData`): with 8 frames per step and 3 workers that is about +0.3 s per step; the cleaner pair of runs (E312 against E305, both joint) shows +0.31 s = +29 % of the step. 3000 steps of the pair stage = 1.15 h (E312) to 1.6 h under load.

## 3. Can the handle-scale fit be cached? Yes (report and proposed patch, NOT applied)
`fit_pair_scales(jg)` reads only `jg['pairs']`, `jg['ent']`, `jg['ext']`, `jg['pair_pts']` and a constant grid `(0.05 .. 1.25)^2`, with `jg = jgt.junction_gt(g)` of the lane GT graph `g`: it does not depend on the paint, the map, the augmentation, the model or any random number, so the result is a pure function of the frame's lane GT (and so of the lane signature `9ad1d4ce30`). Per-frame cache, saving about 0.3 s per step (-22 % of the 1.38 s) for every `--scales` arm (E317 - E320 and all arms of this series): 15 minutes less per run of 3000 steps. Patch (to be applied by the owner of those files, none of them touched here):
1. `training/grid/qjunction.py`: `def fit_pair_scales_cached(token, jg)`: load `data/prep/pair_scales/<lane signature>/<token>.npy` (float32 (P, 2)) if it exists and has `len(jg['pairs'])` rows, else `fit_pair_scales(jg)` (and store it when the folder is writable).
2. `training/data/lane_gt_data.py` line 115: `QJ.fit_pair_scales(jg)` -> `QJ.fit_pair_scales_cached(token, jg)`.
3. a one-off prebuild `data/prep/pair_scales.py` over the TRAIN (8414) and VAL (873) tokens (23 minutes in one process, about 6 minutes in four); derived data, never in git, deletable by `package.py`.
4. a test: cached equals computed bit for bit on 200 frames; the training loss of 20 steps is identical with and without the cache.
(Not in the removal arms: the cache changes no number, only time.)

## 4. GT layers the E312 recipe needs, and what the arms drop
| layer (loader `QData(junction=True, pscale=True)`) | used by | dropped by |
|---|---|---|
| `jz` zone, conflict core (1 m, 2 channels) | maps loss (zone / core); evidence along the curve | E332 arm b (with the evidence: E331) |
| `jheat` entry / exit heat (sigma 1.0 m), `jdir` headings | maps loss; the keypoints | never (the ends are the point) |
| `jpaint` stop lines, crossings (0.5 m) + the paint sidecars of `data/prep/junction_paint.py` for TRAIN and VAL | paint loss; evidence | E332 (both arms): no sidecars needed, no 0.5 m raster |
| `ent`, `ext`, `pairs`, `pdrv` (the end table of `qends`) | pair loss, order features, driven weight | never |
| `pscale` (fitted handle scales) | scale regression (E303) | E335 would; closed by data (section 6) |
| `ppts` (GT connector polylines, 16 points) | E317 / E320 geometry loss only | |
The loader still rasterises zone, core and paint for a run that does not use them (CPU time per frame); a follow-up after E332: skip them in `QData` when the head says so (a file outside this card's list: `training/data/lane_gt_data.py`).

## 5. Hard steps and the differentiable replacement (D84 point 3)
| step | where | replacement |
|---|---|---|
| top-K = 32 peak choice of the keypoints | `JEnds.forward` | none planned: an index choice, the gradient flows through the values (D7); a learned query set would be the alternative (deferred with the unstructured head) |
| rank / group bookkeeping of the keypoints | `JEnds.rank_matrix` | none: features without gradient; if the order rule should be learned instead, it is the unstructured-head question |
| probability threshold p >= 0.1 | read-out | none needed: the planner reads the probability |
| snap of the connector ends (E311) | `qends_eval.add_connectors` | **E320** soft attachment (card E320, running first by D85) |
| close-node step 1.5 m (E275) | read-out of B | Matrix-NMS (E288: TIE with the hard step on the stored dumps, CPU replay); the layer inside the network: card E292, not built |
| node / link / section thresholds e >= 0.5, p >= 0.5 | read-out of B | the probabilities are the output; the thresholds stay a consumer's choice |
| Douglas-Peucker 0.3 m output stage | `sparse_lanes` output | none planned (vertex budget of the planner) |

## 6. The removal arms (one thing at a time out of the E312 recipe; each judged on BOTH read-outs, D85)
Rule of every arm (written before the first run, `evaluation/jrule_removal.py`): the removal is ADOPTED when the delivered graph of the arm loses at most 0.01 topo F1 against the control (E312 of the same seed) on BOTH seeds and on BOTH read-outs, **unsnapped p >= 0.1 (the differentiable read-out, counts)** and **snapped p >= 0.1 (the rule step of E311, reported and must hold)**, with F1@1.0 (<= -0.01), dup_frac (<= +0.01), vertices per 100 m (<= +1.0) and junction F1 (<= -0.03) held; inside the band the cheaper variant wins. If E320 is KEEP the arms are re-based on its read-out (`temp/e330/recipe.sh`, below).
| id | removal | what it saves | cost | READY |
|---|---|---|---|---|
| E331 | the evidence along the curve (S1, `--no-evidence`) | 6 inputs of the pair MLP, the Hermite sampling of the three maps per pair, code | pair stage 1.15 - 1.6 h per seed | `temp/e331/READY` (prefix `e331_e_s`) |
| E332 | the auxiliary maps (S3): arm a no 0.5 m paint branch, arm b also no zone / core supervision (needs E331's no-evidence) | **-1.48 GMACs (-18.6 % of the system)**, the paint sidecars, the paint loss; b: two GT layers | maps stage 0.9 h + pair stage; arm b only if a is adopted | chain `temp/e332/chain.sh` |
| E333 | the width of the maps head (S4): 40 instead of 64 | -1.29 GMACs (-16 %) | maps stage + pair stage | chain `temp/e333/chain.sh` |
| E334 | the separate maps stage (S5): one joint stage from the untrained head, 3000 steps (o3), 6000 conditional (o6) | a whole training stage and its checkpoint hand-over | pair stage only (1.15 - 1.6 h per seed); o6 2.3 - 3.2 h | `temp/e334/READY` (prefix `e334_o_s`) |
| E335 | the handle scales (S2) | scale head, the fit in the loader | none: **closed by data**: the chord read-out of the same E312 pairs loses -0.0118 / -0.0126 topo F1 (snapped), more than the band: not adopted, not run; the cache of section 3 removes the cost instead | no run |
Savings combined (E332 a + E333, if both adopted): 5.464 GMACs, 2.353 M parameters (-31 % / -11 %); the maps head 0.766 GMACs (-76 %). The stack must be re-checked as one arm (removals need not add up): E336, written when two arms are adopted.

## 7. Dead options (listed for a later removal after an `ar-critic` pass; NOT deleted: history keeps them, old checkpoints need some of the flags)
| file | option | origin | verdict | used by the E312 recipe? |
|---|---|---|---|---|
| `training/train/jends_train.py` + `jends.py` | `--capacity`, `JEnds(capacity)`, `sinkhorn_plan`, `degree_classes`, the degree heads (about 60 lines) | E302 | card kept, never run | no |
| same | `--ntok`, `JEnds(ntok)`, B's frozen node head in `JEndModel` | E308 | TIE, not in the final system; **E320 reuses the node-head loader: keep that part** | no |
| same | `--raster exact` | E306 | card kept, never run | no |
| same | `--no-order` | E287 arm p0 | control of the order rule (order helps) | no |
| `training/train/jmaps_train.py` | `--joint` (pyramid trained with the head) | E286 arm jj | stopped at step 1000 (11 s per step) | no |
| same | `--far-w` (far-range weighted heat), `--raster exact` | E304 / E306 | stopped / never run | no |
| `training/grid/jmaps.py` | ctx `none`, ctx `attn` (a self-attention layer at 4 m, 0.52 M parameters) | E286 arms j0 / j2 | j1 (`dil`) is the base | `dil` only |
| `evaluation/jends_eval.py`, `qends_eval.py` | `--snap-all`, `--snap-guard`, `--snap-dedup`, `--snap-nogeom`, `--snap-both` | E311 refinement controls | the verdict was the plain `--snap`; they go when E320 decides the snap | no |
| same | `--chord` | E303 control (scales worth +0.012) | keep until the E320 verdict | no |
| `training/grid/qendnet.py` | the `EndHead` class (E284 end head on node tokens) | E284 | TIE; the helpers (`match_keypoints`, `end_counts`, `_dual_softmax`, MATCH_*) are live | helpers only |
| `evaluation/` | `jrule_control.py`, `jdiag.py`, `jsplits.py` | E287 / E300 / E310 analyses | analysis tools of cited cards: keep | no |
Not audited here: `training/grid/chain*.py`, `hold.py`, `assign.py`, `qassign.py`, `map_sim.py` (the grid chain E272 and the map assignment of B, 1.0 kLOC), `qmnms.py` (Matrix-NMS of E288, used by the replay).

## 8. Schedule proposal (after the E320 result, D85)
GPU hours from the measured step times (pair stage 3000 steps 1.15 - 1.6 h per seed, maps stage 0.9 h, post-processing 0.25 h with the dumps one at a time; both seeds in the two slots at once, so wall = per-seed time):
| order | arm | wall h | slot hours (2 seeds) | window |
|---|---|---|---|---|
| 0 | E320 (lane lead, first) + the booked day11 until about 04:30 Fri | - | - | |
| 1 | E331 | 1.65 | 3.3 | Fri 08:30 - 10:10 |
| 2 | E332 arm a (maps 0.9 + pairs 1.15 - 1.6 + post) | 2.55 | 5.1 | Fri 10:10 - 12:45 |
| 3 | E333 | 2.4 | 4.8 | Fri 12:45 - 15:10 |
| 4 | E334 o3 | 1.65 | 3.3 | Fri 15:10 (start before 15:30) - 16:50 |
| later | E332 arm b (only if a adopted) 5.1; E334 o6 (only if o3 is within 0.03) 6.0; E336 the stack | 2.55; 3.0; 3.0 | | Fri night 22:00 - |
Total 16.5 slot hours = 8.25 h wall for the four main arms (one day window; each start inside the window rule: nights 22:00 - 05:00, Mon-Fri 08:30 - 15:30). By value per hour E332 a (-19 % MACs) and E334 o3 (a stage less) come first, E333 (a pure width cut) last; Mirko's order S1, S3, S4, S5 is the default of the chains. **Re-basing on E320 (D85):** if E320 is KEEP its recipe replaces E312 as the control: change `temp/e330/recipe.sh` only (`RECIPE_FLAGS` += E320's training flags `--geom-w 0.5 --attach`, `CONTROL_PREFIX=e320_a_s`, `RULE_READOUTS="attach"`, `READOUT_EXTRA="--attach"`), nothing in the start scripts; the pair stages get slower by the attachment and the geometry loss (E320 card: +5 - 10 %, 3000 steps 1.7 - 1.8 h): add about 8 % to the wall times above; the maps stages of E332 / E333 and the init of E334 do not change. If E320 is a TIE or a CUT the arms run on the E312 recipe as written.
