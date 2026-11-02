# E271 Grid chain, first card: a 1 m centre-lane head with a differentiable sparsification reaches the vertex coverage of the prototype at the same vertex budget (stages 1 + 2), and the oracle of the whole chain

Card of 2026-11-03, **revised once after the critic pass of 2026-11-03 (REVISE)**. Mirko's wish of 2026-11-02: "a grid output in 1 m res to output centre lane pixels, then a sparsification, then temporal fusion, then assignment of sparse lanes to map lanes ... E2E differentiable only ... copy and adapt, no complex structures". Design, parts and self-tests: `training/grid/README.md`; picture of the assignment: `tools/viz/assign_view.py`.

**Scope after the revision.** One card = one question that `decide.py` can judge. E271 asks only whether stages 1 + 2 (head and sparsification) work against the system of record, plus the oracle gate of the whole chain (G0: GT through stages 1, 2 and 4 must reproduce itself). The hold (stage 3), the assignment trained jointly (stage 4) and the delivered lane graph are **follow-up cards, written after E271 has a verdict** (their requirements are collected in section 4 so that they are not lost). The critic's points on the hold and on `road_acc` live there.

## 0. The card (what the tools run)
```json card
{
 "id": "E271",
 "card": "",
 "title": "grid head 1 m + differentiable sparsification vs the prototype's vertices at the same vertex budget",
 "kind": "block",
 "deterministic": false,
 "metric": "cov5",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": -0.04, "vs": "r"},
 "arms": {
  "r": {"recipe": "reference, no training: the delivered node ids of the prototype recall_sp0_dp (registry name, 10.9 per 100 m on VAL), scored with the same functions; its card is listed once per seed"},
  "a": {"recipe": "grid head on the 39 frozen GaussianCaR channels (x[:39] of BuilderData; the old dense head's 3 channels are dropped), single frame, stages 1 + 2, losses L_p, L_dir, L_off, L_cov, L_prec, L_budget; read-out threshold chosen on TRAIN for 11 +- 0.5 vertices per 100 m"}
 },
 "control": "r",
 "variant": "a",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "cov5", "vs": "r", "max_loss": 0.03},
  {"metric": "prec1", "vs": "r", "max_loss": 0.02},
  {"metric": "vert100", "vs": "r", "max_loss": 1.0}
 ],
 "budget_gpu_h": 3.0,
 "kill": "G0 fails; loss non-finite; the training loss has not fallen below 0.7 of its start by one third of the steps; a run longer than 2 h of training",
 "evaluate": {"split": "val"}
}
```
How the tools read it: headline `cov5`, band **0.03 declared here** (an equivalence margin: a difference smaller than 0.03 is a TIE by declaration, as E269 declared 0.03 for route success; the project's 0.010 is the noise band of S / Q, and no seed spread of `cov5` is measured yet; it is restated after step 1 and can only be tightened by a written commit before the first training). `decide.py` gives: **KEEP** a - r > +0.03; **TIE** inside +-0.03 (the cheaper chain is taken, see below); **CUT** a - r < -0.03 or a guard violated (`prec1` loses more than 0.02, `vert100` more than 1.0 per 100 m above r, i.e. above 11.9). Passing = KEEP or TIE with the guards held and both seeds on the same side. Verdicts are read from the cards by `decide.py --id E271 --write`; `experiment.py check E271` is ready.

Metrics (all VAL 873 frames via `BuilderData.rows` / `data/splits.json`, never by iterating a stored file that also holds TEST rows; counts added over frames, paired scene bootstrap; point-level functions in `evaluation/lane_task/measures.py` where they exist, the missing ones added there with their self-test in the same commit, not as private metrics; `vert100` is the vertex count per 100 m of GT lane as in `sparse_lanes`):
- `cov5`: share of GT lane length that has a vertex within 5 m along the lane and within 1 m across it (5 m = half the budgeted spacing, so sparseness is not punished; fixed here).
- `prec1`: share of vertices within 1 m of a GT lane whose heading differs by less than 30 degrees (undefined heading, e.g. a junction node of r: counted as matching if any GT lane within 1 m has one of its headings within 30 degrees; stated in `grid_eval.py`).
- **The read-out is a named rule step on the OUTPUT, not on the training path:** the vertices are the cells with `w >= thr` (`soft_lines.vertices`). `thr` is fixed **per model on TRAIN frames** so that the model delivers 11 +- 0.5 vertices per 100 m, which makes the comparison one at an equal budget; `vert100` is reported for every arm anyway and guarded. The frontier (cov5 at 8, 11, 14.3 vertices per 100 m) is reported for a, for r only where it has a knob (its DP tolerance).

## 1. Card
| | |
|---|---|
| lens | system review (rule steps in the delivered graph, `training/architecture.md`) + simplification (Mirko 2026-11-02) |
| group | A3 sparse lane graph, A4 differentiable selection |
| sources | `training/grid/README.md`, BEV `plan/head_design.md` 8.5 (D26) and `extract.py` (`soft_line_map`, copied), E31 / E32 / E37 (dense head + tracer, F1 0.885 VAL picture level, weak forks), E44, E265 / E267 (prototype), E269 (gradient measurements), `sd_lib.py`, `assoc_lib.py` |
| status | revised after the critic pass, second pass pending |

- **Statement:** a 1 m centre-lane head on the frozen camera + radar features, with a differentiable across-line soft NMS and a learned keep gate under a vertex budget, delivers vertices whose coverage of the GT lanes is within 0.03 of the prototype's node ids at the same vertex budget and whose precision is within 0.02, with every stage on the gradient path.
- **Mechanism:** (1) a heat / heading / offset target at 1 m is matching-free and carries the information of the 0.5 m U-Net (E37 needed 0.5 m for the tracer); (2) a vertex is a cell with weight selected by a learned gate under a budget, no top-k, no tracer and no threshold on the training path; the threshold exists only at the read-out; (3) the keep gate can learn where a vertex is worth its cost (bends, junctions, spacing) because the coverage loss uses a kernel of 4 m along the line.
- **Prediction** (VAL, against r): `cov5` a - r about -0.04 (a CUT at -0.03 is a live outcome and is reported as such), `prec1` within -0.02, `vert100` 11 vs 10.9. **These are guesses**: no `cov5` exists for any system; r is measured first (step 1) and the prediction is restated before any training.
- **Decision rule:** the section-0 rule, no hand verdict. A TIE takes the cheaper arm **by ms and parameters** (the project's tie rule); the removal of rule steps is a design argument, not a tie-breaker here, because E271 delivers no lane graph and E272+ still needs a rule at lane end points. A CUT closes the head-and-gate recipe on this input only; the remedies named in advance are a second heading channel (crossings) and the sub-cell offset weight.
- **Test design:** arms r and a, two seeds for a; controls: **null** = a head that outputs the constant mean heat, read out by the same TRAIN-fixed budget threshold (a real null: arbitrary cells, cov5 low, `prec1` low); **oracle G0** (below); **matched reference** r at the same vertex budget, plus the informational row r_poly (r's polylines resampled every 5 m, which counts what the graph covers between nodes; r's DP leaves long straights with end nodes only, so the node-id row under-scores r there: both rows reported, the verdict uses r); **seed twin** = second seed; **equal budget** = a single arm, steps fixed by the first run. TRAIN 8414 frames, nothing selected on VAL: the final checkpoint is evaluated once, the mid-training check of the kill rule uses TRAIN frames only. TEST never.
- **Known risks:** `dec32` and `logit7` of GaussianCaR are in-sample on TRAIN (that is why the dense head's out-of-fold maps existed), so a train-to-val gap is expected and reported; the vertex basis (node ids vs per-piece DP) is the open D25: `vert100` is the node-id basis of the prototype and is the one used; one heading per cell loses crossings; the 1 m cell against the 1 m metric tolerance is saved only by the sub-cell offset.
- **Cost:** about 3 slot-hours (a x 2 seeds, ~1 h each plus the null and read-out fits), CPU evaluation on cached features, 2 threads while another chat trains; coding about 1.5 days. ms (head + sparsify, batch 1, idle GPU, excluding the frozen trunk, against r's assembly of 27.9 ms which excludes it too) needs an idle GPU, which the BEV training currently prevents: if it is not idle at the end the verdict is written "provisional: ms not measured" and the tie-break waits.
- **Kill criterion:** in the block; and G0 / G1 before any training.
- **Inputs:** ready: `BuilderData` (dec32 + logit7 = 39 channels plus the old heat and heading as channels 39 - 41, exact GT distance `gd`, GT heading `gdir`, GT points `g1`), `lane_suite.py`, `lane_task`, the parts of `training/grid/` with self-tests, `runs/eval/recall_sp0_dp/val.json`. To build: below.
- **Starts from:** nothing trained; commit ac19de4.
- **Referee:** pass 1 REVISE (2026-11-03: the rule was not executable, cov5 gameable by dense vertices, G0 inconsistent, two controls missing, hold and road_acc evidence); this text is the revision.

## 2. Detail plan
What would make it fail: the vertex metrics are new, so a bug there would look like a result; the keep gate collapses to all-ones or all-zeros; the ridge picks the wrong side at lane splits; the read-out threshold is fitted on TRAIN where the head is in-sample (a train-to-val shift of the vertex count: `vert100` is guarded on VAL for exactly this).

| step | what exactly | where | done when | budget |
|---|---|---|---|---|
| 1 reference, null, freeze | `cov5`, `prec1`, `vert100` of r, r_poly and the null on VAL; restate the prediction and the band in a commit before training | `evaluation/lane_task/measures.py` (missing functions) + `evaluation/grid_eval.py` (new, thin: vertices in, counts out, rows via `BuilderData.rows`) | r reproduces `vert100` 10.936 of `runs/eval/recall_sp0_dp/val.json`; null `cov5` and `prec1` low; a self-test with a mutation (move the GT lane 2 m: cov5 falls) | 0.5 day |
| 2 **G0 oracle** | (a) GT lanes -> heat from the exact GT distance, heading, offset, **gate = 1 on the cells of the GT lanes' DP vertices** (a named off-path oracle step; DP tolerance as the prototype's), read out with the same `thr` -> `prec1` >= 0.98, `cov5` >= 0.95, `vert100` <= 14.3 on 100 VAL frames; degraded copies (shifted 1 m, dropout, heading flipped) fall as they must. (b) the oracle vertices through the START-value `Assigner` with the exact map: share of weight on the right road >= 0.97, roads compared by the **merged road id** (the `route` copies of `map_tokens` count as their road). FAIL = the representation is the limit: redesign, no training | `training/grid/test_oracle.py` (new, real VAL frames, CPU) | both parts pass | 0.5 day |
| 3 head + losses | U-Net 39 -> 6 ch at 1 m, L_p (BCE + soft Dice vs the heat), L_dir (1 - cos on GT cells, travel direction), L_off (L1), L_cov (noisy-or coverage of the 1 m GT points, kernel 1 m across / 4 m along), L_prec (w x clipped GT distance by `grid_sample`), L_budget (`soft_lines.budget`); **G1: gradient from the total loss reaches every parameter group** (the check of `e2e_gradient_report.py` on the new modules); `x[:39]` slice tested (shape and that the three old-head channels are not read) | `training/grid/head.py`, `training/grid/losses.py` (new) | spec test | 1 day |
| 4 trainer | single frame, adapted from `training/train/dense_head.py`; checkpoint every 400 steps with resume | `training/train/grid_train.py` (new) | `run.py` 200-step smoke: loss falls | 0.5 day |
| 5 train | a, seeds 0, 1, steps fixed by the first run | `run.py --require-card`, planned ledger rows `E271.a_s0`, `E271.a_s1` | done flags | 2 slot-h |
| 6 evaluate | vertex metrics on VAL for a, r, r_poly, null; the frontier of a; the picture level through `lane_suite.py dense:<run>` for the 1 m heat (upsampled to the 0.5 m grid) next to `dense:e37_final`, informational | `grid_eval.py`, `lane_suite.py` | metric cards, paired bootstrap | 0.5 day |
| 7 decide | the rule above | `decide.py --id E271 --write` | verdict in the ledger | |

A critic pass (mode code) on `losses.py`, `grid_eval.py` and `measures.py` additions before the first run: they define what the numbers mean. The live chain and `autoresearch/` are not touched.

## 3. Summary
Open (no run started).

## 4. Requirements kept for the follow-up cards (from the critic pass; each is a card of its own, written after E271)
- **Hold (stage 3):** the evidence for a gain is on degraded frames (E112: night +0.34, blur +0.20, not for lost front cameras), not on `val_clean`: predict no gain on clean frames and test on the failure series too; `BuilderData(with_prev=True)` gives one previous frame, so a chain loader over `poses["prev"]` is needed (new code); train / evaluation gap (4-frame sequences against whole scenes of ~40 keyframes) is a named risk; the control is an `a` with the same sequence batches and the hold off (otherwise the hold is confounded with more frames per step); the variant of that card is the hold arm, so `b - a` is its headline.
- **Assignment trained jointly (stage 4):** `road_acc` is near-guaranteed by geometry (the SD map is built from the GT lanes of the same frame and exactly registered; the START-value assigner already gives 0.98 on a synthetic frame): the control is the START-value `Assigner` on the same vertices, the claim is the gain of learning, and the statement says "exact GT-derived map"; absolute floors are not expressible in `decide.py` (every clause is variant - vs): use a control arm whose card carries the floor, or the verdict is labelled "by hand: <reason>".
- **Delivered graph (E272 or later):** one polyline per lane slot and lane-level edges by a geometric rule at the lane ends (a named output rule step); registration (SoftRegister) and the map-trust output come back; the project's Q / topo F1 are the numbers.
