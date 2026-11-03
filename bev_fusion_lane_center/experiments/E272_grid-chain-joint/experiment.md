# E272 The grid chain as ONE differentiable function: blocks fitted from the E271 head, then a joint fine-tune

Card of 2026-11-03. Mirko's instruction after E271: "do not create multiple cards, try to establish an E2E differentiable function once, start with where you are and refine the blocks for a joint fine tune." This is that one card: the chain frames -> head -> sparsify -> hold -> assignment to the SD map as a single module (`training/grid/chain.py`) with a single loss, trained in two phases (the blocks with the E271 head frozen, then everything jointly). Follow-up work (the delivered lane graph with lane ends and edges, degraded-frame tests, TEST) is not a card of its own here: it is named as a gap in the summary when this one is judged.

## 0. The card (what the tools run)
```json card
{
 "id": "E272",
 "card": "",
 "title": "grid chain as one differentiable function (hold + assignment, joint fine-tune) vs the E271 head alone at the same vertex budget",
 "kind": "block",
 "deterministic": false,
 "metric": "cov5",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.02, "vs": "a"},
 "arms": {
  "a": {"recipe": "control, no new training: the E271 head alone (single frame, stages 1 + 2), cards runs/e271/e271_a_s0 and e271_a_s1, read out at 11.2 vertices per 100 m of GT lane"},
  "j": {"recipe": "the whole chain, start from the E271 head of the same seed; phase A (600 steps, lr 1e-3) the head frozen, hold and assigner fitted; phase B (1500 steps, lr 3e-4) everything fitted; chains of T = 3 frames, batch 4; loss = per-frame head loss + sparse-map terms on the fused map + assignment loss; read out from the FUSED map at 11.2 vertices per 100 m (threshold from 300 TRAIN chains)"}
 },
 "control": "a",
 "variant": "j",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "cov5", "vs": "a", "max_loss": 0.03},
  {"metric": "prec1", "vs": "a", "max_loss": 0.02},
  {"metric": "vgt100", "vs": "a", "max_loss": 1.0}
 ],
 "budget_gpu_h": 3.5,
 "kill": "a loss that is not finite; the fused-map `cov5` of phase A more than 0.10 below the head's own at the end of phase A; a run longer than 2.5 h",
 "evaluate": {"split": "val"}
}
```
Verdict rule, written before the run and applied by hand like E271's (`decide.py` cannot read vertex-level cards: `card.validate` demands the project's graph headline metrics; label "by hand: <reason>"): headline `cov5` j - a on the mean of the seeds, band 0.03 declared (an equivalence margin). **KEEP**: > +0.03 with the guards held. **TIE**: inside the band with the guards held: the chain is accepted as THE end-to-end function (it adds the hold and the trained assignment at no loss; the cheaper-arm tie rule does not apply because the point of the card is that the function exists and is trained as one). **CUT**: < -0.03 or a guard violated (`prec1` loses more than 0.02, `vgt100` gains more than 1.0 per 100 m); a CUT sends the phases back to the blocks (named below), not to a new card. Both seeds on the same side. Prediction: +0.02 (the hold adds a little evidence on clean frames, the joint terms pull the head towards the fused objective).

Reported next to the rule, not judged by it (declared here): (1) **hold on / off** on the same weights (`j_nohold`): what the hold contributes on clean frames (E112 expects its gain on degraded frames, not here); (2) **assignment**: the trained assigner against the START-value assigner on the same read-out vertices, `road_acc` (plausible road of the nearest GT lane, within 1 m and 30 degrees) and `dust_off` (vertices more than 2 m from every lane that go to the dustbin); "the assignment learned something" only if `dust_off` rises by 0.10 or more over the start values, otherwise it is geometry alone (the SD map is GT-derived and registered exactly, so `road_acc` is near-guaranteed: the smoke test already gave 0.99 at the start values); (3) the road mass on the plausible roads in training; (4) ms of the whole chain per frame (idle GPU, batch 1, hold with one previous frame).

## 1. Design (what the one function is)
```
x_t (39 ch, 0.5 m) --head--> 6 ch @ 1 m --sparsify--> sparse_t --\
                                                      sparse_t-1 warped by the ego motion --hold--> fused_t  --assigner--> P(road | cell), lateral offset, lane slot
```
- Chain loader (`chain_data.py`): T = 3 frames of a scene through `poses["prev"]`, the ego step of each pair (checked on real VAL chains: the lane heat of frame t-1 warped with it overlaps the lane heat of frame t with IoU 0.94, 0.75 unwarped), the SD map of the last frame, and the plausible road labels of its GT points (98.7 % of the points have one).
- Loss (`chain.py`): mean per-frame head loss of E271 (`losses.grid_loss`) + the sparse-map terms (offset, coverage, precision, vertex budget) on the FUSED map + the assignment loss (probability mass on the plausible roads of the nearest GT point, dustbin for cells without a lane within 2 m), the cell weight detached in that term.
- Differentiable path: no argmax, no matching, no top-k, no threshold on the training path (the assignment skips cells with a weight below 1e-3: a numerical cut of < 0.1 % of the loss weight). Gate **G2** (self-test `chain.py`): the loss reaches the head, the hold's gate and the assigner's scalars; the assignment loss alone moves the head; with the head frozen no gradient reaches it; without predecessors the fused map is the current one. Passed.
- **Named rule steps / gaps:** the read-out (threshold + 3 x 3 maximum) on the OUTPUT; the lane ends and lane-level edges of a delivered graph (none here: the project's Q, topo F1 do not exist for this chain); no registration of the map (the map is exact); ego motion from the stored poses (not estimated); the hold is tested on clean frames only; the SD map is GT-derived.
- Pictures: `tools/viz/grid_view.py` (head, fused map, SD map and the system side by side in road colours); result page `tools/viz/vertex_report.py`.

## 2. Detail plan
| step | what | where | done when |
|---|---|---|---|
| 1 data + warp check | chain loader, ego step, road labels | `training/grid/chain_data.py` | warp IoU 0.94 vs 0.75 on 12 VAL chains; labels 98.7 % (done) |
| 2 chain + loss + G2 | one module, one loss | `training/grid/chain.py` | self-test (done) |
| 3 trainer + scoring | two phases, hold on / off, both assigners | `training/train/chain_train.py`, `evaluation/chain_eval.py` | smoke through `run.py` (done: 20 + 20 steps, 24 frames) |
| 4 train | seeds 0, 1 through `run.py --require-card` | rows `E272.j_s0`, `E272.j_s1` | done flags + cards |
| 5 score, pictures, verdict | cards, paired bootstrap against the E271 cards, side-by-side assignment pictures and videos, result page | `chain_eval.py`, `grid_view.py`, `vertex_report.py` | verdict written by hand with the rule above |

## 3. Summary
**Interim, seed 0 only (2026-11-04): the chain is one trained differentiable function and matches the E271 head; verdict TIE on seed 0, the card's verdict needs seed 1, which waits behind the BEV system's t21 runs on the shared GPU** (`temp/chain/start_j1_gated.sh`; the gate opens when the BEV queue has run, fallback after 04:00).

| VAL, 873 frames, 11.2 vertices per 100 m | cov5 | prec1 | vertices / 100 m |
|---|---|---|---|
| a = E271 head alone, seed 0 (control) | 0.7212 | 0.8711 | 10.279 |
| **j = chain, hold on, seed 0** (E272.j_s0) | **0.7254** | **0.8756** | 10.279 |
| j with the hold off (same weights) | 0.7254 | 0.8756 | 10.279 |

- **Rule (seed 0):** `cov5` j - a = +0.004, paired scene bootstrap 95 % [+0.001, +0.007] (above zero but far inside the 0.03 band), `prec1` +0.0045 [+0.001, +0.008], `vgt100` -0.0004; guards held: **TIE** by the written rule (prediction was +0.02, the gain is smaller). Accepted as the end-to-end function by the card's tie rule once seed 1 agrees.
- **The hold learned to switch itself off on clean frames:** the trained gate has mean 0.001 and maximum 0.22 on real VAL chains, so the fused map equals the current sparse map to 0.008 and the hold-on and hold-off cards are identical (same threshold to 6 digits). This is a result, not a bug: the weights of a fused map add vertex mass under the budget and a clean frame needs no past (E112 expects the gain on night and blur frames). The hold is therefore built, differentiable and trained, but **its value is untested**: that needs a degraded-frame evaluation (not part of this card).
- **Assignment:** `road_acc` (plausible road, 39,152 vertices within 1 m of a lane) trained 0.9837 against start values 0.9842; `dust_off` (841 vertices farther than 2 m from every lane) 0.486 against 0.465 (+0.02, below the declared +0.10): the assignment is geometry alone, nothing learned beyond it (the SD map is GT-derived and exact). In the pictures the trained assigner colours a corridor with the road the map draws there; off-lane vertices still partly go to roads.
- **What the joint fine-tune changed:** the head is slightly better in precision and coverage at the same budget; the phases reached loss 1.27 (blocks, head frozen) and 2.74 (joint, includes the three per-frame head losses).
- **Speed:** training 0.6 s/step (blocks) and 2.1 s/step (joint, batch 4 chains of 3 frames); ms per frame of the chain not measured (the GPU is not idle).
- **Honest negatives / gaps (named):** no delivered lane graph (no lane ends / lane-level edges), so Q and topo F1 do not exist for this chain; the read-out rule step on the output; ego motion from stored poses; clean frames only; the SD map is GT-derived and registered exactly (a real map would be neither); a learned threshold for the read-out is in-sample on TRAIN chains.
- **Pictures:** `runs/grid_view/e272_j_s0/` (scene 7 and 23: SD map and system side by side in road colours); `tools/viz/grid_view.py e272_j_s0 --scene 7 --video`.
