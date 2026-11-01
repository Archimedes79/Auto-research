# E247 Learned input-gated evidence update (minLSTM-type) in place of the fixed log-odds time recursion

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E247",
 "card": "H19.F1-1",
 "title": "learned input-gated evidence update (minLSTM-type, match evidence as input) replaces the fixed log-odds time recursion of the champion; builder frozen",
 "kind": "block",
 "deterministic": false,
 "metric": "R@1_far",
 "direction": "higher",
 "band": 0.01,
 "prediction": {
  "value": 0.02,
  "vs": "c0"
 },
 "variant": "a",
 "arms": {
  "a": {
   "recipe": "gate trainer (new, pattern of p17_gate.py): builder e123_D7_r3_f1 frozen, MinGateCell on the matched nodes, zero-initialised last layer, 1500 steps, lr 1e-3, loss BCE(fused pi, on-lane) + graph_losses on the fused pi; seeds 0 and 1; fallback 2-fold cross-fit over the 22 VAL scenes if the frozen builder is over-confident on TRAIN",
   "base": "e123_D7_r3_f1",
   "seeds": [
    0,
    1
   ],
   "planned_names": [
    "champion_gate",
    "champion_gate_s1"
   ]
  },
  "c1": {
   "recipe": "same trainer with two global scalars (a, b) instead of the conditioned gate (equal budget)",
   "base": "e123_D7_r3_f1",
   "seeds": [
    0
   ],
   "planned_names": [
    "champion_gate_scalar"
   ]
  },
  "c0": {
   "recipe": "the champion with the fixed rule (a = 0.9, b = 1.0), no training",
   "names": [
    "champion"
   ]
  }
 },
 "control": "c0",
 "rule": [
  {
   "metric": "R@1_far",
   "min_effect": 0.02
  },
  {
   "metric": "topo_F1",
   "min_effect": 0.008
  },
  {
   "metric": "Q",
   "max_loss": 0.005
  },
  {
   "metric": "R@1_far",
   "vs": "c1",
   "min_effect": 0.008,
   "note": "conditioning on the match evidence matters, not only the constants; ms per frame +0.5 at most is checked in the timing run"
  }
 ],
 "seeds": [
  0,
  1
 ],
 "budget_gpu_h": 1.0,
 "kill": "fixed rule gains less than +0.010 R@1_far over no recursion (step 0); identity test fails; cell stays within 0.05 of its initial weights or mean fused pi of matched nodes > 0.97; after seed 0 of a gain below +0.005 over c0 and below c1",
 "evaluate": {
  "split": "val",
  "compare": "c0"
 }
}
```


One file per experiment; it grows from card (P2) to detail plan (P3 start, block experiments only) to summary (P3 end). Machine-readable part: `experiment.json` next to this file.

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E247 / H19.F1-1 / A learned gate on the warped previous validity (log-odds), fed with the match evidence, replaces the constants a = 0.9, b = 1.0 of `fuse_prior`; builder frozen, gate trained 1500 steps, champion f1 288 / 144 |
| kind | block (new module `MinGateCell` in `training/models/p16_time.py`, hook in `Builder.fuse_prior`, small trainer in `training/train/`) |
| deterministic | false (gate training seeds 0 and 1) |
| metric, direction, band | R@1_far, higher, 0.010 (also topo_F1, lanes_found, Q, J, ms per frame) |
| prediction | R@1_far +0.020, topo_F1 +0.010, lanes_found +0.015, Q +0.004 against arm c0 (the champion with the fixed rule) |
| arms | a: evidence-conditioned gate (a_t = sigmoid(W_a x), b_t = softplus(W_b x)), seeds 0 and 1; c1: two learned scalars (a, b) trained the same way (equal budget, no conditioning); c0: the champion, fixed rule (no training; registry `champion`) |
| control | c0 (judged), c1 is the equal-budget control for the "conditioning matters" clause |
| rule | R@1_far min_effect 0.020 vs c0; topo_F1 min_effect 0.008 vs c0; Q max_loss 0.005 vs c0; R@1_far min_effect 0.008 vs c1 (ms per frame +0.5 at most is a check, from the timing run) |
| budget_gpu_h, kill | 1.0 (3 x 0.3 gate training incl. evaluation, 0.1 diagnostics), see section 1 |
| evaluate | split val (873), compare c0 |

## 1. Card (written in P2, checked by ar-referee)
| | |
|---|---|
| lens | foundations (F1 building block: gated linear recurrence on the vertex memory) |
| group | A5 temporal fusion (gap 1 coverage at range through evidence accumulation; gap 2 connectivity through consistent validity) |
| sources | deep/2410.01201_mingru.md (sections 5.2 - 5.5, 8 port card: Arm C), deep/2412.06464_gated-deltanet.md (what does not transfer: custom kernels, honesty about a GRU), `training/models/p16_time.py` + `Builder.fuse_prior` (the fixed rule), E199 / `p17_gate.py` (raster gate, zero-initialised, out-of-fold), closed question 2 of `closed_questions.md` ("temporal accumulation" named as an open mechanism); new K id K195 |
| status | proposed |

- **Statement** (falsifiable): Replacing the constants of the log-odds fusion at the nodes that coincide with a warped previous node (`l = 0.9 logit(0.9 pi_prev) + 1.0 logit(pi_now)`) by an input-gated unnormalised update `l = sigmoid(W_a x) logit(pi_prev) + softplus(W_b x) logit(pi_now)`, with x = [distance to the matched previous node, heading difference, previous validity, current validity, a 16-d projection of the node feature] (about 400 parameters, last layer zero-initialised so the untrained cell equals the fixed rule), trained for 1500 steps with the builder frozen, raises R@1 beyond 40 m of the champion (288 / 144, fork pruning 0.3) by at least 0.020 and topo F1 by at least 0.008 on VAL, and beats the same training with two global scalars (a, b) by at least 0.008 in R@1_far.
- **Mechanism** (why it should work): The fixed rule trusts every matched previous node by the same weight, whether the previous and current evidence agree or not, and the previous side is the network's own single-frame prediction of frame t - 1 (`prior_from_prev`), which at 45 m is as uncertain as the current one. A learned weight that reads the match quality (position and heading agreement, the two validities, the node feature) can lean on the previous frame exactly where the current evidence is weak and the match is tight (far, thin radar returns), and ignore it where they disagree (a lane that ended, a fork that appeared); that is the part of a GRU that a pure input-gated minGRU lacks (deep note 5.3), so the gate is fed the match evidence. The unnormalised form keeps the evidence accumulation (a convex minGRU update cannot raise a logit above the larger of the two), and the zero-initialised last layer starts at the present rule (the trick of `p17_gate.py`). Why connectivity: a validity that is consistent between frames removes the flicker of single nodes that breaks a lane into pieces (topo_R / lanes found 0.98 for the champion; the temporal rule is what the champion has and the recall variants and the fast variant j1 do not). Honest limit: the deep note estimates the value for the known gaps as low; the cell costs microseconds, so the card is cheap, and the main evidence it can give is whether temporal evidence has headroom at all.
- **Prediction** (VAL 873 frames, `evaluate.py champion` setting, against c0): R@1_far +0.020 (0.709 -> 0.729), topo_F1 +0.010 (0.741 -> 0.751), lanes_found +0.015, Q +0.004, J not lower, ms per frame +0.3 at most (24 ms). Against c1 (scalars only): R@1_far +0.010. If the fixed rule's own gain over no recursion is below +0.010 in R@1_far (step 0), expect a. to gain at most +0.005 and the card is stopped.
- **Decision rule** (band 0.010, 2 seeds paired, paired scene bootstrap): KEEP if all clauses of section 0 hold with both seeds on the same side and the interval of R@1_far excludes 0. TIE (|dR@1_far| <= 0.010): keep the fixed rule (zero parameters, cheaper). CUT if Q or topo_F1 falls below its guard. INCONCLUSIVE: seed 2 for a and c1. A gain of a over c0 without a gain over c1 means the constants were mis-tuned, not that conditioning matters: report it as such and adopt c1 (two numbers) if it passes the clauses vs c0.
- **Test design**: arms a (seeds 0, 1), c1 (seed 0, 1 if a passes), c0 (no training, existing card `runs/eval/champion/val.json`). Controls: null = recursion switched off at inference (step 0); equal budget c1; seed twin; shuffled association (the match descriptor of another node, layer 2) must remove the gain of a over c1. Layer 0: spec tests of the port card of the deep note (identity at init: the cell with zero last layer reproduces `fuse_prior` to 1e-6 on random tensors; the chain with the cell reproduces the graphs of `champion` on 873 VAL frames, bit-identical up to the golden tolerance 0.0005). Step 0 diagnostics (0.1 GPU h): (i) champion with and without the prior on VAL (R@1_far, topo_F1, Q) = the headroom of the fixed rule; (ii) reliability (ECE) of the frozen builder's pi on 500 TRAIN frames versus the 873 VAL frames. If the TRAIN ECE is more than 0.04 below the VAL ECE, the frozen builder is over-confident on TRAIN (it was fitted there) and a gate trained on TRAIN learns to over-trust the current frame: then the gate is fitted by 2-fold cross-fit over the 22 VAL scenes instead (the cell is tiny, builder outputs are cached; evaluation on the other fold only). Training: loss = BCE(fused pi, on-lane label within 1 m) + the graph recall / precision terms of the builder on the fused pi (`graph_losses`), previous-frame prior with the pose noise and 10 % missing priors of `prior_from_prev`, lr 1e-3, 1500 steps, batch as `p17_gate.py`.
- **Cost** (GPU h / coding days / risk): 1.0 GPU h (a: 2 x 0.3; c1: 0.3; diagnostics 0.1); coding 1 day (cell 30 lines, hook with default off, trainer re-using `p17_gate.py` conventions, registry key `gate_ckpt`, identity test); risk high for the effect (the deep note: value low; fitted on the over-confident frames of TRAIN), low for the cost; a shared-code change (`Builder.fuse_prior`) needs an `ar-reviewer` pass and the golden check.
- **Kill criterion**: step 0 (i) gain of the fixed rule below +0.010 R@1_far; identity test fails; the trained cell stays within 0.05 of its initial weights (nothing learned) or the validity of matched nodes saturates (mean fused pi > 0.97); after seed 0 of a, R@1_far gain below +0.005 over c0 and below c1 (no seed 1).
- **Differentiable path**: yes. Graph-based (per node); the gate is a sigmoid / softplus of a linear layer, the association by distance and heading is the existing coincidence rule (unchanged, no argmax added); the previous side carries no gradient as today. Only the cell is trained; the builder is frozen, so nothing else on the path changes.
- **Inputs**: ready: `e123_D7_r3_f1` (temporal champion), `prior_from_prev`, `p17_gate.py` as the pattern, the champion card. To build: `MinGateCell`, hook, trainer, registry key; the over-confidence diagnostic. New technique **K195** "input-gated unnormalised (minLSTM-type) evidence update on the per-node validity logit, match evidence as input, initialised to the fixed log-odds rule", source 2410.01201 (idea; no scan, no log-space), to be added to `literature/register.md` before the first run.
- **Referee** (2026-10-25, ar-referee): REVISE and LATER (do only the step-0 diagnostic tonight, CPU / 0.1 GPU h: champion with and without the prior on VAL, plus the TRAIN-versus-VAL ECE of the frozen builder). Mechanism is coherent and `fuse_prior(a=0.9, b=1.0)` exists as described, but: the card itself says the deep note rates the value for our gaps as low; the prediction (R@1_far +0.020, topo F1 +0.010) is large for about 400 parameters trained 1500 steps on frames where the frozen builder is likely over-confident (the TRAIN-ECE check may force a cross-fit on 22 scenes, which then both fits and evaluates on VAL); closed question 2 names temporal accumulation as open, so no conflict, but nothing yet shows the fixed rule has headroom. Revise: (1) make step 0 (fixed rule's gain over no recursion >= +0.010 R@1_far) a hard gate before any code; (2) the clause "R@1_far vs c1 min_effect 0.008" is raised by decide.py to the band (0.010): set `"band": 0.008` or accept 0.010; (3) c1 has one seed while a has two: pair a-seed-0 with c1 and state that the c1 clause is unpaired, or train c1 with two seeds (0.3 GPH more); (4) identity at init needs bias initialisation (sigmoid bias = logit 0.9, softplus bias such that b = 1.0), not only a zero last layer: add to the spec test; (5) shared `Builder.fuse_prior`: reviewer + golden check; 1 day of coding does not fit tonight. Dependency: independent of E244 / E245, but runs on the champion (288 / 144 with recursion, 24 ms).

## 2. Detail plan (block experiments only; written by ar-planner when the loop reaches the card)
What would make it fail: the previous frame's single-frame prior adds little evidence at range (so the fixed rule gains nothing and the gate has nothing to weigh); fitting on the over-confident TRAIN frames; the matched fraction of far nodes is small (match radius 1.5 m, heading 30 deg), so the gate acts on few nodes.

| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| diagnostic | prior on / off; ECE TRAIN vs VAL of the frozen builder | `temp/loop19/` | numbers in this file | 0.1 GPU h |
| code | `MinGateCell`, hook in `fuse_prior` (None = old rule) | `training/models/p16_time.py`, `p16_builder.py` | identity test; graphs of `champion` unchanged | 1 day |
| train | arms a0, a1, c1 | gate trainer, `run.py --id E247.a0 ...` | done flags | 0.9 GPU h |
| evaluate / decide | registry with `gate_ckpt`, `--compare champion`, `decide.py --experiment E247 --write` | evaluation entry | verdict | - |

## 3. Summary (2026-10-26)
- **Decision: NOT RUN in loop 19**. The step-0 diagnostic (champion with and without the fixed prior on VAL, gate +0.010 R@1_far for the fixed rule; TRAIN-versus-VAL ECE of the frozen builder) needs GPU inference, and the GPU was shared with other chats most of the night; the cheaper CPU diagnostics (E245, E246) took the time.
- **Next**: run the step-0 diagnostic on a quiet GPU in the next loop; the code (MinGateCell, hook in `Builder.fuse_prior`) only if the gate passes.
