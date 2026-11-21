# E391 - STEP 2 of the SD-map line: the FUSION of the perceived tokens with the encoded logical road graph, as an ASSOCIATION that reads logical map data and leaves the perception untouched

**DRAFT, NOT RUN.** Card of 2026-11-21 (system clock), Fable (the lead's task of ~09:05, Mirko's 'ja'). **Version 1 (~10:15): the first cards critic on the draft (REVISE, 8 items) taken in (section 4).** Nothing of this card runs until (1) step 1 (E390) has a SOUND verdict, (2) Mirko has seen it and says go (D130: "Wenn die Codierung vernünftig ist, kann man über Cross Attention Techniken diskutieren ... Schritt für Schritt"), (3) the nulls of section 1.5 are measured, the bars written from them and a fresh critic has passed the card. Step 3 (the error models and the trust mechanism, 'a wrong map must not hurt below no-map') comes AFTER this step and is not part of it.

**The order and its refinement:** D130 (Mirko, verbatim in E390's header): the map enters as a logical road graph (lanes per direction, connections to further roads and branches), encoded first (E390 step 1), the fusion second (this card), the error models third. D135 (Mirko): "Die Kartenzuordnung soll hauptsächlich logische Daten zuordnen, ggf. weniger Perzeption überschreiben." So: the fusion is an ASSOCIATION of each perceived element to a map segment (or 'unassigned'), the logical attributes are READ THROUGH that assignment, and the delivered graph is NOT changed by the map in the main design. E389b (the report chat, 2026-11-21): ONE learned cross-attention with the metre-scale pair bias reads the route as well as E389's two-attention head (a - z +0.0936 / +0.0928 against m - z +0.0939 / +0.0972), and a plain attention without the bias was close (a - p +0.0118, one seed): this card starts from one attention and keeps the plain one as the simplest arm.

## 0. The card (DRAFT: the bars and the prediction are written only after the nulls are measured)
```json card
{
 "id": "E391",
 "card": "",
 "title": "E391 step 2 (DRAFT): the association of the champion's delivered perceived tokens to the segments of the E390 road-graph encoding (with 'unassigned'; false positives included) and the logical attributes read through it (lanes per direction, the node-level route role, the branch at a route fork), fusions P (plain attention), F1 (one cross-attention with the relative-geometry bias), F2 ([token ; association-pooled map] + MLP), F3 (F1 + a gated residual), the encoder frozen or fine-tuned, against the zero-shot E390 association head, the logistic and nearest association rules and the no-map head, a gradient test that perception is untouched, one deciding map-error level, VAL with a TRAIN-holdout read first",
 "kind": "block",
 "deterministic": false,
 "metric": "assoc_acc_dec",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.0, "vs": "nulls", "note": "DRAFT: written after the nulls are measured, before any training"},
 "design_delta": [{"row": "R9", "change": "after a verdict: basis + 'E391 step 2: <outcome> (the association and attribute numbers against their nulls)'; open to attack when -> 'the bars of the step-3 error models and trust' on PASS"}],
 "arms": {
  "nulls": {"recipe": "no fusion training, measured first on VAL (and the TRAIN holdout): (a) the E390 P6 association head of the SOUND step-1 version applied ZERO-SHOT to the tokens (the token position / direction in place of the proxy); (b) the logistic association null refitted on TRAIN tokens (E390's p6_simple features); (c) the nearest-segment rule (tau fitted on TRAIN tokens); (d) the no-map head z (trained, 2 seeds: the attribute prior from perception alone, no association); the oracles: the GT association and the attributes read through it", "run_ids": []},
  "P": {"recipe": "the simplest fusion: ONE plain cross-attention (q.k, no pair bias, an 'unassigned' key) from the token features to the segment embeddings; association = its weights; attributes = small heads on the weighted read", "run_ids": []},
  "F1": {"recipe": "ONE cross-attention with a per-head bias from the token-segment relative geometry (E390's prox_features for the token: distance, heading, along, lateral, attach position, kind) and the 'unassigned' key", "run_ids": []},
  "F2": {"recipe": "Mirko's wording [feature_spur ; feature_mix_karte] + MLP: the token features concatenated with the F1-association-pooled segment embeddings, an MLP to the attributes (its association = F1's)", "run_ids": []},
  "F3": {"recipe": "F1 with a gated residual (the S2b style) of the read map features onto a COPY of the token features used only by the attribute heads; the gate bias -4 (the no-map behaviour at the start); its association = F1's", "run_ids": []},
  "t": {"recipe": "the arm chosen with the encoder frozen, read again with the E390 encoder FINE-TUNED (lr x 0.1); adopted only if it beats the frozen arm by more than the band on assoc_acc_dec with the paired interval above 0 on both seeds", "run_ids": []},
  "Fx": {"recipe": "a NAMED extra arm, reported only, never adopted by this card: the F3 residual applied to the token features that FEED the champion's delivered decode (the token head continued 3000 steps with the token loss + the map losses, S4 style); its delivered graph scored by the champion's scorer (topo F1, junction F1, lanes found at D115) against the champion on the same VAL frames: the cost or gain of letting the map overwrite perception", "run_ids": []}
 },
 "control": "nulls",
 "variant": "F1",
 "seeds": [0, 1],
 "rule": [{"metric": "assoc_acc_dec", "vs": "nulls", "min_effect": 0.0, "note": "DRAFT: the bars are written as null + 0.25 x (oracle - null) with the STRONGEST null AFTER the nulls are measured (section 1.5), before any training"}],
 "budget_gpu_h": 3.0,
 "kill": "a loss is not finite; no learning (the loss at step 3000 above 0.8 x the loss at step 100, as E389); the perception is touched by a main arm (the gradient test of section 1.4 fails); the step-1 encoder is not the version with the SOUND verdict; fewer labelled tokens than the minimum counts of section 1.3",
 "evaluate": {"split": "val"}
}
```

## 1. The design
### 1.1 The question
Given the step-1 encoding (one embedding per road / intersection segment, invariant, with its ego-frame pose beside it), which FUSION assigns each DELIVERED perceived token of the champion to its map segment (or 'unassigned') and reads the logical data through that assignment best, without touching the delivered graph, and does it beat the step-1 association head applied zero-shot?

### 1.2 Data and labels
- **The scored set = E389's delivered set:** the cached B' tokens (the E365 cache, the same frames as E389 / E390: VAL 873, TRAIN 8414) that the E373 stage-1 head KEEPS at p_none < 0.6 (the champion's decode cell), with their features, position and direction; not all K 512. (TRAIN tokens are in-sample for B': their position error is smaller than on VAL, the shift E381 measured: said, not corrected.)
- **The association label of a token:** its GT node (`gt_id` of the E357 table) -> the E390 road-graph node whose CUT geometry (E390 v6: the node boundaries cut, inner stretches by intersection area) holds the GT node's position (the nearest cut stretch among the nodes that list its piece, within 0.5 m). This labels tokens on multi-node pieces too (the critic: dropping them removed about 30 % of the pieces and biased the set); a token whose GT node lies on no cut stretch within 0.5 m is not labelled (counted). **'Unassigned'** = the label segment is missing from the (error-applied) map, OR the token is a FALSE POSITIVE (gt_id -1: perceived, but no lane of the GT is there: nothing in the map should be read for it): trained as unassigned; its assign rate (how often a false token gets a segment) and the attributes read for it are reported separately (as E389's false-token flag rate).
- **The attributes read through the association:** the segment's lanes per direction (the E390 logical count) for tokens on road segments; the **route role at the NODE level** (the token's GT node lies on a tier-1 route piece, E389's `on_lane`, NOT the segment's route flag: a segment-level flag gives an intersection's flag to every connector in it, including the turns not taken; E389 rejected that label: 55.9 % against 40.0 % on delivered VAL nodes); and at a route fork the **branch label** = the successor segment that holds the route continuation, scored on the delivered tokens past the fork within 30 m (E389's decision frames: 234 VAL frames from 14 scenes).
- **The map side:** E390's encoder at the version with the SOUND verdict, frozen from its run (arm t: fine-tuned), its segment embeddings and the ego-frame pose of each segment; if that version reads the map's stated count-error rate (E390 v7), the stated rate of the level is fed (as in E390).
- **Map errors in step 2 (only the simple ones; the error models and trust are step 3):** ONE deciding level **dec** = map-only pose error 1 m + count error 0.1 + link drop 0.1 + route-flag hide 0.3 + segments missing 0.1 (the route hide keeps route_f1 from being a copy of the map's flag); reported beside: exact, pose 2 m, segments missing 0.2. One generator per frame for all levels (nested, as E390).

### 1.3 Metrics (VAL 873 and the TRAIN holdout; the scene bootstrap B 1000; the same draws for every arm and null)
- **assoc_acc_dec** (deciding): the share of labelled delivered tokens assigned to their segment or, when it is missing or the token is false, left unassigned, at the level dec; its parts: the accuracy on tokens with a present segment, the none recall (missing segment), the false-token assign rate, the false-unassigned rate.
- **lanes_acc** (labelled road tokens), **route_f1** (node level), **fork_acc** (the branch of the delivered tokens past a route fork; unit = the fork, the majority branch of its tokens): the attributes read through the association, at dec.
- **Minimum counts** (else INCONCLUSIVE for that metric): 20000 labelled tokens, 3000 false tokens, 2000 tokens of missing segments, 150 fork frames (VAL), measured with the nulls.
- **The E389 reference (reported, not deciding, not paired with the deciding metric):** the chosen arm read also on E389's own metric `route_F1_road` with E389's dumps, labels and mode `error`, beside E389 (two attentions) and E389b (one attention).
- **cost:** parameters and ms per frame of the fusion with the encoder (CPU and GPU).

### 1.4 Perception untouched (D135)
In the main arms the association and attribute heads read the token features through a stop-gradient; nothing of theirs enters the token head that delivers the graph, so on the cached tokens the delivered graph cannot change: a bitwise comparison of dumps would be vacuous. **The test instead:** the gradients of every map loss on every perception parameter (B' and the token head) and on the token tensors are zero / None, with a mutant without the stop-gradient that the test catches; if the heads are ever attached to the live champion, the delivered graph with and without them is compared on the CPU in deterministic mode (GPU kernels are not bit-stable). Fx is the only arm that may touch perception (section 0).

### 1.5 The nulls, the bars, the choice (DRAFT; the bars are written only after the nulls are measured)
- **The nulls to MEASURE first** (a run `E391.nulls`, VAL and the TRAIN holdout, before any fusion training): the zero-shot E390 P6 head, the logistic association null (TRAIN tokens), the nearest rule (tau fitted on TRAIN tokens), the no-map head z (2 seeds; a null for the attributes only), the oracle reads.
- **Bars** (per metric, at dec): **the strongest null + 0.25 x (oracle - null)**, with the paired interval against it above 0; at dec the false-unassigned rate <= 0.05; the minimum counts.
- **The choice:** (1) the association: P against F1 (they differ in the association); F2 / F3 share F1's association, so (2) among F1 / F2 / F3 the choice is made on the attribute metrics (lanes_acc, route_f1, fork_acc) at dec, on both seeds: the simplest (fewest parameters, then the lowest cost) of the arms that pass every bar, unless another is better by more than the band 0.01 on the mean of the three attribute metrics with the paired interval above 0 (RQ-4 / D106); (3) t only for the chosen frozen arm, as in section 0. **The looks at VAL:** a TRAIN-holdout read (every 10th TRAIN scene, left out of every fusion training, as E390 v7) first; the choice is made on the holdout, VAL is read once for the verdict.
- **Outcomes (in this order):** NOT RUN (step 1 not SOUND or no go), NO FUSION (no arm passes on both seeds: report, step 3 does not start), INCONCLUSIVE (an arm passes on one seed, or a minimum count is missed: a third seed or more frames), PASS (the chosen arm; step 3 is planned on it after Mirko has seen it).

### 1.6 Prediction, cost, limits (DRAFT)
- **Prediction:** written after the nulls are measured. Expectation from E390 (proxies at exact positions: association 0.956 at the exact pose, 0.93 at 2 m, none recall 0.8 with missing segments) and E389 / E389b (one attention reads the route): the zero-shot P6 head already strong; F1 above it by the perception's position error it can learn; P close to F1 (E389b); F2 / F3 differ from F1 only on the attributes.
- **Cost:** training on the cached tokens (no B' forward): about 5 - 10 min per arm on the GPU (E389b: 5 - 7 min per head); arms P, F1, F2, F3 x 2 seeds + t x 2 + z x 2 + Fx x 2 = up to 16 runs, about 2.7 GPU h (budget 3.0); the reads on the CPU. Only after Mirko's go.
- **Not claimed:** anything about wrong maps or trust (step 3), TEST, a change of the champion's delivered graph (only Fx, reported).

## 2. Detail plan (DRAFT)
| step | what | done when |
|---|---|---|
| go | step 1 SOUND + Mirko's go | recorded by the lead |
| data | the token labels (cut-stretch association, false tokens, node-level route, fork branches) on the E365 cache frames of the delivered set, VAL / TRAIN / holdout; the counts | cache + counts |
| nulls | `E391.nulls`: zero-shot P6 head, logistic, nearest, z (2 seeds), oracle reads | numbers -> the bars and the prediction written into section 0 |
| critic | a fresh ar-critic (cards) on the card with the bars; a code critic on the heads and the gradient test | ACCEPT |
| build | the fusion heads P / F1 / F2 / F3 / Fx, the gradient test with its mutant | tests green |
| train / read | the arms, two seeds; the TRAIN-holdout read, the choice; VAL once | cards |
| decide | the rule above | section 3, ledger, report |

## 3. Read-out log
(empty: DRAFT, not run)

## 4. Referee
- **Draft v0 REVISE** (fresh ar-critic, cards, ~09:50), all taken in (v1): (1) the scored set = E389's delivered set (p_none < 0.6), false positives trained as 'unassigned' with their assign rate reported, multi-node pieces labelled by the cut stretch (not dropped), the labelled share and minimum counts; (2) the route role at the node level (E389's on_lane), not the segment flag; fork_acc with a branch label, a unit and a minimum count; (3) the E389 reference read on E389's own metric, labels, dumps and mode, reported and not paired; (4) the nulls: the zero-shot E390 P6 head, the logistic association null, the nearest rule, and the plain attention P as the simplest arm; the JSON compares with the nulls, not with z; (5) the choice: P against F1 on the association, F1 / F2 / F3 on the attributes, f / t set before the run, a TRAIN-holdout read first; (6) one deciding level dec with route hide, the stated rate fed if the step-1 version reads it; (7) the perception test = zero gradients with a no-detach mutant (a bitwise dump comparison is vacuous on cached tokens), Fx defined with its scoring; (8) the budget (16 runs, 3.0 GPU h), a learnability kill, the TRAIN in-sample shift said, the prediction deferred to the nulls (a draft).
