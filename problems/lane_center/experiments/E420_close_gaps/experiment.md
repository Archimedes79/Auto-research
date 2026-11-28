# E420 a gap-closing layer on top of the soft skip-link layer, on the champion's weights

The card is the json block below: `research check <problem> E420` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E420",
 "title": "a gap-closing layer on top of the soft skip-link layer, on the champion's weights",
 "question": "on the champion's weights, without training: do the soft skip-link layer (E399) and a soft layer that delivers a weak link where it is the only way to continue an open end give a topo F1 gain on VAL against the champion without breaking a guard, in particular without the skip layer's loss of lanes found?",
 "hypothesis": "the skip layer gains topo F1 and loses lanes found; many true links are missing between delivered nodes and part of them are weak candidates whose source and target have no other delivered link; delivering the strongest of those wins the lanes found back and adds topo F1 at an unchanged junction F1",
 "prediction": 0.034,
 "kind": "compare",
 "control": "champ",
 "seeds": [0, 1],
 "arms": {
  "champ": {"from": "E396.champ", "note": "the champion's stored runs"},
  "skip": {"from": "E399.skip", "note": "E399's runs with the skip layer alone"},
  "skipgap": {"set": {"import_weights": "runs/e420_skipgap_s{seed}"}, "note": "the champion's weights, the skip layer and the gap-closing layer (lo 0.03, lo_long 0.02, sigma 0.01, sigma_lo 0.01, gamma 16, the read-out's thresholds)"}
 },
 "rule": {},
 "kill": [],
 "note": "asked for by the owner in chat on 2026-11-28 (detail improvements of the existing system, check every stage, look at all errors, fix iteratively); the decode stays the champion's (0.3 / 0.2 / 0.6); the candidate is skipgap; no adoption without the owner"
}
```

## Why (champion with the soft skip layer, TRAIN sample, 291 frames, seeds 0 / 1; scratch scripts, no record)
- Error budget, one thing replaced by the truth at a time: removing the delivered nodes farther than 1.5 m from every lane +0.0005 /
  +0.0004 topo F1; moving delivered nodes within 2 m onto the lane +0.021 / +0.020; perfect wiring of the delivered nodes +0.116 / +0.109
  (junction F1 0.39 -> 0.94). Wiring is the limit.
- Wiring errors: 85 % of the delivered links follow the truth, 9 % touch a node off every lane, 4 % join two nodes of one place, 1.3 % go to
  another lane, 1.1 % jump ahead. 6.5 / 6.4 % of the true links between delivered nodes are missing; adding those alone: +0.074 / +0.068.
- Of the missing links 2 % are excluded by the geometry gate; after the skip layer 53 / 51 % are candidates with probability 0.05 or more.

Against the mechanism, from the same sample:
- only 16 / 13 % of the missing links have both ends open (source without a delivered successor and target without a delivered predecessor);
- the links a hard rule of that kind adds (lower threshold 0.05): 35 / 32 % follow the truth, 49 / 50 % touch a node off every lane, 12 /
  13 % join one place;
- one-sided rules lowered junction F1 by 0.038 to 0.049, "either end open" by 0.058 to 0.079; "both ends open" left it unchanged;
- the layer below gains +0.020 / +0.015 topo F1 on top of the skip layer, about a quarter of the headroom above;
- lanes found counts recall only, and topo F1 also credits links off every lane (dropping the wrong, backward and off-lane links costs
  0.031 topo F1 and 0.074 lanes found, seed 0). A KEEP therefore does not show that gaps are closed correctly; the lifted links of the VAL
  runs are classified after the run and reported in the conclusion.

## Chosen on TRAIN (the same sample; the network code, seeds 0 / 1)
| variant | topo F1 | junction F1 | lanes found |
|---|---|---|---|
| champion | 0.7192 / 0.7225 | 0.3399 / 0.3396 | 0.8095 / 0.8019 |
| skip layer | 0.7364 / 0.7414 | 0.3908 / 0.3821 | 0.7956 / 0.7903 |
| skip and gap, lo 0.02 | 0.7571 / 0.7570 | 0.3935 / 0.3814 | 0.8125 / 0.8055 |
| skip and gap, lo 0.03 | 0.7565 / 0.7567 | 0.3936 / 0.3815 | 0.8122 / 0.8050 |
| skip and gap, lo 0.05 | 0.7556 / 0.7558 | 0.3936 / 0.3813 | 0.8113 / 0.8033 |
| skip and gap, lo 0.03, sigma 0.03 | 0.7542 / 0.7543 | 0.3919 / 0.3811 | 0.8111 / 0.8030 |
| skip and gap, lo 0.03, and thinning (E399) | 0.7582 / 0.7569 | 0.3991 / 0.3932 | 0.8065 / 0.8036 |

Chosen on this sample: both ends open (against the one-sided forms, as hard rules), the lower threshold among 0.02 / 0.03 / 0.05 (they
differ by at most 0.0015; 0.03 is within 0.0006 of the best), sigma 0.01 against 0.03, exponent 16 (against 4, on an earlier form). Token
thinning is left out of the candidate: it drops tokens at a hard threshold inside the forward path (a rule step under R1) and, as built, only
masks them, so it saves no compute.

## The layer
`net/model/jtoken.close_gaps`, no parameter, applied after the skip layer. A pair is open when no OTHER delivered link leaves its source or
reaches its target (soft: sigmoids of probability minus class threshold times both ends kept), eligible above the lower threshold, and counted
only as far as it is the strongest such pair of both its ends (power 16 of its share of the maximum). With w the product of the three, its
probability p becomes p + w t (1 - p) with t the class threshold: at w = 1 above t and still ordered by p; the mass comes from the source's
none slot. The output never falls when the pair's own probability rises (tested for both classes below, at and above the thresholds). Open and
best are maxima (differentiable almost everywhere). At an open, best pair a link is delivered from about p = 0.050 (short) and 0.037 (long).
A delivered link with no other delivered link at its ends is raised too and stays delivered.

## Imported networks
`<runs root>/lane_center/e420_skipgap_s{0,1}`: the champion's `e372_s4_s{0,1}/model.pt` with `skip_suppress` and `close_gaps` added to its
args, weights and B' tensors equal (checked); `import.json` in each folder records both sha256.

## Prediction
VAL topo F1 gain of `skipgap` over the champion +0.034 (TRAIN +0.037 / +0.034; E399's skip layer carried its TRAIN gain to VAL almost one to
one). VAL lanes found about 0.001 below the champion (TRAIN +0.003 / +0.003; E399 lost 0.004 more on VAL than on TRAIN). A pass or a fail of
the lanes-found guard within 0.003 of its 0.010 is an edge result.

## How the verdict is read
The standard rule per arm against the champion; `skip` repeats E399's arm for context. What the gap layer adds to the skip layer is the
companion card E421 (control E399.skip, variant E420.skipgap), committed before the results are read; without it no claim about the third
layer. KEEP: a candidate; the owner decides an adoption.

## Critic (cards mode, 2026-11-28, on the first form of this card): revise
Taken in: (1) the first form's output fell with the pair's own probability below the class threshold: rebuilt (above), with a test; (2) the
thinning arm held a hard step: dropped from the candidate; (3) the marginal question got its own card (E421); (4) prediction with lanes found
and an edge zone; (5) the evidence against the mechanism and the off-lane credit (above); (6) wrong statements corrected; (7) design delta
extended, the earlier hysteresis attempt of v1 (E190, in the old assembly: small lanes-found gain, junction metric down) named.

## Design delta
R1: two fixed soft layers on the wiring, not learned; whether that satisfies R1 or the steps must be learned is the owner's call at an
adoption. R7: the read-out's thresholds are now also inside the forward path; a raised link's probability is p + w t (1 - p). R3: the layers'
time per frame is not measured; to be timed before an adoption. DESIGN.md changes only at an adoption.

## Plan
- Code on branch `lane/skip-links`: `jtoken.close_gaps`, `JTokenHead(gap=)`, `build_head` reads `close_gaps`; test `test_close_gaps`.
- Checks before the start: `test_jtoken` full, layering, drawing, anchor of `E396.champ_s0`.
- Runs: predict and evaluate of `skipgap` only (2 runs). TEST is not touched.

## Conclusion
**Verdicts** (`research judge`, 2026-11-28, `verdict.json`): `skipgap` KEEP, `skip` TIE (E399 again). VAL (873 frames, 22 scenes, scene
bootstrap), pooled over seeds 0 / 1, against the champion's stored runs. TEST not touched. Nothing adopted.

| metric | champion | skipgap |
|---|---|---|
| topo F1 | 0.6748 | 0.7139 (+0.0392 [+0.0333, +0.0462]; seeds +0.0393 / +0.0391) |
| junction F1 | 0.2510 | 0.2881 |
| lanes found | 0.7223 | 0.7275 |
| F1@1 | 0.8467 | 0.8462 |
| duplicate share | 0.0637 | 0.0186 |
| vert100_dp | 23.96 | 22.16 |

- Predictions: topo F1 gain +0.034, observed +0.0392 (met); lanes found 0.001 below the champion, observed 0.005 above it.
- E421 (companion): the gap layer adds +0.0207 [+0.0179, +0.0239] topo F1 to the skip layer and 0.022 lanes found, junction F1 unchanged.
- `tools/diagnose`, seeds 0 / 1: true lane points within 1.5 m of a delivered line 90.7 / 90.5 % (inside intersections 84.6 / 84.2 %); forks
  per frame 5.0 / 4.7, 6 % of them with a jump.
- The links `skipgap` delivers and `skip` does not (scratch script, no record; 3.2 / 3.0 % of the delivered links, 24 lost per seed): 31 /
  30 % follow the truth, 8 / 9 % join two nodes of one place, 2 / 3 % go to another lane, 1 % jump ahead, 56 / 57 % touch a node without a
  GT node. Of the last: 41 % within 1.5 m of a lane, 33 / 31 % at 1.5 - 2.5 m, 26 / 28 % farther: about one added link in seven lies clearly
  off every lane. The gain is therefore partly longer lines that are not exactly right.
- Champion against `skipgap`, seed 0, frames with at least three true lanes (826, scratch): covered share of the true lane length within
  1.5 m 0.905 -> 0.913, up in 323 frames, down in 140; delivered pieces per frame 56.6 -> 38.5. The frame with the largest loss (scene 33)
  shows the skip layer removing true connectors through a complex junction, because their target is also reached over another branch.
- Learned: wiring, not node choice, limits topo F1; two fixed soft steps on the wiring recover a part of it. Open: the skip layer in complex
  junctions; node positions (zig-zag); most of the wiring headroom (perfect wiring: +0.11 on TRAIN) needs a better wiring head.
- Follows: the owner's call on an adoption (R1: the steps are fixed, not learned). An adoption needs the champion measured again, because
  the evaluator on main changed after this branch started; the layers' time per frame is to be measured first.
