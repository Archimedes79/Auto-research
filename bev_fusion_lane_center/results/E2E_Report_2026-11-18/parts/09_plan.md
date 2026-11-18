# 9. Plan: recursive temporal fusion and end-to-end training without more input frames

**The order (Mirko, 2026-11-17 night, relayed by the lane lead):** "mach mal Experimente weiter und einen neuen Bericht für morgen. Plane ggf. zeitliche Fusion und E2E training ein, aber ohne mehr input frames, nur rekursiv." This chapter is a PLAN: it holds no new result and no number that was measured for it. Its evidence is what was measured before; its experiments each get a card with the rule written before the run.

**The constraint (D91, decided 2026-11-13; his words, verbatim):** "Zeitlich möchte ich nur einen Frame als eingang und einen Radarscan, die Vergangenheit soll rekursiv auf Featureebene referenziert werden." One camera frame and one radar scan per step; the past enters only as the state of the previous step at the level of features, moved with the ego motion, never as a second image or a window of frames. D103 (2026-11-15) asked for the same for the lane graph: recursive temporal fusion on the feature level, completely end-to-end differentiable, simple.

## 9.1 What was measured for the lane graph: the past adds almost nothing

A recursive state was tried at three places of the lane system and probed at a fourth (step 0 only). Each was read against the same module with an empty state (a matched control), VAL 873 frames, two seeds, TEST untouched.

| where the state sits | card | result | verdict |
|---|---|---|---|
| a heat-side end state (an end prior carried from the previous frame) | E299 / E336 | the recoverable share of missed junction pairs 0.0483 against the bar 0.05; carried ends add 1 true to 5.5 false end | closed by the data: the heat-side state is not justified |
| a feature-level recursive state inside the junction head | E354 | state minus empty state +0.0018 / +0.0036 topo F1 (unsnapped read-out), +0.0027 / +0.0013 (snapped) | TIE, the system without the state stays |
| the token head reads its own token states of the previous keyframe (ego-warped, one frame back, recursive) | E368 (S2) | +0.0042 / +0.0023 topo F1 against the bar +0.010 (chapter 5, section B4) | TIE, the champion takes no past block |
| a recursive node state in the node head of B' (the previous keyframe's B' output moved with the ego pose, one frame back; step 0 on 851 VAL frames, no training) | E384 (planned under D118, closed at step 0) | oracle coverage ceiling of the previous keyframe's output net of the identity null +0.0021 .. +0.0037 coverage points (bar +0.006); pose-minus-identity AP lift +0.0017 / +0.0018 (bar +0.005); 1 true to 8 false carried-over tokens | CUT by hand: critic step 0 (the bars were written after the numbers were seen); no code, no GPU hour; the card stays ready to run, `closed_questions.md` holds the row |

**Reading.** On the lane graph the past is worth at most about +0.004 topo F1 at every level that was tried, and the head does read it (a wrong past costs 0.026 in E368). The one-frame token set already carries what the previous step could say about the lane network, which is static. This is no argument against recursion in general: the lane graph is the quiet case.

## 9.2 Where a recursive state should pay (hypotheses, not results)

1. **Radar.** One sweep carries far fewer points than the half second of sweeps the earlier system used: 54 radar points in the grid per keyframe against 300, 5.6 times fewer; a vehicle has at least one point in 64 % of the cases against 82 %, 1.7 points on average against 8.9 (D91, nuScenes mini, 384 keyframes). A recursive radar state is the one place of the camera-and-radar system where the single scan loses information the state can give back (motion, speed). In the BEV system the recurrent camera buffer exists (`cam_buffer: "recurrent"`, arm R of t21); a recursive radar state is weak there and is the open part.
2. **The far range and occlusion of the lane graph.** Where the one-frame head misses far lanes (lanes found 0.72 at the champion) or sees a lane only in part, an earlier step may hold the missing piece. The earlier readings (E354, E368) were pooled over all frames; a split by far range, by hidden stretches and by cold start (the first frames of a scene) was not the question of those cards.
3. **The loss that reaches the state.** In all three experiments of 9.1 the lane head B' was frozen and cached: the recursive state sat after B', in a head that gets its gradient only from its own loss. A state inside B', trained through the final graph loss, learns what the graph needs; it was never tried.

## 9.3 End-to-end training: S4 comes first

The champion is staged: the lane head B' is frozen and its predicted nodes are cached, so the loss of the token head never reaches B' (the named gap "staged, B' frozen"). **S4 (E372)** closes it: the node head of B' runs inside the training step (a live token producer), the loss of the token head reaches it, and B' keeps its own loss so that it does not drift. The card (v3.1, after two critic looks) compares it with a matched fine-tune without the gradient (arm z) and with B' trained on its own loss only (arm b); two seeds, VAL. Its code is built and tested (the data path, the live producer), its runs are queued behind a GPU gate. **Status in this report: see chapter 5, section B9** (a result only if its card holds one at the time the report was built).

## 9.4 The plan, in order (each step is a card before it is a run)

1. **S4, the joint fine-tune (E372).** The condition for everything below: the loss must reach B'. Verdict by its own written rule.
2. **A recursive BEV feature state inside B', trained through the final loss (new card, after S4).** One frame and one scan in, the state of the previous step (the pyramid features at 2 m, moved with the ego motion) added through one module; the same module with an empty state is the control; the read-out streams the state through a scene and resets it at the 22 VAL scene starts; two seeds. Prediction written in the card: small, about +0.003 topo F1 (the three earlier results), larger only in the far-range and cold-start splits named in 9.2. **Kill:** S4 shows no gain, or the state module costs more than +20 % per training step without a gain in the splits. Not run before S4 has a verdict. **Status after the night of 2026-11-18:** the first card of this step (E384, the state in the node head, D118) was measured at step 0 and closed by the ceiling (table 9.1, last row): the previous keyframe's B' output adds at most +0.002 to +0.004 coverage points, so a feature-level state is re-opened only with a new mechanism; the E2E part that was trained is S4 (E372).
3. **A recursive radar state in the camera-and-radar BEV system** (the system of the BEV chat, not of this report): the same pattern (a recursive state against an empty state, one scan), judged on moving objects and speed, where a single scan loses most (9.2, item 1).
4. **Cost.** Every added module is read with the cost gate of the project (ms per frame on a quiet host, parameters). The training-step speed path (chapter 5, section D, and the K = 384 token budget card) is the cost half of this plan; the inference milliseconds of a state module are **not measured** until the module exists.

## 9.5 What stays open

- No TEST number exists for any system in this report; a TEST check needs the owner's recorded word.
- The junction F1 of the champion (0.25 at the final decode) stays below the plain head's 0.27; the three evidence experiments (the past, the map, the loss placement) did not lift it, and S4 is the first one that changes what the head sees.
- Four named output thresholds, the top-K token budget and the allowed-pair gate remain hard steps of the delivered graph.
