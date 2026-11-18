# 6. The champion against its challengers, by their written rules

**In short.** Every challenger below was judged by a rule that its card wrote before the run; "by hand" is said where a verdict is not from a rule. The champion since D120 is the two-stage recipe (the E365 token head with none weight 0.25 and the decode D115 as stage 1, then S4: the head and the node head of B' trained together); a challenger replaces it only by its rule. VAL 873 frames, two seeds written s0 / s1, topo F1 unless named; TEST is untouched.

Table: Champion and challengers. A cell that says PENDING is not measured at the time of the build; numbers come from the cards named in the last column.

| card | what it asks | the written rule (short) | result s0 / s1 | verdict | card |
|---|---|---|---|---|---|
| staged champion (E373, E375), the reference of S4 | the token head, none weight 0.25, per-class decode | the gate of E365 + the cell rule of E375 | topo F1 0.6679 / 0.6762, junction F1 0.2517 / 0.2452 | the champion until D120 | E373, E375, D115 |
| E375 | a per-class decode instead of one link threshold | junction F1 up, topo F1 and F1@1.0 inside the band | junction F1 +0.049 / +0.048 | ADOPT-DECODE | E375 |
| E376 | a max-positive wiring loss | junction F1 +0.030 | +0.016 / +0.008 | NO | E376 |
| E377 | the none push placed on the near-node none tokens | junction F1 and lanes found at the TRAIN-chosen cell | targeting only: junction F1 -0.009 / +0.007, lanes found -0.038 / -0.031 | TARGETING-ONLY | E377 |
| E378 | PENDING: the card names its question and rule | PENDING | PENDING | PENDING | E378 |
| E379 | PENDING: ensemble distillation (the KL of the teachers' mean successor distribution) | PENDING | PENDING | PENDING | E379 |
| S2, E368 | cross-attention to the previous keyframe | +0.010 topo F1 | +0.0042 / +0.0023 | TIE, no past block | E368 |
| S3, E374 | cross-attention to a road-level SD map | +0.010 topo F1 | +0.0031 / +0.0026 | TIE, no map block | E374 |
| S4, E372 | the loss reaches the node head of the lane head (joint fine-tune; arms z, b, s4) | S4-HOLDS: s4 - z not lower than -0.01 in topo F1 and junction F1, s4 - ref not lower than -0.01, floors, parity; S4-GAINS needs s4 - z >= +0.01 on both seeds | topo F1 0.6682 / 0.6813; s4 - z -0.0003 / +0.0049; s4 - ref +0.0002 / +0.0051 | **S4-HOLDS, not S4-GAINS: the champion since D120** | E372, D120 |
| E383 | the speed path of the training reproduces the champion (parity) | topo F1 -0.005, F1@1.0 -0.005, lanes found -0.010, junction F1 -0.015 at most, both seeds | PENDING | PENDING | E383 |
| E384 | a recursive node state in the node head: can the previous keyframe's B' output help this frame's nodes at all (step 0, 851 VAL frames, D118) | GO bars +0.006 coverage points, +0.005 AP lift (written after the probe numbers were seen) | oracle coverage ceiling net of the identity null +0.0021 .. +0.0037, pose-minus-identity AP lift +0.0017 / +0.0018, 1 true to 8 false carried-over tokens | CUT by hand: critic step 0 (no code, no GPU hour; card kept ready) | E384 |

**How to read the table.** A TIE keeps the cheaper variant (the project rule): here no block is added to the champion for S2 and S3. A NO or a TARGETING-ONLY keeps the champion. The champion's own number is not a card verdict: it is the reference every row is paired against (position pairing, the same frames, the same B' seed). The noise of a head seed is about +-0.003 in topo F1 and +-0.006 in junction F1 (E378's noise cards), which is why the bars sit at 0.005 and above.

**What this ranking does not say.** It is a ranking of heads on the frames of VAL; no TEST number exists (a TEST run needs the owner's recorded word); the junction F1 of the champion (0.25) stays below the plain head's 0.27.

**sources** E373, E375, E376, E377, E378, E379, E368, E374, E372, E383, E384 cards; D115.
