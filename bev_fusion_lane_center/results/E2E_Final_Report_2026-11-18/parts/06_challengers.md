# 6. The champion against its challengers, by their written rules

**In short.** Every challenger below was judged by a rule that its card wrote before the run; "by hand" is said where a verdict is not from a rule. The champion (the E365 token head, none weight 0.25, decode D115) stays unless a challenger beats it by its rule. VAL 873 frames, two seeds written s0 / s1, topo F1 unless named; TEST is untouched.

Table: Champion and challengers. A cell that says PENDING is not measured at the time of the build; numbers come from the cards named in the last column.

| card | what it asks | the written rule (short) | result s0 / s1 | verdict | card |
|---|---|---|---|---|---|
| champion (E373, E375) | the token head, none weight 0.25, per-class decode | the gate of E365 + the cell rule of E375 | topo F1 0.6679 / 0.6762, junction F1 0.2517 / 0.2452 | **the champion** | E373, E375, D115 |
| E375 | a per-class decode instead of one link threshold | junction F1 up, topo F1 and F1@1.0 inside the band | junction F1 +0.049 / +0.048 | ADOPT-DECODE | E375 |
| E376 | a max-positive wiring loss | junction F1 +0.030 | +0.016 / +0.008 | NO | E376 |
| E377 | the none push placed on the near-node none tokens | junction F1 and lanes found at the TRAIN-chosen cell | targeting only: junction F1 -0.009 / +0.007, lanes found -0.038 / -0.031 | TARGETING-ONLY | E377 |
| E378 | PENDING: the card names its question and rule | PENDING | PENDING | PENDING | E378 |
| E379 | PENDING: ensemble distillation (the KL of the teachers' mean successor distribution) | PENDING | PENDING | PENDING | E379 |
| S2, E368 | cross-attention to the previous keyframe | +0.010 topo F1 | +0.0042 / +0.0023 | TIE, no past block | E368 |
| S3, E374 | cross-attention to a road-level SD map | +0.010 topo F1 | +0.0031 / +0.0026 | TIE, no map block | E374 |
| S4, E372 | the loss reaches the lane head (joint fine-tune) | PENDING: the rule of the card (arms z, b, s4) | PENDING; interim seed 0, live read-out: z 0.6685, s4 0.6682, champion 0.6679 | PENDING | E372 |
| E383 | the speed path of the training reproduces the champion (parity) | topo F1 -0.005, F1@1.0 -0.005, lanes found -0.010, junction F1 -0.015 at most, both seeds | PENDING | PENDING | E383 |
| E384 | PENDING: the combined optimisation (recipe and K) | PENDING | PENDING | PENDING | E384 |

**How to read the table.** A TIE keeps the cheaper variant (the project rule): here no block is added to the champion for S2 and S3. A NO or a TARGETING-ONLY keeps the champion. The champion's own number is not a card verdict: it is the reference every row is paired against (position pairing, the same frames, the same B' seed). The noise of a head seed is about +-0.003 in topo F1 and +-0.006 in junction F1 (E378's noise cards), which is why the bars sit at 0.005 and above.

**What this ranking does not say.** It is a ranking of heads on the frames of VAL; no TEST number exists (a TEST run needs the owner's recorded word); the junction F1 of the champion (0.25) stays below the plain head's 0.27.

**sources** E373, E375, E376, E377, E378, E379, E368, E374, E372, E383, E384 cards; D115.
