# 6. The champion against its challengers, by their written rules

**In short.** Every challenger below was judged by a rule that its card wrote before the run; "by hand" is said where a verdict is not from a rule. The champion since D120 is the two-stage recipe (the E365 token head with none weight 0.25 and the decode D115 as stage 1, then S4: the head and the node head of B' trained together); a challenger replaces it only by its rule. VAL 873 frames, two seeds written s0 / s1, topo F1 unless named; TEST is untouched.

Table: Champion and challengers. A cell that says PENDING is not measured at the time of the build; numbers come from the cards named in the last column.

| card | what it asks | the written rule (short) | result s0 / s1 | verdict | card |
|---|---|---|---|---|---|
| staged champion (E373, E375), the reference of S4 | the token head, none weight 0.25, per-class decode | the gate of E365 + the cell rule of E375 | topo F1 0.6679 / 0.6762, junction F1 0.2517 / 0.2452 | the champion until D120 | E373, E375, D115 |
| E375 | a per-class decode instead of one link threshold | junction F1 up, topo F1 and F1@1.0 inside the band | junction F1 +0.049 / +0.048 | ADOPT-DECODE | E375 |
| E376 | a max-positive wiring loss | junction F1 +0.030 | +0.016 / +0.008 | NO | E376 |
| E377 | the none push placed on the near-node none tokens | junction F1 and lanes found at the TRAIN-chosen cell | targeting only: junction F1 -0.009 / +0.007, lanes found -0.038 / -0.031 | TARGETING-ONLY | E377 |
| E378 | does the max-positive wiring loss of E376 replicate on two fresh heads (the loss that sums the positive destinations stays the champion's) | topo F1 at least +0.005 with the interval above 0 on both heads at the recipe cell; junction F1, F1@1.0, lanes found not lower by more than 0 / 0.005 / 0.010 | topo F1 +0.0030 / +0.0000; junction F1 +0.038 / +0.012 (recipe cell); x_s1 lanes found -0.0106 | **NO** (not a TIE either); seed twins not trained; the champion keeps `--pos-loss sum` | E378 |
| E379 | distillation of a two-head ensemble into one head (the KL to the teachers' mean successor distribution) and the plain two-head ensemble e2 | topo F1 at least +0.005 with the interval above 0 on both heads at the recipe cell, guards on junction F1, F1@1.0, lanes found, duplicates; e2 also the cost gate of 19.7 ms | d at the recipe cell: topo F1 +0.0038 / +0.0008, lanes found -0.0120 / -0.0171; d at D115: +0.0074 / +0.0027 (recovers 65 % / 42 % of the ensemble gain); e2 topo F1 +0.0113 / +0.0065 at 21.84 ms in sequence, 25.19 ms stacked | d: NO by the letter, TIE in substance (the champion keeps one head, no teacher); e2: MISS on cost; the controls l, ds not run | E379, E2E cost table |
| S2, E368 | cross-attention to the previous keyframe | +0.010 topo F1 | +0.0042 / +0.0023 | TIE, no past block | E368 |
| S3, E374 | cross-attention to a road-level SD map | +0.010 topo F1 | +0.0031 / +0.0026 | TIE, no map block | E374 |
| S4, E372 | the loss reaches the node head of the lane head (joint fine-tune; arms z, b, s4) | S4-HOLDS: s4 - z not lower than -0.01 in topo F1 and junction F1, s4 - ref not lower than -0.01, floors, parity; S4-GAINS needs s4 - z >= +0.01 on both seeds | topo F1 0.6682 / 0.6813; s4 - z -0.0003 / +0.0049; s4 - ref +0.0002 / +0.0051 | **S4-HOLDS, not S4-GAINS: the champion since D120** | E372, D120 |
| E383 | the speed path of the training reproduces the champion (parity) | topo F1 -0.005, F1@1.0 -0.005, lanes found -0.010, junction F1 -0.015 at most, both seeds | PENDING | PENDING | E383 |
| E384 | a recursive node state in the node head: can the previous keyframe's B' output help this frame's nodes at all (step 0, 851 VAL frames, D118) | GO bars +0.006 coverage points, +0.005 AP lift (written after the probe numbers were seen) | oracle coverage ceiling net of the identity null +0.0021 .. +0.0037, pose-minus-identity AP lift +0.0017 / +0.0018, 1 true to 8 false carried-over tokens | CUT by hand: critic step 0 (no code, no GPU hour; card kept ready) | E384 |

**How to read the table.** A TIE keeps the cheaper variant (the project rule): here no block is added to the champion for S2 and S3. A NO or a TARGETING-ONLY keeps the champion. The champion's own number is not a card verdict: it is the reference every row is paired against (position pairing, the same frames, the same B' seed). The noise of a head seed is about +-0.003 in topo F1 and +-0.006 in junction F1 (E378's noise cards), which is why the bars sit at 0.005 and above.

**The four max-positive heads at the D115 cell (by hand: stored numbers, not a verdict).** Against the champion head of the same cache, the mean of the four max heads (E376 s0 / s1, E378 s0 / s1; SD in brackets) is junction F1 +0.0110 (0.0102), topo F1 +0.0062 (0.0039), lanes found +0.0081 (0.0047); the champion recipe on another head seed (the noise of the same comparison) gives +0.0063 / +0.0014 junction F1 and +0.0030 / -0.0032 topo F1. One fresh head (x_s1: +0.0015 topo F1) shows almost nothing, and the junction gain at the recipe cell (+0.038) was paid in lanes found; a fresh confirmation pair at D115 with a junction bar of +0.015 was estimated at 10 - 15 % to pass and was not run.

**What this ranking does not say.** It is a ranking of heads on the frames of VAL; no TEST number exists (a TEST run needs the owner's recorded word); the junction F1 of the champion (0.25) stays below the plain head's 0.27.

**S2b: the past fused properly (asked 2026-11-18, D121, run the same day; the card and the runs belong to the lane chat).**

What S2 (E368) did, and what it did not do: the block is one cross-attention after self-attention layer 1 (the current tokens attend to the ego-warped token states of the previous keyframe, a none key for the first frame) and a zero-initialised residual projection back into the token stream. It adds the attention output to the token states and nothing else: the only places where the new information is mixed with the old are the feed-forward blocks of layers 2 and 3. The head reads the past (a wrong past costs 0.026 topo F1) and the gain is a TIE (+0.0042 / +0.0023 against the bar +0.010).

S2b adds the fusion that S2 lacks (Mirko's reading, D121: cross-attention followed by an MLP fusion is the natural form). The cross output c is not added; it is fused with the input x by a small MLP over [x, c and the evidence of the attention (the attention mass on real past keys against the none key)], and a per-token gate decides how much of the fused result replaces x, so that a token passes unchanged where the past does not help. The output is zero-initialised, so the start is the head without the block. The constraint D91 / D118 stands: one frame and one radar scan per step, the past only recursively at feature level.

How it relates to E384: S2b is the token-level form of what E384 asked at the feature level (a recursive state inside the node head of B'). E384 stays closed by the oracle ceiling of its step 0 (chapter 4); S2b asks the same question one level up, where the head already reads the past. If S2b wins its rule, the end-to-end confirmation is the S4 stage 2 over it (the loss reaching the node head with the past block in the head); a win at stage 1 alone is not the champion.

Table: S2b against its control (card, two seeds, VAL 873 frames; the numbers are copied from the card of the lane chat, never typed here).

| arm | topo F1 s0 / s1 | junction F1 s0 / s1 | lanes found | parameters | ms per frame | verdict |
|---|---|---|---|---|---|---|
| control z (E368 loader, no block) | PENDING | PENDING | PENDING | PENDING | PENDING | reference |
| S2 (E368, cross-attention only) | +0.0042 / +0.0023 against z | | | 144,547 (head alone, +17,416) | head alone 7.88 against 6.66 (quiet) | TIE |
| S2b (cross-attention + MLP + gate) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING: the rule of the card |

The report states the card's verdict as it is (a win, a TIE, or a labelled negative). If S2b fails its rule its code leaves again behind an archive tag, nothing is lost (D121). Whatever the verdict, the tokens of the past stay a block of the head: they are not in the champion unless S2b passes its rule and the stage 2 (S4) confirms it.

**sources** E373, E375, E376, E377, E378, E379, E368, E374, E372, E383, E384 cards; D115, D121; the S2b card (lane chat).
