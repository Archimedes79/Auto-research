R = "bev_fusion_lane_center/results/E2E_Final_Report_2026-11-18/parts/"
p = R + "06_challengers.md"
s = open(p, encoding="utf-8").read()
new = """**S2b: the past fused properly (asked 2026-11-18, D121, run the same day; the card and the runs belong to the lane chat).**

What S2 (E368) did, and what it did not do: the block is one cross-attention after self-attention layer 1 (the current tokens attend to the ego-warped token states of the previous keyframe, a none key for the first frame) and a zero-initialised residual projection back into the token stream. It adds the attention output to the token states and nothing else: the only places where the new information is mixed with the old are the feed-forward blocks of layers 2 and 3. The head reads the past (a wrong past costs 0.026 topo F1) and the gain is a TIE (+0.0042 / +0.0023 against the bar +0.010).

S2b adds the fusion that S2 lacks (Mirko's reading, D121: cross-attention followed by an MLP fusion is the natural form). The cross output c is not added; it is fused with the input x by a small MLP over [x, c and the evidence of the attention (the attention mass on real past keys against the none key)], and a per-token gate decides how much of the fused result replaces x, so that a token passes unchanged where the past does not help. The output is zero-initialised, so the start is the head without the block. The constraint D91 / D118 stands: one frame and one radar scan per step, the past only recursively at feature level.

Table: S2b against its control (card, two seeds, VAL 873 frames; the numbers are copied from the card of the lane chat, never typed here).

| arm | topo F1 s0 / s1 | junction F1 s0 / s1 | lanes found | parameters | ms per frame | verdict |
|---|---|---|---|---|---|---|
| control z (E368 loader, no block) | PENDING | PENDING | PENDING | PENDING | PENDING | reference |
| S2 (E368, cross-attention only) | +0.0042 / +0.0023 against z | | | 144,547 (head alone, +17,416) | head alone 7.88 against 6.66 (quiet) | TIE |
| S2b (cross-attention + MLP + gate) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING: the rule of the card |

The report states the card's verdict as it is (a win, a TIE, or a labelled negative). If S2b fails its rule its code leaves again behind an archive tag, nothing is lost (D121). Whatever the verdict, the tokens of the past stay a block of the head: they are not in the champion unless S2b passes its rule and the stage 2 (S4) confirms it.

"""
k = s.index("**sources** E373")
s = s[:k] + new + s[k:]
s = s.replace("**sources** E373, E375, E376, E377, E378, E379, E368, E374, E372, E383, E384 cards; D115.", "**sources** E373, E375, E376, E377, E378, E379, E368, E374, E372, E383, E384 cards; D115, D121; the S2b card (lane chat).")
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = R + "14_plan.md"
s = open(p, encoding="utf-8").read()
k = s.index("## 9.1 What was measured")
q = """**A question asked on 2026-11-18 and run the same day (D121).** Mirko asked whether the past should be fused the proper way, after the cross-attention: "Ja, eigentlich wäre das die ideale Fusion, kannst Du mir da folgen?" (answering the lead's question whether S2b should run). The answer was yes: S2b (cross-attention, an MLP over the input, the cross output and the evidence of the attention, and a per-token gate, zero-initialised) is run on 2026-11-18 under the constraint above, with a card and a written rule before the first run, two seeds and the control of E368; its result or its labelled negative is in chapter 6 (section S2b). It changes item 9.1 below only if it passes its rule.

"""
s = s[:k] + q + s[k:]
a = "4. **Cost.** Every added module"
s = s.replace(a, "4. **S2b, the proper fusion of the past (asked 2026-11-18, run the same day, D121).** Chapter 6 holds the design and the result. A pass adds a past block to the champion and the S4 stage 2 runs over it as a short confirmation; a TIE or a NO leaves the champion as it is.\n5. **Cost.** Every added module", 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
