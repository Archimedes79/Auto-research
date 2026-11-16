import io
p = "experiments/E319_heading-guided-exchange/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
i = s.index("## 5. Result")
head = s[:i]
new = '''## 5. Result (2026-11-15, by the written gate of section 0 / `temp/e319/kill_check.py`, exit 1 on both seeds; the arm is closed at stage 1; runs `E319.m_s1` 01:28 - 02:25 (3391 s) and `E319.m_s0` 05:53 - 06:48 (3261 s), VAL, TEST untouched)
**GATE NOT MET on BOTH seeds: the pair stage was not run on either seed, so there is no topo F1 and no verdict token; the kill (below -0.03 keypoint F1) did not fire.** By the card ("if the topo F1 ties, the head without the block stays", and here the maps do not even gain what the gate asks) the head without the block stays; the idea is closed at the gate. D87 / D88 do not apply (a maps block, not an E2E replacement of a rule step).

| maps stage after 3000 steps, against E307 of the same seed (`temp/e319/kill_s<S>.log`) | seed 0 | seed 1 |
|---|---|---|
| keypoint F1: E319 / E307 / difference (gate bar +0.01) | 0.7259 / 0.7233 / **+0.0026** | 0.7132 / 0.7051 / **+0.0082** |
| entry recall beyond 35 m: E319 / E307 / difference (gate bar +0.01) | 0.6487 / 0.6522 / **-0.0035** | 0.6437 / 0.6509 / **-0.0073** |
| gate | not met | not met |

**Reading (section 3 applies).** The gains of the maps stage are inside the seed spread of E307 itself (keypoint F1 0.7233 / 0.7051 = 0.018, section 3), the far-range entry recall FELL on both seeds (-0.0035 / -0.0073, the spread of E307 there is 0.0013: the fall is larger than that spread on seed 1 and about three times on seed 0), and the pattern that section 3 says is read, a gain of 0.03 or more in the far-range AND the crowded entry recall on both seeds, is absent: the crowded entry recall was not read because the gate was not met. **Prediction (section 3, revised 2026-11-14):** keypoint F1 +0.008 hit on seed 1 (+0.0082) and missed on seed 0 (+0.0026, by 0.0054); entry recall beyond 35 m +0.03 missed on both seeds (the sign is wrong); topo F1 +0.005 not measured (no pair stage). A block that changes the DIRECTION of the context (heading-guided forward / backward exchange along the centre-lane heading, 0.330 GMAC, 33 218 parameters, two rounds up to 16 m each way) did not help the maps that the pair head reads, on top of dilated blocks that already see 38 m. Cost of the arm: about 1.8 GPU hours (the two maps stages), the pair stages (about 4 GPU hours each) were not spent. Nothing here supports or contradicts the E2E argument for graph-side direction logic; no rule step is involved.
'''
io.open(p, "w", encoding="utf-8", newline="").write(head + new)
print("ok")
