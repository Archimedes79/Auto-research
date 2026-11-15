import re
p = "experiments/E344_e2e-completion-plan/experiment.md"
s = open(p, encoding="utf-8").read()
i = s.index("## 7. Questions for Mirko")
j = s.index("## 8. Referee")
new7 = """## 7. Questions for Mirko, with the DEFAULT taken (2026-11-14 20:40; he can correct each: D86 says the lane chats do not wait)
The lane lead's order of 2026-11-14 (D86: never wait for Mirko): for each open question the safe default is the cheaper, more end-to-end variant with the D86 weights (detection up, simple, little compute, E2E differentiable). Each default is labelled "default, he can correct"; a correction changes the plan, not anything already run.
1. **Grid-anchored dense query head (section 4) or free DETR-style queries (B2)?** Default: **the grid-anchored head (Q-Lane, with B0 on the 2 m grid as the cheaper start)**; B2 stays parked, D84 point 6 (no unstructured transformer head) is not lifted for B2 and is read as not applying to a grid-anchored, convolution-first head with local attention. Reason: anchored queries train on 8 414 frames (the CenterNet-style heat of E307 shows it), free queries need far more data and cost more compute and code (D86). Q0 decides 2 m against 1 m on the ground truth before any training.
2. **The price for removing the selections: the equal-accuracy bar** (snapped and unsnapped topo F1 within 0.01 of E324 s on both seeds, guards held) and **may Q-Lane replace B's node / link heads at the end?** Default: **yes to both**, with the speed condition of section 4 (ms no worse than B's node + link read-out plus 1 ms) checked at every stage; until Q2 passes its gate B's heads stay and nothing is switched off. Reason: a bar tighter than the seed noise (0.01 in S) would make every stage INCONCLUSIVE, and a replacement that is slower fails D86 (little compute).
3. **Priority against D99 (smart pruning) and track A on the GPU.** Default: **track A first (E325, E334, E317 - E319 are booked and running through day20), D99 step 0 (E345) on the CPU and the first free GPU slot, Q0 on the CPU in parallel, Q1 and later only after the critic pass of the Q0 card and a free slot**; run.py's first-come-first-served line decides the GPU order (D100: two slots, no weekend rule). Reason: nothing on track B needs the GPU before Q1, and the CPU is free for Q0.
4. **What the planner needs from the output.** Default: **probabilities and the position uncertainty per node and edge** (existence, type, mean, sigma, heading; edge probability), the consumer's threshold the only selection left (as written in section 4). Reason: the D90 / D91 time fusion and a planner use uncertainty; the plan costs no extra for it.

"""
s = s[:i] + new7 + s[j:]
s = s.replace("Not yet: a fresh `ar-critic` (cards) on this plan and on the Q0 card before any code.", "A fresh `ar-critic` (cards) on this plan (started 2026-11-14 20:40 after the defaults of section 7 were written) and on the Q0 card (not written yet) before any code. Q0 is the first card of track B: CPU, ground truth only, grid size as a parameter (2 m and 1 m).")
open(p, "w", encoding="utf-8").write(s)
print(len(s))
