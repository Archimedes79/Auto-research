R = "bev_fusion_lane_center/results/E2E_Final_Report_2026-11-18/parts/"
p = R + "06_challengers.md"
s = open(p, encoding="utf-8").read()
old = "If S2b wins its rule, the end-to-end confirmation is the S4 stage 2 over it (the loss reaching the node head with the past block in the head); a win at stage 1 alone is not the champion."
new = "S2b is tested on the cached B' tokens (stage 1, card E387). The end-to-end confirmation, the S4 stage 2 with the past block in the head, is open and not part of this report: a stateful head needs sequences of three frames through B' in the live path, which is new code. A win at stage 1 alone is therefore labelled as such and is not the champion."
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("the card and the runs belong to the lane chat).", "card E387, `experiments/E387_fused-cross-attention-block-s2b`; the card and the runs belong to the lane chat).")
s = s.replace("(card, two seeds, VAL 873 frames; the numbers are copied from the card of the lane chat, never typed here)", "(card E387, two seeds, VAL 873 frames; the numbers come from the card only, never typed here)")
s = s.replace("the S2b card (lane chat).", "E387 (the S2b card, lane chat).")
open(p, "w", encoding="utf-8", newline="\n").write(s)
p = R + "14_plan.md"
s = open(p, encoding="utf-8").read()
old = "A pass adds a past block to the champion and the S4 stage 2 runs over it as a short confirmation; a TIE or a NO leaves the champion as it is."
new = "Card E387, tested on the cached B' tokens (stage 1). A pass is labelled a stage-1 result; the S4 stage 2 with the past block (a stateful head, three-frame sequences through B' in the live path, new code) stays open; a TIE or a NO leaves the champion as it is."
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("A pass adds", "A pass adds")
open(p, "w", encoding="utf-8", newline="\n").write(s)
