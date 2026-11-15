p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
a = s.index("## Roadmap (D87 order"); b = s.index("5. Simplification:")
new = '''## Roadmap (D87 order; one at a time; card + fresh `ar-critic` before any code or GPU) - state 2026-11-13 21:30
1. DONE: D87 status lines in the cards E320 and E322; **E325** = the learned wiring head (card v2 after the critic: decisive kill with margin 0.02 against the zero-parameter twin, prediction 0.79 +- 0.02 that excludes the rule's 0.75, in-sample guard = pilot + heavy-tailed jitter + distance-only ablation + VAL cross-validation, places for ties, 545 parameters, 1.5 GPU h); order of building: twin + `wire_choice` path first, pilot, TRAIN features, head.
2. **E324** = E320 refined (card v2, second critic pass running): CPU diagnosis done (`evaluation/jdup.py`, six read-outs): the guard excess of E320 = three equal steps (the read-out choice nogeom u' - c +0.0062 / +0.0053, the recipe +0.0064 / +0.0030, the attached ends +0.0058 / +0.0011) and 71-78 % of the connector duplicate points overlap another connector; false curves +788 / +352 (E316: -716 / -1219). Arms: s = E320 recipe at 6000 pair steps (no new code, read against E316), n = a connector-level Matrix-NMS layer on the pair probabilities (new code). Then ONE stack card on the recipes that already ran.
3. **E292** card v3 (critic REVISE answered): c2 = the same fine-tune without the layer in the rule and run FIRST, D87 premise + E279 stated, verdict cells, scorer plumbing (tag, refuse to overwrite, pyramid assert), phase 9 plumbing; code only after a second look at the card is not needed unless the design changes, the code gets its own critic pass.
4. **E299** time card v2 (critic REVISE answered): OUTPUT-space state from the network's own no-grad forward, metric = precision at c's recall (jpair_pr, logit 0 and -2.2), c2 = flag-off retrain as the noise control, step 0 (CPU: ceiling of logit-only flips + replay at weight 0.5) is the kill; budget 9 GPU h; no gradient through time.
'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s); print("ok")
