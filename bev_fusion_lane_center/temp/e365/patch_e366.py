p = "experiments/E366_e365-decode-selected-on-train/experiment.md"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


rep("Card of 2026-11-17 (version 1, 01:10 system clock, before any sweep;",
    "Card of 2026-11-17 (version 2, 01:45 system clock, after the first critic pass (fresh ar-critic, cards: REVISE, 7 findings, taken in: the JSON rule is not the verdict, no second cell, the grid edge, the sample fixed, the verdict aligned with E365, the TRAIN bias said, the two seeds are two heads); before any sweep;")
rep("the sweep code is being added by a coder; nothing run.", "the sweep code is being added by a coder (`score --link-thr --keep-thr --no-none-gate` and `sweep` are not yet in `jtoken_eval.py`); nothing run.")
rep("TRAIN sample (873 evenly spaced TRAIN frames, link threshold x keep threshold x none gate on / off, 24 cells)",
    "TRAIN sample (every third frame of the 873 evenly spaced TRAIN frames: 291 frames, fixed now; link threshold x keep threshold x none gate on / off, 24 cells)")
rep("**The limit, said now:** the TRAIN sample is cleaner than VAL (B' recall 0.926 against 0.862, S1 topo F1 0.61 against 0.55), so a decode selected there may be too loose or too tight for VAL; the VAL result is the honest test, a VAL sweep (in-sample) is reported only as a diagnosis next to it, never chosen from.",
    "**The limit, said now:** the TRAIN sample is cleaner than VAL (B' recall 0.926 against 0.862, S1 topo F1 0.61 against 0.55) and the heads were trained on those frames (their probabilities are more confident there), so a decode selected there is biased TIGHTER than the VAL optimum; the VAL result is the honest test, a VAL sweep (in-sample) is reported only as a diagnosis next to it, never chosen from. **The chosen cell is applied ONCE. No second cell is tried, the grid is not extended afterwards, and if the result on VAL is not better than the record (a gain below +0.01 on a seed) the verdict stands and E367 reads its ref at the record decode.** **What will probably happen, said before the sweep:** precision 0.86 against recall 0.40 - 0.47 means topo F1 rises with every loosening, so the rule will probably pick the loosest cell the guard allows (T = 0.05, K = 0.7, none gate off); the guard dup_frac <= 0.05 is measured on TRAIN (dup_frac 0.005 there), VAL duplicates can be higher: the VAL dup_frac is reported next to the result. The two seeds are two different trained heads, not noise draws.")
rep("PASS-DECODE = G-topo and G-found hold on both seeds (the cost and simplicity gates are those of E365 S1, unchanged: same heads); MISS-DECODE = G-topo fails on both seeds; else INCONCLUSIVE. The chosen cell is the decode of record of every E367 arm (written there before any E367 result exists).",
    "PASS-DECODE = G-topo and G-found hold on both seeds (the cost and simplicity gates are those of E365 S1, unchanged: same heads; the thresholds are on the output only, so G-simple holds); MISS-DECODE = G-topo fails on both seeds or G-found fails on both seeds (as E365); else INCONCLUSIVE (reported with the diagnosis; no action beyond the reading of E367's ref). **The JSON rule above is NOT the verdict:** `decide.py` is not used (a gain of +0.011 would read KEEP while the gate needs +0.06 to +0.09); the verdict is by hand: the gate of E365 section 0. The paired scene-bootstrap interval dec - ref is reported per seed. E367 does not take over this cell: it selects the decode PER HEAD by this rule on that head's own TRAIN sample; this card's VAL result of the S1 heads is E367's `ref`.")
rep("the chosen cell is looser than the record (T <= 0.2) with probability 80 %;", "the chosen cell is looser than the record (T <= 0.2) with probability 85 % (the grid edge, see above);")
rep("24 cells x 2 seeds x 873 TRAIN frames at about 2 - 3 minutes each: about 40 minutes per seed (niced, two seeds in parallel), or a reduced sample of 300 frames (stated if used: the sample is the first 300 of the 873 evenly spaced frames).", "24 cells x 2 seeds x 291 TRAIN frames (fixed above): about 13 minutes per seed, niced, two seeds in parallel.")
open(p, "w", encoding="utf-8").write(s)
print("ok")
