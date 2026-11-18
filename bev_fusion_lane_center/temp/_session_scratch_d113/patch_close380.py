import os, re
root = os.environ["BFLC_ROOT"]
p = os.path.join(root, "experiments", "E380_chord-evidence-in-the-token-head", "experiment.md")
s = open(p, encoding="utf-8").read()
i0 = s.index("**Status: DRAFT")
i1 = s.index("**", i0 + 3) + 2
new_status = "**Status: CLOSED 2026-11-17 20:40 by the written gate (AUC), on the lead's decision: the step-0 CPU chord probe did not meet it (AUC gain >= 0.03 on 3 of 5 folds over the geometry baseline, 0 of 5 over geometry + node-head features); nothing built, no GPU hour spent.** A LEAD, not a verdict (below): average precision over geometry + the tokens' node-head features rises on 5 of 5 folds (+0.034 .. +0.143; in-sample VAL, trees, GT-assigned tokens only, a post-hoc observation, labelled so). It may be re-opened ONLY as a NEW card with its own rule written before any run (a new id, e.g. E382: an AP gate, a held-out fold design that does not reuse VAL, the 4.5 GPU h priced against the E379 / E372 results); the lead does not queue it now."
s = s[:i0] + new_status + s[i1:]
a = "**An amended gate ('AP gain >= +0.03 over geometry + tok on at least four of five folds': met, 5 of 5) is proposed here AFTER the probe, by hand: it is post-hoc, labelled so, and is a build gate (spend GPU or not), never a verdict; the lead / the critic decides whether to accept it before any GPU hour; if it is not accepted the card is closed as written.**"
assert s.count(a) == 1
s = s.replace(a, "**Decision (the lead, 2026-11-17 ~20:40): CLOSED AS WRITTEN. The gate was written before the probe and is not met; an AP gate chosen after seeing the AP numbers is a changed bar after the result (a forking path) and is not accepted as the gate of this card; with the in-sample / trees / GT-assigned limits above the AP gain is an upper estimate anyway. The AP observation stays here as a LEAD for a new card (E382, its own rule before any run, a held-out design, priced against E379 / E372), not as a result of this one.**")
a = "(RESULT below: the gate as written is not met; an amendment is proposed)"
assert s.count(a) == 1
s = s.replace(a, "(RESULT below: the gate as written is NOT met: the card is closed)")
open(p, "w", encoding="utf-8").write(s)
p = os.path.join(root, "experiments", "E379_next_candidates.md")
s = open(p, encoding="utf-8").read()
a = "the written AUC gate NOT met (3 of 5 folds), AP over geometry + tok +0.034 .. +0.143 on 5 of 5 folds: an amended gate is the lead's call |"
assert s.count(a) == 1
s = s.replace(a, "**CLOSED by the written AUC gate** (3 of 5 folds, lead 20:40); AP over geometry + tok +0.034 .. +0.143 on 5 of 5 folds is a post-hoc LEAD for a new card (E382) |")
a = "the amended gate (AP >= +0.03 on 4 of 5 folds) is met 5 of 5 but it is post-hoc: the lead decides."
assert s.count(a) == 1
s = s.replace(a, "an AP gate chosen after the result is a changed bar: the lead closed E380 by the written AUC gate (20:40); the AP gain is a LEAD for a new card with its own rule before any run.")
a = "| 3 | **E380 chord evidence**"
s = s.replace(a, "| 3 (CLOSED) | **E380 chord evidence**")
open(p, "w", encoding="utf-8").write(s)
print("ok")
