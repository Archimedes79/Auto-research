import os
root = os.environ["BFLC_ROOT"]
p = os.path.join(root, "experiments", "E380_chord-evidence-in-the-token-head", "experiment.md")
s = open(p, encoding="utf-8").read()
a = "## 1. Card text\n"
b = '''**Step 0 RESULT (MEASURED 2026-11-17 20:12, CPU, scratch `temp/e380/chord_probe.py`, output `temp/e380/chord_probe.out` / `.json`; run as specified above, with one addition written before the full run: a STRONGER baseline that also gives the classifier the two tokens' 64-channel node-head features `tok`, because the head reads them and the geometry-only baseline is far weaker than the head): VAL 873 frames, B' seed 0, 356,805 pairs of GT-assigned tokens in the allowed gate with a chord above 8 m, 7,062 positives (2.0 %), 5 contiguous folds. AUC of the evidence over the geometry-only baseline (13 pair features + 2 existence logits): +0.0906 / +0.0717 / +0.0300 / +0.0211 / +0.0461 (mean +0.052; folds >= 0.03: 3 of 5); the evidence ALONE separates with AUC 0.951 / 0.924 / 0.954 / 0.977 / 0.946. Over the stronger baseline (geometry + the tokens' node-head features, AUC 0.942 - 0.994): AUC +0.0101 / +0.0049 / -0.0025 / +0.0016 / +0.0014 (mean +0.0031; folds >= 0.03: 0 of 5), but AVERAGE PRECISION (the metric of a task with 2 % positives; AUC saturates) 0.458 -> 0.601, 0.409 -> 0.522, 0.742 -> 0.775, 0.736 -> 0.826, 0.676 -> 0.728: +0.143 / +0.113 / +0.034 / +0.089 / +0.053 (mean +0.086, positive on 5 of 5 folds).
**Reading, with its limits, and what the written gate says.** By the gate AS WRITTEN above (AUC gain >= 0.03 on at least four of five folds, over the geometry-only baseline) the probe is NOT met: 3 of 5 (0 of 5 over the stronger baseline): by the letter the card is closed with these numbers. The author's reading: the AUC criterion was the wrong metric for a 2 % positive rate (it saturates at 0.99 once `tok` is given) and the geometry-only baseline the wrong reference (the head has `tok`); the AP gain over the stronger baseline is positive on all five folds (+0.034 .. +0.143) and large next to the long-link AP of the champion head (0.26 - 0.29 on ALL valid candidate pairs, a different set, not comparable). **An amended gate ('AP gain >= +0.03 over geometry + tok on at least four of five folds': met, 5 of 5) is proposed here AFTER the probe, by hand: it is post-hoc, labelled so, and is a build gate (spend GPU or not), never a verdict; the lead / the critic decides whether to accept it before any GPU hour; if it is not accepted the card is closed as written.** Limits of the probe: in-sample on VAL (a probe in the critic's pattern of E374, folds are contiguous blocks of frames), a boosted-tree classifier on pairs of GT-ASSIGNED tokens (the existence question is removed, the head also faces unassigned tokens and duplicates), `tok` in a tree model is not the attention context the head has (the real baseline is stronger than the probe's: the AP gain is an upper estimate of what the head can still gain), the 2 % positives come from all frames including those without a junction.

## 1. Card text
'''
assert s.count(a) == 1
s = s.replace(a, b)
a2 = "**Build only if the AUC gain is >= 0.03 on at least four of five folds** (a probe in the critic's pattern of E374: in-sample on VAL, never a verdict); else the card is closed with the numbers"
assert s.count(a2) == 1
s = s.replace(a2, a2 + " (RESULT below: the gate as written is not met; an amendment is proposed)")
s = s.replace("**Status: DRAFT, not run, no critic yet**", "**Status: DRAFT, not run, no critic yet; step 0 (CPU chord probe) DONE 20:12: gate as written NOT met (3 of 5 folds over the geometry baseline), AP gain over the stronger baseline +0.034 .. +0.143 on 5 of 5 folds: an amended gate is proposed, the lead decides**", 1)
open(p, "w", encoding="utf-8").write(s)
# the candidates file
p = os.path.join(root, "experiments", "E379_next_candidates.md")
s = open(p, encoding="utf-8").read()
a = "## Considered and not written as cards"
b = '''4. **Chord probe of E380** (`temp/e380/chord_probe.py`, VAL 873 frames of B' seed 0, 356,805 pairs of GT-assigned tokens with a chord above 8 m, 7,062 GT long links; boosted trees, 5 contiguous folds): the eight chord numbers of B''s pyramid maps alone separate true from wrong long pairs with AUC 0.92 - 0.98; over geometry + existence the AUC gain is +0.052 (3 of 5 folds >= 0.03: the written gate of the card, >= 4 of 5, is NOT met); over geometry + the tokens' 64-channel node-head features the AUC gain is +0.003 (saturated at 0.94 - 0.99) but the average precision rises on all five folds, +0.034 .. +0.143 (mean +0.086, 0.46 -> 0.60, 0.41 -> 0.52, 0.74 -> 0.78, 0.74 -> 0.83, 0.68 -> 0.73). Reading: the evidence carries information about WHICH pairs are real that the node-head features do not (AP), the written AUC gate was the wrong metric; the amended gate (AP >= +0.03 on 4 of 5 folds) is met 5 of 5 but it is post-hoc: the lead decides. In-sample on VAL, GT-assigned tokens only, trees not attention: an upper estimate of the head's gain.

## Considered and not written as cards'''
assert s.count(a) == 1
s = s.replace(a, b)
a = "| 3 | **E380 chord evidence**"
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s)
print("ok")
