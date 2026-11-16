p = "experiments/E320_soft-attach-replaces-snap/experiment.md"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


# N9: the git dates are authoritative
rep("(A2, 2026-11-13 00:15, after the second critic and the lane lead's measurement above)", "(A2, commits 00:13 - 00:14 of 2026-11-13: the git dates are authoritative; after the second critic and the lane lead's measurement above)")
rep("**Rule (the lane lead's final form, 2026-11-13 00:25).**", "**Rule (the lane lead's final form, commit 2de8614036).**")
rep("## 0. The card (what the tools run)", "(A3, before the run, after the critic re-check of d406ad9c: card ACCEPT, code SHIP) N1: the cover radius of B's long links follows the tight wiring radius; N2: the tight read-outs are a contact threshold, not a learned wiring, and carry the guards; N4: the kill criterion reads the mean of the last 10 log rows (`jdiag.py attlog`); N5: `jmaneuver.parse_variants` with asserts; N6: the within-1 m share of the snap cards is about the keypoints; N10: the dup_frac margin, the prediction's own verdict and the re-score of the E312 cards (below). The by-hand paragraph is not edited after the run starts.\n\n## 0. The card (what the tools run)")

# N10: topo precision, the dup_frac margin, the prediction's verdict
rep("topo precision of u' equals c's)", "topo precision of u' is within 0.005 of c's: 0.6758 / 0.6854 against 0.6804 / 0.6863). **The dup_frac guard is tight for u' itself:** u' uses 0.0062 / 0.0053 of the 0.01 margin (0.1809 / 0.1610 against c's 0.1747 / 0.1557), so a dup_frac guard CUT with topo F1 passing is plausible and is a CUT by the written rule")
rep("(2) The prediction (a - c = -0.006, i.e. a - u' = +0.0059 / +0.0044) sits at the KEEP bound: close to a coin flip near the bound, by the numbers alone.", "(2) The prediction (a - c = -0.006, i.e. a - u' = +0.0059 on seed 0 and +0.0044 on seed 1) sits at the KEEP bound: close to a coin flip near the bound, by the numbers alone, and **the stated prediction itself would be a TIE** by the rule (KEEP on seed 0, TIE on seed 1: KEEP needs both seeds).")

# N2: the tight read-outs
rep("**If either tight read-out reaches c - 0.01 on both seeds (above 0.6725 / 0.6764) the wiring is learned (soft) and not a rule; if neither does, the follow-up card E322 \"soft wiring\" (`experiments/E322_soft-wiring/experiment.md`, a stub) is the next end-to-end step after E320.**", "**These are a contact wiring (a threshold around the attached end) instead of choosing among several pieces: still a distance rule on the output (a contact radius of 1.5 m / 1.0 m plus the 45 degree gate), NOT a learned wiring, and nothing in this card calls it learned or soft. The criterion of a tight read-out: topo F1 at least c - 0.01 (0.6725 / 0.6764) AND the guards F1@1.0, dup_frac and junction F1 against c, on both seeds (the junction guard may be evaluable there: it is reported and it counts if it fails; the vertex guard as in the rule). A tight read-out that meets it shows that a contact threshold can replace the one-piece argmin at the read-out; it does not make the wiring learned. The follow-up card E322 \"soft wiring\" (`experiments/E322_soft-wiring/experiment.md`, a stub) stays the next end-to-end step after E320 either way, and is the more urgent one if neither tight read-out meets the criterion.**")

# N4: the kill criterion reads the mean of the last 10 log rows
rep("or at the last logged step of train.log the attachment accuracy over the keypoints with at least one candidate (`att_acc`) is not above the nearest-candidate baseline (`att_acc_nearest`) by 0.02;", "or the attachment accuracy over the keypoints with at least one candidate (`att_acc`) is not above the nearest-candidate baseline (`att_acc_nearest`) by 0.02, both the mean of the last 10 rows of `runs/e320_a_s<S>/log.jsonl` weighted by `att_n` (`jdiag.py attlog e320_a_s<S>`);")
rep("the criterion above is read from `train.log` afterwards, and a run that meets it is reported as CUT-by-kill with its numbers.", "the criterion above is read afterwards, and a run that meets it is reported as CUT-by-kill with its numbers. **The statistic is the mean of the last 10 rows of `runs/e320_a_s<S>/log.jsonl` (steps 2100 - 3000) weighted by `att_n`, `bash tools/env/_py14.sh jdiag.py attlog e320_a_s<S>`**: one row is one batch of 8 frames, the standard error of `att_acc - att_acc_nearest` there is about 0.05 against the margin 0.02, so a single row decides by noise.")

# N6: the within-1 m share of the snap cards is about the keypoints
rep("**Reported next to the verdict:** the share of connector ends within 1 m of a delivered piece end (`ends_within_1m_of_a_delivered_piece_end` in every card the scorer writes: attached ends and raw keypoints),", "**Reported next to the verdict:** the share of connector ends within 1 m of a delivered piece end (`ends_within_1m_of_a_delivered_piece_end` in every card the scorer writes: the `entry` / `exit` fields are the ends the connector is built from BEFORE any snap, attached ends under `--attach`, keypoints otherwise, and `keypoint_entry` / `keypoint_exit` are the raw keypoints; **on a `--snap` card (c, u') the share is therefore about the keypoints, the snapped end is the piece end itself: compare c and u' with the `keypoint_*` fields of the attached card**),")

# N1 (named difference) and the tooling
rep("**Differences to the snap that the result must be read with:**", "**Cover radius (N1, amendment A3):** under a tight wiring radius r the long links of B are dropped only when a connector's ends are within min(4 m, r) of the link's ends (`add_connectors(wire_pos=r)`; with `wire_pos=None` it stays COVER_POS = 4 m, bit for bit), so a connector that is not wired to the pieces does not delete B's long link either; `long_links_dropped` is reported next to `wired_both_ends` for every card.\n**Differences to the snap that the result must be read with:**")

# N10: the re-score of the E312 cards
rep("- **Start** (not started;", "- **Re-score of the E312 cards (lane lead, `temp/day12/day12e_rescore.sh`, waits for DAY12D_DONE):** it re-scores E312's `--snap` (c) and `--snap --snap-nogeom` (u') cards once to add the per-frame counts (the old cards are kept as `.pre_counts`, the numbers are checked equal), so that the paired bootstrap of a - c and a - u' (`jdiag.py boot`) can run, and it gives `long_links_dropped` and the within-1 m share for c and u'.\n- **Start** (not started;")
open(p, "w", encoding="utf-8").write(s)
print("card5 ok")
