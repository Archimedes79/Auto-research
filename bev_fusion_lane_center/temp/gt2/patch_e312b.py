p = "experiments/E312_combined-end-maps/experiment.md"
s = open(p, encoding="utf-8").read()
old = "\n\n## 1. Order of work"
assert old in s
add = "\n\n**Amendment 2 2026-11-12 12:00 (before any run of this card):** E307 (heat sigma 1.0 m alone, run 10:57 - 11:48, scored 11:56) is a KEEP by its own rule and far larger than predicted: keypoint F1 0.723 / 0.705 against 0.630 / 0.642 of E286 j1, crowded entry recall 0.789 / 0.789 against 0.542 / 0.576, entry recall beyond 35 m 0.652 / 0.651 against 0.551 / 0.562, zone IoU +0.007 / +0.002 (E307 card). The far-range weight of E304 was stopped at step 200 (a single-factor test on the old sigma is no longer the question; the arm built from both is round 2, card E313 if time allows). Arm m of this card is therefore: the pair head `--e2e --scales` on the E307 maps (sigma 1.0, no far-range weight) instead of the maps m2; the rule, the control c (E305) and the read-out are unchanged. The maps arm m2 (sigma 1.0 + far-w 2) stays the follow-up."
s = s.replace(old, add + old, 1)
s = s.replace("junction maps head with heat sigma 1.0 m and far-range weight 2 (arm m2), then", "junction maps head with heat sigma 1.0 m (E307; the first version of this card added a far-range weight 2, see amendment 2), then", 1)
open(p, "w", encoding="utf-8").write(s)
p = "temp/e312/start_pairs.sh"
s = open(p, encoding="utf-8").read()
assert "runs/e312_m2_s$S/model.pt" in s
s = s.replace("runs/e312_m2_s$S/model.pt", "runs/e307_n_s$S/model.pt")
s = s.replace("on the m2 maps", "on the E307 maps (sigma 1.0)")
open(p, "w", encoding="utf-8").write(s)
print("ok")
