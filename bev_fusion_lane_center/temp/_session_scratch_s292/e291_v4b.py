import io

p = "experiments/E291_sd-road-level-lane-counts/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


# 6 m -> 10 m in the named steps (critic leftover)
rep("the nearest-arm `argmin`, the 6 m presence gate, the rank sort", "the nearest-arm `argmin` (an arm within 10 m and 60 degrees), the 10 m presence gate, the rank sort")
# the folder names of the dumps as built
rep("`runs/e291_<arm>_<run>_<readout>/`", "`runs/e291_<a|b|c>_s<S>_<readout>/` (read-outs clean, noise10, noise30, pm1, u, s, sc)")
# arm b recipe: the two 20 % cases as built
rep("15 % of the arms are marked UNKNOWN, 20 % of the junctions have no map: `--prior-noise mix`", "15 % of the arms are marked UNKNOWN, 20 % of the junctions have no map and another 20 % have ALL counts unknown with the geometry kept (so that the U read-out is in distribution): `--prior-noise mix`")
# the C clause: U also zeroes the count-normalised features
rep("**C-: the gain, if any, is the arm geometry, not the count.**", "**C-: the gain, if any, is the arm geometry, not the count.** (Wording after the code critic of 11:50: U zeroes the count n / 4 AND the two count-normalised features of the keypoints in that arm, rank / n and (found - n) / 4; so C+ credits 'the count together with the within-arm rank and found-minus-count features', which the E312 rank matrix already carries in part; it is not the count alone.)")
# the kill (5): s and sc
rep("(5) the lower end of the 95 % interval of (the SHUFFLED-table read-out of b minus the twin) is above 0 (the input would carry something other than the table's count)", "(5) the lower end of the 95 % interval of (the SHUFFLED-table read-out of b minus the twin) is above 0, for the read-out s (the table of another frame) OR sc (the frame's own arm centres and axes, the counts of another frame's arms; added after the code critic because s is close to 'no map': the partner's arms lie in the partner's ego frame, so most keypoints of s have no arm within 10 m; the dump reports the share of s keypoints with an arm) (the input would carry something other than the table's count)")
# validity caveat + arm a note, before the section 4 referee
cav = """- **Validity caveats of the build (code critic 2026-11-15 11:50, for the report):** (1) the arm axis passes through the mean of the GT end centroids, so the per-keypoint lateral offset (lat / 20) and the heading cosine measured against it carry the GT geometry of the junction (kept in U, under at most 2 m / 5 degrees of noise in b's training): G (b clean minus the twin) can be a POSITION oracle, not a count effect; the C clause (clean minus U) is what separates the two. (2) In training the missing ends are injected as valid keypoints (`jends.py`), so 'found < n' co-occurs with an injected true keypoint in training and never at evaluation: this probably biases G downward and no read-out can show it. (3) The clean row is an upper bound (the table is built from GT end clusters), not a real-map result. (4) Arm a (trained clean) is NOT in the tokens (the rule uses b and the twin only): under the GPU budget of 2026-11-15 it is queued last and may not run today; its read-outs would fill the 'trained at noise X, read at noise Y' table only. (5) A retried arm resumes from `ckpt.pt` with a different data order: the ledger row says 'resumed from step N'.
"""
rep("## 4. Referee", cav + "\n## 4. Referee")
rep("A pass on the code before any GPU run.", "A pass on the code before any GPU run: code critic 2026-11-15 11:50 (mode code, staged build): REVISE (small), no finding blocked the twin's first start; taken: the stage guard (`BFLC_STAGE` set unconditionally, the guard requires the path), the sha256 freeze of the staged files with a refusal of an existing run directory, the read-out sc, atomic dump folders, the README rows, the C-clause wording and the caveats above; no finding changes the training of the arms.")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
