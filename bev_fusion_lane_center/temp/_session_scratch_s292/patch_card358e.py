import io
p = "experiments/E358_wiring-head-on-attached-ends/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
a = "## 6. Summary"
assert s.count(a) == 1
txt = ("**Live check on the GPU (the lead's deviation, 2026-11-16 12:08, run.py job `E358.live_s0`, same tolerance and fields): REFUSED on seed 0: `frames that differ: {46: ['pairs']}`** (23 of 24 frames equal in every field; frame 46 differs in the field `pairs` only; `kp_p` and `gts` are equal, so the pair lists have the same length and the difference lies in the VALUES of an attached end or its tangent, not in a pair that appears or disappears near p = 0.1, which the tolerated exception of the rule covers). Seed 1 was not run: `E358.live_s1`, the two training runs and the scoring jobs are blocked by `--after` (run.py status `blocked`), as designed. **By the rule written before any number (REFUSED: no GPU training job, no change of the tolerance) the card stops here; the loader question for the lead:** the live pass uses `map_mode=\"exact\"` and batch size 4, the stored dump `\"error\"` and batch size 8 (`jwire_data.py:158`, `jends_eval.py:42`); the attached end of a pair is a softmax over the nearest 8 candidate nodes (`jends.py:117-126`), so a near-tie of the candidate distances or a different batch size can move one attached end by more than 2e-3 in one frame. Options for the lead (none taken): (A) run the check once more with the dump's own settings (batch size 8 and the dump's map mode) to test the loader hypothesis; (B) a documented shift (one frame of 24, field `pairs` only), which is a change of the rule and is not mine to make; (C) stop the card.**" + nl + nl)
s = s.replace(a, txt + a)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
