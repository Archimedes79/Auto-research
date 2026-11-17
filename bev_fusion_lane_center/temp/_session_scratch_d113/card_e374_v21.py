import io
p = "experiments/E374_s3-cross-attention-to-the-sd-map/experiment.md"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("at most 512 samples per frame; when more exist the samples nearest the ego are kept; the build counts the frames that are truncated (TRAIN and VAL) and the card is revised if more than 0.1 % are (the measured maxima of the v1 lane map were 773 TRAIN / 687 VAL; the road map is smaller: to be measured by the build, not assumed).",
    "at most **704** samples per frame (v2 said 512; **revised 2026-11-17 10:00 by the card's own trigger**: the build measured, TRAIN 8414 frames, exact map mean 275 / max 653, 225 frames (2.67 %) above 512, the `error` map mean 246 / max 614, 92 frames (1.09 %) above 512; VAL 873 frames: max 487 / 460, none above 512; the dropped samples of a capped frame lie 41 m and farther from the ego); when more than 704 exist the samples nearest the ego are kept (a named hard step; the rebuilt `check` reports the new counts).")
rep("about 20 k parameters. No state, no recursion.", "20,264 parameters as built (embedding MLP width 32; the 'presence flag' of the embedding and of the geometry = the lane count is known, count > 0). No state, no recursion.")
rep("(an MLP of the fields above -> d, LayerNorm)", "(an MLP of the fields above -> d, LayerNorm)") if False else None
io.open(p, "w", encoding="utf-8").write(s)
s = io.open(p, encoding="utf-8").read()
s = s.rstrip("\n") + """

**Build notes (2026-11-17 10:00, the coder's report, reproduced here so that the card matches the code):** the data module reproduces the step 0 of this card to the digit (71 numbers, largest difference 0.0, full VAL); the fixture on 100 VAL frames: the `error` map's rigid shift 1.001 - 1.989 m, yaw std 1.23 degrees (the draws equal `map_sim.make_map`'s), nearest-sample distance of the corrupted to the exact map median 0.12 / p90 0.71 / p99 1.37 m against 1.07 / 2.02 / 3.17 when the pose error is not undone, 1.52 / 3.06 / 3.97 with the shift sign flipped, 0.75 / 1.97 / 3.66 with the rotation transposed (the three wrong conventions fail), the S map is another scene's (own-roads share 0.985 for `error`, 0.484 for S); the `train` mix: the 'counts unknown' draws keep the error-like geometry, 'gross' = a 4 - 8 m rigid shift of the error-like map (half) or the error-like map of another scene's frame (half); the read-out `gross` direction is seeded per frame by the token; arm z equals `jtoken_train`'s loss and final weights bit for bit over 20 steps (both heads, two init seeds); the kill measure v is `jpast.PastBlock.use` / `UseMeter` (the other-sample reference; in the stream-like read-out the donor is the S map, so v is 0 in mode S by construction); the capacity control c is not built (a loader mode where every map is another scene's); the hook `map_block` after `layers[1]` exists in `jtoken.py` (27eb572f); `{}` / None / all-invalid maps = the none key only (the block is the identity only at zero initialisation). Code: `training/data/lane_map_data.py`, `training/grid/jmap.py`, `training/train/jmap_train.py`, `evaluation/jmap_eval.py` + three test files with 33 + 60 + 42 hand mutants (cap raised: rerun).
"""
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
