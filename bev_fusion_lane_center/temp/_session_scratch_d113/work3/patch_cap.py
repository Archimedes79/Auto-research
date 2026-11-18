import os

R = os.environ["BFLC_ROOT"]


def sub(path, pairs):
    p = os.path.join(R, path)
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:70])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


sub("training/data/lane_map_data.py", [
    ("At most MAX_SAMPLES = 512\nsamples per frame", "At most MAX_SAMPLES = 704\nsamples per frame"),
    ("MAX_SAMPLES = 512                                                            # samples per frame: the nearest the ego are kept",
     "MAX_SAMPLES = 704                                                            # samples per frame: the nearest the ego are kept (512 truncated 2.7 % of the TRAIN frames, the exact-map maximum is 653)"),
    ("== (2.0, 0.1, 512, 5, 5.0, 97)", "== (2.0, 0.1, 704, 5, 5.0, 97)"),
    ("far = (np.stack([np.arange(600.0), np.zeros(600)], 1), np.tile([1.0, 0.0], (600, 1)), np.full(600, 7.0))",
     "far = (np.stack([np.arange(800.0), np.zeros(800)], 1), np.tile([1.0, 0.0], (800, 1)), np.full(800, 7.0))"),
    ("np.arange(512.0)) and (t[\"count\"]", "np.arange(704.0)) and (t[\"count\"]"),
    ("np.arange(511.0, -1.0, -1.0)", "np.arange(703.0, -1.0, -1.0)"),
    ("ring = (np.concatenate([np.full((300, 2), 100.0), np.zeros((300, 2))]), far[1], far[2])", "ring = (np.concatenate([np.full((500, 2), 100.0), np.zeros((300, 2))]), far[1], far[2])"),
    ("len(finish(ring)[\"valid\"]) == 512", "len(finish(ring)[\"valid\"]) == 704"),
    ("np.array([0.2] * 511 + [3.0] + [0.2] * 88)))[\"count\"][[0, 1, 511]]", "np.array([0.2] * 703 + [3.0] + [0.2] * 96)))[\"count\"][[0, 1, 703]]"),
    ("cb[\"pos\"].shape == (3, 512, 2) and cb[\"valid\"].sum(1).tolist() == [512, 0, 83]", "cb[\"pos\"].shape == (3, 704, 2) and cb[\"valid\"].sum(1).tolist() == [704, 0, 83]"),
])
sub("training/grid/jmap.py", [
    ("# a padding-only batch row and K = M = 512 (memory below 2.5 GB on the CPU, bf16 autocast finite)", "# a padding-only batch row and K = 512 tokens x M = 704 map samples (the cap of `lane_map_data`; memory below 3 GB on the CPU, bf16 autocast finite)"),
    ("mb = _maps(2, 512, 5, n_invalid=50)", "mb = _maps(2, 704, 5, n_invalid=50)"),
    ("ru_maxrss / 1024 < 2560, \"K = M = 512, batch 2, forward and backward: more than 2.5 GB\"", "ru_maxrss / 1024 < 3072, \"K = 512, M = 704, batch 2, forward and backward: more than 3 GB\""),
])
sub("training/train/jmap_train.py", [
    ("SMOKE_M = 40                                                                # map samples per synthetic frame of the smoke run",
     "SMOKE_M = LMD.MAX_SAMPLES                                                   # map samples per synthetic frame of the smoke run: the cap of `lane_map_data` (the worst case of the memory check)"),
    ("up to n samples near its lane tokens (the token's position + 0.5 m of noise, its heading, 1 - 3 lanes)", "n samples near its lane tokens (the lane tokens used again and again: the token's position + 0.5 m of noise, its heading, 1 - 3 lanes)"),
    ("[:n]\n        tabs.append", "[:n]\n        idx = np.resize(idx, n)\n        tabs.append"),
    ("(K 512, 40 map samples near the lane tokens, no cache)", "(K 512, 704 map samples = the cap, near the lane tokens, no cache)"),
])
sub("training/grid/test_lane_map.py", [
    ('("the cap is 600", "MAX_SAMPLES = 512 ", "MAX_SAMPLES = 600 "),', '("the cap is 600", "MAX_SAMPLES = 704 ", "MAX_SAMPLES = 600 "),\n    ("the cap is 512 again", "MAX_SAMPLES = 704 ", "MAX_SAMPLES = 512 "),\n    ("the cap is 512 again (the real maps are cut)", "MAX_SAMPLES = 704 ", "MAX_SAMPLES = 512 ", "check"),'),
])
sub("training/grid/test_jmap_train.py", [
    ("bm[\"maps\"][\"pos\"].shape == (B, 30, 2), f\"{label}: the smoke maps (3 lanes of 10 tokens, below SMOKE_M)\"", "bm[\"maps\"][\"pos\"].shape == (B, T.SMOKE_M, 2) and T.SMOKE_M == 704, f\"{label}: the smoke maps (the cap)\""),
])
print("patched")
