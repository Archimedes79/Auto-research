import os
R = os.environ["BFLC_ROOT"]


def sub_file(path, pairs):
    p = os.path.join(R, path)
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


sub_file("training/grid/test_lane_map.py", [
    ('"raw = error_like(self.pts(i), rng, *TRAIN_SHIFT)", "raw = error_like(self.pts(i), rng, *ERROR_SHIFT)"', '"raw = error_like(self.pts(src), rng, *TRAIN_SHIFT)", "raw = error_like(self.pts(src), rng, *ERROR_SHIFT)"'),
    ('"return error_like(self.pts(i), rng, *GROSS_SHIFT)", "return error_like(self.pts(i), rng, *TRAIN_SHIFT)"', '"return error_like(self.pts(src), rng, *GROSS_SHIFT)", "return error_like(self.pts(src), rng, *TRAIN_SHIFT)"'),
    ('("the draw seed ignores the draw index", "zlib.crc32(self.tokens[i].encode()), n])", "zlib.crc32(self.tokens[i].encode()), 0])")', '("the draw seed ignores the draw index", "zlib.crc32(self.tokens[i].encode()), n] +", "zlib.crc32(self.tokens[i].encode()), 0] +")'),
    ('("the draw seed is the frame index, not the token", "zlib.crc32(self.tokens[i].encode()), n])", "i, n])")', '("the draw seed is the frame index, not the token", "zlib.crc32(self.tokens[i].encode()), n] +", "i, n] +")'),
    ("""'.share_memory_() if mode == "train"', ' if mode == "train"'""", """'.share_memory_() if mode in ("train", "other")', ' if mode in ("train", "other")'"""),
])
sub_file("training/grid/test_jmap_train.py", [
    ('"init * 100003 + step, a.no_map)", "a.seed * 100003 + step, a.no_map)"', '"init * 100003 + step, a.no_map, a.map_mode == \\"other\\")", "a.seed * 100003 + step, a.no_map, a.map_mode == \\"other\\")"'),
    ("""'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_map")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr")'""", """'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_map", "map_mode")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "map_mode")'"""),
])
print("ok")
