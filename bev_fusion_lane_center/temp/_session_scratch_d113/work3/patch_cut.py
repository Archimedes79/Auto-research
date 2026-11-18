import os

R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "training/data/lane_map_data.py")
s = open(p, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---------------------------------------------------------------- docstring
sub("""Map of a frame (`sample_roads`): `sd_lib.build_sd(dict(pts=[the points of every lane GT piece]), [])` (carriageway centre lines, a lane count at every vertex, the route is NOT passed), optionally degraded""",
    """Map of a frame (`sample_roads`): `sd_lib.build_sd(dict(pts=[the ROAD RUNS of the lane GT pieces]), [])` (carriageway centre lines, a lane count at every vertex, the route is NOT passed); a road run (`road_runs`) is a piece's points WITHOUT its in-intersection connector
ranges (`piece['segments']`: kind 'connector' and in_intersection, indices i0..i1 cut out; a piece cut in the middle gives two polylines, a run of one point is dropped, a piece that lies wholly in an intersection gives none): the map has the road centre lines up to the junction and
NO turn path through it, as a navigation map (the junction turn paths are lane-level information). The map is optionally degraded""")
sub("""  U       the `error` map with every count 0 (unknown)       C2   the `error` map with every count 2
""", """  U       the `error` map with every count 0 (unknown)       C2   the `error` map with every count 2
  other   the capacity control of the card (arm c): like `train` (the same mix of exact / error-like / unknown / absent / gross draws) but the draw is made on a RANDOM frame of ANOTHER scene, a fresh random choice per read (not a fixed frame-to-map plan, which could be memorised): same
          block, same compute, no information about this frame; the stream is (init seed, token, draw index, 1)
""")
sub("""The scene of a frame (S, the other-scene gross draw) comes from the S2 extension""", """The scene of a frame (S, other, the other-scene gross draw) comes from the S2 extension""")
sub("""MODES = ("exact", "error", "train", "gross", "none", "S", "U", "C2")""", """MODES = ("exact", "error", "train", "gross", "none", "S", "U", "C2", "other")""")

# ---------------------------------------------------------------- road runs
sub('''def roads_of(pts):''', '''def road_runs(piece):
    """the road-level points of a lane GT piece: its points WITHOUT the in-intersection connector ranges (`piece['segments']`: kind 'connector' and in_intersection, the indices i0..i1 inclusive), i.e. the road centre line up to the junction and none through it (the turn paths
    are lane-level information that a navigation map does not hold) -> a list of (n, 2) float arrays, one per run of at least 2 points (a piece cut in the middle gives two, a piece that lies wholly in an intersection none)"""
    pts = np.asarray(piece["pts"], float)
    keep = np.ones(len(pts), bool)
    for sg in piece["segments"]:
        if sg["kind"] == "connector" and sg["in_intersection"]:
            keep[sg["i0"]: sg["i1"] + 1] = False
    edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))                       # run starts and (exclusive) run ends, alternating
    return [pts[a:b] for a, b in zip(edges[::2], edges[1::2]) if b - a >= 2]


def frame_pts(pieces):
    """the polylines `roads_of` gets for a frame: the road runs of every lane GT piece"""
    return [r for pc in pieces for r in road_runs(pc)]


def connector_interior(pieces, margin=2.5):
    """the points of the in-intersection connector ranges of a frame that lie at least `margin` m (along the range) from both its ends: what a road-level map must stay away from (the ends of a range are where the road run stops) -> (n, 2)"""
    out = []
    for pc in pieces:
        pts = np.asarray(pc["pts"], float)
        for sg in pc["segments"]:
            if sg["kind"] == "connector" and sg["in_intersection"]:
                seg = pts[sg["i0"]: sg["i1"] + 1]
                arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
                out.append(seg[(arc >= margin) & (arc[-1] - arc >= margin)])
    return np.concatenate(out) if out else np.zeros((0, 2))


def roads_of(pts):''')

# ---------------------------------------------------------------- dataset
sub('''`gt(token)` -> the point arrays of the lane GT
    pieces of a frame (default: the lane GT store of this checkout);''', '''`gt(token)` -> the lane GT
    pieces of a frame (the dicts with 'pts' and 'segments'; default: the lane GT store of this checkout);''')
sub("""        if self.maps and (mode in ("S", "train") or with_donor):""", """        if self.maps and (mode in ("S", "train", "other") or with_donor):""")
sub("""if mode == "train" else None       # the draw index of every frame (shared with the workers)""", """if mode in ("train", "other") else None       # the draw index of every frame (shared with the workers)""")
sub('''    def pts(self, i):
        """the points of the lane GT pieces of frame i"""
        if self._gt is not None:
            return self._gt(self.tokens[i])
        if self._lane_gt is None:
            import lane_gt_data as LD
            self._lane_gt = LD.lane_store()[0]
        return [np.asarray(p["pts"], float) for p in self._lane_gt.load(self.tokens[i])["pieces"]]
''', '''    def pieces(self, i):
        """the lane GT pieces of frame i"""
        if self._gt is not None:
            return self._gt(self.tokens[i])
        if self._lane_gt is None:
            import lane_gt_data as LD
            self._lane_gt = LD.lane_store()[0]
        return self._lane_gt.load(self.tokens[i])["pieces"]

    def pts(self, i):
        """the road runs of the lane GT pieces of frame i: what `build_sd` gets (`road_runs`: the in-intersection connector ranges are cut out)"""
        return frame_pts(self.pieces(i))
''')
sub('''        if mode == "train":
            return self._draw(i)
        if mode == "S":
            return error_map''', '''        if mode in ("train", "other"):
            return self._draw(i, mode == "other")
        if mode == "S":
            assert self.plan is not None, "mode S needs the scene plan: build the dataset with maps=True and mode S / train / other, or with_donor=True"
            return error_map''')
sub('''    def _draw(self, i):
        n = int(self.draws[i])
        self.draws[i] = n + 1
        rng = np.random.RandomState([self.init_seed % 2 ** 32, zlib.crc32(self.tokens[i].encode()), n])
        cls = draw_class(rng.rand())
        if cls == "exact":
            return exact_map(self.pts(i))
        if cls == "absent":
            return EMPTY
        if cls in ("error", "unknown"):
            raw = error_like(self.pts(i), rng, *TRAIN_SHIFT)
            return with_count(raw, 0) if cls == "unknown" else raw
        if rng.rand() < 0.5:                                              # gross: a rigid shift of 4 - 8 m ...
            return error_like(self.pts(i), rng, *GROSS_SHIFT)''', '''    def _draw(self, i, other=False):
        """one draw of the mix of the module docstring for frame i; other: on a fresh random frame of another scene (the stream gets a fourth seed entry: it is another stream than the plain draw of the same read)"""
        assert self.draws is not None and self.scene is not None, "the draws need a dataset of mode train / other with maps (the counters and the scene plan)"
        n = int(self.draws[i])
        self.draws[i] = n + 1
        rng = np.random.RandomState([self.init_seed % 2 ** 32, zlib.crc32(self.tokens[i].encode()), n] + ([1] if other else []))
        src = i
        if other:
            src = int(rng.randint(len(self.tokens)))
            while self.scene[src] == self.scene[i]:
                src = int(rng.randint(len(self.tokens)))
        cls = draw_class(rng.rand())
        if cls == "exact":
            return exact_map(self.pts(src))
        if cls == "absent":
            return EMPTY
        if cls in ("error", "unknown"):
            raw = error_like(self.pts(src), rng, *TRAIN_SHIFT)
            return with_count(raw, 0) if cls == "unknown" else raw
        if rng.rand() < 0.5:                                              # gross: a rigid shift of 4 - 8 m ...
            return error_like(self.pts(src), rng, *GROSS_SHIFT)''')

# ---------------------------------------------------------------- check: the TRAIN guard
sub('''def check(seed=0, ident="e292_a", n=None, cache=LC.CACHE, log=print):''', '''def train_guard(rows, tokens, splits, train_tokens, val_tokens):
    """the TRAIN guard: the rows of the TRAIN cache are exactly `data/splits.json['train']` (indices into the train feature file), its frame tokens are those the train feature file holds at these rows, and none of them is the token of a VAL or TEST frame (`splits['val']` and
    `splits['test']` index the VAL feature file: another index space, so the disjointness is checked on the sample tokens) -> dict(train_frames, val_test_frames)"""
    assert len(set(rows)) == len(rows) and sorted(int(r) for r in rows) == sorted(int(r) for r in splits["train"]), "the rows of the TRAIN cache are not data/splits.json['train']"
    assert all(train_tokens[int(r)] == t for r, t in zip(rows, tokens)), "the tokens of the TRAIN cache are not those of the train feature file at its rows"
    held = {val_tokens[int(r)] for r in list(splits["val"]) + list(splits["test"])}
    both = held & set(tokens)
    assert not both, f"{len(both)} TRAIN frames are VAL / TEST frames (data/splits.json val + test)"
    return dict(train_frames=len(rows), val_test_frames=len(held))


def check(seed=0, ident="e292_a", n=None, cache=LC.CACHE, log=print):''')
sub('''    res = dict(seed=seed, ident=ident, cap=MAX_SAMPLES, n=n)
    for split, expect in (("val", EXPECT_VAL), ("train", EXPECT_TRAIN)):
        ds = MapData(split, seed, ident, None, "exact", cache=cache)
        assert n is not None or len(ds) == expect, f"{split}: {len(ds)} frames, expected {expect}"''', '''    res = dict(seed=seed, ident=ident, cap=MAX_SAMPLES, n=n)
    import p16_data as PD
    splits = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    tok_files = [json.load(open(f"{PD.D}/feats/{part}_tokens.json")) for part in ("train", "val")]
    for split, expect in (("val", EXPECT_VAL), ("train", EXPECT_TRAIN)):
        ds = MapData(split, seed, ident, None, "exact", cache=cache)
        assert n is not None or len(ds) == expect, f"{split}: {len(ds)} frames, expected {expect}"
        if split == "train":
            res["train_guard"] = train_guard(ds.tok.rows, ds.tokens, splits, *tok_files)
            log(f"TRAIN guard: the {res['train_guard']['train_frames']} TRAIN frames are data/splits.json['train'] and none is one of the {res['train_guard']['val_test_frames']} VAL / TEST frames")''')
sub('''    """the TEST guard (a VAL cache is `data/splits.json['val']` exactly: `MapData` asserts it), the leak check on every table built''', '''    """the TEST guards (a VAL cache is `data/splits.json['val']` exactly: `MapData` asserts it; the TRAIN cache is `data/splits.json['train']`, disjoint from the VAL and TEST frames: `train_guard`), the leak check on every table built''')

# ---------------------------------------------------------------- fixture: the cut on real frames
sub('''FIX_TOL = dict(shift_lo=1.0, shift_hi=2.0, yaw_std_lo=0.6, yaw_std_hi=1.5, undo_median=0.5, undo_p90=1.2, sample_on_road=1e-6, own_share_error=0.9, own_share_gap=0.3)''',
    '''FIX_TOL = dict(shift_lo=1.0, shift_hi=2.0, yaw_std_lo=0.6, yaw_std_hi=1.5, undo_median=0.5, undo_p90=1.2, sample_on_road=1e-6, own_share_error=0.9, own_share_gap=0.3, connector_cut=0.05, connector_uncut=0.15)''')
sub('''    ok["gross = the error map shifted by 5 m"]''', '''    ok["the exact map keeps off the in-intersection connector ranges"] = s["connector_share_cut"] <= tol["connector_cut"]
    ok["the same test fails on the uncut map (it has power)"] = s["connector_share_uncut"] >= tol["connector_uncut"]
    ok["gross = the error map shifted by 5 m"]''')
sub('''    gross, off, n_ex, n_exp, own_e, own_s = [], 0.0, 0, 0, [], []''', '''    gross, off, n_ex, n_exp, own_e, own_s = [], 0.0, 0, 0, [], []
    near_conn = {"cut": [], "uncut": []}''')
sub('''        lines = [LineString(p) for p, _ in roads_of(pts)["roads"] if len(p) >= 2 and LineString(p).length >= 1e-6]''', '''        conn = connector_interior(ds.pieces(i))                                              # the interior of the in-intersection connector ranges, straight from piece['segments']
        if len(conn):
            for name, P_ in (("cut", ex[0]), ("uncut", exact_map([np.asarray(pc["pts"], float) for pc in ds.pieces(i)])[0])):
                if len(P_):
                    near_conn[name].append(cKDTree(conn).query(P_)[0] <= 0.5)
        lines = [LineString(p) for p, _ in roads_of(pts)["roads"] if len(p) >= 2 and LineString(p).length >= 1e-6]''')
sub('''               own_share_error=float(np.mean(own_e)), own_share_S=float(np.mean(own_s)))''', '''               own_share_error=float(np.mean(own_e)), own_share_S=float(np.mean(own_s)), connector_share_cut=float(np.concatenate(near_conn["cut"]).mean()), connector_share_uncut=float(np.concatenate(near_conn["uncut"]).mean()),
               connector_samples=int(sum(len(x) for x in near_conn["cut"])))''')
sub('''    log(f"  exact map: {n_ex} samples (expected {n_exp}), farthest from a sd_lib road {off:.2e} m;''', '''    log(f"  share of the exact-map samples within 0.5 m of the interior of an in-intersection connector range: cut map {res['connector_share_cut']:.4f}, uncut map {res['connector_share_uncut']:.4f} ({res['connector_samples']} samples)")
    log(f"  exact map: {n_ex} samples (expected {n_exp}), farthest from a sd_lib road {off:.2e} m;''')

# ---------------------------------------------------------------- selftest: toy pieces
sub('''class ToyTok:''', '''def toy_pieces(k):
    """the lane GT pieces of a toy frame: the lanes of `toy_pts` as lane pieces (one lane segment each, nothing in an intersection)"""
    return [dict(pts=a, segments=[dict(kind="lane", i0=0, i1=len(a) - 1, in_intersection=False)]) for a in toy_pts(k)]


class ToyTok:''')
sub('''gt=lambda t: toy_pts(int(t[3:]) % 5), scene=scene, **kw)''', '''gt=lambda t: toy_pieces(int(t[3:]) % 5), scene=scene, **kw)''')

# selftest: rounding of the counts
sub('''    # the pose error by hand: p' = R(yaw) p + shift''', '''    ramp = dict(roads=[(np.array([[0.0, 0.0], [0.0, 10.0]]), np.array([1.0, 2.0])), (np.array([[5.0, 0.0], [5.0, 5.0], [5.0, 10.0]]), np.array([1.0, 3.0, 3.0]))])      # counts change along a road: interpolated along the vertices, ROUNDED to the nearest
    assert sample_roads(ramp)[2].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0], "counts: interpolated, rounded to the nearest (not down, not up)"
    # the cut of the in-intersection connector ranges (road level: no turn path through a junction), by hand on a piece of 10 points
    seg = lambda kind, i0, i1, inter: dict(kind=kind, i0=i0, i1=i1, in_intersection=inter)  # noqa: E731
    line = np.stack([np.arange(10.0), np.zeros(10)], 1)
    runs_of = lambda *segs: [r[:, 0].astype(int).tolist() for r in road_runs(dict(pts=line, segments=list(segs)))]  # noqa: E731
    assert runs_of(seg("lane", 0, 2, False), seg("connector", 3, 5, True), seg("lane", 6, 9, False)) == [[0, 1, 2], [6, 7, 8, 9]], "a piece cut in the middle gives two runs"
    assert runs_of(seg("connector", 2, 3, True), seg("connector", 6, 7, True)) == [[0, 1], [4, 5], [8, 9]] and runs_of(seg("connector", 0, 9, True)) == [] and runs_of(seg("lane", 0, 0, False), seg("connector", 1, 9, True)) == [], "a run of one point and a piece in an intersection give nothing"
    assert runs_of(seg("connector", 0, 9, False)) == [list(range(10))] and runs_of(seg("lane", 3, 5, True)) == [list(range(10))] and runs_of() == [list(range(10))], "only a connector range IN an intersection is cut"
    assert runs_of(seg("connector", 0, 3, True), seg("lane", 4, 9, False)) == [[4, 5, 6, 7, 8, 9]] and runs_of(seg("lane", 0, 5, False), seg("connector", 6, 9, True)) == [[0, 1, 2, 3, 4, 5]], "ranges at the ends"
    ys_ = np.arange(-20.0, 30.001, 0.5)
    street = [dict(pts=np.stack([np.zeros_like(ys_), ys_], 1), segments=[seg("lane", 0, 39, False), seg("connector", 40, 59, True), seg("lane", 60, 100, False)])]       # a street with a junction between y = 0 and y = 9.5
    ds_cut = MapData(tok=ToyTok(2), mode="exact", gt=lambda t: street)
    pc_ = ds_cut.table(0)["pos"]
    assert len(pc_) > 20 and ((pc_[:, 1] <= -0.5 + 1e-6) | (pc_[:, 1] >= 10.0 - 1e-6)).all() and (pc_[:, 1] < -0.5 + 1e-6).any() and (pc_[:, 1] > 10.0 - 1e-6).any(), "the exact map has the road up to the junction and none through it"
    assert ((exact_map([street[0]["pts"]])[0][:, 1] > 0.5) & (exact_map([street[0]["pts"]])[0][:, 1] < 9.0)).any(), "the uncut street has samples in the junction"
    ci = connector_interior(street)
    assert np.allclose(ci[:, 1], np.arange(2.5, 7.01, 0.5)) and len(ci) == 10 and len(connector_interior(toy_pieces(0))) == 0, "the interior of a connector range: 2.5 m from both ends"
    # the pose error by hand: p' = R(yaw) p + shift''')

# selftest: other mode + messages + guards
sub('''    # the dataset item: arm z builds nothing (the plain item), arm m adds the maps, with a donor the S map''', '''    # mode other (the capacity control): the draws of `train` on a FRESH random frame of another scene at every read (the stream has a fourth seed entry), not a fixed plan
    do, seen_src, n_other = toy_data("other", init_seed=2), set(), 400
    for k in range(n_other):
        rr = np.random.RandomState([2, zlib.crc32(b"toy5"), k, 1])
        src = int(rr.randint(12))
        while do.scene[src] == do.scene[5]:
            src = int(rr.randint(12))
        seen_src.add(src)
        cl = draw_class(rr.rand())
        raw = do.raw_map(5)
        if cl == "exact":
            want = exact_map(pts_of(src))
        elif cl == "absent":
            want = EMPTY
        elif cl in ("error", "unknown"):
            want = error_like(pts_of(src), rr, *TRAIN_SHIFT)
            want = with_count(want, 0) if cl == "unknown" else want
        elif rr.rand() < 0.5:
            want = error_like(pts_of(src), rr, *GROSS_SHIFT)
        else:
            j = int(rr.randint(12))
            while do.scene[j] == do.scene[5]:
                j = int(rr.randint(12))
            want = error_like(pts_of(j), rr, *TRAIN_SHIFT)
        assert len(raw[0]) == len(want[0]) and all(np.array_equal(x, y) for x, y in zip(raw, want)), f"read {k} of mode other is not the replay of its stream"
    assert len(seen_src) >= 6 and all(do.scene[j] != do.scene[5] for j in seen_src) and int(do.draws[5]) == n_other, "every read draws a fresh random frame of another scene"
    # clear messages: a mode that needs the scene plan or the counters on a dataset that has none
    refuses(lambda: toy_data("error").raw_map(0, "S"), "needs the scene plan")
    refuses(lambda: toy_data("error").raw_map(0, "train"), "mode train / other")
    refuses(lambda: toy_data("error").raw_map(0, "other"), "mode train / other")
    # the TRAIN guard: the rows are splits['train'], the tokens those of the train feature file, no token is a VAL / TEST frame's
    sp = dict(train=[0, 1, 2], val=[0, 1], test=[2])
    tt, vt = ["a", "b", "c"], ["x", "y", "z"]
    assert train_guard([0, 1, 2], ["a", "b", "c"], sp, tt, vt) == dict(train_frames=3, val_test_frames=3)
    refuses(lambda: train_guard([0, 1], ["a", "b"], sp, tt, vt), "not data/splits.json['train']")
    refuses(lambda: train_guard([0, 1, 1], ["a", "b", "b"], sp, tt, vt), "not data/splits.json['train']")
    refuses(lambda: train_guard([0, 1, 2], ["a", "b", "x"], sp, tt, vt), "not those of the train feature file")
    refuses(lambda: train_guard([0, 1, 2], ["a", "b", "c"], sp, tt, ["x", "b", "z"]), "are VAL / TEST frames")
    refuses(lambda: train_guard([0, 1, 2], ["a", "b", "c"], sp, tt, ["x", "y", "c"]), "are VAL / TEST frames")
    assert train_guard([0, 1, 2], ["a", "b", "c"], sp, tt, ["x", "y", "z", "b"]) == dict(train_frames=3, val_test_frames=3), "only the val and test ROWS count, not every token of the val file"
    # the dataset item: arm z builds nothing (the plain item), arm m adds the maps, with a donor the S map''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
