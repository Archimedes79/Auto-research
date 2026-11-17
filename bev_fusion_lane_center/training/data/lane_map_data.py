"""E374 (stage S3 of the token champion, D113): the data of the cross-attention to the SD MAP: ROAD-level map tokens of a frame (one centre line per carriageway with a lane count), the simulated map errors, the loader on top of
`lane_ptok_data`, the leak check, the step-0 measurement and the fixture. A navigation map knows roads, not lanes: the lane-level fields of `map_sim` (piece ids, successor / predecessor / left / right / opposite, lane index, kind, arclength
fraction, the junction turn paths) and the route NEVER enter; the map is a GT-derived SIMULATION (the lane GT pieces of the frame, merged into carriageways by `sd_lib.build_sd`).

    wsl bash tools/env/_py14.sh lane_map_data.py check     [--seed 0] [--ident e292_a] [--n N] [--out FILE]      # the TEST guard (VAL rows = data/splits.json['val']), the leak check on every built table, the truncation counts of VAL and TRAIN (exact and error map)
    wsl bash tools/env/_py14.sh lane_map_data.py potential --split val --seed 0 [--n N] [--out FILE]            # step 0 of the E374 card (JSON + the table of the card): supported share per token class for the exact / error / other-scene map, by existence bin
    wsl bash tools/env/_py14.sh lane_map_data.py fixture   --split val [--n 100] [--seed 0] [--out FILE]        # the built maps against their specification on real frames: PASS / FAIL + numbers; the wrong conventions must FAIL
    wsl bash tools/env/_py14.sh lane_map_data.py selftest                                                       # no cache needed; tests: grid/test_lane_map.py (the hand mutants)
    ds, dl = loader("train" | "val", seed=0, bs=8, workers=4, init_seed=0, mode=None, maps=True)                 # the batches of `lane_ptok_data.loader` + batch["maps"]
    batch["maps"]   dict(pos (B, M, 2) float32 [lane frame, m], dir (B, M, 2) the unit direction of travel (signed), count (B, M) int64 the lane count 1 - 5 (0 = unknown), valid (B, M) bool); M = the largest map of the batch (at least 1; padding is not valid)
    batch["donor"]  the same dict, the `S` map of the frame (only with `with_donor`: the read-out's other-scene reference of the kill measure)

Map of a frame (`sample_roads`): `sd_lib.build_sd(dict(pts=[the points of every lane GT piece]), [])` (carriageway centre lines, a lane count at every vertex, the route is NOT passed), optionally degraded (`sd_lib.degrade_sd`, drop 0.1: Douglas-Peucker
sparsification 0.5 - 2 m, count errors 0.1, missing side roads), then the pose error p -> R(yaw) p + shift, then every STEP = 2 m along every centre line: pos, the unit tangent in the direction of travel, the lane count (rounded, 1 - 5 in the table). At most MAX_SAMPLES = 512
samples per frame: when more exist the ones nearest the ego are kept (a named hard step; `check` counts the truncated frames). Modes (`MapData(mode=)`):
  exact   no error                                           error   a FIXED map per frame, a generator seeded by the sample token (md5, as `map_sim`): shift 1 - 2 m in a random direction, yaw N(0, 1 degree), degrade_sd (drop 0.1); the VAL record
  train   one DRAW per read of a frame (TRAIN): exact 25 %, error-like 47 % (shift U(0, 2) m), counts unknown 8 % (the error-like map with count 0), absent 10 % (no samples), gross 10 % (50 %: the error-like map with a shift of U(4, 8) m; 50 %: the error-like map of a frame of ANOTHER scene)
          the generator is seeded by (the init seed, the frame token, the draw index of the frame: how often it was read before, a counter array in shared memory so that persistent workers count together); a RESUMED run starts its counters at 0 again: its draws are not those of
          the straight run (it is not bit-reproducible; report the resume steps)
  gross   the `error` map shifted rigidly by 5 m (a direction seeded by the token)      none  no samples (the none key only)      S  the `error` map of ANOTHER SCENE (frame (i + 97) mod N, moved on to the next frame of another scene: a fixed plan)
  U       the `error` map with every count 0 (unknown)       C2   the `error` map with every count 2
FIXED KEY SET: a table has exactly KEYS = (pos, dir, count, valid); `check_keys` refuses any other key (a piece id, adjacency, lane index, kind, frac or route field is a leak). The scene of a frame (S, the other-scene gross draw) comes from the S2 extension of the cache
(`lane_past_data.build`: prev.npy, a new scene at every frame without a previous keyframe). Arm z (`maps=False`) builds nothing: its batch is `lane_ptok_data`'s, bit for bit.
"""
import argparse
import hashlib
import json
import os
import sys
import zlib

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "training/data/prep", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import sd_lib as SD  # noqa: E402

KEYS = ("pos", "dir", "count", "valid")                                      # the fixed key set of a frame's map table
MAX_SAMPLES = 512                                                            # samples per frame: the nearest the ego are kept
STEP = 2.0                                                                   # m between the samples of a centre line
DROP = 0.1                                                                   # missing side roads (`sd_lib.degrade_sd`; its sparsification 0.5 - 2 m and count errors 0.1 are its defaults)
COUNT_MAX = 5                                                                # the lane counts of the table are 1 - COUNT_MAX (0 = unknown)
ERROR_SHIFT, TRAIN_SHIFT, GROSS_SHIFT = (1.0, 2.0), (0.0, 2.0), (4.0, 8.0)   # m: the shift of the `error` map, of the error-like draws of `train`, of the gross draws
GROSS_M = 5.0                                                                # read-out `gross`: the `error` map shifted rigidly by this (m)
OTHER = 97                                                                   # the plan of `S`: frame (i + OTHER) mod N, moved on to another scene
MIX = (("exact", 0.25), ("error", 0.47), ("unknown", 0.08), ("absent", 0.10), ("gross", 0.10))     # the draw classes of mode `train` (card section 1)
MODES = ("exact", "error", "train", "gross", "none", "S", "U", "C2")
EXPECT_TRAIN, EXPECT_VAL = 8414, 873
RAD, COS = 1.5, 0.5                                                          # `potential`: a token is near a GT node within 1.5 m with a heading cosine >= 0.5 (the card's step 0, as `lane_past_data`)
SUP_D, SUP_COS = 3.0, 0.8                                                    # `potential`: a token is supported by the map within 3 m of a sample with |cos| >= 0.8
BINS = ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5))
EMPTY = (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))                    # a map without samples (pos, dir, count)


# ------------------------------------------------------------------------------------------------------------------------------ the map of a frame
def token_rng(token, salt=""):
    """the generator of a frame token (md5, as `map_sim._rng_for`; a salt gives another stream of the same token)"""
    return np.random.RandomState(int(hashlib.md5((str(token) + salt).encode()).hexdigest()[:8], 16))


def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def pose_error(rng, lo, hi):
    """the rigid pose error of a map, the draws of `map_sim.make_map` in its order: the direction, the length U(lo, hi) m, the yaw N(0, 1 degree) -> (shift (2,), yaw rad)"""
    ang = rng.uniform(0, 2 * np.pi)
    shift = rng.uniform(lo, hi) * np.array([np.cos(ang), np.sin(ang)])
    return shift, np.deg2rad(rng.randn())


def roads_of(pts):
    """the SD roads of a frame: `sd_lib.build_sd` on the lane GT pieces' points (lane pieces side by side in one direction = one carriageway with its lane count); the route is not passed"""
    return SD.build_sd(dict(pts=[np.asarray(p, float) for p in pts]), [])


def sample_roads(sd, shift=None, yaw=0.0):
    """the roads of `sd` (dict with 'roads' = [(polyline (n, 2), lane count (n,))]), after the pose error p -> R(yaw) p + shift when `shift` is given, sampled every STEP m along every centre line -> (pos (S, 2), dir (S, 2) the unit tangent in the direction of travel,
    count (S,) the lane count interpolated along the vertices and rounded; not yet clipped); a road of one point is left out"""
    P, T, C = [], [], []
    for poly, cnt in sd["roads"]:
        poly = np.asarray(poly, float)
        if len(poly) < 2:
            continue
        if shift is not None:
            poly = poly @ rot(yaw).T + shift
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
        if s[-1] < 1e-6:
            continue
        t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / STEP)), 1) + 1)
        q = np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1)
        tg = np.gradient(q, axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        P.append(q)
        T.append(tg)
        C.append(np.interp(t, s, np.asarray(cnt, float)))
    return (np.concatenate(P), np.concatenate(T), np.rint(np.concatenate(C))) if P else EMPTY


def exact_map(pts):
    return sample_roads(roads_of(pts))


def error_like(pts, rng, lo, hi):
    """a degraded and displaced map: the pose error first (`pose_error`), then `sd_lib.degrade_sd` (drop DROP) on the same generator, then the samples"""
    shift, yaw = pose_error(rng, lo, hi)
    return sample_roads(SD.degrade_sd(roads_of(pts), rng, drop=DROP), shift, yaw)


def error_map(pts, token):
    """the FIXED `error` map of a frame: shift 1 - 2 m, yaw N(0, 1 degree), degrade_sd, the generator seeded by the token"""
    return error_like(pts, token_rng(token), *ERROR_SHIFT)


def shifted(raw, vec):
    return (raw[0] + vec, raw[1], raw[2])


def with_count(raw, c):
    return (raw[0], raw[1], np.full(len(raw[0]), float(c)))


def truncate(raw, n=MAX_SAMPLES):
    """at most n samples: the n nearest the ego (the origin), in their own order"""
    pos, d, c = raw
    if len(pos) <= n:
        return raw
    keep = np.sort(np.argsort(np.linalg.norm(pos, axis=1), kind="stable")[:n])
    return pos[keep], d[keep], c[keep]


def check_keys(m):
    """THE leak check of a map table: exactly the keys of KEYS (a piece id, adjacency, lane index, kind, frac or route field is refused), one row per sample"""
    extra, missing = sorted(set(m) - set(KEYS)), sorted(set(KEYS) - set(m))
    assert not extra and not missing, f"a road-level map table has exactly the keys {KEYS}: extra {extra} (a lane-level field is a leak), missing {missing}"
    n = len(m["valid"])
    assert tuple(np.shape(m["pos"])) == (n, 2) and tuple(np.shape(m["dir"])) == (n, 2) and tuple(np.shape(m["count"])) == (n,), f"map table shapes {[np.shape(m[k]) for k in KEYS]}"
    return m


def finish(raw):
    """a raw map -> the table of a frame (numpy; `check_keys` holds): truncated to MAX_SAMPLES (nearest the ego), the counts clipped to 1 - COUNT_MAX (0 stays 0 = unknown), all samples valid"""
    pos, d, c = truncate(raw)
    c = np.where(c > 0, np.clip(c, 1, COUNT_MAX), 0)
    return check_keys(dict(pos=pos.astype(np.float32), dir=d.astype(np.float32), count=c.astype(np.int64), valid=np.ones(len(pos), bool)))


def collate_maps(tables):
    """the tables of the frames of a batch -> dict of tensors padded to the largest map (at least 1 column; the padding is not valid): pos / dir float32, count int64, valid bool"""
    for t in tables:
        check_keys(t)
    M = max(1, max(len(t["valid"]) for t in tables))
    out = dict(pos=np.zeros((len(tables), M, 2), np.float32), dir=np.zeros((len(tables), M, 2), np.float32), count=np.zeros((len(tables), M), np.int64), valid=np.zeros((len(tables), M), bool))
    for b, t in enumerate(tables):
        n = len(t["valid"])
        for k in KEYS:
            out[k][b, :n] = t[k]
    return {k: torch.from_numpy(v) for k, v in out.items()}


def draw_class(u):
    """the class of a `train` draw from u in [0, 1): MIX in order (exact 25 %, error 47 %, unknown 8 %, absent 10 %, gross 10 %)"""
    c = 0.0
    for name, p in MIX:
        c += p
        if u < c - 1e-12:
            return name
    return MIX[-1][0]


def other_scene_plan(scene, shift=OTHER):
    """(N,) int: the frame whose map frame i borrows (`S`, the read-out's donor): (i + OTHER) mod N, moved on frame by frame until it lies in another scene. A fixed plan"""
    scene = np.asarray(scene)
    assert len(set(scene.tolist())) >= 2, "one scene only: no map of another scene"
    out = []
    for i in range(len(scene)):
        j = (i + shift) % len(scene)
        while scene[j] == scene[i]:
            j = (j + 1) % len(scene)
        out.append(j)
    return np.array(out, np.int64)


# ------------------------------------------------------------------------------------------------------------------------------ the dataset
class MapData(torch.utils.data.Dataset):
    """the frames of `lane_ptok_data.PTokData` (the same batch dict) + `maps` (+ `donor`): mode of the module docstring; maps False = no map is built (arm z: the item is the plain PTokData item); with_donor: item['donor'] = the `S` map. `gt(token)` -> the point arrays of the lane GT
    pieces of a frame (default: the lane GT store of this checkout); `scene` (N,) the scene of every frame (default: from the S2 extension of the cache); tok / gt / scene are injected by the selftest only"""

    def __init__(self, split="train", seed=0, ident="e292_a", K=None, mode="error", init_seed=0, maps=True, with_donor=False, cache=LC.CACHE, dir=None, tok=None, gt=None, scene=None):
        assert mode in MODES, mode
        if tok is None:
            self.tok = LP.PTokData(split, seed, ident, K, 0.0, cache, dir)
            if dir is None:
                import lane_past_data as LPD
                LPD.check_rows(split, self.tok.rows)                       # the TEST guard: a VAL cache is data/splits.json['val'] exactly (the poses file and the raw data also hold TEST rows)
        else:
            self.tok = tok
        self.mode, self.init_seed, self.maps, self.with_donor, self._gt, self._lane_gt = mode, int(init_seed), bool(maps), bool(with_donor), gt, None
        self.tokens, self.scene, self.plan = list(self.tok.tokens), None, None
        if self.maps and (mode in ("S", "train") or with_donor):
            if scene is None:
                import lane_past_data as LPD
                scene = LPD.scene_ids(LPD.load_ext(self.tok.dir)["prev"])
            self.scene = np.asarray(scene)
            assert len(self.scene) == len(self.tokens), "one scene id per frame"
            self.plan = other_scene_plan(self.scene)
        self.draws = torch.zeros(len(self.tokens), dtype=torch.int32).share_memory_() if mode == "train" else None       # the draw index of every frame (shared with the workers)

    def __len__(self):
        return len(self.tok)

    def pts(self, i):
        """the points of the lane GT pieces of frame i"""
        if self._gt is not None:
            return self._gt(self.tokens[i])
        if self._lane_gt is None:
            import lane_gt_data as LD
            self._lane_gt = LD.lane_store()[0]
        return [np.asarray(p["pts"], float) for p in self._lane_gt.load(self.tokens[i])["pieces"]]

    def raw_map(self, i, mode=None):
        """the raw map (pos, dir, count; before the cap) of frame i in `mode` (default: the dataset's); `train` makes one draw (and counts it)"""
        mode = mode or self.mode
        tok = self.tokens[i]
        if mode == "none":
            return EMPTY
        if mode == "exact":
            return exact_map(self.pts(i))
        if mode == "train":
            return self._draw(i)
        if mode == "S":
            return error_map(self.pts(self.plan[i]), self.tokens[self.plan[i]])
        base = error_map(self.pts(i), tok)
        if mode == "error":
            return base
        if mode == "gross":
            ang = token_rng(tok, "|gross").uniform(0, 2 * np.pi)
            return shifted(base, GROSS_M * np.array([np.cos(ang), np.sin(ang)]))
        return with_count(base, 0 if mode == "U" else 2)

    def _draw(self, i):
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
            return error_like(self.pts(i), rng, *GROSS_SHIFT)
        j = int(rng.randint(len(self.tokens)))                            # ... or the map of a frame of another scene
        while self.scene[j] == self.scene[i]:
            j = int(rng.randint(len(self.tokens)))
        return error_like(self.pts(j), rng, *TRAIN_SHIFT)

    def table(self, i, mode=None):
        return finish(self.raw_map(i, mode))

    def __getitem__(self, i):
        out = self.tok[i]
        if self.maps:
            out["maps"] = self.table(i)
            if self.with_donor:
                out["donor"] = self.table(i, "S")
        return out


def collate(b):
    """`lane_ptok_data.collate` + the padded maps (and donors) of the items that have them"""
    out = LP.collate(b)
    for k in ("maps", "donor"):
        if k in b[0]:
            out[k] = collate_maps([o[k] for o in b])
    return out


def loader(split="train", seed=0, bs=8, workers=4, init_seed=0, mode=None, maps=True, with_donor=False, ident="e292_a", K=None, shuffle=None, cache=LC.CACHE, dir=None):
    """as `lane_ptok_data.loader` (the same sampler, batch size, workers) + the maps; mode: TRAIN `train`, VAL `error` by default"""
    shuffle = split == "train" if shuffle is None else shuffle
    ds = MapData(split, seed, ident, K, mode or ("train" if split == "train" else "error"), init_seed, maps, with_donor, cache, dir)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)


# ------------------------------------------------------------------------------------------------------------------------------ check: the leak check and the truncation counts
def _sizes(ds, modes, n=None, log=print):
    idx = np.arange(len(ds)) if n is None else np.linspace(0, len(ds) - 1, n).astype(int)
    out = {m: [] for m in modes}
    over = 0
    for i in idx:
        for m in modes:
            raw = ds.raw_map(int(i), m)
            out[m].append(len(raw[0]))
            if m == "exact":
                over += int((raw[2] > COUNT_MAX).sum())
            ds.table(int(i), m)                                            # the leak check of every table (`finish` -> `check_keys`)
    return {m: dict(frames=len(v), mean=float(np.mean(v)), max=int(np.max(v)), truncated=int((np.asarray(v) > MAX_SAMPLES).sum()), share_truncated=float((np.asarray(v) > MAX_SAMPLES).mean())) for m, v in out.items()}, over


def check(seed=0, ident="e292_a", n=None, cache=LC.CACHE, log=print):
    """the TEST guard (a VAL cache is `data/splits.json['val']` exactly: `MapData` asserts it), the leak check on every table built, the truncation counts of the exact and the error map on VAL and TRAIN -> dict. The card is revised when more than 0.1 % of the frames are truncated"""
    res = dict(seed=seed, ident=ident, cap=MAX_SAMPLES, n=n)
    for split, expect in (("val", EXPECT_VAL), ("train", EXPECT_TRAIN)):
        ds = MapData(split, seed, ident, None, "exact", cache=cache)
        assert n is not None or len(ds) == expect, f"{split}: {len(ds)} frames, expected {expect}"
        st, over = _sizes(ds, ("exact", "error"), n)
        res[split] = dict(frames=len(ds), sizes=st, samples_with_count_above_max=over)
        log(f"{split}: {len(ds)} frames; exact map: mean {st['exact']['mean']:.1f} max {st['exact']['max']} samples, truncated {st['exact']['truncated']} ({st['exact']['share_truncated']:.4%}); "
            f"error map: mean {st['error']['mean']:.1f} max {st['error']['max']}, truncated {st['error']['truncated']} ({st['error']['share_truncated']:.4%}); samples whose rounded count exceeds {COUNT_MAX} (clipped): {over}")
    res["truncated_over_0.1pct"] = any(res[s]["sizes"][m]["share_truncated"] > 0.001 for s in ("val", "train") for m in ("exact", "error"))
    log("leak check: every table built has exactly the keys " + str(KEYS))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ potential: step 0 of the E374 card
def potential(split="val", seed=0, ident="e292_a", n=None, cache=LC.CACHE, dir=None, log=print):
    """step 0 of the card (the scratch script step0_map2.py, promoted): the tokens of B' of every frame (valid, p = e_p) against the GT nodes of the frame (`lane_tok_data.TokData` table) and against the road maps; a weak token = valid and p < 0.5, true = near a GT node (within RAD, heading
    cosine >= COS), 'novel' = near a GT node that NO delivered token (p >= 0.5) found, 'duplicate' = near found nodes only, 'false' = near none; supported = within SUP_D of a sample of the map with |cos| >= SUP_COS to its direction. For the exact map, the `error` map and the `error` map of
    another scene (the plan of `S` = the null) -> dict: per class n, the supported share per map, the median distance to the nearest sample (exact / error); by existence bin (the error map): n, the true share, supported if true / if false, the novel share of the supported / unsupported"""
    from scipy.spatial import cKDTree
    import lane_tok_data as LT
    ds = MapData(split, seed, ident, None, "error", cache=cache, dir=dir, with_donor=True)
    c = ds.tok.c
    td = LT.TokData(split, res=2.0, feats=False)
    assert td.tokens == c.tokens, "the GT tables are not in the order of the cache"
    idx = np.arange(len(c)) if n is None else np.linspace(0, len(c) - 1, n).astype(int)
    cls = ("novel", "dup", "false", "deliv_true", "deliv_false")
    acc = {(mode, k): [] for mode in ("exact", "error", "null_error") for k in cls}
    bins, nsd = [], []
    for i in idx:
        t = td.table(int(i))
        gpos, gdir = t["pos"], t["dirv"]
        if len(gpos) == 0:
            continue
        pos, dirv, valid, ep = (np.array(c.f[k][i], float) if k != "valid" else np.array(c.f[k][i]) for k in ("pos", "dirv", "valid", "e_p"))
        deliv = valid & (ep >= 0.5)
        weak = valid & ~deliv
        near = (np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD) & (dirv @ gdir.T >= COS)
        miss = ~((near & deliv[:, None]).any(0))
        sel = dict(novel=weak & (near & miss[None]).any(1), dup=weak & near.any(1) & ~(near & miss[None]).any(1), false=weak & ~near.any(1), deliv_true=deliv & near.any(1), deliv_false=deliv & ~near.any(1))
        maps = dict(exact=ds.raw_map(int(i), "exact"), error=ds.raw_map(int(i), "error"), null_error=ds.raw_map(int(ds.plan[i]), "error"))
        nsd.append(len(maps["exact"][0]))
        for mode, (P, T, _) in maps.items():
            if len(P) == 0:
                sup, dist = np.zeros(len(pos), bool), np.full(len(pos), 99.0)
            else:
                dist, jj = cKDTree(P).query(pos)
                sup = (dist <= SUP_D) & (np.abs((dirv * T[jj]).sum(1)) >= SUP_COS)
            for k in cls:
                acc[(mode, k)].append(np.stack([dist[sel[k]], sup[sel[k]]], 1))
            if mode == "error":
                bins.append(np.stack([ep[weak], sup[weak], near.any(1)[weak], (near & miss[None]).any(1)[weak]], 1))
    res = dict(split=split, seed=seed, ident=ident, frames=len(idx), cache=os.path.relpath(ds.tok.dir, ROOT), samples_per_frame=float(np.mean(nsd)), classes={})
    for k in cls:
        a = {m: np.concatenate(acc[(m, k)]) for m in ("exact", "error", "null_error")}
        res["classes"][k] = dict(n=len(a["exact"]), supported_exact=float(a["exact"][:, 1].mean()), supported_error=float(a["error"][:, 1].mean()), supported_other_scene=float(a["null_error"][:, 1].mean()),
                                 median_dist_exact=float(np.median(a["exact"][:, 0])), median_dist_error=float(np.median(a["error"][:, 0])))
    B = np.concatenate(bins)
    res["by_bin"] = []
    for lo, hi in BINS:
        m = (B[:, 0] >= lo) & (B[:, 0] < hi)
        t_, f_, s_, u_ = m & (B[:, 2] > 0.5), m & (B[:, 2] < 0.5), m & (B[:, 1] > 0.5), m & (B[:, 1] < 0.5)
        res["by_bin"].append(dict(lo=lo, hi=hi, n=int(m.sum()), true_share=float(t_.sum() / m.sum()), supported_if_true=float(B[t_, 1].mean()), supported_if_false=float(B[f_, 1].mean()),
                                  novel_share_supported=float(B[s_, 3].mean()), novel_share_unsupported=float(B[u_, 3].mean())))
    log(f"VAL seed {seed}, road-level SD map (sd_lib.build_sd; error = degrade_sd drop {DROP} + shift 1-2 m + yaw 1 degree), samples per frame {res['samples_per_frame']:.0f}")
    log("class | n | within 3 m and aligned (|cos| >= 0.8): exact / error / another scene's error map | median distance exact / error")
    for k, v in res["classes"].items():
        log("  %-12s %7d | %.3f / %.3f / %.3f | %.2f / %.2f" % (k, v["n"], v["supported_exact"], v["supported_error"], v["supported_other_scene"], v["median_dist_exact"], v["median_dist_error"]))
    log("weak tokens by existence bin, error map: p range | n | share near a GT node | map-supported if near GT / if not | novel share (supported / unsupported)")
    for b in res["by_bin"]:
        log("  %.2f - %.2f | %6d | %.3f | %.3f / %.3f | %.4f / %.4f" % (b["lo"], b["hi"], b["n"], b["true_share"], b["supported_if_true"], b["supported_if_false"], b["novel_share_supported"], b["novel_share_unsupported"]))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ fixture: the built maps against their specification
FIX_TOL = dict(shift_lo=1.0, shift_hi=2.0, yaw_std_lo=0.6, yaw_std_hi=1.5, undo_median=0.5, undo_p90=1.2, sample_on_road=1e-6, own_share_error=0.5, own_share_other=0.3)
HOWS = ("undo", "identity", "shift_sign", "rotation_transposed")                # the pose error undone as it was made, and the wrong ways the fixture must reject


def undo(P, shift, yaw, how="undo"):
    """the samples of an `error` map put back by `how`: undo = R^T (p - shift) (the inverse of p -> R p + shift), identity = left as they are, shift_sign = R^T (p + shift), rotation_transposed = R (p - shift)"""
    R = rot(yaw)
    if how == "identity":
        return P
    if how == "shift_sign":
        return (P + shift) @ R
    if how == "rotation_transposed":
        return (P - shift) @ R.T
    return (P - shift) @ R


def fixture_checks(s, hows=HOWS, tol=FIX_TOL):
    """the rule of the fixture on its numbers (dict of `fixture`) -> name -> bool: the pose error is map_sim's and within its bounds, the undone error map lies on the exact map (median / p90 of the nearest-sample distance), every wrong way FAILS that, the exact samples lie on the sd_lib roads,
    the sample counts are those of the polylines, gross is a 5 m rigid shift, the S map is another scene's and farther from this frame's own roads than the error map"""
    ok = {"the pose error is map_sim's (shift, yaw)": s["pose_equals_map_sim"],
          "the shift is 1 - 2 m on every frame": tol["shift_lo"] - 1e-9 <= s["shift_min"] and s["shift_max"] <= tol["shift_hi"] + 1e-9,
          "the yaw is N(0, 1 degree)": tol["yaw_std_lo"] <= s["yaw_std_deg"] <= tol["yaw_std_hi"]}
    good = lambda r: r["median"] <= tol["undo_median"] and r["p90"] <= tol["undo_p90"]  # noqa: E731
    ok["the error map undone lies on the exact map"] = good(s["undo"]["undo"])
    for how in hows:
        if how != "undo":
            ok[f"{how} fails that"] = not good(s["undo"][how])
    ok["the exact samples lie on the sd_lib roads"] = s["exact_off_road_max"] <= tol["sample_on_road"]
    ok["the exact map has the sample count of the roads"] = s["exact_samples"] == s["exact_samples_expected"]
    ok["gross = the error map shifted by 5 m"] = abs(s["gross_shift_min"] - GROSS_M) < 1e-9 and abs(s["gross_shift_max"] - GROSS_M) < 1e-9
    ok["the S map is the error map of another scene"] = s["S_scene_differs"] and s["S_is_error_map_of_plan"] and not s["S_is_own_error_map"]
    ok["the S map lies off this frame's roads, the error map on them"] = s["own_share_error"] >= tol["own_share_error"] and s["own_share_S"] <= tol["own_share_other"]
    return ok


def fixture(split="val", n=100, seed=0, ident="e292_a", cache=LC.CACHE, dir=None, hows=HOWS, log=print):
    """the module docstring's fixture on n evenly spaced frames of the cache of B' seed `seed` (GT store of this checkout): see `fixture_checks`; passed = every check -> dict"""
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, Point
    import map_sim as MS
    ds = MapData(split, seed, ident, None, "error", cache=cache, dir=dir, with_donor=True)
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    assert all(ds.plan[i] == (i + OTHER) % len(ds) for i in range(len(ds))), "the S plan is not (i + 97) mod N on this split"
    sh, yw, equal = [], [], True
    gross, off, n_ex, n_exp, own_e, own_s = [], 0.0, 0, 0, [], []
    near = {h: [] for h in hows}
    s_scene, s_plan, s_own = True, True, False
    for i in idx:
        i = int(i)
        tok = ds.tokens[i]
        pts = ds.pts(i)
        rng = token_rng(tok)
        shift, yaw = pose_error(rng, *ERROR_SHIFT)
        mp = MS.make_map(dict(pieces=[], sample=tok), "error", token=tok)                  # map_sim's draws for the same token (the pose error does not depend on the pieces)
        equal = equal and np.allclose(mp["shift"], shift, atol=1e-12) and abs(mp["yaw"] - yaw) < 1e-12
        sh.append(float(np.linalg.norm(shift)))
        yw.append(np.degrees(yaw))
        ex, er = ds.raw_map(i, "exact"), ds.raw_map(i, "error")
        sd = roads_of(pts)
        for P_, hw in ((ex[0], "exact"),):
            exp = sum(max(int(np.ceil(LineString(p).length / STEP)), 1) + 1 for p, _ in sd["roads"] if len(p) >= 2 and LineString(p).length >= 1e-6)
            n_ex += len(P_)
            n_exp += exp
            for q in P_:
                off = max(off, min(LineString(p).distance(Point(q)) for p, _ in sd["roads"] if len(p) >= 2))
        if len(er[0]) and len(ex[0]):
            tree = cKDTree(ex[0])
            for h in hows:
                near[h].append(tree.query(undo(er[0], shift, yaw, h))[0])
            own_e.append(float((tree.query(er[0])[0] <= SUP_D).mean()))
        gr = ds.raw_map(i, "gross")
        gross.append(float(np.linalg.norm(gr[0][0] - er[0][0]))) if len(er[0]) else None
        sm, j = ds.raw_map(i, "S"), int(ds.plan[i])
        s_scene = s_scene and ds.scene[j] != ds.scene[i]
        s_plan = s_plan and all(np.array_equal(a, b) for a, b in zip(sm, error_map(ds.pts(j), ds.tokens[j])))
        s_own = s_own or (len(sm[0]) == len(er[0]) and len(er[0]) > 0 and all(np.array_equal(a, b) for a, b in zip(sm, er)))
        if len(sm[0]) and len(ex[0]):
            own_s.append(float((tree.query(sm[0])[0] <= SUP_D).mean()))
    cat = lambda hw: np.concatenate(near[hw])  # noqa: E731
    res = dict(split=split, n=n, seed=seed, cache=os.path.relpath(ds.tok.dir, ROOT), tolerances=FIX_TOL, pose_equals_map_sim=bool(equal), shift_min=float(min(sh)), shift_max=float(max(sh)), yaw_std_deg=float(np.std(yw)), yaw_mean_deg=float(np.mean(yw)),
               undo={h: dict(n=len(cat(h)), median=float(np.median(cat(h))), p90=float(np.percentile(cat(h), 90)), p99=float(np.percentile(cat(h), 99))) for h in hows},
               exact_off_road_max=float(off), exact_samples=int(n_ex), exact_samples_expected=int(n_exp), gross_shift_min=float(min(gross)), gross_shift_max=float(max(gross)), S_scene_differs=bool(s_scene), S_is_error_map_of_plan=bool(s_plan), S_is_own_error_map=bool(s_own),
               own_share_error=float(np.mean(own_e)), own_share_S=float(np.mean(own_s)))
    ok = res["checks"] = fixture_checks(res, hows)
    res["passed"] = all(ok.values())
    log(f"fixture {split} {len(ds)} frames, {n} checked: shift {res['shift_min']:.3f} - {res['shift_max']:.3f} m, yaw std {res['yaw_std_deg']:.3f} degrees (mean {res['yaw_mean_deg']:.3f}); map_sim's draws equal: {equal}")
    for h in hows:
        u = res["undo"][h]
        log(f"  {h:<20} nearest-sample distance to the exact map: n {u['n']:6d} median {u['median']:.3f} p90 {u['p90']:.3f} p99 {u['p99']:.3f} m")
    log(f"  exact map: {n_ex} samples (expected {n_exp}), farthest from a sd_lib road {off:.2e} m; gross shift {res['gross_shift_min']:.6f} - {res['gross_shift_max']:.6f} m; share of the samples within 3 m of this frame's exact map: error {res['own_share_error']:.3f}, S {res['own_share_S']:.3f}")
    for k, v in ok.items():
        log(f"  {'ok  ' if v else 'FAIL'} {k}")
    log("PASS" if res["passed"] else "FAIL")
    return res


# ------------------------------------------------------------------------------------------------------------------------------ selftest
def refuses(fn, text):
    """fn must fail with an AssertionError whose message holds `text` (a call that is accepted raises a RuntimeError, which no test catches)"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    raise RuntimeError(f"accepted although it must be refused ({text})")


def toy_pts(k):
    """the lane GT of a toy frame: two lanes side by side (x = -1.75 and 1.75) going north, y -20 .. 30 (one carriageway of 2 lanes), an opposite lane at x = 9 going south (1 lane), a crossing road at y = 12 + k going east (1 lane): merged by sd_lib into 3 roads"""
    ys = np.arange(-20.0, 30.001, 0.5)
    xs = np.arange(-30.0, 30.001, 0.5)
    return [np.stack([np.full_like(ys, -1.75), ys], 1), np.stack([np.full_like(ys, 1.75), ys], 1), np.stack([np.full_like(ys, 9.0), ys[::-1]], 1), np.stack([xs, np.full_like(xs, 12.0 + k)], 1)]


class ToyTok:
    """the stand-in of `PTokData` for the selftest: tokens, one item per frame"""

    def __init__(self, n):
        self.tokens, self.rows, self.K, self.dir = [f"toy{i}" for i in range(n)], list(range(n)), 8, None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return dict(token=self.tokens[i])


def toy_data(mode, n=12, scene=None, **kw):
    scene = [i // 4 for i in range(n)] if scene is None else scene                                          # 3 scenes of 4 frames
    return MapData(tok=ToyTok(n), mode=mode, gt=lambda t: toy_pts(int(t[3:]) % 5), scene=scene, **kw)


def selftest():
    # the exact map by hand: three roads (the crossing one first: the longest), the 2-lane carriageway at x = 0 with count 2, samples every 2 m
    sd = roads_of(toy_pts(0))
    assert [len(p) for p, _ in sd["roads"]] == [31, 26, 26] and [sorted(set(c.tolist())) for _, c in sd["roads"]] == [[1.0], [2.0], [1.0]] and np.allclose(sd["roads"][1][0][:, 0], 0.0)
    pos, d, c = exact_map(toy_pts(0))
    assert len(pos) == 31 + 26 + 26 and np.allclose(pos[31:57, 0], 0.0) and np.allclose(pos[31:57, 1], np.arange(-20.0, 30.001, 2.0)) and np.allclose(d[31:57], [0.0, 1.0]) and (c[31:57] == 2).all(), "the carriageway of 2 lanes: samples every 2 m, direction north"
    assert np.allclose(pos[:31, 1], 12.0) and np.allclose(d[:31], [1.0, 0.0]) and np.allclose(pos[57:, 0], 9.0) and np.allclose(d[57:], [0.0, -1.0]) and (c[57:] == 1).all(), "the crossing road goes east, the opposite lane south (the SIGNED direction of travel)"
    # the pose error by hand: p' = R(yaw) p + shift (a rotation by +yaw: north -> west for 90 degrees), the directions rotate and are not translated
    sh0 = np.array([3.0, 4.0])
    p2, d2, c2 = sample_roads(sd, sh0, np.pi / 2)
    assert np.allclose(p2[31:57], np.stack([-pos[31:57, 1] + 3.0, pos[31:57, 0] + 4.0], 1), atol=1e-9) and np.allclose(d2[31:57], [-1.0, 0.0]) and np.allclose(d2[:31], [0.0, 1.0]) and (c2 == c).all()
    assert np.allclose(sample_roads(sd, sh0, 0.0)[0], pos + sh0) and np.allclose(sample_roads(sd, None, 0.3)[0], pos), "no shift: the yaw alone is not applied"
    assert np.allclose(undo(p2, sh0, np.pi / 2), pos, atol=1e-9) and np.array_equal(undo(p2, sh0, np.pi / 2, "identity"), p2)
    assert not np.allclose(undo(p2, sh0, np.pi / 2, "shift_sign"), pos) and not np.allclose(undo(p2, sh0, np.pi / 2, "rotation_transposed"), pos)
    # the pose error draws: the order of map_sim (direction, length, yaw), the bounds, the token generator is map_sim's
    r1, r2 = np.random.RandomState(5), np.random.RandomState(5)
    sh, yw = pose_error(r1, 1.0, 2.0)
    ang = r2.uniform(0, 2 * np.pi)
    ln = r2.uniform(1.0, 2.0)
    assert np.allclose(sh, ln * np.array([np.cos(ang), np.sin(ang)])) and abs(yw - np.deg2rad(r2.randn())) < 1e-15 and 1.0 <= np.linalg.norm(sh) <= 2.0
    draws = [pose_error(token_rng(f"t{k}"), *ERROR_SHIFT) for k in range(400)]
    assert all(1.0 <= np.linalg.norm(s_) <= 2.0 for s_, _ in draws) and abs(np.std([np.degrees(y_) for _, y_ in draws]) - 1.0) < 0.15 and abs(np.mean([np.degrees(y_) for _, y_ in draws])) < 0.15
    assert token_rng("a").rand() == token_rng("a").rand() != token_rng("b").rand() and token_rng("a").rand() != token_rng("a", "|gross").rand()
    import map_sim as MS
    assert token_rng("tok1").rand() == MS._rng_for("tok1").rand() and np.allclose(MS.make_map(dict(pieces=[], sample="x"), "error", token="tok1")["shift"], pose_error(token_rng("tok1"), *ERROR_SHIFT)[0]), "the token generator and the pose error are map_sim's"
    # the error map: fixed per token, another token differs; each of the 3 roads is dropped with probability 0.1
    e1, e2, e3 = (error_map(toy_pts(0), t) for t in ("a", "a", "b"))
    assert all(np.array_equal(x, y) for x, y in zip(e1, e2)) and not np.array_equal(e1[0], e3[0])
    sizes = [len(error_map(toy_pts(0), f"t{k}")[0]) for k in range(300)]
    assert 83 <= max(sizes) <= 86 and 0.2 < np.mean([s_ < 80 for s_ in sizes]) < 0.35, "a side road is missing with probability 0.1 (1 - 0.9^3 = 0.27 of the frames lose one; a rotated road of 60 m can get one sample more by rounding)"
    # truncation: the nearest the ego, in the original order; counts clipped to 1 - 5, 0 stays unknown; the table has the fixed keys, one row per sample
    far = (np.stack([np.arange(600.0), np.zeros(600)], 1), np.tile([1.0, 0.0], (600, 1)), np.full(600, 7.0))
    t = finish(far)
    assert len(t["valid"]) == MAX_SAMPLES and np.array_equal(t["pos"][:, 0], np.arange(512.0)) and (t["count"] == COUNT_MAX).all() and sorted(t) == sorted(KEYS) and t["count"].dtype == np.int64 and t["valid"].all()
    assert np.array_equal(finish((far[0][::-1].copy(), far[1], far[2]))["pos"][:, 0], np.arange(511.0, -1.0, -1.0)), "the nearest are kept, in their own order"
    ring = (np.concatenate([np.full((300, 2), 100.0), np.zeros((300, 2))]), far[1], far[2])
    assert (np.linalg.norm(finish(ring)["pos"], axis=1) < 1).sum() == 300 and len(finish(ring)["valid"]) == 512
    assert (finish(with_count(far, 0))["count"] == 0).all() and (finish(with_count(far, 2))["count"] == 2).all() and finish((far[0], far[1], np.array([0.2] * 511 + [3.0] + [0.2] * 88)))["count"][[0, 1, 511]].tolist() == [1, 1, 3]
    e = finish(EMPTY)
    assert len(e["valid"]) == 0 and e["pos"].shape == (0, 2)
    # the leak check: extra keys of any lane-level kind are refused, missing keys too, a wrong shape too
    good = finish(exact_map(toy_pts(0)))
    for extra in ("frac", "piece", "ids", "succ", "adj", "kind", "lane_index", "route", "road", "pidx", "tan"):
        refuses(lambda: check_keys(dict(good, **{extra: np.zeros(len(good["valid"]))})), "a lane-level field is a leak")
    refuses(lambda: check_keys({k: v for k, v in good.items() if k != "count"}), "missing ['count']")
    refuses(lambda: check_keys(dict(good, pos=good["pos"][:, :1])), "map table shapes")
    refuses(lambda: collate_maps([dict(good, frac=good["count"])]), "a lane-level field is a leak")
    # collate: padding to the largest map (at least one column, not valid), the values in place
    cb = collate_maps([finish(far), finish(EMPTY), good])
    assert cb["pos"].shape == (3, 512, 2) and cb["valid"].sum(1).tolist() == [512, 0, 83] and not cb["valid"][2, 83:].any() and cb["count"][2, :31].eq(1).all() and cb["pos"].dtype == torch.float32 and cb["count"].dtype == torch.int64 and cb["valid"].dtype == torch.bool
    assert torch.equal(cb["pos"][2, 31:57, 1], torch.from_numpy(np.arange(-20.0, 30.001, 2.0).astype(np.float32))) and (cb["pos"][1] == 0).all()
    ce = collate_maps([finish(EMPTY)] * 2)
    assert ce["pos"].shape == (2, 1, 2) and not ce["valid"].any(), "all absent: one padding column"
    # the draw classes: the boundaries of MIX and the shares over many draws
    assert [draw_class(u) for u in (0.0, 0.2499, 0.25, 0.7199, 0.72, 0.7999, 0.80, 0.8999, 0.90, 0.9999)] == ["exact", "exact", "error", "error", "unknown", "unknown", "absent", "absent", "gross", "gross"]
    cls = [draw_class(np.random.RandomState([0, k, 0]).rand()) for k in range(20000)]
    assert all(abs(cls.count(n_) / len(cls) - p) < 0.012 for n_, p in MIX), {n_: cls.count(n_) / len(cls) for n_, _ in MIX}
    # the S plan: (i + 97) mod N, the next frame of another scene when that is the same scene; a single scene is refused
    sc = np.repeat(np.arange(5), 40)
    pl = other_scene_plan(sc)
    assert (pl == (np.arange(200) + 97) % 200).all() and (sc[pl] != sc).all()
    assert other_scene_plan([0, 0, 0, 1, 1, 1], 1).tolist() == [3, 3, 3, 0, 0, 0] and other_scene_plan([0, 0, 1, 1], 1).tolist() == [2, 2, 0, 0]
    refuses(lambda: other_scene_plan([0, 0, 0]), "one scene only")
    # the dataset on the toy world, every read-out mode by hand
    pts_of = lambda i: toy_pts(i % 5)  # noqa: E731
    assert all(np.array_equal(toy_data("exact").table(0)[k], finish(exact_map(toy_pts(0)))[k]) for k in KEYS) and len(toy_data("exact").table(0)["valid"]) == 83
    dS = toy_data("S")
    for i in (0, 5, 11):
        raw_e = error_map(pts_of(i), f"toy{i}")
        assert all(np.array_equal(x, y) for x, y in zip(toy_data("error").raw_map(i), raw_e)), "error = the seeded error map, the same on every call"
        g = toy_data("gross").raw_map(i)
        assert len(g[0]) == len(raw_e[0]) and np.allclose(np.linalg.norm(g[0] - raw_e[0], axis=1), GROSS_M) and np.array_equal(g[1], raw_e[1]) and np.array_equal(g[2], raw_e[2]), "gross = the error map shifted rigidly by 5 m"
        u, c2_, nn = toy_data("U").raw_map(i), toy_data("C2").raw_map(i), toy_data("none").raw_map(i)
        assert np.array_equal(u[0], raw_e[0]) and (u[2] == 0).all() and np.array_equal(c2_[0], raw_e[0]) and (c2_[2] == 2).all() and len(nn[0]) == 0
        j = int(dS.plan[i])
        assert dS.scene[j] != dS.scene[i] and all(np.array_equal(x, y) for x, y in zip(dS.raw_map(i), error_map(pts_of(j), f"toy{j}"))), "S = the error map of the frame of another scene (the plan)"
    assert len({tuple(toy_data("gross").raw_map(i)[0][0] - toy_data("error").raw_map(i)[0][0]) for i in range(12)}) > 6, "the gross direction differs per frame"
    # mode train: the draw counter counts the reads of a frame; each read replays the stream of (init seed, token, draw index): the class, then the pose error, then the degradation; another init seed, another map
    dt = toy_data("train", init_seed=3)
    a0, a1 = dt.raw_map(2), dt.raw_map(2)
    assert int(dt.draws[2]) == 2 and int(dt.draws[3]) == 0, "the draw counter of a frame counts its reads"
    dt2 = toy_data("train", init_seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(dt2.raw_map(2), a0)) and all(np.array_equal(x, y) for x, y in zip(dt2.raw_map(2), a1)), "the same (init seed, token, draw index) = the same map"
    assert toy_data("train", init_seed=4).raw_map(2)[0].tobytes() != a0[0].tobytes() or toy_data("train", init_seed=4).raw_map(2)[2].tobytes() != a0[2].tobytes(), "another init seed, another draw"
    big, seen, n_read = toy_data("train", init_seed=2), dict.fromkeys([n_ for n_, _ in MIX], 0), 2000
    ex_raw = exact_map(toy_pts(0))
    for k in range(n_read):
        rr = np.random.RandomState([2, zlib.crc32(b"toy0"), k])                                          # the stream of read k of frame 0
        cl = draw_class(rr.rand())
        raw = big.raw_map(0)
        seen[cl] += 1
        if cl == "exact":
            want = ex_raw
        elif cl == "absent":
            want = EMPTY
        elif cl in ("error", "unknown"):
            want = error_like(toy_pts(0), rr, *TRAIN_SHIFT)
            want = with_count(want, 0) if cl == "unknown" else want
        elif rr.rand() < 0.5:
            want = error_like(toy_pts(0), rr, *GROSS_SHIFT)
        else:
            j = int(rr.randint(12))
            while big.scene[j] == big.scene[0]:
                j = int(rr.randint(12))
            want = error_like(pts_of(j), rr, *TRAIN_SHIFT)
        assert len(raw[0]) == len(want[0]) and all(np.array_equal(x, y) for x, y in zip(raw, want)), f"read {k} ({cl}) is not the replay of its stream"
    assert all(abs(seen[n_] / n_read - p) < 0.03 for n_, p in MIX), seen
    # the dataset item: arm z builds nothing (the plain item), arm m adds the maps, with a donor the S map
    dz, dm, dd2 = toy_data("error", maps=False), toy_data("error"), toy_data("error", with_donor=True)
    assert sorted(dz[3]) == ["token"] and sorted(dm[3]) == ["maps", "token"] and sorted(dd2[3]) == ["donor", "maps", "token"] and dz.draws is None
    assert all(np.array_equal(dd2[3]["donor"][k], dS.table(3)[k]) for k in KEYS) and all(np.array_equal(dm[3]["maps"][k], dm.table(3)[k]) for k in KEYS)
    refuses(lambda: toy_data("train", scene=[0] * 12), "one scene only")
    refuses(lambda: MapData(tok=ToyTok(3), mode="nope"), "nope")
    print("lane_map_data selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ident", default="e292_a")
    s.add_argument("--n", type=int)
    s.add_argument("--out")
    for name, n_default in (("potential", None), ("fixture", 100)):
        s = sub.add_parser(name)
        s.add_argument("--split", default="val", choices=("val", "train"))
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--ident", default="e292_a")
        s.add_argument("--n", type=int, default=n_default)
        s.add_argument("--out")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    res = check(a.seed, a.ident, a.n) if a.cmd == "check" else (potential if a.cmd == "potential" else fixture)(a.split, n=a.n, seed=a.seed, ident=a.ident)
    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        with open(p, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    if a.cmd == "fixture":
        assert res["passed"], "the fixture FAILED"


if __name__ == "__main__":
    main()
