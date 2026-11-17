"""Tests of the data of the cross-attention to the SD map (`data/lane_map_data.py`, E374).

    wsl bash tools/env/_py14.sh test_lane_map.py            # the selftest, `check` on 60 frames of VAL and TRAIN, the fixture on 100 real VAL frames, the pinned numbers of `check` (60 frames), of the fixture (20 frames) and of `potential` (60 frames); then the hand mutants
    wsl bash tools/env/_py14.sh test_lane_map.py selftests  # the same without the mutants

`potential` is also compared, in the unmutated run, with an INDEPENDENT re-implementation of the card's step 0 (`reference_potential`: the scratch script step0_map2.py with the cut of the in-intersection connector ranges written out again, loop by loop) to 1e-12.

A mutant = one textual change of the source (the cut of the in-intersection connector ranges skipped, the pose error applied with the transposed rotation, the translation before the rotation, a wrong cap or the farthest samples kept, a lane-level key let through, the S plan in the same scene, the draw counter not advanced, a class boundary moved, a threshold of the supported shares
changed, ...); the mutated copy runs in a temporary folder with BFLC_ROOT pointing at this project, four at a time, and every mutant must make ITS COMMAND fail: the selftest (default), or a real-data command named in the mutant's 4th field: `fixture` (20 frames: exit code and the pinned numbers), `potential`
(60 frames, the pinned numbers), `check` (60 frames of VAL and TRAIN, the pinned numbers). The pins are numbers of the caches of B' seed 0 (VAL and TRAIN, GT signature 9ad1d4ce30) and of the lane GT store of this checkout; a rebuilt cache or GT needs new pins.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "training", "data")
MOD = "lane_map_data.py"
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")      # a shared box: one thread per test process

PINS = json.loads("""{
 "fixture": {
  "shift_min": 1.0116577526632158,
  "shift_max": 1.9569271212690489,
  "yaw_std_deg": 1.1353025598341813,
  "exact_samples": 2654,
  "own_share_error": 0.9937902562209062,
  "own_share_S": 0.32300807326583714,
  "connector_share_cut": 0.01685613484907879,
  "connector_share_uncut": 0.24062713019768234,
  "connector_samples": 2551,
  "undo.undo.n": 2409,
  "undo.undo.median": 0.06031006403702066,
  "undo.undo.p90": 0.7252420228813172,
  "undo.undo.p99": 1.4494209052096356,
  "undo.identity.n": 2409,
  "undo.identity.median": 1.0651538749169398,
  "undo.identity.p90": 2.0720429890962797,
  "undo.identity.p99": 2.949767706786176,
  "undo.shift_sign.n": 2409,
  "undo.shift_sign.median": 1.7702277926278345,
  "undo.shift_sign.p90": 3.431760516246763,
  "undo.shift_sign.p99": 4.137514273091624,
  "undo.rotation_transposed.n": 2409,
  "undo.rotation_transposed.median": 0.7443555567543796,
  "undo.rotation_transposed.p90": 2.1804156443526477,
  "undo.rotation_transposed.p99": 3.619831279571746
 },
 "potential": {
  "samples_per_frame": 132.83333333333334,
  "classes.novel.n": 1412,
  "classes.novel.supported_exact": 0.9058073654390935,
  "classes.novel.supported_error": 0.7924929178470255,
  "classes.novel.supported_other_scene": 0.12039660056657224,
  "classes.novel.median_dist_exact": 0.9851950644026312,
  "classes.novel.median_dist_error": 1.5104750423681186,
  "classes.dup.n": 5222,
  "classes.dup.supported_exact": 0.9442742244350824,
  "classes.dup.supported_error": 0.8008425890463424,
  "classes.dup.supported_other_scene": 0.23994638069705093,
  "classes.dup.median_dist_exact": 1.22332971924152,
  "classes.dup.median_dist_error": 1.6241235352893297,
  "classes.false.n": 9594,
  "classes.false.supported_exact": 0.2892432770481551,
  "classes.false.supported_error": 0.24379820721284137,
  "classes.false.supported_other_scene": 0.16885553470919323,
  "classes.false.median_dist_exact": 4.100335484769882,
  "classes.false.median_dist_error": 4.681706817807978,
  "classes.deliv_true.n": 8891,
  "classes.deliv_true.supported_exact": 0.9562478911258576,
  "classes.deliv_true.supported_error": 0.7840512878191429,
  "classes.deliv_true.supported_other_scene": 0.23844336969969632,
  "classes.deliv_true.median_dist_exact": 1.3131608459046975,
  "classes.deliv_true.median_dist_error": 1.592057673888792,
  "classes.deliv_false.n": 1092,
  "classes.deliv_false.supported_exact": 0.4981684981684982,
  "classes.deliv_false.supported_error": 0.41025641025641024,
  "classes.deliv_false.supported_other_scene": 0.18223443223443223,
  "classes.deliv_false.median_dist_exact": 2.776830851235781,
  "classes.deliv_false.median_dist_error": 3.219798311976238,
  "by_bin.0.n": 3983,
  "by_bin.0.true_share": 0.3341702234496611,
  "by_bin.0.supported_if_true": 0.7911344853493614,
  "by_bin.0.supported_if_false": 0.1858974358974359,
  "by_bin.0.novel_share_supported": 0.0517464424320828,
  "by_bin.0.novel_share_unsupported": 0.006975789905621666,
  "by_bin.1.n": 5246,
  "by_bin.1.true_share": 0.3966831871902402,
  "by_bin.1.supported_if_true": 0.814512253724171,
  "by_bin.1.supported_if_false": 0.2344391785150079,
  "by_bin.1.novel_share_supported": 0.07386130488305294,
  "by_bin.1.novel_share_unsupported": 0.01423994304022784,
  "by_bin.2.n": 2900,
  "by_bin.2.true_share": 0.4113793103448276,
  "by_bin.2.supported_if_true": 0.8113998323554066,
  "by_bin.2.supported_if_false": 0.27006444053895723,
  "by_bin.2.novel_share_supported": 0.15395381385584325,
  "by_bin.2.novel_share_unsupported": 0.03467029231815092,
  "by_bin.3.n": 2262,
  "by_bin.3.true_share": 0.47126436781609193,
  "by_bin.3.supported_if_true": 0.7936210131332082,
  "by_bin.3.supported_if_false": 0.28595317725752506,
  "by_bin.3.novel_share_supported": 0.2297979797979798,
  "by_bin.3.novel_share_unsupported": 0.07076350093109869,
  "by_bin.4.n": 1837,
  "by_bin.4.true_share": 0.5242242787152966,
  "by_bin.4.supported_if_true": 0.7673935617860852,
  "by_bin.4.supported_if_false": 0.34439359267734554,
  "by_bin.4.novel_share_supported": 0.35192307692307695,
  "by_bin.4.novel_share_unsupported": 0.13676286072772897
 },
 "check": {
  "val.frames": 873,
  "val.samples_with_count_above_max": 0,
  "val.sizes.exact.frames": 60,
  "val.sizes.exact.mean": 132.83333333333334,
  "val.sizes.exact.max": 206,
  "val.sizes.exact.truncated": 0,
  "val.sizes.error.frames": 60,
  "val.sizes.error.mean": 119.78333333333333,
  "val.sizes.error.max": 200,
  "val.sizes.error.truncated": 0,
  "train.frames": 8414,
  "train.samples_with_count_above_max": 0,
  "train.sizes.exact.frames": 60,
  "train.sizes.exact.mean": 139.73333333333332,
  "train.sizes.exact.max": 261,
  "train.sizes.exact.truncated": 0,
  "train.sizes.error.frames": 60,
  "train.sizes.error.mean": 126.18333333333334,
  "train.sizes.error.max": 234,
  "train.sizes.error.truncated": 0,
  "train_guard.train_frames": 8414,
  "train_guard.val_test_frames": 1706
 }
}""")

MUTANTS = [
    ("the cut of the in-intersection connector ranges is skipped (the real-frame test)", '            keep[sg["i0"]: sg["i1"] + 1] = False', "            pass", "fixture"),
    ("the cut is skipped", '            keep[sg["i0"]: sg["i1"] + 1] = False', "            pass"),
    ("the frame's points bypass the cut", "return frame_pts(self.pieces(i))", 'return [np.asarray(pc["pts"], float) for pc in self.pieces(i)]', "fixture"),
    ("a connector range outside an intersection is cut", 'if sg["kind"] == "connector" and sg["in_intersection"]:\n            keep[', 'if sg["kind"] == "connector":\n            keep['),
    ("a lane range in an intersection is cut", 'if sg["kind"] == "connector" and sg["in_intersection"]:\n            keep[', 'if sg["in_intersection"]:\n            keep['),
    ("the end of a connector range is not cut", 'keep[sg["i0"]: sg["i1"] + 1] = False', 'keep[sg["i0"]: sg["i1"]] = False'),
    ("a run of one point is kept", "if b - a >= 2]", "if b - a >= 1]"),
    ("the counts are rounded down", "np.rint(np.concatenate(C))", "np.floor(np.concatenate(C))"),
    ("the counts are rounded up", "np.rint(np.concatenate(C))", "np.ceil(np.concatenate(C))"),
    ("mode other takes a fixed frame", "            src = int(rng.randint(len(self.tokens)))\n            while self.scene[src] == self.scene[i]:\n                src = int(rng.randint(len(self.tokens)))", "            src = int(self.plan[i])"),
    ("mode other may take a frame of the same scene", "            while self.scene[src] == self.scene[i]:", "            while False:"),
    ("mode other has the stream of mode train", "+ ([1] if other else [])", "+ []"),
    ("mode other has no counters", 'if mode in ("train", "other") else None       # the draw', 'if mode == "train" else None       # the draw'),
    ("mode S without a plan fails without a message", 'assert self.plan is not None, "mode S needs the scene plan', 'assert True or "mode S needs the scene plan'),
    ("the TRAIN guard ignores the TEST frames", 'for r in list(splits["val"]) + list(splits["test"])}', 'for r in list(splits["val"])}'),
    ("the TRAIN guard lets a VAL frame through", "assert not both,", "assert True or not both,"),
    ("the TRAIN guard does not check the rows", 'assert len(set(rows)) == len(rows) and sorted(int(r) for r in rows) == sorted(int(r) for r in splits["train"]),', "assert True or len(set(rows)) == len(rows),"),
    ("the TRAIN guard does not check the tokens", 'assert all(train_tokens[int(r)] == t for r, t in zip(rows, tokens)),', "assert True or all(train_tokens[int(r)] == t for r, t in zip(rows, tokens)),"),
    ("the TRAIN guard is not run by check", 'res["train_guard"] = train_guard(ds.tok.rows, ds.tokens, splits, *tok_files)', 'res["train_guard"] = dict(train_frames=0, val_test_frames=0)', "check"),
    ("the pose error is applied with the transposed rotation", "poly = poly @ rot(yaw).T + shift", "poly = poly @ rot(yaw) + shift"),
    ("the shift is added before the rotation", "poly = poly @ rot(yaw).T + shift", "poly = (poly + shift) @ rot(yaw).T"),
    ("the yaw is not applied", "poly = poly @ rot(yaw).T + shift", "poly = poly + shift"),
    ("the direction of the shift is drawn on half the circle", "    ang = rng.uniform(0, 2 * np.pi)\n    shift = rng.uniform(lo, hi)", "    ang = rng.uniform(0, np.pi)\n    shift = rng.uniform(lo, hi)"),
    ("the yaw is in degrees", "return shift, np.deg2rad(rng.randn())", "return shift, rng.randn()"),
    ("the token generator takes seven digits", "hexdigest()[:8]", "hexdigest()[:7]"),
    ("the exact map is degraded", "def exact_map(pts):\n    return sample_roads(roads_of(pts))", "def exact_map(pts):\n    return sample_roads(SD.degrade_sd(roads_of(pts), np.random.RandomState(0), drop=DROP))"),
    ("the samples are 1 m apart", "STEP = 2.0 ", "STEP = 1.0 "),
    ("no side road is dropped", "DROP = 0.1 ", "DROP = 0.0 "),
    ("the cap is 600", "MAX_SAMPLES = 704 ", "MAX_SAMPLES = 600 "),
    ("the cap is 512 again", "MAX_SAMPLES = 704 ", "MAX_SAMPLES = 512 "),
    ("the cap is 200 (the real maps are cut)", "MAX_SAMPLES = 704 ", "MAX_SAMPLES = 200 ", "check"),
    ("the truncation keeps the farthest samples", 'kind="stable")[:n])', 'kind="stable")[-n:])'),
    ("there is no truncation", "    if len(pos) <= n:\n        return raw", "    if True:\n        return raw"),
    ("the kept samples are not in their own order", 'keep = np.sort(np.argsort(np.linalg.norm(pos, axis=1), kind="stable")[:n])', 'keep = np.argsort(np.linalg.norm(pos, axis=1), kind="stable")[:n]'),
    ("the counts are not clipped", "c = np.where(c > 0, np.clip(c, 1, COUNT_MAX), 0)", "c = c"),
    ("unknown counts become 1", "c = np.where(c > 0, np.clip(c, 1, COUNT_MAX), 0)", "c = np.clip(c, 1, COUNT_MAX)"),
    ("the leak check lets a lane-level key through", "assert not extra and not missing,", "assert not missing,"),
    ("the leak check lets a missing key through", "assert not extra and not missing,", "assert not extra,"),
    ("collate does not check the keys", "    for t in tables:\n        check_keys(t)", "    for t in tables:\n        pass"),
    ("the padding is valid", "valid=np.zeros((len(tables), M), bool))", "valid=np.ones((len(tables), M), bool))"),
    ("an all-absent batch has no padding column", 'M = max(1, max(len(t["valid"]) for t in tables))', 'M = max(0, max(len(t["valid"]) for t in tables))'),
    ("the exact class is 20 %", 'MIX = (("exact", 0.25), ("error", 0.47),', 'MIX = (("exact", 0.20), ("error", 0.52),'),
    ("the unknown class is 10 % and the absent class 8 %", '("unknown", 0.08), ("absent", 0.10), ("gross", 0.10))     # the draw', '("unknown", 0.10), ("absent", 0.08), ("gross", 0.10))     # the draw'),
    ("the S plan is 96", "OTHER = 97 ", "OTHER = 96 "),
    ("the S plan stays in the scene", "        while scene[j] == scene[i]:\n            j = (j + 1) % len(scene)", "        while False:\n            j = (j + 1) % len(scene)"),
    ("S is the frame's own error map", "return error_map(self.pts(self.plan[i]), self.tokens[self.plan[i]])", "return error_map(self.pts(i), self.tokens[i])"),
    ("S is the exact map of another scene", "return error_map(self.pts(self.plan[i]), self.tokens[self.plan[i]])", "return exact_map(self.pts(self.plan[i]))"),
    ("gross is 4 m", "GROSS_M = 5.0 ", "GROSS_M = 4.0 "),
    ("gross shifts every frame the same way", 'ang = token_rng(tok, "|gross").uniform(0, 2 * np.pi)', "ang = 0.0"),
    ("U sets the counts to 1", 'with_count(base, 0 if mode == "U" else 2)', 'with_count(base, 1 if mode == "U" else 2)'),
    ("C2 sets the counts to 3", 'with_count(base, 0 if mode == "U" else 2)', 'with_count(base, 0 if mode == "U" else 3)'),
    ("the error map is the same for every token", "return error_like(pts, token_rng(token), *ERROR_SHIFT)", 'return error_like(pts, token_rng("x"), *ERROR_SHIFT)'),
    ("the error shift is 0 - 2 m", "ERROR_SHIFT, TRAIN_SHIFT, GROSS_SHIFT = (1.0, 2.0), (0.0, 2.0), (4.0, 8.0)", "ERROR_SHIFT, TRAIN_SHIFT, GROSS_SHIFT = (0.0, 2.0), (0.0, 2.0), (4.0, 8.0)"),
    ("the train draws use the error shift", "raw = error_like(self.pts(src), rng, *TRAIN_SHIFT)", "raw = error_like(self.pts(src), rng, *ERROR_SHIFT)"),
    ("the gross draw uses the train shift", "return error_like(self.pts(src), rng, *GROSS_SHIFT)", "return error_like(self.pts(src), rng, *TRAIN_SHIFT)"),
    ("the gross draw may take a frame of the same scene", "        while self.scene[j] == self.scene[i]:", "        while False:"),
    ("the draw counter does not advance", "self.draws[i] = n + 1", "self.draws[i] = n"),
    ("the draw seed ignores the draw index", "zlib.crc32(self.tokens[i].encode()), n] +", "zlib.crc32(self.tokens[i].encode()), 0] +"),
    ("the draw seed ignores the init seed", "RandomState([self.init_seed % 2 ** 32, zlib", "RandomState([0, zlib"),
    ("the draw seed is the frame index, not the token", "zlib.crc32(self.tokens[i].encode()), n] +", "i, n] +"),
    ("unknown counts keep their values", 'return with_count(raw, 0) if cls == "unknown" else raw', "return raw"),
    ("an absent draw returns the exact map", '        if cls == "absent":\n            return EMPTY', '        if cls == "absent":\n            return exact_map(self.pts(i))'),
    ("the counters are not in shared memory", '.share_memory_() if mode in ("train", "other")', ' if mode in ("train", "other")'),
    ("arm z builds the maps", '        if self.maps:\n            out["maps"]', '        if True:\n            out["maps"]'),
    ("the donor is missing", '            if self.with_donor:\n                out["donor"]', '            if False:\n                out["donor"]'),
    ("the TEST guard is removed", "LPD.check_rows(split, self.tok.rows)", "pass"),
    ("collate drops the donor", 'for k in ("maps", "donor"):', 'for k in ("maps",):', "check"),
    ("the truncation cap of the check is 256", "            if len(raw[0]) > MAX_SAMPLES:\n                cut[m].append", "            if len(raw[0]) > 256:\n                cut[m].append", "check"),
    ("the check counts the error map as the exact one", "            raw = ds.raw_map(int(i), m)\n            out[m].append(len(raw[0]))", '            raw = ds.raw_map(int(i), "exact")\n            out[m].append(len(raw[0]))', "check"),
    ("the fixture undoes the pose with the transposed rotation", "    return (P - shift) @ R\n\n\ndef fixture_checks", "    return (P - shift) @ R.T\n\n\ndef fixture_checks", "fixture"),
    ("the fixture's S map is the frame's own", 'sm, j = ds.raw_map(i, "S"), int(ds.plan[i])', 'sm, j = ds.raw_map(i, "error"), int(ds.plan[i])', "fixture"),
    ("the supported distance is 2 m", "SUP_D, SUP_COS = 3.0, 0.8 ", "SUP_D, SUP_COS = 2.0, 0.8 ", "potential"),
    ("the supported cosine is signed", "np.abs((dirv * T[jj]).sum(1)) >= SUP_COS", "(dirv * T[jj]).sum(1) >= SUP_COS", "potential"),
    ("a found node is one a valid token is near", "miss = ~((near & deliv[:, None]).any(0))", "miss = ~((near & valid[:, None]).any(0))", "potential"),
    ("a padding token is a weak token", "weak = valid & ~deliv", "weak = ~deliv", "potential"),
    ("a duplicate is near any node", "dup=weak & near.any(1) & ~nov,", "dup=weak & near.any(1),", "potential"),
    ("the delivered threshold is exclusive", "deliv = valid & (ep >= 0.5)", "deliv = valid & (ep > 0.5)"),
    ("the heading cosine is exclusive", "(dirv @ gdir.T >= COS)", "(dirv @ gdir.T > COS)"),
    ("the null map is the frame's own error map", 'null_error=ds.raw_map(int(ds.plan[i]), "error")', 'null_error=ds.raw_map(int(i), "error")', "potential"),
    ("the error map of potential is the exact map", 'error=ds.raw_map(int(i), "error")', 'error=ds.raw_map(int(i), "exact")', "potential"),
    ("the near radius is exclusive", "(np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD)", "(np.linalg.norm(pos[:, None] - gpos[None], axis=2) < RAD)"),
]


def run(args):
    """selftest, check (60 frames) and fixture (100 frames) of the module in `DATA` itself: the unmutated code must pass"""
    for a in args:
        r = subprocess.run([sys.executable, os.path.join(DATA, MOD)] + a, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{MOD} {' '.join(a)} failed:\n{r.stdout[-1200:]}{r.stderr[-1500:]}"
        print(r.stdout.strip().splitlines()[-1], flush=True)


def reference_potential(n):
    """an INDEPENDENT re-implementation of the card's step 0 on the cut map (the scratch script step0_map2.py of the card, with the cut of the in-intersection connector ranges written out again point by point; it shares nothing with `lane_map_data` but `sd_lib`, the GT store and the cache readers)
    for n evenly spaced VAL frames of B' seed 0 -> the numbers of `potential`'s JSON (classes, by_bin, samples_per_frame)"""
    import hashlib
    from scipy.spatial import cKDTree
    for sub_ in ("training/data", "training/data/prep", "training/grid", ".."):
        sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
    import lane_gt_data as LD
    import lane_ptok_data as LP
    import lane_tok_data as LT
    import lane_tokcache as LC
    import sd_lib as SD
    c = LC.TokCache(LP.cache_dir("val", 0, "e292_a"))
    ds = LT.TokData("val", res=2.0, feats=False)
    lane_gt, _ = LD.lane_store()

    def cut_points(piece):
        pts, drop, runs, cur = np.asarray(piece["pts"], float), set(), [], []
        for s_ in piece["segments"]:
            if s_["kind"] == "connector" and s_["in_intersection"]:
                drop.update(range(s_["i0"], s_["i1"] + 1))
        for k in range(len(pts)):
            if k in drop:
                if len(cur) >= 2:
                    runs.append(np.array(cur))
                cur = []
            else:
                cur.append(pts[k])
        if len(cur) >= 2:
            runs.append(np.array(cur))
        return runs

    def samples(g, rng, degrade, shift=None, yaw=0.0):
        sd = SD.build_sd(dict(pts=[r for pc in g["pieces"] for r in cut_points(pc)]), [])
        if degrade:
            sd = SD.degrade_sd(sd, rng, drop=0.1)
        P, T = [], []
        for poly, cnt in sd["roads"]:
            poly = np.asarray(poly, float)
            if len(poly) < 2:
                continue
            if degrade:
                cy, sy = np.cos(yaw), np.sin(yaw)
                poly = poly @ np.array([[cy, -sy], [sy, cy]]).T + shift
            seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
            s_ = np.concatenate([[0], np.cumsum(seg)])
            if s_[-1] < 1e-6:
                continue
            t = np.linspace(0, s_[-1], max(int(np.ceil(s_[-1] / 2.0)), 1) + 1)
            q = np.stack([np.interp(t, s_, poly[:, 0]), np.interp(t, s_, poly[:, 1])], 1)
            tg = np.gradient(q, axis=0)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            P.append(q)
            T.append(tg)
        return (np.concatenate(P), np.concatenate(T)) if P else (np.zeros((0, 2)), np.zeros((0, 2)))

    def maps_of(i, degrade):
        g = lane_gt.load(c.tokens[i])
        rng = np.random.RandomState(int(hashlib.md5(c.tokens[i].encode()).hexdigest()[:8], 16))
        ang = rng.uniform(0, 2 * np.pi)
        sh = rng.uniform(1, 2) * np.array([np.cos(ang), np.sin(ang)])
        return samples(g, rng, degrade, sh, np.deg2rad(rng.randn()))

    N = len(c)
    cls = ("novel", "dup", "false", "deliv_true", "deliv_false")
    res = {(mode, k): [] for mode in ("exact", "error", "null") for k in cls}
    bins, nsd = [], []
    for i in np.linspace(0, N - 1, n).astype(int):
        t = ds.table(int(i))
        gpos, gdir = t["pos"], t["dirv"]
        if len(gpos) == 0:
            continue
        pos, dirv, valid, ep = (np.array(c.f[k][i], float) if k != "valid" else np.array(c.f[k][i]) for k in ("pos", "dirv", "valid", "e_p"))
        deliv, weak = valid & (ep >= 0.5), valid & ~(valid & (ep >= 0.5))
        near = (np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= 1.5) & (dirv @ gdir.T >= 0.5)
        miss = ~((near & deliv[:, None]).any(0))
        sel = dict(novel=weak & (near & miss[None]).any(1), dup=weak & near.any(1) & ~(near & miss[None]).any(1), false=weak & ~near.any(1), deliv_true=deliv & near.any(1), deliv_false=deliv & ~near.any(1))
        maps = dict(exact=maps_of(i, False), error=maps_of(i, True), null=maps_of((i + 97) % N, True))
        nsd.append(len(maps["exact"][0]))
        for mode, (P, T) in maps.items():
            if len(P) == 0:
                sup, dist = np.zeros(len(pos), bool), np.full(len(pos), 99.0)
            else:
                dist, jj = cKDTree(P).query(pos)
                sup = (dist <= 3.0) & (np.abs((dirv * T[jj]).sum(1)) >= 0.8)
            for k in cls:
                res[(mode, k)].append(np.stack([dist[sel[k]], sup[sel[k]]], 1))
            if mode == "error":
                bins.append(np.stack([ep[weak], sup[weak], near.any(1)[weak], (near & miss[None]).any(1)[weak]], 1))
    out = dict(samples_per_frame=float(np.mean(nsd)), classes={}, by_bin=[])
    for k in cls:
        a = {m: np.concatenate(res[(m, k)]) for m in ("exact", "error", "null")}
        out["classes"][k] = dict(n=len(a["exact"]), supported_exact=float(a["exact"][:, 1].mean()), supported_error=float(a["error"][:, 1].mean()), supported_other_scene=float(a["null"][:, 1].mean()), median_dist_exact=float(np.median(a["exact"][:, 0])), median_dist_error=float(np.median(a["error"][:, 0])))
    B = np.concatenate(bins)
    for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)):
        m = (B[:, 0] >= lo) & (B[:, 0] < hi)
        t_, f_, s_, u_ = m & (B[:, 2] > 0.5), m & (B[:, 2] < 0.5), m & (B[:, 1] > 0.5), m & (B[:, 1] < 0.5)
        out["by_bin"].append(dict(n=int(m.sum()), true_share=float(t_.sum() / m.sum()), supported_if_true=float(B[t_, 1].mean()), supported_if_false=float(B[f_, 1].mean()), novel_share_supported=float(B[s_, 3].mean()), novel_share_unsupported=float(B[u_, 3].mean())))
    return out


def compare_reference(path, n=60):
    """`potential --n 60`'s JSON against the independent re-implementation: every number to 1e-12"""
    r, ref = json.load(open(path)), reference_potential(n)
    pairs = [(r["samples_per_frame"], ref["samples_per_frame"], "samples_per_frame")]
    pairs += [(r["classes"][k][kk], vv, f"{k}.{kk}") for k, v in ref["classes"].items() for kk, vv in v.items()]
    pairs += [(r["by_bin"][i][kk], vv, f"bin{i}.{kk}") for i, b in enumerate(ref["by_bin"]) for kk, vv in b.items()]
    bad = [(t, a, b) for a, b, t in pairs if abs(a - b) > 1e-12]
    assert not bad, f"potential differs from its independent re-implementation: {bad[:3]}"
    return len(pairs)


def get(r, path):
    """the number at a dotted path of a result dict (list entries by their index)"""
    for k in path.split("."):
        r = r[int(k)] if isinstance(r, list) else r[k]
    return r


def check_pins(kind, path):
    """the numbers of a real-data command against the pins (dotted paths of its JSON; integers exactly, floats to 1e-7 relative)"""
    r = json.load(open(path))
    for p, want in PINS[kind].items():
        try:
            got = get(r, p)
        except (KeyError, IndexError, TypeError) as e:
            raise AssertionError(f"{kind} {p}: not in the result ({e!r})")
        assert abs(got - want) <= 1e-7 * max(abs(want), 1.0), f"{kind} {p}: {got} against the pin {want}"


def command(kind, path, tmp):
    """the real-data command of a mutant on the module at `path` -> (returncode, text); fixture / potential / check also check the pinned numbers"""
    out = os.path.join(tmp, f"{kind}.json")
    args = dict(selftest=["selftest"], check=["check", "--n", "60", "--out", out], fixture=["fixture", "--split", "val", "--n", "20", "--out", out], potential=["potential", "--split", "val", "--seed", "0", "--n", "60", "--out", out])[kind]
    r = subprocess.run([sys.executable, path] + args, capture_output=True, text=True, env=ENV)
    if r.returncode == 0 and kind != "selftest":
        try:
            check_pins(kind, out)
        except AssertionError as e:
            return 1, str(e)
    return r.returncode, r.stdout[-800:] + r.stderr[-1500:]


def one(args):
    k, (name, a, b, *rest), tmp = args
    kind = rest[0] if rest else "selftest"
    src = open(os.path.join(DATA, MOD), encoding="utf-8").read()
    if src.count(a) != 1:
        return name, f"NOT APPLIED ({src.count(a)} matches)"
    d = os.path.join(tmp, str(k))
    os.makedirs(d)
    p = os.path.join(d, MOD)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src.replace(a, b))
    rc, _ = command(kind, p, d)
    return name, "killed" if rc != 0 else "SURVIVED"


def run_mutants(workers=4):
    left = []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(workers) as ex:
        for name, res in ex.map(one, [(k, m, tmp) for k, m in enumerate(MUTANTS)]):
            print(f"{res:<12} {name}", flush=True)
            if res != "killed":
                left.append(name)
    print(f"{len(MUTANTS) - len(left)} of {len(MUTANTS)} killed", flush=True)
    return left


if __name__ == "__main__":
    run([["selftest"], ["fixture", "--split", "val", "--n", "100"]])
    with tempfile.TemporaryDirectory() as tmp_:                                                              # the unmutated code passes the pinned commands
        for kind_ in ("check", "fixture", "potential"):
            rc_, txt_ = command(kind_, os.path.join(DATA, MOD), tmp_)
            assert rc_ == 0, f"{kind_} against its pin failed:\n{txt_}"
            print(f"{kind_} pins ok", flush=True)
        print(f"potential = its independent re-implementation on the cut map: {compare_reference(os.path.join(tmp_, 'potential.json'))} numbers to 1e-12", flush=True)
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_lane_map ok")
