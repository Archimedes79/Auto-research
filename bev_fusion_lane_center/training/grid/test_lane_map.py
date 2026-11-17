"""Tests of the data of the cross-attention to the SD map (`data/lane_map_data.py`, E374).

    wsl bash tools/env/_py14.sh test_lane_map.py            # the selftest, `check` on 60 frames of VAL and TRAIN, the fixture on 100 real VAL frames, the pinned numbers of `check` (60 frames), of the fixture (20 frames) and of `potential` (60 frames); then the hand mutants
    wsl bash tools/env/_py14.sh test_lane_map.py selftests  # the same without the mutants

A mutant = one textual change of the source (the pose error applied with the transposed rotation, the translation before the rotation, a wrong cap or the farthest samples kept, a lane-level key let through, the S plan in the same scene, the draw counter not advanced, a class boundary moved, a threshold of the supported shares
changed, ...); the mutated copy runs in a temporary folder with BFLC_ROOT pointing at this project, four at a time, and every mutant must make ITS COMMAND fail: the selftest (default), or a real-data command named in the mutant's 4th field: `fixture` (20 frames: exit code and the pinned numbers), `potential`
(60 frames, the pinned numbers), `check` (60 frames of VAL and TRAIN, the pinned numbers). The pins are numbers of the caches of B' seed 0 (VAL and TRAIN, GT signature 9ad1d4ce30) and of the lane GT store of this checkout; a rebuilt cache or GT needs new pins.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "training", "data")
MOD = "lane_map_data.py"
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")      # a shared box: one thread per test process

PINS = json.loads("""{
 "fixture": {
  "shift_min": 1.0116577526632158,
  "shift_max": 1.9569271212690489,
  "yaw_std_deg": 1.1353025598341813,
  "exact_samples": 4513,
  "own_share_error": 0.9874786302382115,
  "own_share_S": 0.47063583916169555,
  "undo.undo.n": 4150,
  "undo.undo.median": 0.1479181029014856,
  "undo.undo.p90": 0.7758979807197818,
  "undo.undo.p99": 1.4202811085541491,
  "undo.identity.n": 4150,
  "undo.identity.median": 1.0068370881652555,
  "undo.identity.p90": 2.0890423399177513,
  "undo.identity.p99": 3.0604590724266494,
  "undo.shift_sign.n": 4150,
  "undo.shift_sign.median": 1.5149172890339035,
  "undo.shift_sign.p90": 3.3561501392987028,
  "undo.shift_sign.p99": 4.287327721695744,
  "undo.rotation_transposed.n": 4150,
  "undo.rotation_transposed.median": 0.7017890718000681,
  "undo.rotation_transposed.p90": 2.079446319537164,
  "undo.rotation_transposed.p99": 3.559274891068083
 },
 "potential": {
  "samples_per_frame": 235.65,
  "classes.novel.n": 1412,
  "classes.novel.supported_exact": 0.9348441926345609,
  "classes.novel.supported_error": 0.8781869688385269,
  "classes.novel.supported_other_scene": 0.2046742209631728,
  "classes.novel.median_dist_exact": 0.9180839711628174,
  "classes.novel.median_dist_error": 1.2334108120718796,
  "classes.dup.n": 5222,
  "classes.dup.supported_exact": 0.952508617387974,
  "classes.dup.supported_error": 0.8240137878207583,
  "classes.dup.supported_other_scene": 0.3320566832631176,
  "classes.dup.median_dist_exact": 1.1640799555520895,
  "classes.dup.median_dist_error": 1.4983145291666526,
  "classes.false.n": 9594,
  "classes.false.supported_exact": 0.6096518657494268,
  "classes.false.supported_error": 0.5337711069418386,
  "classes.false.supported_other_scene": 0.22597456743798208,
  "classes.false.median_dist_exact": 1.9352248397034684,
  "classes.false.median_dist_error": 2.1335617452180093,
  "classes.deliv_true.n": 8891,
  "classes.deliv_true.supported_exact": 0.9637836013946688,
  "classes.deliv_true.supported_error": 0.8171184343718367,
  "classes.deliv_true.supported_other_scene": 0.32819705319986503,
  "classes.deliv_true.median_dist_exact": 1.1372565928331393,
  "classes.deliv_true.median_dist_error": 1.503201730932145,
  "classes.deliv_false.n": 1092,
  "classes.deliv_false.supported_exact": 0.7793040293040293,
  "classes.deliv_false.supported_error": 0.6639194139194139,
  "classes.deliv_false.supported_other_scene": 0.21611721611721613,
  "classes.deliv_false.median_dist_exact": 1.6415757368867372,
  "classes.deliv_false.median_dist_error": 1.7660653252032925,
  "by_bin.0.n": 3983,
  "by_bin.0.true_share": 0.3341702234496611,
  "by_bin.0.supported_if_true": 0.7791134485349361,
  "by_bin.0.supported_if_false": 0.4242081447963801,
  "by_bin.0.novel_share_supported": 0.0393154486586494,
  "by_bin.0.novel_share_unsupported": 0.006589785831960461,
  "by_bin.1.n": 5246,
  "by_bin.1.true_share": 0.3966831871902402,
  "by_bin.1.supported_if_true": 0.8438250840941854,
  "by_bin.1.supported_if_false": 0.5636650868878357,
  "by_bin.1.novel_share_supported": 0.05621468926553672,
  "by_bin.1.novel_share_unsupported": 0.0123094958968347,
  "by_bin.2.n": 2900,
  "by_bin.2.true_share": 0.4113793103448276,
  "by_bin.2.supported_if_true": 0.8642078792958927,
  "by_bin.2.supported_if_false": 0.5729349736379613,
  "by_bin.2.novel_share_supported": 0.11896465903434544,
  "by_bin.2.novel_share_unsupported": 0.03591470258136925,
  "by_bin.3.n": 2262,
  "by_bin.3.true_share": 0.47126436781609193,
  "by_bin.3.supported_if_true": 0.8452157598499062,
  "by_bin.3.supported_if_false": 0.5994983277591973,
  "by_bin.3.novel_share_supported": 0.18788627935723115,
  "by_bin.3.novel_share_unsupported": 0.06987577639751552,
  "by_bin.4.n": 1837,
  "by_bin.4.true_share": 0.5242242787152966,
  "by_bin.4.supported_if_true": 0.8494288681204569,
  "by_bin.4.supported_if_false": 0.5915331807780321,
  "by_bin.4.novel_share_supported": 0.3093632958801498,
  "by_bin.4.novel_share_unsupported": 0.12350597609561753
 },
 "check": {
  "val.frames": 873,
  "val.samples_with_count_above_max": 0,
  "val.sizes.exact.frames": 60,
  "val.sizes.exact.mean": 235.65,
  "val.sizes.exact.max": 454,
  "val.sizes.exact.truncated": 0,
  "val.sizes.error.frames": 60,
  "val.sizes.error.mean": 214.18333333333334,
  "val.sizes.error.max": 432,
  "val.sizes.error.truncated": 0,
  "train.frames": 8414,
  "train.samples_with_count_above_max": 0,
  "train.sizes.exact.frames": 60,
  "train.sizes.exact.mean": 296.45,
  "train.sizes.exact.max": 607,
  "train.sizes.exact.truncated": 3,
  "train.sizes.error.frames": 60,
  "train.sizes.error.mean": 268.15,
  "train.sizes.error.max": 573,
  "train.sizes.error.truncated": 2
 }
}""")

MUTANTS = [
    ("the pose error is applied with the transposed rotation", "poly = poly @ rot(yaw).T + shift", "poly = poly @ rot(yaw) + shift"),
    ("the shift is added before the rotation", "poly = poly @ rot(yaw).T + shift", "poly = (poly + shift) @ rot(yaw).T"),
    ("the yaw is not applied", "poly = poly @ rot(yaw).T + shift", "poly = poly + shift"),
    ("the direction of the shift is drawn on half the circle", "    ang = rng.uniform(0, 2 * np.pi)\n    shift = rng.uniform(lo, hi)", "    ang = rng.uniform(0, np.pi)\n    shift = rng.uniform(lo, hi)"),
    ("the yaw is in degrees", "return shift, np.deg2rad(rng.randn())", "return shift, rng.randn()"),
    ("the token generator takes seven digits", "hexdigest()[:8]", "hexdigest()[:7]"),
    ("the exact map is degraded", "def exact_map(pts):\n    return sample_roads(roads_of(pts))", "def exact_map(pts):\n    return sample_roads(SD.degrade_sd(roads_of(pts), np.random.RandomState(0), drop=DROP))"),
    ("the samples are 1 m apart", "STEP = 2.0 ", "STEP = 1.0 "),
    ("no side road is dropped", "DROP = 0.1 ", "DROP = 0.0 "),
    ("the cap is 600", "MAX_SAMPLES = 512 ", "MAX_SAMPLES = 600 "),
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
    ("the train draws use the error shift", "raw = error_like(self.pts(i), rng, *TRAIN_SHIFT)", "raw = error_like(self.pts(i), rng, *ERROR_SHIFT)"),
    ("the gross draw uses the train shift", "return error_like(self.pts(i), rng, *GROSS_SHIFT)", "return error_like(self.pts(i), rng, *TRAIN_SHIFT)"),
    ("the gross draw may take a frame of the same scene", "        while self.scene[j] == self.scene[i]:", "        while False:"),
    ("the draw counter does not advance", "self.draws[i] = n + 1", "self.draws[i] = n"),
    ("the draw seed ignores the draw index", "zlib.crc32(self.tokens[i].encode()), n])", "zlib.crc32(self.tokens[i].encode()), 0])"),
    ("the draw seed ignores the init seed", "RandomState([self.init_seed % 2 ** 32, zlib", "RandomState([0, zlib"),
    ("the draw seed is the frame index, not the token", "zlib.crc32(self.tokens[i].encode()), n])", "i, n])"),
    ("unknown counts keep their values", 'return with_count(raw, 0) if cls == "unknown" else raw', "return raw"),
    ("an absent draw returns the exact map", '        if cls == "absent":\n            return EMPTY', '        if cls == "absent":\n            return exact_map(self.pts(i))'),
    ("the counters are not in shared memory", '.share_memory_() if mode == "train"', ' if mode == "train"'),
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


def get(r, path):
    """the number at a dotted path of a result dict (list entries by their index)"""
    for k in path.split("."):
        r = r[int(k)] if isinstance(r, list) else r[k]
    return r


def check_pins(kind, path):
    """the numbers of a real-data command against the pins (dotted paths of its JSON; integers exactly, floats to 1e-7 relative)"""
    r = json.load(open(path))
    for p, want in PINS[kind].items():
        got = get(r, p)
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
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_lane_map ok")
