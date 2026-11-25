"""Tests of the data of the SD-route association (`net/prep/lane_route_data.py`, E389).

    wsl bash tools/env/_py14.sh test_lane_route.py            # the selftest, `check` on 20 frames of the built VAL cache, the pinned `stats` of the VAL cache; then the textual mutants
    wsl bash tools/env/_py14.sh test_lane_route.py selftests  # the same without the mutants
    wsl bash tools/env/_py14.sh test_lane_route.py only "merge" "seven digits"   # those mutants alone (a part of the name)

A mutant = one textual change of the source (the route taken from the path instead of the tier, a wrong tier, successor order dropped, the band around the route too wide, too narrow, without the heading, exclusive, with the wrong lane count, the OR on_lane dropped, the road-id label by id instead of by the road's piece list, merges counted as forks, the 40 m bound exclusive, the pose error with the
transposed rotation, the farthest samples kept, a lane-level key let through, the wrong-route plan in the same scene, the draw counter not advanced or not shared, ...); the mutated copy runs in a temporary folder with BFLC_ROOT pointing at this project, two at a time,
and every mutant must make ITS COMMAND fail by an assertion: the selftest (default) or, for the mutants of the GT side (tier, road list, fork rule, duplicate cut) that the toy world cannot see on real frames, `check --n 20` (it recomputes the frames and compares them with the
cache). A mutant that does not apply counts as failed. The pins are the numbers of `stats --split val` on the route cache of the VAL frames of B' seed 0 (GT signature 9ad1d4ce30): a rebuilt cache or GT needs new pins.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "net", "prep")
MOD = "lane_route_data.py"
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")      # a shared box: one thread per test process

PINS = json.loads("""{
 "frames": 873,
 "frames_with_route": 873,
 "frames_without_route": 0,
 "decision_frames": 234,
 "scenes": 22,
 "scenes_with_fork": 14,
 "route_pieces_mean": 2.9942726231386025,
 "covered.nodes": 149976,
 "covered.on_road": 0.387688696858164,
 "covered.on_lane": 0.26456232997279566,
 "covered.on_road_id": 0.5439536992585481,
 "delivered.nodes": 125971,
 "delivered.on_road": 0.4004651864317978,
 "delivered.on_lane": 0.2699033904628843,
 "delivered.on_road_id": 0.5594144684093958,
 "delivered_on_decision_frames.nodes": 29980,
 "delivered_on_decision_frames.on_road": 0.38432288192128083,
 "delivered_on_decision_frames.on_lane": 0.2823215476984656,
 "delivered_on_decision_frames.on_road_id": 0.5253168779186124,
 "route_tokens_mean": 65.57273768613975,
 "route_tokens_max": 136,
 "frames_truncated": 0
}""")

MUTANTS = [
    ("the route is what the path follows, not the tier", '    return driving_order(g, [k for k, p in enumerate(g["pieces"]) if (np.asarray(p["tier"]) == 1).any()])', "    return viterbi_route(g)"),
    ("the route is tier 2", '(np.asarray(p["tier"]) == 1).any()])', '(np.asarray(p["tier"]) == 2).any()])'),
    ("a route piece needs all its points on tier 1", '(np.asarray(p["tier"]) == 1).any()])', '(np.asarray(p["tier"]) == 1).all()])'),
    ("the driving order ignores the successor links", "or list(left)\n", "or list(left)\n        free = list(left)\n"),
    ("the driving order is by the key only, last first", "k = min(free, key=lambda k_: (key[k_], k_))", "k = max(free, key=lambda k_: (key[k_], k_))"),
    ("on_road_id by the road id of the pieces, not by the road's piece list", 'return {p for r in g["roads"] if ids & set(r["pieces"]) for p in r["pieces"]}', 'return {p["id"] for p in g["pieces"] if p["road"] in {g["pieces"][k]["road"] for k in route}}'),
    ("on_road_id does not include on_lane", "for p in g[\"pieces\"]], bool) | lane_ok", "for p in g[\"pieces\"]], bool)"),
    ("on_road does not include on_lane", "return (road_band(polys, table[\"pos\"], table[\"dirv\"]) | on_lane).astype(np.int8)", "return (road_band(polys, table[\"pos\"], table[\"dirv\"])).astype(np.int8)"),
    ("on_road is the road-id label", "return (road_band(polys, table[\"pos\"], table[\"dirv\"]) | on_lane).astype(np.int8)", "return (id_ok[piece] | on_lane).astype(np.int8)"),
    ("on_lane is the road-id label", "    on_lane = lane_ok[piece]\n", "    on_lane = id_ok[piece]\n"),
    ("the band is 2 m per lane", "BAND_LANE, BAND_BASE, BAND_COS = 1.75, 0.5, 0.7 ", "BAND_LANE, BAND_BASE, BAND_COS = 2.0, 0.5, 0.7 "),
    ("the band has no base width", "BAND_LANE, BAND_BASE, BAND_COS = 1.75, 0.5, 0.7 ", "BAND_LANE, BAND_BASE, BAND_COS = 1.75, 0.0, 0.7 "),
    ("the band ignores the heading", "BAND_LANE, BAND_BASE, BAND_COS = 1.75, 0.5, 0.7 ", "BAND_LANE, BAND_BASE, BAND_COS = 1.75, 0.5, -1.0 "),
    ("the band heading cosine is 0.9", "BAND_LANE, BAND_BASE, BAND_COS = 1.75, 0.5, 0.7 ", "BAND_LANE, BAND_BASE, BAND_COS = 1.75, 0.5, 0.9 "),
    ("the band edge is exclusive", "(d <= BAND_LANE * cv[2][j] + BAND_BASE)", "(d < BAND_LANE * cv[2][j] + BAND_BASE)"),
    ("the band takes the lane count of the first route point", "BAND_LANE * cv[2][j] + BAND_BASE", "BAND_LANE * cv[2][0] + BAND_BASE"),
    ("the band takes the absolute heading cosine", "((np.asarray(dirv, float).reshape(-1, 2) * cv[1][j]).sum(1) >= BAND_COS)", "(np.abs((np.asarray(dirv, float).reshape(-1, 2) * cv[1][j]).sum(1)) >= BAND_COS)"),
    ("the band lane counts are not clipped", "C.append(np.clip(np.rint(np.interp(t, s_, np.asarray(cnt, float))), 1, COUNT_MAX))", "C.append(np.rint(np.interp(t, s_, np.asarray(cnt, float))))"),
    ("the band lane count of an unknown road is 0", "C.append(np.clip(np.rint(np.interp(t, s_, np.asarray(cnt, float))), 1, COUNT_MAX))", "C.append(np.clip(np.rint(np.interp(t, s_, np.asarray(cnt, float))), 0, COUNT_MAX))"),
    ("the band measures from nodes shifted by 1 m", "d, j = cv[0].query(pos)\n    return (d <=", "d, j = cv[0].query(pos + 1.0)\n    return (d <="),
    ("no route: the labels are 0", "on_road=-np.ones(n, np.int8), on_lane=-np.ones(n, np.int8), on_road_id=-np.ones(n, np.int8), polys=[])", "on_road=np.zeros(n, np.int8), on_lane=-np.ones(n, np.int8), on_road_id=-np.ones(n, np.int8), polys=[])"),
    ("no route: the road-id labels are 0", "on_road=-np.ones(n, np.int8), on_lane=-np.ones(n, np.int8), on_road_id=-np.ones(n, np.int8), polys=[])", "on_road=-np.ones(n, np.int8), on_lane=-np.ones(n, np.int8), on_road_id=np.zeros(n, np.int8), polys=[])"),
    ("a fork needs only one successor", 'len(g["pieces"][k]["succ"]) > 1 and 0.0 <=', 'len(g["pieces"][k]["succ"]) > 0 and 0.0 <='),
    ("a merge is a decision too", 'return any(len(g["pieces"][k]["succ"]) > 1 and 0.0 <= base[j + 1] - s_ego <= ahead for j, k in enumerate(route))',
     'return any((len(g["pieces"][k]["succ"]) > 1 or len(g["pieces"][k]["pred"]) > 1) and 0.0 <= base[j + 1] - s_ego <= ahead for j, k in enumerate(route))'),
    ("the 40 m bound is exclusive", "0.0 <= base[j + 1] - s_ego <= ahead", "0.0 <= base[j + 1] - s_ego < ahead"),
    ("a fork behind the ego counts", "0.0 <= base[j + 1] - s_ego <= ahead", "-1e9 <= base[j + 1] - s_ego <= ahead"),
    ("the ego is the first route vertex", "s_ego = off[int(np.argmin(np.linalg.norm(V, axis=1)))]", "s_ego = off[0]"),
    ("the ahead bound is 50 m", "AHEAD = 40.0 ", "AHEAD = 50.0 "),
    ("the samples are 1 m apart", "STEP = 2.0 ", "STEP = 1.0 "),
    ("the cap is 32", "MAX_SAMPLES = 160 ", "MAX_SAMPLES = 32 "),
    ("the cap is the old 64", "MAX_SAMPLES = 160 ", "MAX_SAMPLES = 64 "),
    ("the cap is 128", "MAX_SAMPLES = 160 ", "MAX_SAMPLES = 128 "),
    ("the pose error is applied with the transposed rotation", "poly = poly @ rot(yaw).T + shift", "poly = poly @ rot(yaw) + shift"),
    ("the shift is added before the rotation", "poly = poly @ rot(yaw).T + shift", "poly = (poly + shift) @ rot(yaw).T"),
    ("the yaw is not applied", "poly = poly @ rot(yaw).T + shift", "poly = poly + shift"),
    ("the yaw is in degrees", "return shift, np.deg2rad(rng.randn())", "return shift, rng.randn()"),
    ("the direction of the shift is drawn on half the circle", "    ang = rng.uniform(0, 2 * np.pi)\n    shift = rng.uniform(lo, hi)", "    ang = rng.uniform(0, np.pi)\n    shift = rng.uniform(lo, hi)"),
    ("the error draws the degradation before the pose error", "    shift, yaw = pose_error(rng, lo, hi)\n    return sample_route(SD.degrade_sd(dict(route=polys, roads=[]), rng)[\"route\"], shift, yaw)",
     "    d_ = SD.degrade_sd(dict(route=polys, roads=[]), rng)[\"route\"]\n    shift, yaw = pose_error(rng, lo, hi)\n    return sample_route(d_, shift, yaw)"),
    ("the token generator takes seven digits", "hexdigest()[:8]", "hexdigest()[:7]"),
    ("the joint of two pieces keeps both samples", "if P and np.linalg.norm(q[0] - P[-1][-1]) < 0.5:", "if False:"),
    ("the truncation keeps the farthest samples", 'kind="stable")[:n])', 'kind="stable")[-n:])'),
    ("the kept samples are not in their own order", 'keep = np.sort(np.argsort(np.linalg.norm(pos, axis=1), kind="stable")[:n])', 'keep = np.argsort(np.linalg.norm(pos, axis=1), kind="stable")[:n]'),
    ("the counts are not clipped", "c = np.where(c > 0, np.clip(c, 1, COUNT_MAX), 0)", "c = c"),
    ("the leak check lets a lane-level key through", "assert not extra and not missing,", "assert not missing,"),
    ("the leak check lets a missing key through", "assert not extra and not missing,", "assert not extra,"),
    ("collate does not check the keys", "    for t in tables:\n        check_keys(t)", "    for t in tables:\n        pass"),
    ("the padding is valid", "valid=np.zeros((len(tables), M), bool))", "valid=np.ones((len(tables), M), bool))"),
    ("an all-absent batch has no padding column", 'M = max(1, max(len(t["valid"]) for t in tables))', 'M = max(0, max(len(t["valid"]) for t in tables))'),
    ("the exact class is 20 %", 'MIX = (("exact", 0.25), ("error", 0.47),', 'MIX = (("exact", 0.20), ("error", 0.52),'),
    ("the wrong plan stays in the scene", "if scene[j % n] != scene[i] and ok[j % n]:", "if ok[j % n]:"),
    ("the wrong plan takes a frame without a route", "if scene[j % n] != scene[i] and ok[j % n]:", "if scene[j % n] != scene[i]:"),
    ("the wrong plan is 96", "OTHER = 97 ", "OTHER = 96 "),
    ("wrong is the frame's own error route", "return error_route(self.polys(int(self.plan[i])), self.tokens[self.plan[i]])", "return error_route(self.polys(i), self.tokens[i])"),
    ("gross is 4 m", "GROSS_M = 5.0 ", "GROSS_M = 4.0 "),
    ("gross shifts every frame the same way", 'ang = token_rng(tok, "|gross").uniform(0, 2 * np.pi)', "ang = 0.0"),
    ("the none mode keeps the route", '        if mode == "none":\n            return EMPTY', '        if mode == "none":\n            return self.raw_route(i, "error")'),
    ("the error route is the same for every token", "return error_like(polys, token_rng(token), *ERROR_SHIFT)", 'return error_like(polys, token_rng("x"), *ERROR_SHIFT)'),
    ("the error shift is 0 - 2 m", "ERROR_SHIFT, TRAIN_SHIFT, GROSS_SHIFT = (1.0, 2.0), (0.0, 2.0), (4.0, 8.0)", "ERROR_SHIFT, TRAIN_SHIFT, GROSS_SHIFT = (0.0, 2.0), (0.0, 2.0), (4.0, 8.0)"),
    ("the train draws use the error shift", "raw = error_like(polys, rng, *TRAIN_SHIFT)", "raw = error_like(polys, rng, *ERROR_SHIFT)"),
    ("the gross draw uses the train shift", "return error_like(polys, rng, *GROSS_SHIFT)", "return error_like(polys, rng, *TRAIN_SHIFT)"),
    ("the draw counter does not advance", "self.draws[i] = n + 1", "self.draws[i] = n"),
    ("the draw seed ignores the draw index", "zlib.crc32(self.tokens[i].encode()), n])", "zlib.crc32(self.tokens[i].encode()), 0])"),
    ("the draw seed is the frame index, not the token", "zlib.crc32(self.tokens[i].encode()), n])", "i, n])"),
    ("the draw seed ignores the init seed", "RandomState([self.init_seed % 2 ** 32, zlib", "RandomState([0, zlib"),
    ("the counters are not in shared memory", ".share_memory_() if mode == \"train\"", " if mode == \"train\""),
    ("the token label is the label of the next node", "t[got] = lab[np.asarray(gt_id)[got]] if len(lab) else t[got]", "t[got] = lab[(np.asarray(gt_id)[got] + 1) % len(lab)] if len(lab) else t[got]"),
    ("a token without a node gets a label", "t = -np.ones(len(gt_id), np.int8)", "t = np.zeros(len(gt_id), np.int8)"),
    ("the loader item loses the route key", '        if self.mode is not None:\n            out["route"] = self.table(i)', '        if False:\n            out["route"] = self.table(i)'),
    ("the TEST guard accepts a foreign row", "assert set(int(r) for r in rows) <= set(int(r) for r in sp) and len(rows) == len(sp),", "assert len(rows) == len(sp),"),
    ("the TEST guard accepts a missing row", "assert set(int(r) for r in rows) <= set(int(r) for r in sp) and len(rows) == len(sp),", "assert set(int(r) for r in rows) <= set(int(r) for r in sp),"),
    ("the cache reader accepts another version", 'assert self.manifest["version"] == VERSION,', "assert True or self.manifest[\"version\"] == VERSION,"),
    ("the cache reader accepts another lane GT", 'assert lane_gt is None or self.manifest["lane_gt"] == lane_gt,', 'assert True or self.manifest["lane_gt"] == lane_gt,'),
    ("the cache reader accepts other constants", 'assert self.manifest["params"] == cache_params(),', 'assert True or self.manifest["params"] == cache_params(),'),
    ("the dataset does not hand the lane GT to the reader", "TokenRoute(rdir or route_dir(split), token_lane_gt(tok))", "TokenRoute(rdir or route_dir(split))"),
    ("the cache reader accepts another frame list", 'assert tokens_sha(self.tokens) == self.manifest["tokens_sha"] and', "assert True and"),
    ("the dataset accepts another frame list", 'assert list(self.tok.tokens) == self.rc.tokens,', "assert True or list(self.tok.tokens) == self.rc.tokens,"),
    ("the duplicate cut tolerance is 3 m", "DUP_D, DUP_COS = 1.0, 0.9 ", "DUP_D, DUP_COS = 3.0, 0.9 "),
    ("the duplicate cut ignores the heading", "DUP_D, DUP_COS = 1.0, 0.9 ", "DUP_D, DUP_COS = 1.0, -1.0 "),
    # the defaults of the loader on the real caches (`loadercheck`)
    ("TRAIN is not shuffled", 'shuffle = split == "train" if shuffle is None else shuffle', 'shuffle = False if shuffle is None else shuffle', "loadercheck"),
    ("VAL is shuffled", 'shuffle = split == "train" if shuffle is None else shuffle', 'shuffle = True if shuffle is None else shuffle', "loadercheck"),
    ("TRAIN keeps the last short batch", "drop_last=shuffle", "drop_last=False", "loadercheck"),
    ("the auto mode map is swapped", '("train" if split == "train" else "error") if mode == "auto"', '("error" if split == "train" else "train") if mode == "auto"', "loadercheck"),
    ("the auto mode of VAL is the exact route", '("train" if split == "train" else "error") if mode == "auto"', '("train" if split == "train" else "exact") if mode == "auto"', "loadercheck"),
    # the GT side on real frames: `check` recomputes frames and compares them with the cache
    ("real frames: the route is tier 2", '(np.asarray(p["tier"]) == 1).any()])', '(np.asarray(p["tier"]) == 2).any()])', "check"),
    ("real frames: the band base is 1 m", "BAND_LANE, BAND_BASE, BAND_COS = 1.75, 0.5, 0.7 ", "BAND_LANE, BAND_BASE, BAND_COS = 1.75, 1.0, 0.7 ", "check"),
    ("real frames: the road-id label by the road id", 'return {p for r in g["roads"] if ids & set(r["pieces"]) for p in r["pieces"]}', 'return {p["id"] for p in g["pieces"] if p["road"] in {g["pieces"][k]["road"] for k in route}}', "check"),
    ("real frames: a fork needs three successors", 'len(g["pieces"][k]["succ"]) > 1 and 0.0 <=', 'len(g["pieces"][k]["succ"]) > 2 and 0.0 <=', "check"),
    ("real frames: the route polylines are not cut", "return dedupe_polys(SD.build_sd(", "return (lambda x: x)(SD.build_sd(", "check"),
]


def run(args):
    """selftest, check (20 frames) and stats of the module in `DATA` itself: the unmutated code must pass"""
    for a in args:
        r = subprocess.run([sys.executable, os.path.join(DATA, MOD)] + a, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{MOD} {' '.join(a)} failed:\n{r.stdout[-1200:]}{r.stderr[-1500:]}"
        print(r.stdout.strip().splitlines()[-1], flush=True)


def get(r, path):
    """the number at a dotted path of a result dict"""
    for k in path.split("."):
        r = r[k]
    return r


def check_pins(path):
    """the numbers of `stats --split val` against the pins (integers exactly, floats to 1e-7 relative)"""
    r = json.load(open(path))
    for p, want in PINS.items():
        got = get(r, p)
        assert abs(got - want) <= 1e-7 * max(abs(want), 1.0), f"stats {p}: {got} against the pin {want}"


def command(kind, path, tmp):
    """the command of a mutant on the module at `path` -> (returncode, text)"""
    out = os.path.join(tmp, f"{kind}.json")
    args = dict(selftest=["selftest"], loadercheck=["loadercheck"], check=["check", "--n", "20", "--out", out])[kind]
    r = subprocess.run([sys.executable, path] + args, capture_output=True, text=True, env=ENV)
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
    rc, txt = command(kind, p, d)
    return name, "killed" if rc != 0 and "AssertionError" in txt else ("CRASHED (no assertion)" if rc != 0 else "SURVIVED")


def run_mutants(workers=2, only=()):
    """all mutants, or those whose name contains one of the strings of `only`"""
    todo = [m for m in MUTANTS if not only or any(o in m[0] for o in only)]
    left = []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(workers) as ex:
        for name, res in ex.map(one, [(k, m, tmp) for k, m in enumerate(todo)]):
            print(f"{res:<24} {name}", flush=True)
            if res != "killed":
                left.append(name)
    print(f"{len(todo) - len(left)} of {len(todo)} killed", flush=True)
    return left


if __name__ == "__main__":
    if sys.argv[1:2] == ["only"]:                                                                            # python test_lane_route.py only "<part of a mutant name>" ...: those mutants alone
        left = run_mutants(only=sys.argv[2:])
        assert not left, f"mutants survived: {left}"
        sys.exit(0)
    run([["selftest"], ["check", "--n", "20"], ["loadercheck"]])
    with tempfile.TemporaryDirectory() as tmp_:                                                              # the unmutated code passes the pinned numbers
        out_ = os.path.join(tmp_, "stats.json")
        run([["stats", "--split", "val", "--out", out_]])
        check_pins(out_)
        print("stats pins ok", flush=True)
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_lane_route ok")
