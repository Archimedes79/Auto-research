"""Tests of the data of the cross-attention to the past keyframe (`data/lane_past_data.py`, E368).

    wsl bash tools/env/_py14.sh test_lane_past.py            # the selftest, `check` on the built cache directories, the fixture on 40 real VAL pairs, the loader fixture, the pinned numbers of the fixture (20 pairs) and of `potential` (60 frames); then the hand mutants
    wsl bash tools/env/_py14.sh test_lane_past.py selftests  # the same without the mutants (about 5 minutes)

A mutant = one textual change of the source (the pose of the wrong pair, the row looked up by the cache index, the TEST guard removed, a chain across a scene start, the wrong warp, a tolerance loosened, a role of the weak tokens mixed up, ...); the mutated copy runs in a
temporary folder with BFLC_ROOT pointing at this project, four at a time, and every mutant must make ITS COMMAND fail: the selftest (default), or a real-data command named in the mutant's 4th field: `fixture` (20 pairs: exit code and the pinned numbers), `potential` (60 frames, the pinned
numbers), `loader` (the loader fixture). The pins are numbers of the VAL cache of B' seed 0 (GT signature 9ad1d4ce30); a rebuilt cache needs new pins.
"""
import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "training", "data")
MOD = "lane_past_data.py"
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")      # a shared box: one thread per test process

FIXTURE_PIN = dict(points=dict(n=19835, p90=0.028297071, p99=0.147805338), headings=dict(n=19835, p90=2.600123428), tokens=dict(n=2864, p90=1.058689265))           # `fixture --n-pairs 20`, the right pose
POTENTIAL_PIN = dict(frames_with_past=56, gt_nodes=9871, found=0.8645527302198359, missed_weak_near=0.7479431563201197, missed_delivered_past_near=0.2505609573672401, duplicates_of_the_true_weak=0.7845205257179945)       # `potential --n 60`
POTENTIAL_PIN_NESTED = {("weak", "true"): 6163, ("weak", "false"): 9022, ("weak", "true_supported"): 0.7045270160636053, ("delivered", "true"): 8075, ("roles", "novel", "n"): 1328, ("roles", "novel", "pose"): 0.34036144578313254, ("roles", "dup", "pose"): 0.8045501551189245,
                        ("roles", "false", "pose"): 0.14398137885169585, ("roles", "novel", "random"): 0.12424698795180723, ("recall_ceiling", "all"): 0.03069597811771857, ("recall_ceiling", "topology"): 0.006078411508459123}
MUTANTS = [
    ("the pose is the inverse (current and previous swapped)", """LS.rel_pose(pose["R"][rows[i]], pose["t"][rows[i]], pose["R"][pr], pose["t"][pr])""", """LS.rel_pose(pose["R"][pr], pose["t"][pr], pose["R"][rows[i]], pose["t"][rows[i]])"""),
    ("the pose row is looked up by the cache index", """LS.rel_pose(pose["R"][rows[i]], pose["t"][rows[i]], pose["R"][pr], pose["t"][pr])""", """LS.rel_pose(pose["R"][i], pose["t"][i], pose["R"][pr], pose["t"][pr])"""),
    ("prev holds the poses row, not the cache index", "prev[i] = idx_of[int(pr)]", "prev[i] = int(pr)"),
    ("a previous row outside the cache is accepted", """assert int(pr) in idx_of, f"frame {i} (row {rows[i]}): the previous keyframe (row {pr}) is not in the cache\"""", "pass"),
    ("the VAL guard accepts any rows", """assert [int(r) for r in rows] == sv,""", "assert True or [int(r) for r in rows] == sv,"),
    ("the VAL guard ignores the order", "assert [int(r) for r in rows] == sv,", "assert sorted(int(r) for r in rows) == sorted(sv),"),
    ("a row twice is accepted", """assert len(set(rows)) == len(rows), "a row twice in the cache\"""", "pass"),
    ("the stream order is not checked", """assert [int(r) for r, _ in so] == [int(r) for r in rows],""", """assert True or [int(r) for r, _ in so] == [int(r) for r in rows],"""),
    ("the scene starts are not checked", "assert [bool(x) for _, x in so] == [bool(p < 0) for p in prev],", "assert True or [bool(x) for _, x in so] == [bool(p < 0) for p in prev],"),
    ("the previous keyframe need not be the row before", "assert (prev[has] == (np.arange(len(prev)) - 1)[has]).all(),", "assert True or (prev[has] == (np.arange(len(prev)) - 1)[has]).all(),"),
    ("past.json carries a time stamp (not idempotent)", "info = dict(format=FORMAT, split=split,", """info = dict(format=FORMAT, built=__import__("time").time(), split=split,"""),
    ("a temporary file stays", "    os.replace(tmp, path)\n", """    os.replace(tmp, path)\n    open(tmp, "w").close()\n"""),
    ("a changed cache is accepted", """info["frames_sha"] == _sha(os.path.join(d, "frames.json"))""", "True"),
    ("a chain may start at a frame without a past, the next ones need none", "has[s + 1: s + K_seq].all()", "has[s: s + K_seq].all()"),
    ("the last chain is dropped", "range(len(prev) - K_seq + 1)", "range(len(prev) - K_seq)"),
    ("the scene ids start at 1", "np.cumsum(np.asarray(prev) < 0) - 1", "np.cumsum(np.asarray(prev) < 0)"),
    ("a stream item holds K_seq frames", "self.K_seq = 1 if self.stream else int(K_seq)", "self.K_seq = int(K_seq)"),
    ("has_prev is prev > 0", "has_prev=self.prev[ix] >= 0", "has_prev=self.prev[ix] > 0"),
    ("the poses of a chain are in reverse order", "rel_t=self.rel_t[ix], has_prev", "rel_t=self.rel_t[ix[::-1]], has_prev"),
    ("the frames of a batch are in reverse time order", """[LP.collate([o["frames"][k] for o in b]) for k in range(n)]""", """[LP.collate([o["frames"][k] for o in b]) for k in reversed(range(n))]"""),
    ("the warp uses R instead of R^T", "return P @ R.T + t\n    if how == \"identity\"", "return P @ R + t\n    if how == \"identity\""),
    ("the warp forgets the translation", "return P @ R.T + t\n    if how == \"identity\"", "return P @ R.T\n    if how == \"identity\""),
    ("the inverse pose is the identity", "return (P - t) @ R", "return P"),
    ("the box is 450 m", "BOX, NEAR = 45.0, 6.0", "BOX, NEAR = 450.0, 6.0"),
    ("the neighbour radius is 60 m", "BOX, NEAR = 45.0, 6.0", "BOX, NEAR = 45.0, 60.0"),
    ("the p90 tolerance is 1 m", "TOL = dict(p90=0.10, p99=0.25,", "TOL = dict(p90=1.0, p99=0.25,"),
    ("the p99 tolerance is 2.5 m", "p99=0.25, head_p90_deg", "p99=2.5, head_p90_deg"),
    ("the heading angle is in radians", "np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))", "np.arccos(np.clip(cs, -1.0, 1.0))"),
    ("the tangents are not normalised", "return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)", "return g"),
    ("the percentile p90 is p80", "p90=float(np.percentile(x, 90))", "p90=float(np.percentile(x, 80))"),
    ("the shifted pair is not checked for a past", "(shift == 0 or prev[prev[i]] >= 0)", "(shift == 0 or True)"),
    ("support ignores the heading", "(D @ B.T >= COS)", "(D @ B.T >= -1)"),
    ("near ignores the heading", "(D @ gdir.T >= COS)", "(D @ gdir.T >= -1)"),
    ("the near radius is exclusive", "(np.linalg.norm(P[:, None] - gpos[None], axis=2) <= RAD)", "(np.linalg.norm(P[:, None] - gpos[None], axis=2) < RAD)"),
    ("the delivered threshold is exclusive", "deliv = valid & (ep >= 0.5)", "deliv = valid & (ep > 0.5)"),
    ("a padding token is a weak token", "weak = valid & ~deliv", "weak = ~deliv"),
    ("novel = near any GT node", "novel = (near(sel) & miss[None]).any(1) if M else", "novel = near(sel).any(1) if M else"),
    ("found = a weak token is near", "found = near(deliv).any(0) if deliv.any() else", "found = near(valid).any(0) if deliv.any() else"),
    ("the bin edges are (lo, hi]", "m = (W[:, 0] >= lo) & (W[:, 0] < hi)", "m = (W[:, 0] > lo) & (W[:, 0] <= hi)"),
    ("the TEST guard of build is skipped", "    if check:\n        check_rows(split, rows)\n", "    if False:\n        check_rows(split, rows)\n"),
    ("pose_of reads the previous frame's rotation", 'return ext["rel_R"][i], ext["rel_t"][i]', 'return ext["rel_R"][i - 1], ext["rel_t"][i]'),
    ("pose_of reads the previous frame's translation", 'return ext["rel_R"][i], ext["rel_t"][i]', 'return ext["rel_R"][i], ext["rel_t"][i - 1]'),
    ("the temporary file of an atomic write has a double dot", 'tmp = f"{root}.{os.getpid()}.tmp{ext}"', 'tmp = f"{root}..{os.getpid()}.tmp{ext}"'),
    ("the fixture reads the pose of the previous frame", "R, t = pose_of(ext, ic)\n            (P0, T0), (P1, T1) = gt(ip), gt(ic)", "R, t = pose_of(ext, ip)\n            (P0, T0), (P1, T1) = gt(ip), gt(ic)", "fixture"),
    ("potential reads the pose of the previous frame", "R, tv = (a_.astype(float) for a_ in pose_of(ext, i))", "R, tv = (a_.astype(float) for a_ in pose_of(ext, ip))", "potential"),
    ("potential's random null reads the frame itself", "j = int(rng.integers(len(c)))", "j = int(i)", "potential"),
    ("the loader fixture gets the pose entry of the previous item", 'R, t = bc["rel_R"][:, 0], bc["rel_t"][:, 0]', 'R, t = bp["rel_R"][:, 0], bp["rel_t"][:, 0]', "loader"),
    ("the loader fixture moves the current frame's tokens", "JP.make_past(dict(x=torch.zeros(1, K, JP.D), w=torch.zeros(1, K)), fp, fc, R, t,", "JP.make_past(dict(x=torch.zeros(1, K, JP.D), w=torch.zeros(1, K)), fc, fp, R, t,", "loader"),
    ("the loader fixture does not check the pose entry", "perr[how] = max(perr[how], float(np.abs(R[0].numpy() - Rf).max()), float(np.abs(t[0].numpy() - tf).max()))", "perr[how] = 0.0", "loader"),
]


def run(args):
    """selftest, check, fixture (40 pairs) of the module in `DATA` itself: the unmutated code must pass"""
    for a in args:
        r = subprocess.run([sys.executable, os.path.join(DATA, MOD)] + a, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{MOD} {' '.join(a)} failed:\n{r.stdout[-1200:]}{r.stderr[-1500:]}"
        print(r.stdout.strip().splitlines()[-1], flush=True)


def check_pins(kind, path):
    """the numbers of a real-data command against the pins (integers exactly, floats to 1e-7 relative)"""
    import json
    r = json.load(open(path))
    if kind == "fixture":
        got = dict(points=r["points"]["warp"], headings=r["headings"]["warp"], tokens=r["tokens"]["warp"])
        for grp, want in FIXTURE_PIN.items():
            for k, v in want.items():
                assert abs(got[grp][k] - v) <= 1e-7 * max(abs(v), 1.0), f"fixture {grp} {k}: {got[grp][k]} against the pin {v}"
    else:
        for k, v in POTENTIAL_PIN.items():
            assert abs(r[k] - v) <= 1e-7 * max(abs(v), 1.0), f"potential {k}: {r[k]} against the pin {v}"
        for key, v in POTENTIAL_PIN_NESTED.items():
            x = r
            for part in key:
                x = x[part]
            assert abs(x - v) <= 1e-7 * max(abs(v), 1.0), f"potential {key}: {x} against the pin {v}"


def command(kind, path, tmp):
    """the real-data command of a mutant on the module at `path` -> (returncode, text); fixture / potential also check the pinned numbers"""
    out = os.path.join(tmp, f"{kind}.json")
    args = dict(selftest=["selftest"], check=["check"], fixture=["fixture", "--split", "val", "--n-pairs", "20", "--out", out], potential=["potential", "--split", "val", "--seed", "0", "--n", "60", "--out", out],
                loader=["loader-fixture", "--split", "val", "--n-pairs", "20"])[kind]
    r = subprocess.run([sys.executable, path] + args, capture_output=True, text=True, env=ENV)
    if r.returncode == 0 and kind in ("fixture", "potential"):
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
        return name, "NOT APPLIED"
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
    run([["selftest"], ["check"], ["fixture", "--split", "val", "--n-pairs", "40"], ["loader-fixture", "--split", "val", "--n-pairs", "40"]])
    with tempfile.TemporaryDirectory() as tmp_:                                                              # the unmutated code passes the pinned commands
        for kind_ in ("fixture", "potential"):
            rc_, txt_ = command(kind_, os.path.join(DATA, MOD), tmp_)
            assert rc_ == 0, f"{kind_} against its pin failed:\n{txt_}"
            print(f"{kind_} pins ok", flush=True)
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_lane_past ok")
