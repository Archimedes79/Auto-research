p = "training/grid/test_lane_past.py"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep('''    wsl bash tools/env/_py14.sh test_lane_past.py            # the selftest, the checks on the built cache directories and the fixture on 40 real VAL pairs, then the hand mutants: every mutant must make the selftest FAIL
    wsl bash tools/env/_py14.sh test_lane_past.py selftests  # the selftest, `check` and the fixture only (about 3 minutes)

A mutant = one textual change of the source (the pose of the wrong pair, the row looked up by the cache index, the TEST guard removed, a chain across a scene start, the wrong warp, a tolerance loosened, a role of the weak tokens
mixed up, ...); the mutated copy runs in a temporary folder with BFLC_ROOT pointing at this project, four at a time.''', '''    wsl bash tools/env/_py14.sh test_lane_past.py            # the selftest, `check` on the built cache directories, the fixture on 40 real VAL pairs, the loader fixture, the pinned numbers of the fixture (20 pairs) and of `potential` (60 frames); then the hand mutants
    wsl bash tools/env/_py14.sh test_lane_past.py selftests  # the same without the mutants (about 5 minutes)

A mutant = one textual change of the source (the pose of the wrong pair, the row looked up by the cache index, the TEST guard removed, a chain across a scene start, the wrong warp, a tolerance loosened, a role of the weak tokens mixed up, ...); the mutated copy runs in a
temporary folder with BFLC_ROOT pointing at this project, four at a time, and every mutant must make ITS COMMAND fail: the selftest (default), or a real-data command named in the mutant's 4th field: `fixture` (20 pairs: exit code and the pinned numbers), `potential` (60 frames, the pinned
numbers), `loader` (the loader fixture). The pins are numbers of the VAL cache of B' seed 0 (GT signature 9ad1d4ce30); a rebuilt cache needs new pins.''')
rep('''MUTANTS = [''', '''FIXTURE_PIN = dict(points=dict(n=19835, p90=0.028297071, p99=0.147805338), headings=dict(n=19835, p90=2.600123428), tokens=dict(n=2864, p90=1.058689265))           # `fixture --n-pairs 20`, the right pose
POTENTIAL_PIN = dict(frames_with_past=56, gt_nodes=9871, found=0.8645527302198359, missed_weak_near=0.7479431563201197, missed_delivered_past_near=0.2505609573672401, duplicates_of_the_true_weak=0.7845205257179945)       # `potential --n 60`
POTENTIAL_PIN_NESTED = {("weak", "true"): 6163, ("weak", "false"): 9022, ("weak", "true_supported"): 0.7045270160636053, ("delivered", "true"): 8075, ("roles", "novel", "n"): 1328, ("roles", "novel", "pose"): 0.34036144578313254, ("roles", "dup", "pose"): 0.8045501551189245,
                        ("roles", "false", "pose"): 0.14398137885169585, ("roles", "novel", "random"): 0.12424698795180723, ("recall_ceiling", "all"): 0.03069597811771857, ("recall_ceiling", "topology"): 0.006078411508459123}
MUTANTS = [''')
rep('''    ("the bin edges are (lo, hi]", "m = (W[:, 0] >= lo) & (W[:, 0] < hi)", "m = (W[:, 0] > lo) & (W[:, 0] <= hi)"),
]''', '''    ("the bin edges are (lo, hi]", "m = (W[:, 0] >= lo) & (W[:, 0] < hi)", "m = (W[:, 0] > lo) & (W[:, 0] <= hi)"),
    ("the TEST guard of build is skipped", "    if check:\\n        check_rows(split, rows)\\n", "    if False:\\n        check_rows(split, rows)\\n"),
    ("pose_of reads the previous frame's rotation", 'return ext["rel_R"][i], ext["rel_t"][i]', 'return ext["rel_R"][i - 1], ext["rel_t"][i]'),
    ("pose_of reads the previous frame's translation", 'return ext["rel_R"][i], ext["rel_t"][i]', 'return ext["rel_R"][i], ext["rel_t"][i - 1]'),
    ("the temporary file of an atomic write has a double dot", 'tmp = f"{root}.{os.getpid()}.tmp{ext}"', 'tmp = f"{root}..{os.getpid()}.tmp{ext}"'),
    ("the fixture reads the pose of the previous frame", "R, t = pose_of(ext, ic)\\n            (P0, T0), (P1, T1) = gt(ip), gt(ic)", "R, t = pose_of(ext, ip)\\n            (P0, T0), (P1, T1) = gt(ip), gt(ic)", "fixture"),
    ("potential reads the pose of the previous frame", "R, tv = (a_.astype(float) for a_ in pose_of(ext, i))", "R, tv = (a_.astype(float) for a_ in pose_of(ext, ip))", "potential"),
    ("potential's random null reads the frame itself", "j = int(rng.integers(len(c)))", "j = int(i)", "potential"),
    ("the loader fixture gets the pose entry of the previous item", 'R, t = bc["rel_R"][:, 0], bc["rel_t"][:, 0]', 'R, t = bp["rel_R"][:, 0], bp["rel_t"][:, 0]', "loader"),
    ("the loader fixture moves the current frame's tokens", "JP.make_past(dict(x=torch.zeros(1, K, JP.D), w=torch.zeros(1, K)), fp, fc, R, t,", "JP.make_past(dict(x=torch.zeros(1, K, JP.D), w=torch.zeros(1, K)), fc, fp, R, t,", "loader"),
    ("the loader fixture does not check the pose entry", "perr[how] = max(perr[how], float(np.abs(R[0].numpy() - Rf).max()), float(np.abs(t[0].numpy() - tf).max()))", "perr[how] = 0.0", "loader"),
]''')
rep('''def one(args):
    k, (name, a, b), tmp = args''', '''def check_pins(kind, path):
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
    kind = rest[0] if rest else "selftest"''')
rep('''    r = subprocess.run([sys.executable, p, "selftest"], capture_output=True, text=True, env=ENV)
    return name, "killed" if r.returncode != 0 else "SURVIVED"''', '''    rc, _ = command(kind, p, d)
    return name, "killed" if rc != 0 else "SURVIVED"''')
rep('''    run([["selftest"], ["check"], ["fixture", "--split", "val", "--n-pairs", "40"]])''', '''    run([["selftest"], ["check"], ["fixture", "--split", "val", "--n-pairs", "40"], ["loader-fixture", "--split", "val", "--n-pairs", "40"]])
    with tempfile.TemporaryDirectory() as tmp_:                                                              # the unmutated code passes the pinned commands
        for kind_ in ("fixture", "potential"):
            rc_, txt_ = command(kind_, os.path.join(DATA, MOD), tmp_)
            assert rc_ == 0, f"{kind_} against its pin failed:\\n{txt_}"
            print(f"{kind_} pins ok", flush=True)''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
