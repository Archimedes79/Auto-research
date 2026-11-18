"""E352 step-0 tests of the B-side pruning probe (`jprune_b.py`), CPU only (the GPU is hidden), 2 threads, the stored E324 s and B (`e273_p_s<S>`) and the first 8 / 4 VAL frames; about 8 minutes on a loaded machine.
The tests need the SCRATCH inputs temp/e276/predfull_e273_p_s0.pkl, temp/e345/e324_s_s0/calib.pt and temp/e345/e324_s_s1/calib.pt (not in git) and runs/e345_p000_e324_s_s0/ends_val.pkl; the real-card part of the gate test runs only when the hand-made diagnostic nulls
(`runs/e352diag_null_s<S>` -> temp/e352/diag_null_s<S>) exist.

    wsl bash tools/env/_py14.sh test_jprune_b.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

Known answers: the point table (the sets T1 / T2 on the E345 ladder, the channel counts, the ranked channels against a numpy reference on the calibration, the random draws against the pinned CRC32 seed rule, nested draws, independent seeds per unit and variant); the group variant (whole groups, ranked by the summed
importance, the means aligned with the channels, random whole groups against a numpy reference); run ids, folders, tags and scoring lines (no collision, each line parsed back into the namespace of `jends_eval.py score` and put through the stored checks); the premise (B's 57 pyramid tensors are the junction system's) and the shim; the EMPTY set through
the pass against the stored B dump and the stored E345 null (as sets); the comparison is not blind; an edited pass changes B' and the junction dump in both variants, the random draw differs from the ranked set, seed 1 uses its own calibration (sha256 pinned); the point folder (files, hashes, the row order of B', the partial flag,
the link to the unmodified checkpoint, point.json last, no overwrite); a partial dump cannot land in runs/ (checked before any pass); the null gate (tolerances, synthetic and real cards) and the step-0 table (known answers, the rule's lines). The stored runs' folders must stay as they were.
"""
import json
import os
import pickle
import sys
import tempfile
import time
import zlib
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                      # this test never touches the GPU
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import jends_eval as JEV  # noqa: E402
import jprune_b as PB  # noqa: E402
import jprune_cost as PC  # noqa: E402
import jprune_probe as PP  # noqa: E402
import qends_eval as QE  # noqa: E402
import qpredict as QP  # noqa: E402

torch.set_num_threads(2)
ROOT = PB.ROOT
_FX = {}
COUNTS = {"pyr.e2": 72, "pyr.d2": 36, "pyr.e3": 48, "pyr.d0": 12, "pyr.e1": 24, "pyr.d1": 24}                  # int(f * C + 0.5) of the E345 ladder: 0.5 * 144, 0.25 * 144, 0.25 * 192, 0.25 * 48, 0.25 * 96, 0.25 * 96
GROUP = {"pyr.e2": 18, "pyr.d2": 18, "pyr.e3": 24, "pyr.d0": 6, "pyr.e1": 12, "pyr.d1": 12}                      # channels per GroupNorm(8) group


def fx():
    if not _FX:
        jm, args = JEV.load_model("e324_s_s0", "cpu")
        qg = QP.load_model(os.path.join(ROOT, "runs", "e273_p_s0"), "cpu")
        calib = PP.load_calib(PP.rootpath("temp/e345/e324_s_s0/calib.pt"), "cpu")
        _FX.update(jm=jm, args=args, qg=qg, calib=calib, taps=PP.tap_table(jm), passes={})
    return _FX


def listing(run):
    d = os.path.join(ROOT, "runs", run)
    return sorted((f, os.path.getmtime(os.path.join(d, f))) for f in os.listdir(d))


def one_pass(name, seed=0, groups=False, n=4):
    """a pass of `n` frames of a point, kept for the tests that need it"""
    f = fx()
    if (name, seed, groups, n) not in f["passes"]:
        f["passes"][(name, seed, groups, n)] = PB.run_pass(seed, name, "cpu", max_frames=n, groups=groups, log=lambda *a, **k: None)
    return f["passes"][(name, seed, groups, n)]


def crc_rng(variant, draw, unit):
    """the pinned seed rule of the random draws (a CRC32 of `e352|<variant>|<draw>|<unit>`), written out again here"""
    return np.random.RandomState(zlib.crc32(f"e352|{variant}|{draw}|{unit}".encode()))


# ---------------------------------------------------------------- the point table
def test_points_ladder_counts_and_reference_channels():
    f = fx()
    assert len({PB.IDS[n] for n in PB.POINTS}) == len(PB.POINTS) == 9 and PB.T2.items() >= PB.T1.items() and list(PB.POINTS["t1"][0]) == ["pyr.e2", "pyr.d2", "pyr.e3"]
    units = {u.name: u for u in PP.build_units(f["jm"])}
    assert all(u in units and fr in units[u].fracs for n in PB.POINTS for u, fr in PB.POINTS[n][0].items()), "every set is on the E345 ladder"
    for name in ("t1", "t2"):
        edits = PB.point_edits(f["jm"], f["calib"], name)
        assert {e.tap: len(e.idx) for e in edits} == {u: COUNTS[u] for u in PB.POINTS[name][0]}, name
        for e in edits:                                                                                  # ranked: the lowest importance, numpy reference (a stable sort, as the probe)
            s = f["calib"].s[e.tap].numpy()
            assert set(e.idx) == set(np.argsort(s, kind="stable")[:len(e.idx)].tolist()) and torch.equal(e.val, f["calib"].mean[e.tap][list(e.idx)])
    for name, draw in (("r1d0", 0), ("r1d2", 2), ("r2d1", 1)):                                          # random: the pinned CRC32 rule, the same counts as the ranked set
        for e in PB.point_edits(f["jm"], f["calib"], name):
            assert set(e.idx) == set(crc_rng("channels", draw, e.tap).permutation(len(f["calib"].s[e.tap]))[:COUNTS[e.tap]].tolist()), (name, e.tap)
    assert PB.rng_for("pyr.d2", 0, False).randint(1 << 30) != PB.rng_for("pyr.e1", 0, False).randint(1 << 30), "E345's sum(ord) seed gives these two units the same seed"
    assert sum(map(ord, "pyr.d2")) == sum(map(ord, "pyr.e1")) == 543
    ranked, rnd = PB.point_edits(f["jm"], f["calib"], "t1"), PB.point_edits(f["jm"], f["calib"], "r1d0")
    assert [set(a.idx) for a in ranked] != [set(a.idx) for a in rnd] and [len(a.idx) for a in ranked] == [len(a.idx) for a in rnd]
    u = units["pyr.e2"]                                                                                  # nested draws: the 0.25 set of a unit is inside its 0.5 set
    lo, hi = (PP.edits_for(u, fr, f["calib"], "random", np.random.RandomState(7))[0].idx for fr in (0.25, 0.5))
    assert set(lo) < set(hi)
    cost = PC.Cost(f["jm"])
    s1, s2 = (cost.saved(PB.point_edits(f["jm"], f["calib"], n)) for n in ("t1", "t2"))
    assert 0 < s1["macs"] < s2["macs"] and s1["macs_maps"] == 0 and s2["macs_maps"] > 0 and s1["macs_pyr"] > 0 and cost.saved(PB.point_edits(f["jm"], f["calib"], "r1d1")) == s1, "T1 = e2, d2, e3 are not read by the maps head (its d1 / d0 inputs are in T2); a random draw of the same counts costs the same"
    print("T1 saves", {k: round(v / 1e9, 4) for k, v in s1.items() if k.startswith("macs")}, "T2 saves", {k: round(v / 1e9, 4) for k, v in s2.items() if k.startswith("macs")})


def test_groups_variant_references():
    f = fx()
    for name in ("t1", "t2", "r1d0", "r2d2"):
        for e in PB.point_edits(f["jm"], f["calib"], name, groups=True):
            gs = GROUP[e.tap]
            assert len(e.idx) == COUNTS[e.tap] and sorted(e.idx) == list(e.idx) and all(set(range(c // gs * gs, c // gs * gs + gs)) <= set(e.idx) for c in e.idx), (name, e.tap)
            assert torch.equal(e.val, f["calib"].mean[e.tap][list(e.idx)]), "the means are aligned with the channels"
            G = PC.sub(f["jm"], e.tap)[-2].num_groups
            if name in ("t1", "t2"):                                                                     # ranked: the lowest summed importance
                gsum = f["calib"].s[e.tap].double().numpy().reshape(G, gs).sum(1)
                assert {c // gs for c in e.idx} == set(np.argsort(gsum, kind="stable")[:len(e.idx) // gs].tolist())
            else:                                                                                        # random WHOLE groups: the pinned rule on the group index
                draw = int(name[-1])
                assert {c // gs for c in e.idx} == set(crc_rng("groups", draw, e.tap).permutation(G)[:len(e.idx) // gs].tolist()), (name, e.tap)
    a, b = PB.point_edits(f["jm"], f["calib"], "r1d0", groups=True), PB.point_edits(f["jm"], f["calib"], "r1d0")
    assert [x.idx for x in a] != [x.idx for x in b], "the group and the channel variant draw independently"


def test_names_ids_folders_and_lines():
    assert PB.run_id(0, "null") == "E352.s0_null_s0" and PB.run_id(1, "null", True, "snap") == "E352.s0_null_snap_s1" and PB.run_id(1, "t1", True, "snap") == "E352.s0g_t1_snap_s1" and PB.run_id(0, "t2", False) == "E352.s0c_t2_s0"
    ids = {(s, n, g if n != "null" else None, k): PB.run_id(s, n, g, k) for s in (0, 1) for n in PB.POINTS for g in (True, False) for k in (None, *PB.READOUTS)}          # the null is the same pass in both variants
    assert len(set(ids.values())) == len(ids), "run ids collide"
    assert PB.point_run(0, "null", True) == PB.point_run(0, "null", False) == "e352_null_s0" and PB.point_run(1, "t1", True) == "e352g_t1_s1" and PB.point_run(1, "t1", False) == "e352c_t1_s1"
    assert PB.point_tag("null", True) == "null" and PB.point_tag("t1", True) == "gt1" and PB.point_tag("r1d0", False) == "cr1d0"
    for g in (True, False):
        for name in PB.POINTS:
            for k, line in PB.score_lines(1, name, g).items():
                assert f"--run {PB.point_run(1, name, g)} " in line and f"--tag {PB.point_tag(name, g)} " in line + " " and f"predfull_{PB.point_tag(name, g)}_s1.pkl" in line, (g, name, k)
    assert PB.card_files("gt1")["primary"] == "ends_card_gt1_snap_nogeom_attach_thr0.1.json"


# ---------------------------------------------------------------- the premise and the shim
def test_pyramid_premise_and_shim():
    f = fx()
    assert PB.same_pyramid(f["qg"], f["jm"]) == 57
    bad = QP.load_model(os.path.join(ROOT, "runs", "e273_p_s0"), "cpu")
    with torch.no_grad():
        bad.pyr.e2[0].weight[0, 0, 0, 0] += 1.0
    try:
        PB.same_pyramid(bad, f["jm"])
        raise RuntimeError("a changed pyramid must be refused")
    except AssertionError:
        pass
    sh, w = PP.ablate(PB.Shim(f["qg"]), PB.point_edits(f["jm"], f["calib"], "t1"), f["taps"])
    assert sh.qg.pyr is sh.jmodel.pyr and sh.qg is not f["qg"] and len(sh.qg.pyr.e2._forward_hooks) == 1
    assert not any(len(m._forward_hooks) for m in f["qg"].modules()), "the stored QGraph gets no hook"
    assert set(w.edit) and all(e.kind == "chan" for e in w.edit.values())


# ---------------------------------------------------------------- the unedited pass is the stored dump
def test_empty_set_reproduces_the_stored_dumps():
    with tempfile.TemporaryDirectory() as tmp:
        r = PB.check(0, 8, "cpu", out=tmp)
    print(json.dumps(r, default=float))
    b, j = r["b"], r["junction"]
    assert b["frames"] == 8 and b["nodes_matched"] > 3000 and b["nodes_only_in_ref"] + b["nodes_only_in_new"] <= 0.005 * b["nodes_matched"] and b["e_max_unmatched"] < 0.4, b            # the node sets (e >= 0.2) are the stored ones but for the threshold
    assert b["nodes"]["e"] < 2e-3 and b["nodes"]["pos"] < 1e-2 and b["nodes"]["dirv"] < 2e-2 and b["nodes"]["flags"] < 1e-2 and b["long_equal"], b["nodes"]                           # float32 on the CPU against the stored dump of the GPU
    assert b["links_matched"] > 20000 and b["links_only_in_ref"] + b["links_only_in_new"] <= 0.003 * b["links_matched"] and b["links"]["p"] < 2e-2 and b["links"]["a"] < 1e-2 and b["links"]["b"] < 1e-2, b
    assert b["sections_compared"] > 3000 and b["sections_differ"] <= 0.01 * b["sections_compared"], b                                  # the section start is an argmax: a few near ties flip between devices
    assert j["keypoints_matched"] > 100 and j["keypoints_only_in_ref"] + j["keypoints_only_in_new"] <= 0.01 * j["keypoints_matched"] and j["allowed_equal"], j
    assert j["frames_with_unmatched"] < j["frames"] and j["max_abs"]["kp_pos"] < 5e-3 and j["max_abs"]["kp_p"] < 5e-3 and j["max_abs"]["logit"] < 5e-2 and j["max_abs"]["scale"] < 5e-2, j["max_abs"]     # the logits ARE compared (a key error otherwise)
    assert j["att_moved"] <= 0.02 * j["keypoints_matched"] and j["att_moved_max_m"] < 1.0 and j["max_abs"]["att_p"] < 0.1, j       # the attached ends: a near tie of the attachment softmax moves a few of them to the next node


def test_dump_comparison_catches_a_changed_dump():
    """the comparison is not blind: a permuted dump is equal, a moved node / a changed link probability / a dropped link / a changed section start / a dropped keypoint is reported"""
    import copy
    ref = pickle.load(open(os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl"), "rb"))[:2]
    perm = copy.deepcopy(ref)
    for d in perm:                                                                                       # the same graph with its nodes in another order (the links renumbered)
        n = len(d["e"])
        o = np.random.RandomState(0).permutation(n)
        inv = np.empty(n, int)
        inv[o] = np.arange(n)
        for k in ("e", "pos", "dirv", "flags", "s"):
            d[k] = d[k][o]
        d["link_src"], d["link_dst"] = inv[d["link_src"]], inv[d["link_dst"]]
        d["section"] = d["section"][o]                                                                   # section starts are indices of the unfiltered list, `idx` maps them to the kept nodes
        d["idx"] = d["idx"][o]
    same = PB.compare_dumps(ref, perm)
    assert same["nodes_only_in_ref"] == same["nodes_only_in_new"] == same["links_only_in_ref"] == same["links_only_in_new"] == same["sections_differ"] == 0 and max(same["nodes"].values()) == 0.0 and same["links"]["p"] == 0.0 and same["sections_compared"] > 500, same
    bad = copy.deepcopy(ref)
    bad[0]["pos"][3] += 3.0
    bad[1]["link_p"][5] += 0.1
    r = PB.compare_dumps(ref, bad)
    assert r["nodes_only_in_ref"] == 1 and r["nodes_only_in_new"] == 1 and r["links"]["p"] >= 0.1 - 1e-9, r
    gone = copy.deepcopy(ref)
    for k in ("link_src", "link_dst", "link_p", "link_long", "link_a", "link_b"):
        gone[0][k] = gone[0][k][:-1]                                                                     # the last link of frame 0 is missing in the new dump
    r = PB.compare_dumps(ref, gone)
    assert r["links_only_in_ref"] == 1 and r["links_only_in_new"] == 0 and r["nodes_only_in_ref"] == 0, r
    sec = copy.deepcopy(ref)
    j = int(np.flatnonzero(sec[0]["section"] != sec[0]["section"][0])[0])
    sec[0]["section"][0] = sec[0]["section"][j]                                                          # node 0 starts its section at another node
    assert PB.compare_dumps(ref, sec)["sections_differ"] >= 1
    jf = pickle.load(open(os.path.join(ROOT, "runs", "e345_p000_e324_s_s0", "ends_val.pkl"), "rb"))[:2]
    assert PB.compare_frames(jf, copy.deepcopy(jf))["max_abs"]["kp_pos"] == 0.0
    drop = copy.deepcopy(jf)
    drop[0]["valid"][0][np.flatnonzero(drop[0]["valid"][0])[0]] = False
    assert PB.compare_frames(jf, drop)["keypoints_only_in_ref"] == 1
    moved = copy.deepcopy(jf)
    k = int(np.flatnonzero(moved[0]["valid"][0])[0])
    moved[0]["att_pos"][k] += 0.5
    m = PB.compare_frames(jf, moved)
    assert m["att_moved"] == 1 and abs(m["att_moved_max_m"] - float(np.sqrt(2) * 0.5)) < 1e-4, m


# ---------------------------------------------------------------- an edited pass, the point folder
def test_edited_pass_changes_both_models_and_the_random_draw_differs():
    nul, t1, rnd, tg = one_pass("null"), one_pass("t1"), one_pass("r1d0"), one_pass("t1", groups=True)
    e0 = [p["nodes"]["e"] for p in nul["preds"]]
    for res in (t1, rnd, tg):
        assert any(len(a) != len(b) or not np.allclose(a, b, atol=1e-4) for a, b in zip(e0, [p["nodes"]["e"] for p in res["preds"]])), "B' must differ from B: the edit reached B's QGraph"
        assert any(not np.allclose(a["logit"], b["logit"], atol=1e-4) or not np.allclose(a["kp_pos"], b["kp_pos"], atol=1e-4) for a, b in zip(nul["frames"], res["frames"])), "the junction dump must differ from the null"
    assert any(len(a["nodes"]["e"]) != len(b["nodes"]["e"]) or not np.allclose(a["nodes"]["e"], b["nodes"]["e"], atol=1e-4) for a, b in zip(t1["preds"], rnd["preds"])), "the random draw is another edit"
    assert any(len(a["nodes"]["e"]) != len(b["nodes"]["e"]) or not np.allclose(a["nodes"]["e"], b["nodes"]["e"], atol=1e-4) for a, b in zip(t1["preds"], tg["preds"])), "whole groups are another edit than the lowest channels"
    assert t1["rows"] == nul["rows"] == rnd["rows"] == tg["rows"] and [fr["row"] for fr in t1["frames"]] == t1["rows"] and len(t1["frames"]) == len(t1["preds"]) == 4
    assert nul["saved"]["macs"] == 0 and t1["saved"]["macs"] == rnd["saved"]["macs"] == tg["saved"]["macs"] > 0 and tg["groups"] and not t1["groups"]


def test_seed_one_uses_its_own_calibration():
    s0, s1 = one_pass("t1"), one_pass("t1", seed=1)
    c0, c1 = (PP.load_calib(PP.rootpath(f"temp/e345/e324_s_s{s}/calib.pt"), "cpu") for s in (0, 1))
    assert PB.sha256_file(PB.calib_path(1)) != PB.sha256_file(PB.calib_path(0))
    for e in s1["edits"]:                                                                                # seed 1's channels: the lowest importance of seed 1's calibration (and not seed 0's)
        want = set(np.argsort(c1.s[e.tap].numpy(), kind="stable")[:len(e.idx)].tolist())
        assert set(e.idx) == want and torch.equal(e.val, c1.mean[e.tap][list(e.idx)]), e.tap
    assert [set(e.idx) for e in s1["edits"]] != [set(e.idx) for e in s0["edits"]]
    with tempfile.TemporaryDirectory() as tmp:
        meta = PB.write_point(os.path.join(tmp, "e352c_t1_s1"), 1, "t1", s1)
    assert meta["seed"] == 1 and meta["sha256"]["calib"] == PB.sha256_file(PB.calib_path(1)) and meta["sha256"]["model_junction"] == PB.sha256_file(os.path.join(ROOT, "runs", "e324_s_s1", "model.pt")) and meta["sha256"]["model_b"] == PB.sha256_file(os.path.join(ROOT, "runs", "e273_p_s1", "model.pt"))


def test_point_folder_files_hashes_and_refusals():
    t1 = one_pass("t1")
    before = {r: listing(r) for r in ("e324_s_s0", "e273_p_s0")}
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "e352c_t1_s0")
        meta = PB.write_point(folder, 0, "t1", t1)
        assert sorted(os.listdir(folder)) == ["ends_val.pkl", "model.pt", "point.json", "predfull_ct1_s0.pkl"] and os.path.realpath(os.path.join(folder, "model.pt")) == os.path.realpath(os.path.join(ROOT, "runs", "e324_s_s0", "model.pt"))
        assert meta == json.load(open(os.path.join(folder, "point.json"))) and meta["sha256"]["ends_val"] == PB.sha256_file(os.path.join(folder, "ends_val.pkl")) and meta["sha256"]["predfull"] == PB.sha256_file(os.path.join(folder, "predfull_ct1_s0.pkl"))
        assert meta["set"] == PB.T1 and meta["id"] == 201 and meta["tag"] == "ct1" and meta["frames"] == 4 and meta["frames_total"] == 873 and meta["partial"] is True and [e["n"] for e in meta["edits"]] == [72, 36, 48] and meta["saved"]["macs"] == t1["saved"]["macs"] > 0
        assert meta["sha256"]["calib"] == PB.sha256_file(PB.calib_path(0))
        frames = pickle.load(open(os.path.join(folder, "ends_val.pkl"), "rb"))
        d = pickle.load(open(os.path.join(folder, "predfull_ct1_s0.pkl"), "rb"))
        assert len(frames) == len(d) == 4 and [x["row"] for x in d] == t1["rows"] == [x["row"] for x in frames] and {"e", "pos", "link_p", "assign_P", "road_ids"} <= set(d[0]) and "road2" not in d[0], "the format of the stored B dump"
        for x, p in zip(d, t1["preds"]):                                                                 # the row order of B': entry i holds the prediction of frame i (a reversed list would carry the right rows over the wrong predictions)
            e = np.asarray(p["nodes"]["e"], float)
            assert np.array_equal(x["e"], e[e >= 0.2]) and np.array_equal(x["pos"], np.asarray(p["nodes"]["pos"], float)[e >= 0.2])
        try:
            PB.write_point(folder, 0, "t1", t1)
            raise RuntimeError("a complete folder must not be overwritten")
        except SystemExit:
            pass
        os.remove(os.path.join(folder, "point.json"))                                                    # a crash before point.json: the folder is incomplete and is redone
        assert PB.write_point(folder, 0, "t1", t1)["sha256"]["predfull"] == meta["sha256"]["predfull"], "the dump is deterministic"
    assert before == {r: listing(r) for r in before}, "a stored run folder changed"


def test_partial_dump_cannot_land_in_runs():
    PB.check_out(0, None)
    PB.check_out(4, os.path.join(tempfile.gettempdir(), "x"))
    for out in (None, "runs/e352g_t1_s0", os.path.join(ROOT, "runs", "e352_zz_s0"), "temp/../runs/e352_zz_s0"):
        try:
            PB.check_out(4, out)
            raise RuntimeError(f"--max {out} must be refused")
        except SystemExit:
            pass
    argv, run_pass, called = sys.argv, PB.run_pass, []                                                   # `main` checks before it runs a pass: a stub that records the call
    PB.run_pass = lambda *a, **k: called.append(a) or (_ for _ in ()).throw(AssertionError("run_pass was called"))
    try:
        for extra in ([], ["--out", "runs/e352_zz_s0"]):
            sys.argv = ["jprune_b.py", "dump", "--seed", "0", "--point", "t1", "--max", "4", *extra]
            try:
                PB.main()
                raise RuntimeError("main must refuse")
            except SystemExit:
                pass
    finally:
        sys.argv, PB.run_pass = argv, run_pass
    assert not called and not os.path.exists(os.path.join(ROOT, "runs", "e352_zz_s0"))


# ---------------------------------------------------------------- the scoring lines
def parse_line(line, root):
    """a printed scoring line as the namespace `jends_eval.py score` builds from it (`$R` = root): `--key value` pairs and bare flags"""
    toks = line.replace("$R", root).split()
    toks = toks[toks.index("score") + 1:]
    ns, i = {}, 0
    while i < len(toks):
        k = toks[i][2:].replace("-", "_")
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            ns[k], i = toks[i + 1], i + 2
        else:
            ns[k], i = True, i + 1
    ns["seed"], ns["thr"] = int(ns["seed"]), float(ns["thr"])
    return SimpleNamespace(**ns)


def test_scoring_lines_pass_the_stored_checks():
    t1, tg = one_pass("t1"), one_pass("t1", groups=True)
    flags = {"snap": {"snap"}, "primary": {"snap", "snap_nogeom", "attach"}, "attach": {"attach"}, "unsnapped": set()}
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "runs"))
        os.symlink(os.path.join(ROOT, "runs", "e273_p_s0"), os.path.join(tmp, "runs", "e273_p_s0"))
        for res, groups in ((t1, False), (tg, True)):
            folder = os.path.join(tmp, "runs", PB.point_run(0, "t1", groups))
            PB.write_point(folder, 0, "t1", res)
            files = PB.card_files(PB.point_tag("t1", groups))
            for k, line in PB.score_lines(0, "t1", groups).items():
                ns = parse_line(line, tmp)                                                               # the printed line itself goes through the stored checks
                assert {f for f in ("snap", "snap_nogeom", "attach") if getattr(ns, f, False)} == flags[k] and ns.run == PB.point_run(0, "t1", groups) and ns.tag == PB.point_tag("t1", groups) and ns.spec == QE.DEFAULT_SPEC == "close:1.5@0.5", (k, ns)
                assert os.path.isfile(ns.full) and os.path.isfile(ns.base), (k, ns.full, ns.base)
                name, spec, tag = QE.score_target(ns, folder)
                assert (name, spec, tag) == (files[k], "close:1.5@0.5", PB.point_tag("t1", groups)), (k, name)
        old, JEV.ROOT = JEV.ROOT, tmp                                                                  # `jends_eval.score`: the seed / pyramid checks pass on the point folder, the scorer gets the dump of B'
        seen, saved = [], QE.cmd_score
        QE.cmd_score = lambda ns: seen.append(ns)
        try:
            JEV.score(parse_line(PB.score_lines(0, "t1", True)["snap"], tmp))
        finally:
            QE.cmd_score, JEV.ROOT = saved, old
        assert len(seen) == 1 and seen[0].full.endswith("predfull_gt1_s0.pkl") and seen[0].tag == "gt1" and seen[0].skip_base_check


# ---------------------------------------------------------------- the null gate and the table
def fake_card(base, head):
    full = lambda v: dict(zip(PB.KEYS, v))  # noqa: E731
    return dict(numbers=dict(base=full(base), head=full(head)))


def put_cards(runs, run, tag, base, heads):
    os.makedirs(os.path.join(runs, run), exist_ok=True)
    for k, h in zip(PB.READOUTS, heads):
        json.dump(fake_card(base, h), open(os.path.join(runs, run, PB.card_files(tag)[k]), "w"))


def vec(topo, other=0.5, vert=19.0):
    return [other, other, other, topo, other, other, other, other, other, other, vert]


def test_gate_tolerances_and_real_cards():
    assert PB.tolerance("topo_F1", 0.6) == 2e-4 and PB.tolerance("vert100_dp", 19.0) == max(PB.GATE_ABS, PB.GATE_REL * 19.0) and PB.tolerance("junction_R", 0.55) == max(PB.GATE_ABS, PB.GATE_REL * 0.55)
    with tempfile.TemporaryDirectory() as tmp:
        runs = os.path.join(tmp, "runs")
        for k in PB.READOUTS:                                                                            # stored cards of E324 s and the null's cards
            os.makedirs(os.path.join(runs, "e324_s_s0"), exist_ok=True)
            json.dump(fake_card(vec(0.6259), vec(0.6981)), open(os.path.join(runs, "e324_s_s0", PB.STORED[k]), "w"))
        put_cards(runs, "e352_null_s0", "null", vec(0.62594, vert=19.002), [vec(0.69807)] * 3)
        rows, ok = PB.gate(0, runs_dir=runs)
        assert ok and len(rows) == 3 * 2 * 11 and "| 0 | snap | base |" in PB.gate_md(rows)
        put_cards(runs, "e352_null_s0", "null", vec(0.62594), [vec(0.69807), vec(0.69807), vec(0.6978)])        # the attach card's head topo F1 off by 3e-4
        rows, ok = PB.gate(0, runs_dir=runs)
        assert not ok and [r[1:4] for r in rows if not r[-1]] == [("attach", "head", "topo_F1")]
        put_cards(runs, "e352_null_s0", "null", vec(0.6259, vert=19.1), [vec(0.6981)] * 3)                    # vert100_dp off by 0.1: above max(2e-3, 1e-3 * 19)
        assert not PB.gate(0, runs_dir=runs)[1]
    for seed in (0, 1):                                                                                  # the hand-made diagnostic nulls (when they exist): the gate holds with margin
        run = f"e352diag_null_s{seed}"
        if os.path.exists(os.path.join(ROOT, "runs", run, PB.card_files("null")["attach"])):
            rows, ok = PB.gate(seed, run)
            worst = max(abs(r[6]) / r[7] for r in rows)
            print(f"null gate seed {seed}: worst |difference| / tolerance {worst:.2f}")
            assert ok and worst < 0.8, (seed, worst)


def test_table_known_answers_and_the_rule_lines():
    with tempfile.TemporaryDirectory() as tmp:
        runs = os.path.join(tmp, "runs")
        put_cards(runs, "e352_null_s0", "null", vec(0.6259), [vec(0.6981), vec(0.6912), vec(0.6616)])
        put_cards(runs, "e352g_t1_s0", "gt1", vec(0.6219), [vec(0.6941), vec(0.6872), vec(0.6576)])             # T1 ranked: -0.004 everywhere
        for d, b in enumerate((0.6150, 0.6190, 0.6170)):                                                  # three random draws: mean 0.6170
            put_cards(runs, f"e352g_r1d{d}_s0", f"gr1d{d}", vec(b), [vec(0.6900), vec(0.6800), vec(0.6500)])
        put_cards(runs, "e352c_t1_s0", "ct1", vec(0.6200), [vec(0.6950), vec(0.6880), vec(0.6580)])
        data = PB.table_data((0, 1), runs_dir=runs)
        assert data[1]["null"] == [None] * 4 and data[0]["null"] == [0.6259, 0.6981, 0.6912, 0.6616]
        assert abs(data[0]["random_mean"][("groups", "t1")][0] - 0.6170) < 1e-12 and ("groups", "t2") not in data[0]["random_mean"] and ("channels", "t1") in data[0]["rows"] and ("channels", "t1") not in data[0]["random_mean"]
        md = PB.table_md(data)
        assert "| groups | t1 ranked - random | +0.0049 | -" in md and "| groups | t1 | 0.6219 | -0.0040 | 0.6941 | -0.0040 |" in md and "| groups | t1 random mean | 0.6170 | -0.0089 |" in md and "| channels | t1 | 0.6200 | -0.0059 |" in md and "seed 1: the null cards are missing" in md, md
        lines = PB.checks(data)
        assert any("seed 0 t1: B plain -0.0040" in x and "HIT" in x for x in lines) and any("ranked minus mean of the 3 random draws on B plain +0.0049" in x and "HIT" in x for x in lines) and any("KILL" in x and x.endswith("no") for x in lines), lines
        put_cards(runs, "e352g_t1_s0", "gt1", vec(0.58), [vec(0.55)] * 3)                               # a T1 that loses 0.0459 on B plain while the random mean (0.6170) loses 0.0089: the ranking is worse than random: KILL
        assert any("KILL" in x and "YES" in x for x in PB.checks(PB.table_data((0,), runs_dir=runs)))
        for d in range(3):
            put_cards(runs, f"e352g_r1d{d}_s0", f"gr1d{d}", vec(0.55), [vec(0.50)] * 3)                  # the random draws lose 0.0759: more than T1: no kill
        assert any("KILL" in x and x.endswith("no") for x in PB.checks(PB.table_data((0,), runs_dir=runs)))
        put_cards(runs, "e352g_t1_s0", "gt1", vec(0.60), [vec(0.55)] * 3)                               # T1 loses 0.0259 (not more than 0.03): no kill whatever the draws do
        for d in range(3):
            put_cards(runs, f"e352g_r1d{d}_s0", f"gr1d{d}", vec(0.62), [vec(0.50)] * 3)
        assert any("KILL" in x and x.endswith("no") for x in PB.checks(PB.table_data((0,), runs_dir=runs)))


TESTS = [test_points_ladder_counts_and_reference_channels, test_groups_variant_references, test_names_ids_folders_and_lines, test_pyramid_premise_and_shim, test_empty_set_reproduces_the_stored_dumps, test_dump_comparison_catches_a_changed_dump,
         test_edited_pass_changes_both_models_and_the_random_draw_differs, test_seed_one_uses_its_own_calibration, test_point_folder_files_hashes_and_refusals, test_partial_dump_cannot_land_in_runs, test_scoring_lines_pass_the_stored_checks,
         test_gate_tolerances_and_real_cards, test_table_known_answers_and_the_rule_lines]


if __name__ == "__main__":
    before = {r: listing(r) for r in ("e324_s_s0", "e273_p_s0")}
    want = sys.argv[1:]
    for t in TESTS:
        if not want or t.__name__ in want or t.__name__.replace("test_", "") in want:
            t0 = time.time()
            t()
            print(f"{t.__name__} ok ({time.time() - t0:.0f} s)")
    assert before == {r: listing(r) for r in before}, "a stored run folder changed"
    print("test_jprune_b: all ok; the stored runs' folders are unchanged")
