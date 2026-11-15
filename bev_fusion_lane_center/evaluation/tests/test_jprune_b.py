"""E352 step-0 tests of the B-side pruning probe (`jprune_b.py`), CPU only (the GPU is hidden), 2 threads, the stored E324 s and B (`e273_p_s0`) and the first 8 / 4 VAL frames; about N minutes on a loaded machine.

    wsl bash tools/env/_py14.sh test_jprune_b.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

Known answers: the point table (the sets T1 / T2 on the E345 ladder, the channel counts of the units, the ranked channels against a numpy reference on the calibration, the random draws against `RandomState(1000 * draw + sum(ord))`, nested draws, whole groups); the premise (B's 57 pyramid tensors are the
junction system's); the shim (the hooks of the copy sit on the QGraph that predicts, the stored model gets none); the EMPTY set through the pass against the stored B dump and the stored E345 null (the tolerance reached on the CPU against the device of the stored dumps); an edited pass changes B' and the junction dump,
the random draw differs from the ranked set; the point folder (files, hashes, the link to the unmodified checkpoint, point.json last, no overwrite, no partial dump into runs/); the scoring lines pass the stored checks of `qends_eval.score_target` and the pyramid assertion of `jends_eval.score`. The stored runs' folders must stay as they were.
"""
import json
import os
import pickle
import sys
import tempfile
import time
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


def one_pass(name, n=4):
    """a pass of `n` frames of a point, kept for the tests that need it"""
    f = fx()
    if (name, n) not in f["passes"]:
        f["passes"][(name, n)] = PB.run_pass(0, name, "cpu", max_frames=n, log=lambda *a, **k: None)
    return f["passes"][(name, n)]


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
    for name, draw in (("r1d0", 0), ("r1d2", 2), ("r2d1", 1)):                                          # random: the probe's per-unit RandomState, the same counts as the ranked set
        edits = PB.point_edits(f["jm"], f["calib"], name)
        for e in edits:
            rng = np.random.RandomState(1000 * draw + sum(map(ord, e.tap)))
            assert set(e.idx) == set(rng.permutation(len(f["calib"].s[e.tap]))[:COUNTS[e.tap]].tolist()), (name, e.tap)
    ranked, rnd = PB.point_edits(f["jm"], f["calib"], "t1"), PB.point_edits(f["jm"], f["calib"], "r1d0")
    assert [set(a.idx) for a in ranked] != [set(a.idx) for a in rnd] and [len(a.idx) for a in ranked] == [len(a.idx) for a in rnd]
    u = units["pyr.e2"]                                                                                  # nested draws: the 0.25 set of a unit is inside its 0.5 set
    lo, hi = (PP.edits_for(u, fr, f["calib"], "random", np.random.RandomState(7))[0].idx for fr in (0.25, 0.5))
    assert set(lo) < set(hi)
    for name in ("t1", "t2", "r2d2"):                                                                    # whole groups: the same counts, complete groups, ranked by the summed importance
        for e in PB.point_edits(f["jm"], f["calib"], name, groups=True):
            gs = PC.sub(f["jm"], e.tap)[-2].num_channels // 8
            assert len(e.idx) == COUNTS[e.tap] and all(set(range(c // gs * gs, c // gs * gs + gs)) <= set(e.idx) for c in e.idx)
            if name != "r2d2":
                gsum = f["calib"].s[e.tap].double().numpy().reshape(8, gs).sum(1)
                assert {c // gs for c in e.idx} == set(np.argsort(gsum, kind="stable")[:len(e.idx) // gs].tolist())
    cost = PC.Cost(f["jm"])
    s1, s2 = (cost.saved(PB.point_edits(f["jm"], f["calib"], n)) for n in ("t1", "t2"))
    assert 0 < s1["macs"] < s2["macs"] and s1["macs_maps"] == 0 and s2["macs_maps"] > 0 and s1["macs_pyr"] > 0 and cost.saved(PB.point_edits(f["jm"], f["calib"], "r1d1")) == s1, "T1 = e2, d2, e3 are not read by the maps head (its d1 / d0 inputs are in T2); a random draw of the same counts costs the same"
    print("T1 saves", {k: round(v / 1e9, 4) for k, v in s1.items() if k.startswith("macs")}, "T2 saves", {k: round(v / 1e9, 4) for k, v in s2.items() if k.startswith("macs")})


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
    assert j["keypoints_matched"] > 100 and j["keypoints_only_in_ref"] + j["keypoints_only_in_new"] <= 0.01 * j["keypoints_matched"] and j["allowed_equal"], j
    assert j["max_abs"]["kp_pos"] < 5e-3 and j["max_abs"]["kp_p"] < 5e-3 and j["max_abs"].get("logit", 0.0) < 5e-2, j["max_abs"]


def test_dump_comparison_catches_a_changed_dump():
    """the comparison is not blind: a permuted dump is equal, a moved node / a changed link probability / a dropped keypoint is reported"""
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
    assert same["nodes_only_in_ref"] == same["nodes_only_in_new"] == same["links_only_in_ref"] == same["links_only_in_new"] == 0 and max(same["nodes"].values()) == 0.0 and same["links"]["p"] == 0.0, same
    bad = copy.deepcopy(ref)
    bad[0]["pos"][3] += 3.0
    bad[1]["link_p"][5] += 0.1
    r = PB.compare_dumps(ref, bad)
    assert r["nodes_only_in_ref"] == 1 and r["nodes_only_in_new"] == 1 and r["links"]["p"] >= 0.1 - 1e-9, r
    jf = pickle.load(open(os.path.join(ROOT, "runs", "e345_p000_e324_s_s0", "ends_val.pkl"), "rb"))[:2]
    assert PB.compare_frames(jf, copy.deepcopy(jf))["max_abs"]["kp_pos"] == 0.0
    drop = copy.deepcopy(jf)
    drop[0]["valid"][0][np.flatnonzero(drop[0]["valid"][0])[0]] = False
    assert PB.compare_frames(jf, drop)["keypoints_only_in_ref"] == 1


# ---------------------------------------------------------------- an edited pass, the point folder
def test_edited_pass_changes_both_models_and_the_random_draw_differs():
    nul, t1, rnd = one_pass("null"), one_pass("t1"), one_pass("r1d0")
    e0 = [p["nodes"]["e"] for p in nul["preds"]]
    for res in (t1, rnd):
        assert any(len(a) != len(b) or not np.allclose(a, b, atol=1e-4) for a, b in zip(e0, [p["nodes"]["e"] for p in res["preds"]])), "B' must differ from B: the edit reached B's QGraph"
        assert any(not np.allclose(a["logit"], b["logit"], atol=1e-4) or not np.allclose(a["kp_pos"], b["kp_pos"], atol=1e-4) for a, b in zip(nul["frames"], res["frames"])), "the junction dump must differ from the null"
    assert any(len(a["nodes"]["e"]) != len(b["nodes"]["e"]) or not np.allclose(a["nodes"]["e"], b["nodes"]["e"], atol=1e-4) for a, b in zip(t1["preds"], rnd["preds"])), "the random draw is another edit"
    assert t1["rows"] == nul["rows"] == rnd["rows"] and [fr["row"] for fr in t1["frames"]] == t1["rows"] and len(t1["frames"]) == len(t1["preds"]) == 4
    assert nul["saved"]["macs"] == 0 and t1["saved"]["macs"] == rnd["saved"]["macs"] > 0


def test_point_folder_files_hashes_and_refusals():
    t1 = one_pass("t1")
    before = {r: listing(r) for r in ("e324_s_s0", "e273_p_s0")}
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "e352_t1_s0")
        meta = PB.write_point(folder, 0, "t1", t1)
        assert sorted(os.listdir(folder)) == ["ends_val.pkl", "model.pt", "point.json", "predfull_t1_s0.pkl"] and os.path.realpath(os.path.join(folder, "model.pt")) == os.path.realpath(os.path.join(ROOT, "runs", "e324_s_s0", "model.pt"))
        assert meta == json.load(open(os.path.join(folder, "point.json"))) and meta["sha256"]["ends_val"] == PB.sha256_file(os.path.join(folder, "ends_val.pkl")) and meta["sha256"]["predfull"] == PB.sha256_file(os.path.join(folder, "predfull_t1_s0.pkl"))
        assert meta["set"] == PB.T1 and meta["id"] == 201 and meta["frames"] == 4 and [e["n"] for e in meta["edits"]] == [72, 36, 48] and meta["saved"]["macs"] == t1["saved"]["macs"] > 0
        frames = pickle.load(open(os.path.join(folder, "ends_val.pkl"), "rb"))
        d = pickle.load(open(os.path.join(folder, "predfull_t1_s0.pkl"), "rb"))
        assert len(frames) == len(d) == 4 and [x["row"] for x in d] == t1["rows"] and {"e", "pos", "link_p", "assign_P", "road_ids"} <= set(d[0]) and "road2" not in d[0], "the format of the stored B dump"
        try:
            PB.write_point(folder, 0, "t1", t1)
            raise RuntimeError("a complete folder must not be overwritten")
        except SystemExit:
            pass
        os.remove(os.path.join(folder, "point.json"))                                                    # a crash before point.json: the folder is incomplete and is redone
        assert PB.write_point(folder, 0, "t1", t1)["sha256"]["predfull"] == meta["sha256"]["predfull"], "the dump is deterministic"
    assert before == {r: listing(r) for r in before}, "a stored run folder changed"


# ---------------------------------------------------------------- the scoring lines
def test_scoring_lines_pass_the_stored_checks():
    t1 = one_pass("t1")
    lines = PB.score_lines(0, "t1")
    assert list(lines) == ["snap", "primary", "attach", "unsnapped"] and all("--full $R/runs/e352_t1_s0/predfull_t1_s0.pkl --base $R/runs/e273_p_s0/model.pt --spec close:1.5@0.5 --tag t1" in v for v in lines.values())
    flags = {"snap": dict(snap=True), "primary": dict(snap=True, snap_nogeom=True, attach=True), "attach": dict(attach=True), "unsnapped": {}}
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "runs", "e352_t1_s0")
        PB.write_point(folder, 0, "t1", t1)
        os.symlink(os.path.join(ROOT, "runs", "e273_p_s0"), os.path.join(tmp, "runs", "e273_p_s0"))
        files = PB.card_files("t1")
        for k, fl in flags.items():
            assert all(w in lines[k].split() for w in ("--" + o.replace("_", "-") for o in fl)), (k, lines[k])
            ns = SimpleNamespace(run="e352_t1_s0", seed=0, thr=0.1, full=os.path.join(folder, "predfull_t1_s0.pkl"), base=os.path.join(ROOT, "runs", "e273_p_s0", "model.pt"), spec="close:1.5@0.5", tag="t1", **fl)
            name, spec, tag = QE.score_target(ns, folder)
            assert (name, spec, tag) == (files[k], "close:1.5@0.5", "t1") and spec == QE.DEFAULT_SPEC, (k, name)
        old, JEV.ROOT = JEV.ROOT, tmp                                                                  # `jends_eval.score`: the seed / pyramid checks pass on the point folder, the scorer gets the dump of B'
        seen, saved = [], QE.cmd_score
        QE.cmd_score = lambda ns: seen.append(ns)
        try:
            JEV.score(SimpleNamespace(run="e352_t1_s0", seed=0, thr=0.1, full=os.path.join(folder, "predfull_t1_s0.pkl"), base=os.path.join(ROOT, "runs", "e273_p_s0", "model.pt"), spec="close:1.5@0.5", tag="t1", snap=True))
        finally:
            QE.cmd_score, JEV.ROOT = saved, old
        assert len(seen) == 1 and seen[0].full.endswith("predfull_t1_s0.pkl") and seen[0].tag == "t1" and seen[0].skip_base_check


def test_partial_dump_cannot_land_in_runs():
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "jprune_b.py"), "dump", "--seed", "0", "--point", "t1", "--max", "4"], capture_output=True, text=True, env=dict(os.environ, BFLC_ROOT=ROOT))
    assert r.returncode != 0 and "--max writes a partial dump" in r.stderr, r.stderr[-300:]


TESTS = [test_points_ladder_counts_and_reference_channels, test_pyramid_premise_and_shim, test_empty_set_reproduces_the_stored_dumps, test_dump_comparison_catches_a_changed_dump, test_edited_pass_changes_both_models_and_the_random_draw_differs, test_point_folder_files_hashes_and_refusals,
         test_scoring_lines_pass_the_stored_checks, test_partial_dump_cannot_land_in_runs]


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
