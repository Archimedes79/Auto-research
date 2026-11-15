"""scratch (E343 code step, round 4): the tests for the round-4 findings"""
import os

P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests", "test_jtest_eval.py")
s = open(P, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''import subprocess
import sys
import tempfile
''', '''import subprocess
import sys
import tempfile
import types
''')

# ---------------------------------------------------------------------------------------------------- run_toy_seed: E321's hashed artefacts on TEST
rep('''    with patched(*defaults, (J, "dump_dir", lambda split, n=0: d), (J, "dump_b", boom),''', '''    rel = lambda p: os.path.relpath(os.path.realpath(p), os.path.realpath(J.ROOT)).replace(os.sep, "/")  # noqa: E731
    e321 = {rel(os.path.join(d, "ctl_ends.pkl")): "0"} if split == "test" else dict(J.E321_SHA256)             # TEST: the flagged pair dump without provenance must be one of E321's hashed artefacts
    with patched(*defaults, (J, "E321_SHA256", e321), (J, "dump_dir", lambda split, n=0: d), (J, "dump_b", boom),''')

# ---------------------------------------------------------------------------------------------------- precheck test: a toy control pair dump (the stored VAL dump is 13 MB and is now validated), a config without a seed
a = s.index("def test_precheck_refuses_before_any_dump_is_made():")
b = s.index("OLD_HEAD = dict(")
s = s[:a] + '''def test_precheck_refuses_before_any_dump_is_made():
    ns = lambda **k: argparse.Namespace(**dict(dict(split="val", max=0, device="cpu", run_prefix="e324_s_s", b_dump=None, control_ends_dump=None, control_card=None), **k))  # noqa: E731
    setup = J.SETUPS["e324_s_s"]
    with tempfile.TemporaryDirectory() as d, patched((J, "pyramid_same", lambda *x: True), (J, "dump_dir", lambda split, n=0: d), (J, "provenance", PROV)):
        card = toy_card_file(d, 0.6, 0.5)
        ce = os.path.join(d, "toy_ctl_ends.pkl")                                                       # the control's pair dump of these calls: a toy one (the stored VAL dump would be loaded and validated)
        pickle.dump(toy_dump(), open(ce, "wb"))
        ok = lambda **k: ns(**dict(dict(control_card=card, control_ends_dump=ce), **k))  # noqa: E731
        i, ctl = J.precheck(ok(), setup, 0, True)
        assert i["run"] == "e324_s_s0" and ctl["variants"]["snapped"]["curves"] == 7, "the control card is read at the start"
        seen = []
        with patched((J, "check_seed", lambda run, seed: seen.append((run, seed)))):
            J.precheck(ok(), setup, 1, True)
        assert seen == [("e324_s_s1", 1)], "the seed of the pair run is checked in the precheck"
        assert raises(SystemExit, J.precheck, ok(control_card=os.path.join(d, "missing.json")), setup, 0, True), "a missing control card stops before the dumping"
        assert raises(SystemExit, J.precheck, ok(b_dump=os.path.join(d, "missing_b.pkl")), setup, 0, True), "a flagged B dump that is not there"
        assert raises(SystemExit, J.precheck, ok(control_ends_dump=os.path.join(d, "missing_ends.pkl")), setup, 0, True), "a flagged control pair dump that is not there"
        assert raises(SystemExit, J.precheck, ns(split="test", control_card=card, control_ends_dump=ce), setup, 0, True), "TEST without --b-dump and without a dump at the default path: B's dump is never made on TEST"
        with patched((J, "pyramid_same", lambda *x: False)):
            assert raises(SystemExit, J.precheck, ok(), setup, 0, True), "arm k fails the pyramid check: stop, to be run again with --no-stack"
            assert J.precheck(ok(), setup, 0, False)[0]["run"] == "e324_s_s0", "--no-stack does not look at the pyramid"
    with tempfile.TemporaryDirectory() as d, patched((J, "ROOT", d)):
        os.makedirs(os.path.join(d, "runs", "e324_s_s0"))
        config = os.path.join(d, "runs", "e324_s_s0", "config.json")
        json.dump(dict(seed=1), open(config, "w"))
        assert raises(SystemExit, J.check_seed, "e324_s_s0", 0), "the run of another seed: its keypoints would sit on another seed's graph"
        J.check_seed("e324_s_s0", 1)
        json.dump(dict(args=dict(name="x")), open(config, "w"))
        assert raises(RuntimeError, J.check_seed, "e324_s_s0", 0) and raises(RuntimeError, J.check_seed, "e324_s_s0", 1), "a config without a seed cannot be tied to a seed: refused (B and c2 have no config.json at all; the pair runs must have one)"


''' + s[b:]

# ---------------------------------------------------------------------------------------------------- the old-style head test: the message
rep('''            assert "made by other code OR interrupted between the pickle and its sidecar" in str(e) and "move it away" in str(e), str(e)''',
    '''            assert "has no provenance" in str(e) and "made by other code or by an older version of this module" in str(e) and "move it away" in str(e) and "interrupted" not in str(e), str(e)''')
rep('''        a = ns(control_ends_dump=p, control_card=toy_card_file(d, 0.6, 0.5))
        assert J.precheck(a, setup, 0, False)[0]["ctl_ends"] == p, "the R1 line''', '''        a = ns(control_ends_dump=p, control_card=toy_card_file(d, 0.6, 0.5))
        assert J.precheck(a, setup, 0, False)[0]["ctl_ends"] == p, "the R1 line''')

# ---------------------------------------------------------------------------------------------------- new tests after the provenance-before-the-model test: TEST artefact rule, rows in the precheck, GPU gate, stub smoke, early control
rep('''def test_card_arms_and_results_md():''', '''def test_on_test_a_flagged_dump_without_provenance_is_one_of_e321s_artefacts_and_rows_are_validated_in_the_precheck():
    """TEST: a flagged dump without provenance must be one of E321's hashed artefacts (else it could be any file); on VAL it need not. `precheck` validates the rows of every existing dump too, so a bad dump of seed 1 does not show up after seed 0's
    dumps were made"""
    setup = J.SETUPS["e324_s_s"]
    val_rows, test_rows = J.split_rows("val").tolist(), J.split_rows("test").tolist()
    with tempfile.TemporaryDirectory() as t:
        root = os.path.join(t, "proj")
        os.makedirs(os.path.join(root, "data"))
        os.makedirs(os.path.join(root, "runs", "e324_s_s0"))
        json.dump(dict(val=val_rows, test=test_rows), open(os.path.join(root, "data", "splits.json"), "w"))
        json.dump(dict(seed=0), open(os.path.join(root, "runs", "e324_s_s0", "config.json"), "w"))
        e321, other = os.path.join(root, "temp", "e321", "test"), os.path.join(root, "temp", "other")
        os.makedirs(e321)
        os.makedirs(other)
        full_t, full_v = toy_dump(split="test"), toy_dump(split="val")
        paths = dict(ends=os.path.join(e321, "ends_e312_p_s0.pkl"), b=os.path.join(e321, "predfull_e273_p_s0.pkl"), other=os.path.join(other, "ends.pkl"), other_val=os.path.join(other, "ends_val.pkl"))
        for k in ("ends", "b", "other"):
            pickle.dump(full_t, open(paths[k], "wb"))
        pickle.dump(full_v, open(paths["other_val"], "wb"))
        card = toy_card_file(root, 0.6, 0.5)
        want = {"temp/e321/test/ends_e312_p_s0.pkl": "x", "temp/e321/test/predfull_e273_p_s0.pkl": "x"}
        ns = lambda **k: argparse.Namespace(**dict(dict(split="test", max=0, device="cpu", run_prefix="e324_s_s", b_dump=paths["b"], control_ends_dump=paths["ends"], control_card=card), **k))  # noqa: E731
        with patched((J, "ROOT", root), (J, "E321_SHA256", want), (J, "pyramid_same", lambda *x: True), (J, "provenance", PROV), (J, "dump_b", boom), (J, "dump_ends", boom)):
            assert J.precheck(ns(), setup, 0, False)[0]["ends"].endswith("ends_e324_s_s0.pkl"), "E321's own dumps pass on TEST"
            assert raises(RuntimeError, J.precheck, ns(control_ends_dump=paths["other"]), setup, 0, False), "TEST: a pair dump without provenance that is not E321's is refused"
            assert raises(RuntimeError, J.precheck, ns(b_dump=paths["other"]), setup, 0, False), "TEST: so is a B dump"
            json.dump(dict(provenance=PROV("e312_p_s0", "cpu")), open(paths["other"][:-4] + "_head.json", "w"))
            assert J.precheck(ns(control_ends_dump=paths["other"]), setup, 0, False), "a dump WITH provenance (made by this module) is checked, not refused as foreign"
            assert J.precheck(ns(split="val", control_ends_dump=paths["other_val"], b_dump=paths["other_val"]), setup, 0, False), "on VAL a flagged dump of another origin passes"
            pickle.dump(full_t[:-1], open(paths["ends"], "wb"))
            assert raises(RuntimeError, J.precheck, ns(), setup, 0, False), "E321's pair dump with a missing frame: refused in the precheck, before anything is dumped"
            pickle.dump(full_t, open(paths["ends"], "wb"))
            own = os.path.join(root, "temp", "e343", "test", "ends_e324_s_s0.pkl")                      # a default dump that exists with the right provenance but a bad row order
            os.makedirs(os.path.dirname(own))
            pickle.dump(full_t[::-1], open(own, "wb"))
            json.dump(dict(provenance=PROV("e324_s_s0", "cpu")), open(own[:-4] + "_head.json", "w"))
            assert raises(RuntimeError, J.precheck, ns(), setup, 0, False), "a default dump with the rows in another order: refused in the precheck"


def test_a_dump_on_a_gpu_waits_for_room_like_jends_eval():
    """D100 / hard rule 6: a dump on a device other than cpu calls `gpu_gate.wait_for_room` (as `jends_eval.py dump` does) before it loads anything; on the cpu it does not"""
    calls = []
    stub = types.ModuleType("gpu_gate")
    stub.wait_for_room = lambda who="", **k: calls.append(who)

    class Stop(Exception):
        pass

    def stop(split):
        raise Stop()
    old = sys.modules.get("gpu_gate")
    sys.modules["gpu_gate"] = stub
    try:
        with tempfile.TemporaryDirectory() as d, patched((J, "provenance", PROV), (J, "frames_of", stop), (J, "EXTRA_WRITABLE", (d,))):
            for dev in ("cuda", "cpu"):
                for fn, run in ((J.dump_b, "e273_p_s0"), (J.dump_ends, "e324_s_s0")):
                    assert raises(Stop, fn, "val", run, os.path.join(d, f"{dev}_{run}.pkl"), dev)
    finally:
        if old is None:
            sys.modules.pop("gpu_gate")
        else:
            sys.modules["gpu_gate"] = old
    assert calls == ["jtest_eval dump e273_p_s0", "jtest_eval dump e324_s_s0"], calls


def test_dump_b_and_dump_ends_on_a_stub_model():
    """the dump functions run end to end on a stub model, loader and predict (the rehearsal covers the real ones): the sidecar with the provenance exists BEFORE the pickle is written, the pickle goes in without overwriting, the pair dump has the attached
    ends and the position scales, the head file has the provenance and the counts"""
    import torch
    import jends_eval as JE
    import lane_gt_data as LD
    import qendnet as QEN
    import qgraph_readout as RO
    import qpredict as QP
    rows = J.split_rows("val").tolist()[:6]
    frames = [Frame(r, 0) for r in rows]
    ds = types.SimpleNamespace(bd=types.SimpleNamespace(rows=rows), ks=list(range(6)))
    batches = [dict(x=torch.zeros(len(rows[lo: lo + 4]), 1), token=[f"t{x}" for x in rows[lo: lo + 4]], row=torch.tensor(rows[lo: lo + 4])) for lo in (0, 4)]
    order = []

    def model(x, gt=None):
        k = x.shape[0]
        return None, None, dict(kp_pos=torch.zeros(k, 4, 2), kp_dir=torch.ones(k, 4, 2), kp_p=torch.ones(k, 4), valid=torch.ones(k, 2, 2, dtype=torch.bool), logit=torch.zeros(k, 2, 2), allowed=torch.ones(k, 2, 2),
                                scale=torch.zeros(k, 2, 2, 2), att_pos=torch.ones(k, 4, 2), att_dir=torch.ones(k, 4, 2), att_p=torch.ones(k, 4), kp_logsig=torch.zeros(k, 4))
    counts = dict(ent_hit=1, ent_n=2, ent_ok=1, ent_kp=2, ext_hit=1, ext_n=2, ext_ok=1, ext_kp=2, pairs_hit=1, pairs_n=2, pairs_ok=1, pairs_kp=2)
    with tempfile.TemporaryDirectory() as d:
        b_out, e_out = os.path.join(d, "b.pkl"), os.path.join(d, "e.pkl")

        def dump_full(preds, rows_, path):
            order.append(("dump_full", os.path.exists(b_out[:-4] + "_head.json")))
            pickle.dump([dict(row=int(r)) for r in rows_], open(path, "wb"))
        orig_save = J.save_pickle

        def save_pickle(path, obj, overwrite=True):
            order.append(("save_pickle", os.path.exists(e_out[:-4] + "_head.json"), overwrite))
            orig_save(path, obj, overwrite)
        with patched((J, "provenance", PROV), (J, "frames_of", lambda split: frames), (J, "EXTRA_WRITABLE", (d,)), (J, "save_pickle", save_pickle), (LD, "loader", lambda *a, **k: (ds, batches)), (QP, "load_model", lambda run_dir, device: "model"),
                     (QP, "predict", lambda m, b: [dict(row=int(x)) for x in b["row"]]), (RO, "dump_full", dump_full), (JE, "load_model", lambda run, device: (model, {})), (QEN, "end_counts", lambda oe, b, thr: counts)):
            J.dump_b("val", "e273_p_s0", b_out, "cpu", 0)
            J.dump_ends("val", "e324_s_s0", e_out, "cpu", 0)
            assert raises(SystemExit, J.dump_b, "val", "e273_p_s0", b_out) and raises(SystemExit, J.dump_ends, "val", "e324_s_s0", e_out), "an existing dump is not dumped again"
        assert order == [("dump_full", True), ("save_pickle", True, False)], "the sidecar is there before the pickle; the pair dump's pickle is written without overwriting"
        assert [d_["row"] for d_ in pickle.load(open(b_out, "rb"))] == rows
        head_b, head_e = json.load(open(b_out[:-4] + "_head.json")), json.load(open(e_out[:-4] + "_head.json"))
        assert head_b["provenance"] == PROV("e273_p_s0", "cpu") and head_b["frames"] == 6 and head_e["provenance"] == PROV("e324_s_s0", "cpu") and head_e["frames"] == 6 and head_e["ent_recall"] == 0.5
        ends = pickle.load(open(e_out, "rb"))
        assert [e["row"] for e in ends] == rows and all(k in ends[0] for k in ("att_pos", "att_dir", "att_p", "logsig", "scale", "kp_pos", "logit")), "the attached ends and the position scales are stored"


def test_the_control_is_checked_before_any_new_dump():
    """a KILL after hours must not be possible: with B's dump and E312's pair dump named by flags, the plain graph and E312's snapped read-out are scored and checked FIRST (a wrong control card stops the run before one dump is made); `score_seed` then
    scores only the other read-outs and takes the early counts over"""
    import qgraph_eval as QE
    full = toy_dump()
    frames = [Frame(r["row"], r["row"] % 3) for r in full]

    def run(card_base):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as d2:
            calls = []
            for name in ("b", "ctl_ends"):
                pickle.dump(full, open(os.path.join(d, name + ".pkl"), "wb"))
            card = toy_card_file(d, 0.6, card_base)
            a = toy_args(max=0, boot=20, seeds=[0], b_dump=os.path.join(d, "b.pkl"), control_ends_dump=os.path.join(d, "ctl_ends.pkl"), control_card=card)

            def fake(kind):
                def f(split, run_, out, device="cpu", n=0, **k):
                    calls.append((kind, run_))
                    pickle.dump(full, open(out, "wb"))
                    json.dump(dict(provenance=PROV(run_, device)), open(out[:-4] + "_head.json", "w"))
                return f

            def spy(fr, graphs, ends, variants, thr=J.THR, n=0, log=print):
                calls.append(("score_dumps", [v["name"] for v in variants]))
                return toy_sc(variants, XS, fr)
            buf, old = io.StringIO(), sys.stdout
            with patched((J, "EXTRA_WRITABLE", (d, d2)), (J, "dump_dir", lambda split, n=0: d), (J, "out_dir", lambda st, split, n=0: d2), (J, "frames_of", lambda split: frames), (J, "guard_e321", lambda w: None),
                         (J, "code_sha256", lambda entry=None: {"a.py": "1"}), (J, "environment", lambda *x: ENV), (J, "repo_modules", lambda: set()), (J, "provenance", PROV), (J, "pyramid_same", lambda *x: True),
                         (J, "check_stored", lambda *x, **k: (True, [], [])), (J, "dump_b", fake("dump_b")), (J, "dump_ends", fake("dump_ends")), (J, "score_dumps", spy), (QE, "graph_numbers", toy_numbers)):
                sys.stdout = buf
                try:
                    rc = J.cmd_all(a)
                except SystemExit as e:
                    rc = e
                finally:
                    sys.stdout = old
            card0 = json.load(open(os.path.join(d2, "card_s0.json"))) if os.path.exists(os.path.join(d2, "card_s0.json")) else None
            return rc, calls, card0
    rc, calls, _ = run(0.51)
    assert isinstance(rc, SystemExit) and "KILL" in str(rc) and calls == [("score_dumps", ["plain", "e312_snapped"])], (rc, calls)                 # the card says 0.51, the plain graph is 0.5: killed, no dump was made
    rc, calls, card0 = run(0.5)
    assert rc == 0 and calls[0] == ("score_dumps", ["plain", "e312_snapped"]) and [c[0] for c in calls[1:4]] == ["dump_ends", "dump_b", "score_dumps"], calls
    assert "plain" not in calls[3][1] and "e312_snapped" not in calls[3][1] and "snapped" in calls[3][1] and "stack" in calls[3][1], "the second pass scores the other read-outs only"
    assert card0["compare"]["control"]["repro"]["ok"] and card0["compare"]["control"]["per_frame_counts_used"] and card0["variants"]["snapped"]["numbers"]["base"]["topo_F1"] == 0.5 and "e312_snapped" not in card0["variants"], "the early counts are taken over"


def test_card_arms_and_results_md():''')

# ---------------------------------------------------------------------------------------------------- cmd_all block: toy_cmd_all, the code pin, the TEST line, undigested modules, the early protected-folder refusal
a = s.index("# ------------------------------------------------------------------------------------------------------------------------------ cmd_all end to end")
b = s.index("def test_variants_are_the_stored_readouts():")
s = s[:a] + '''# ------------------------------------------------------------------------------------------------------------------------------ cmd_all end to end
ENV = dict(files={"data/splits.json": "s1", "val_tokens.json": "t1", "poses_val.npz": "p1", "runs/e324_s_s0/config.json": "c0"}, versions={"torch": "2.8.0", "numpy": "2.0.0", "shapely": "2.0.0"})


def toy_args(**k):
    return argparse.Namespace(**dict(dict(split="val", max=873, run_prefix="e324_s_s", seeds=[0, 1], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="", expect_code_digest=None, expect_code_card=None), **k))


def toy_test_args(d, **k):
    """the TEST line's arguments (the rehearsal card is `d/rehearsal.json`)"""
    return toy_args(**dict(dict(split="test", max=0, boot=1000, b_dump="b{seed}.pkl", control_ends_dump="e{seed}.pkl", control_card="c{seed}.json", test_id="E343.t", expect_code_card=os.path.join(d, "rehearsal.json")), **k))


def rehearsal_card(files, **over):
    """the card of a good VAL rehearsal (full, passed, no drift, a digest that is that of its file list)"""
    return dict(dict(run="e324_s_s0", seed=0, split="val", frames=873, rehearsal_check=dict(ok=True, lines=[], failed=[]), code_sha256=files, code_digest=J.code_digest(files), code_drift=[], environment=ENV, device="cpu"), **over)


def toy_cmd_all(d, a, counts1=True, check=None, codes=None, env=None, loaded=None, scenes=None):
    """`cmd_all` with the dumps, the checks of the inputs and the scorer replaced -> (rc or the SystemExit, text, files of the output folder, calls of guard_e321, calls of score_seed); `toy_cmd_all.order` lists precheck / score calls"""
    import qgraph_eval as QE
    setup = J.SETUPS["e324_s_s"]
    n_scenes = scenes or (21 if a.split == "test" else 3)
    frames = [Frame(r["row"], r["row"] % n_scenes) for r in toy_dump(split=a.split)]
    scored, guards, buf, order = [], [], io.StringIO(), []
    code_iter = iter(codes or [{"a.py": "1"}, {"a.py": "1"}])
    load_iter = iter(loaded or [set(), set()])

    def precheck(a_, st, seed, stack):
        order.append(("precheck", seed))
        return None, None

    def score(a_, st, seed, fr, stack, pre, early=None):
        order.append(("score", seed))
        scored.append(seed)
        variants = J.variants_of(setup, True, True)
        sc = toy_sc(variants, dict(XS, snapped=0.60 + (0.016 if seed == 0 else 0.014), e312_snapped=0.6), fr)
        ok = seed == 0 or counts1
        control = dict(card="control.json", numbers=toy_numbers([dict(x=0.6)]), plain_check=dict(max_abs_diff=0.0, where="", tolerance="-", ok=True), repro=dict(max_abs_diff=0.0, where="", ok=ok), counts=sc["counts"]["e312_snapped"] if ok else None)
        return dict(seed=seed, run=f"e324_s_s{seed}", sc=sc, variants=variants, control=control)
    pats = [(J, "EXTRA_WRITABLE", (d,)), (J, "frames_of", lambda split: frames), (J, "precheck", precheck), (J, "control_early", lambda *x: None), (J, "score_seed", score), (J, "out_dir", lambda st, split, n=0: d), (J, "guard_e321", guards.append),
            (QE, "graph_numbers", toy_numbers), (J, "code_sha256", lambda entry=None: next(code_iter)), (J, "environment", lambda *x: env or ENV), (J, "repo_modules", lambda: next(load_iter)), (J, "test_gate_open", lambda *x, **k: (True, ""))]
    if check:
        pats.append((J, "check_stored", check))
    with patched(*pats):
        old, sys.stdout = sys.stdout, buf
        try:
            rc = J.cmd_all(a)
        except SystemExit as e:
            rc = e
        finally:
            sys.stdout = old
    toy_cmd_all.order = order
    return rc, buf.getvalue(), sorted(os.listdir(d)), guards, scored


def test_cmd_all_end_to_end_on_toy_data():
    """`cmd_all`: the prechecks of BOTH seeds before the first seed is scored, both seeds scored, the intervals dropped on BOTH seeds when one seed's E312 counts did not reproduce, cards and results.md written with the device, the code hashes and the
    environment, E321's hashes checked before and after, the label printed; a changed code file is named, the rehearsal branch compares with the stored cards (exit 1 on a miss, the hint when only arm k fails), existing results are not overwritten,
    a failed precheck scores nothing"""
    a = toy_args()
    with tempfile.TemporaryDirectory() as d:
        rc, text, files, guards, scored = toy_cmd_all(d, a)
        card0, md = json.load(open(os.path.join(d, "card_s0.json"))), open(os.path.join(d, "results.md")).read()
        assert rc == 0 and files == ["card_s0.json", "card_s1.json", "results.md"] and guards == ["before the run", "after the run"] and "paired" in card0["compare"]["vs_e312"]["w"] and scored == [0, 1]
        assert toy_cmd_all.order == [("precheck", 0), ("precheck", 1), ("score", 0), ("score", 1)], "every precheck comes before the first score: a bad input of seed 1 shows before seed 0's dumps are made"
        assert "**FULL RETENTION**" in md and "READING: FULL RETENTION" in text and "CODE DRIFT" not in text, text[-300:]
        assert f"code digest: {J.code_digest({'a.py': '1'})} (1 files)" in text.splitlines()[0], "the digest is printed at the start of every run"
        assert card0["device"] == "cpu" and card0["code_sha256"] == {"a.py": "1"} and card0["code_digest"] == J.code_digest({"a.py": "1"}) and card0["code_drift"] == [] and card0["environment"] == ENV and card0["undigested_modules"] == []
        assert "Device of the dumps and the scoring: cpu" in md and "no change of those files during the run" in md and "Environment: torch 2.8.0, numpy 2.0.0, shapely 2.0.0" in md
    with tempfile.TemporaryDirectory() as d:                                                           # seed 1 did not reproduce: no interval on seed 0 either
        rc, text, files, guards, scored = toy_cmd_all(d, a, counts1=False)
        card0, md = json.load(open(os.path.join(d, "card_s0.json"))), open(os.path.join(d, "results.md")).read()
        assert "paired" not in card0["compare"]["vs_e312"]["w"] and "**FULL RETENTION (without interval)**" in md and "NO interval against E312 on any seed" in text
    with tempfile.TemporaryDirectory() as d:                                                           # a lane code file changed during the run: named, not fatal
        rc, text, files, guards, scored = toy_cmd_all(d, a, codes=[{"a.py": "1", "b.py": "1"}, {"a.py": "2", "b.py": "1"}])
        card0, md = json.load(open(os.path.join(d, "card_s0.json"))), open(os.path.join(d, "results.md")).read()
        assert rc == 0 and "CODE DRIFT: a.py" in text and card0["code_drift"] == ["a.py"] and "CODE DRIFT during the run: a.py" in md
    rehearsal = toy_args(max=0)
    for verdict, rc_want, shown, hint in (((False, ["CHECK x"], ["snapped"]), 1, "REHEARSAL CHECK: FAIL", False), ((False, [], ["stack"]), 1, "REHEARSAL CHECK: FAIL", True), ((True, [], []), 0, "REHEARSAL CHECK: PASS", False)):
        with tempfile.TemporaryDirectory() as d:
            rc, text, files, guards, scored = toy_cmd_all(d, rehearsal, check=lambda *x, **k: verdict)
            card0 = json.load(open(os.path.join(d, "card_s0.json")))
            assert rc == rc_want and shown in text and ("only arm k fails" in text) == hint and card0["rehearsal_check"]["ok"] == verdict[0] and files == ["card_s0.json", "card_s1.json", "results.md"], (verdict, rc, text[-200:])
        with tempfile.TemporaryDirectory() as d:                                                       # the rehearsal rule is not applied to a smoke run
            rc, text, *_ = toy_cmd_all(d, a, check=lambda *x, **k: (False, [], ["snapped"]))
            assert rc == 0 and "REHEARSAL CHECK" not in text
    with tempfile.TemporaryDirectory() as d:                                                           # results are written once: nothing scored, the existing file untouched
        open(os.path.join(d, "card_s0.json"), "w").write("keep")
        rc, text, files, guards, scored = toy_cmd_all(d, rehearsal, check=lambda *x, **k: (True, [], []))
        assert isinstance(rc, SystemExit) and scored == [] and guards == [] and open(os.path.join(d, "card_s0.json")).read() == "keep", "refused before any work"
        rc, text, files, guards, scored = toy_cmd_all(d, a)                                           # a smoke run may overwrite
        assert rc == 0 and json.load(open(os.path.join(d, "card_s0.json")))["seed"] == 0
    with tempfile.TemporaryDirectory() as d:                                                           # a failed precheck: refused before the frames are read and before anything is scored
        with patched((J, "frames_of", boom), (J, "score_seed", boom), (J, "guard_e321", lambda w: None), (J, "out_dir", lambda st, split, n=0: d), (J, "precheck", lambda *x: sys.exit("precheck refused")),
                     (J, "code_sha256", lambda entry=None: {"a.py": "1"}), (J, "environment", lambda *x: ENV), (J, "repo_modules", lambda: set())):
            assert raises(SystemExit, J.cmd_all, a)


def test_the_code_pin_cannot_be_bypassed():
    """on TEST the pin is the card of a good VAL rehearsal and nothing else: the card is REQUIRED (an empty string is no card), the expected digest is taken from it (no second typed value), a typed digest is an extra check that an empty string does not skip;
    the card must be a full, passed, undisturbed VAL rehearsal of this system with a digest that is its own file list's and an environment record; other code or another environment is refused before any work"""
    files = {"a.py": "1", "b.py": "1"}

    def run(card=None, **k):
        with tempfile.TemporaryDirectory() as d:
            json.dump(card if card is not None else rehearsal_card(files), open(os.path.join(d, "rehearsal.json"), "w"))
            out = toy_cmd_all(d, toy_test_args(d, **k), codes=[files, files])
            return out + ((json.load(open(os.path.join(d, "card_s0.json"))) if os.path.exists(os.path.join(d, "card_s0.json")) else None),)
    rc, text, out_files, guards, scored, card0 = run()
    assert rc == 0 and scored == [0, 1] and guards == ["before the run", "after the run"] and card0["code_digest"] == J.code_digest(files) and card0["environment"] == ENV, (rc, text[-300:])
    assert raises(RuntimeError, run, expect_code_card=None) and raises(RuntimeError, run, expect_code_card=""), "TEST without the rehearsal card: refused"
    rc, text, out_files, guards, scored, _ = run(expect_code_digest="")
    assert isinstance(rc, SystemExit) and scored == [] and guards == [], "an empty typed digest refuses (it does not skip the check)"
    assert run(expect_code_digest=J.code_digest(files))[0] == 0 and run(expect_code_digest=J.code_digest(files)[:16])[0] == 0, "the typed digest is an extra check: the right one passes"
    rc, *_ = run(expect_code_digest="0" * 64)
    assert isinstance(rc, SystemExit), "a wrong typed digest refuses although the card is right"
    bad_cards = [rehearsal_card(files, code_drift=["a.py"]), rehearsal_card(files, rehearsal_check=dict(ok=False, lines=[], failed=["snapped"])), rehearsal_card(files, frames=5, rehearsal_check=None), rehearsal_card(files, split="test"),
                 rehearsal_card(files, code_digest="0" * 64), rehearsal_card({"a.py": "1", "b.py": "9"}, code_digest=J.code_digest(files)), rehearsal_card(files, run="e312_p_s0"), rehearsal_card(files, environment=None)]
    for name, bad in zip(("code drift", "failed rehearsal", "smoke card", "TEST card", "tampered digest", "tampered file list", "another system", "no environment"), bad_cards):
        assert raises(RuntimeError, run, card=bad), name
    no_drift_key = rehearsal_card(files)
    del no_drift_key["code_drift"]
    assert raises(RuntimeError, run, card=no_drift_key), "a card without the drift record cannot vouch for the code"
    rc, *_ = run(card=rehearsal_card({"a.py": "1", "b.py": "0", "c.py": "5"}))
    assert isinstance(rc, SystemExit) and "changed files against" in str(rc) and "b.py, c.py" in str(rc), "other code than the card's: refused, the files named"
    rc, text, out_files, guards, scored, _ = run(card=rehearsal_card(files, environment=dict(ENV, versions=dict(ENV["versions"], torch="9.9"))))
    assert isinstance(rc, SystemExit) and "environment differs" in str(rc) and "versions/torch" in str(rc) and scored == [] and guards == [], "another environment than the rehearsal's: refused before any work"
    rc, *_ = run(card=rehearsal_card(files, environment=dict(ENV, files=dict(ENV["files"], **{"poses_val.npz": "other"}))))
    assert isinstance(rc, SystemExit) and "files/poses_val.npz" in str(rc)


def test_the_test_line_is_fixed_by_the_card():
    """on TEST the card fixes 1000 bootstrap draws, the seeds 0 1 and the control inputs (E321's dumps and card), and 21 scenes in the frames"""
    files = {"a.py": "1"}

    def run(scenes=None, **k):
        with tempfile.TemporaryDirectory() as d:
            json.dump(rehearsal_card(files), open(os.path.join(d, "rehearsal.json"), "w"))
            return toy_cmd_all(d, toy_test_args(d, **k), codes=[files, files], scenes=scenes)[0]
    assert run() == 0
    for k in (dict(boot=999), dict(boot=1001), dict(seeds=[0]), dict(seeds=[1, 0]), dict(seeds=[0, 1, 2]), dict(b_dump=None), dict(b_dump=""), dict(control_ends_dump=None), dict(control_ends_dump=""), dict(control_card=None), dict(control_card="")):
        assert raises(RuntimeError, run, **k), k
    assert raises(RuntimeError, run, scenes=1) and raises(RuntimeError, run, scenes=20), "a degenerate or other scene column on TEST: refused (a single scene id would collapse the paired interval to a point)"
    assert run(scenes=21) == 0


def test_cmd_all_refuses_e321s_output_folder_before_any_work():
    """E312 on TEST would write E321's cards: refused with the REAL `out_dir`, before anything else is touched"""
    a = toy_test_args("unused", run_prefix="e312_p_s", expect_code_card="rehearsal.json")
    with patched((J, "test_gate_open", lambda *x, **k: (True, "")), (J, "frames_of", boom), (J, "guard_e321", boom), (J, "code_sha256", boom), (J, "precheck", boom), (J, "repo_modules", boom), (J, "environment", boom)):
        try:
            J.cmd_all(a)
            raise AssertionError("E312 on TEST must be refused")
        except SystemExit as e:
            assert "lies under" in str(e), str(e)


def test_undigested_modules_the_environment_and_the_repository_modules():
    """a repository module that a run loads and the digest does not cover is named in the log, the cards and results.md; modules loaded before the run (the gate's) and modules of the digest are not; `repo_modules` leaves out temp/, scratch and the GT package;
    the environment record hashes the small files and names absent ones"""
    with tempfile.TemporaryDirectory() as d:
        rc, text, files, guards, scored = toy_cmd_all(d, toy_args(), codes=[{"a.py": "1", "b.py": "1"}, {"a.py": "1", "b.py": "1"}], loaded=[{"a.py", "x.py"}, {"a.py", "x.py", "new_mod.py", "b.py"}])
        card0, md = json.load(open(os.path.join(d, "card_s0.json"))), open(os.path.join(d, "results.md")).read()
        assert rc == 0 and card0["undigested_modules"] == ["new_mod.py"] and "UNDIGESTED MODULE: new_mod.py" in text and "UNDIGESTED MODULE: new_mod.py" in md and "x.py" not in md.replace("UNDIGESTED MODULE: new_mod.py", ""), (card0["undigested_modules"], text[-300:])
    mods = J.repo_modules()
    assert any(x.endswith("evaluation/jtest_eval.py") for x in mods) and not any(x.split("/")[0] == "GtGenerator" or "/temp/" in "/" + x or "_session_scratch_" in x for x in mods)
    repo = os.path.realpath(os.path.join(J.ROOT, ".."))
    fake = {"fake_ok": os.path.join(J.ROOT, "evaluation", "zz_fake_ok.py"), "fake_temp": os.path.join(J.ROOT, "temp", "zz_fake.py"), "fake_scratch": os.path.join(J.ROOT, "evaluation", "_session_scratch_", "zz_fake.py"),
            "fake_gt": os.path.join(repo, "GtGenerator", "zz_fake.py"), "fake_rel": "zz_relative.py", "fake_out": os.path.join(os.path.dirname(repo), "zz_fake_outside.py")}
    try:
        for name, path in fake.items():
            m = types.ModuleType(name)
            m.__file__ = path
            sys.modules[name] = m
        got = J.repo_modules()
    finally:
        for name in fake:
            sys.modules.pop(name, None)
    assert any(x.endswith("evaluation/zz_fake_ok.py") for x in got) and not any("zz_fake.py" in x or "zz_relative" in x or "zz_fake_outside" in x for x in got), sorted(x for x in got if "zz_" in x)
    with tempfile.TemporaryDirectory() as t:                                                           # the environment: small files hashed, absent ones None, the versions
        import qgraph_eval as QE
        os.makedirs(os.path.join(t, "data"))
        os.makedirs(os.path.join(t, "feats"))
        os.makedirs(os.path.join(t, "runs", "e324_s_s0"))
        for rel, c in (("data/splits.json", b"S"), ("feats/val_tokens.json", b"T"), ("poses_val.npz", b"P"), ("runs/e324_s_s0/config.json", b"C")):
            open(os.path.join(t, rel), "wb").write(c)
        with patched((J, "ROOT", t), (QE.LS, "B123", t)):
            env = J.environment(J.SETUPS["e324_s_s"], "e324_s_s", [0])
        assert env["files"]["data/splits.json"] == hashlib.sha256(b"S").hexdigest() and env["files"]["val_tokens.json"] == hashlib.sha256(b"T").hexdigest() and env["files"]["poses_val.npz"] == hashlib.sha256(b"P").hexdigest()
        assert env["files"]["runs/e324_s_s0/config.json"] == hashlib.sha256(b"C").hexdigest() and env["files"]["runs/e273_p_s0/config.json"] is None and env["files"]["runs/e292_c2_s0/config.json"] is None and env["files"]["runs/e312_p_s0/config.json"] is None
        assert set(env["versions"]) == {"torch", "numpy", "shapely"} and env["versions"]["numpy"] == np.__version__
        assert J.environment_diff(env, env) == [] and J.environment_diff(env, dict(env, versions=dict(env["versions"], numpy="0"))) == ["versions/numpy"] and J.environment_diff(env, dict(env, files={})) == sorted(f"files/{k}" for k in env["files"] if env["files"][k] is not None)


def test_expect_code_digest_refuses_other_code_before_any_work():
    """the optional extra check `--expect-code-digest` (the full digest or its first 16 characters) and the card's digest on VAL: other code is refused before the E321 hashes, the precheck and any dump; the card names the changed files"""
    files = {"a.py": "1", "b.py": "1"}
    digest = J.code_digest(files)
    with tempfile.TemporaryDirectory() as d:
        rc, text, _, guards, scored = toy_cmd_all(d, toy_args(expect_code_digest=digest), codes=[files, files])
        assert rc == 0 and scored == [0, 1] and text.startswith(f"code digest: {digest} (2 files)"), text[:200]
        rc, text, _, guards, scored = toy_cmd_all(d, toy_args(expect_code_digest=digest[:16]), codes=[files, files])
        assert rc == 0, "the first 16 characters (as results.md prints them) are enough"
    for bad in (digest[:12], "", "0" * 64, digest[:16] + "0"):
        with tempfile.TemporaryDirectory() as d:
            rc, text, out_files, guards, scored = toy_cmd_all(d, toy_args(expect_code_digest=bad), codes=[files, files])
            assert isinstance(rc, SystemExit) and scored == [] and guards == [] and "--expect-code-card" in str(rc), (bad, rc)
    with tempfile.TemporaryDirectory() as d:                                                           # a typed digest with a card of other code: the files are named
        json.dump(rehearsal_card({"a.py": "1", "b.py": "0", "c.py": "5"}), open(os.path.join(d, "card.json"), "w"))
        rc, *_ = toy_cmd_all(d, toy_args(expect_code_digest="0" * 64, expect_code_card=os.path.join(d, "card.json")), codes=[files, files])
        assert isinstance(rc, SystemExit) and f"changed files against {os.path.join(d, 'card.json')}: b.py, c.py" in str(rc), str(rc)
    with tempfile.TemporaryDirectory() as d:                                                           # a typed digest that is not the card's although the card's hashes are the current ones
        json.dump(rehearsal_card(files), open(os.path.join(d, "same.json"), "w"))
        rc, *_ = toy_cmd_all(d, toy_args(expect_code_digest="0" * 64, expect_code_card=os.path.join(d, "same.json")), codes=[files, files])
        assert isinstance(rc, SystemExit) and "equal the current ones" in str(rc), str(rc)


# ------------------------------------------------------------------------------------------------------------------------------ the stored cards
''' + s[b:]

open(P, "w", encoding="utf-8").write(s)
print("ok")
