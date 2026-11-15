"""E353 tests of the narrowing (`jnarrow.py`, `train/jnarrow_train.py`, `train/jnarrow_eval.py`), CPU only (the GPU is hidden), 2 threads, the stored E324 s model, 8 real VAL frames and real TRAIN frames; about 7 minutes on a loaded machine.

    wsl bash tools/env/_py14.sh grid/test_jnarrow.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

Known answers: the groups removed are the lowest summed-importance groups of a numpy reference on the calibration (not the probe's per-channel set); the refusals; the narrowed maps head against the hook-ablated model (`jprune_probe.ablate`) that removes the SAME channels:
the output of the consumer conv agrees inside the map to float tolerance, the border ring differs by exactly the zero-padding term (computed by hand from the weights), the heat and the keypoints do not depend on the branch at all (difference 0), the pair logits move by
less than 1e-2 and no sign flips; the MACs and parameters of the narrowed model against `jprune_cost.Cost` and torch FlopCounterMode (1.10592 GMACs by arithmetic); the spec round trip (a checkpoint is read without the calibration file; one new bias tensor); the loader of a stored run is the stored loader;
the dump wrapper: the UNNARROWED stored run through the wrapper equals `jends_eval.dump` frame for frame, a narrowed run dumps what its model computes, `jends_eval.score` accepts the narrowed checkpoint; the frozen set of the trainer, its checks at the start (flags, seed, BFLC_SIG, narrowed init), a fine-tune of
2 steps, the record in the run folder (spec, calibration copy and sha256), the resume (step 2 -> 3), the refusal of another spec, the n1z run (`--steps 0`) and the cp run (no spec); the timing machinery of `jtiming.py` on a narrowed toy model. The stored runs' folders must stay as they were (checked at the end).
"""
import copy
import json
import os
import pickle
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                      # this test never touches the GPU
os.environ["BFLC_SIG"] = "1.0"                                               # the sigma of the maps run (the trainer checks it)
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jnarrow as JN  # noqa: E402
import jprune_cost as PC  # noqa: E402
import jprune_probe as PP  # noqa: E402

torch.set_num_threads(2)
ROOT = JN.ROOT
_FX = {}
_TMP = tempfile.TemporaryDirectory(prefix="jnarrow_test_")


def fx():
    """the shared fixture: the stored E324 s on the CPU, 8 VAL frames (one batch), a calibration on them (kept in a temporary file), the n1 spec (jm.m05.hid, 75 % removed) resolved on it, the narrowed model"""
    if not _FX:
        full, args = JN.JEV.load_model("e324_s_s0", "cpu")
        cache = PP.build_cache("val", workers=0, max_frames=8)
        taps = PP.tap_table(full)
        calib = PP.calibrate(full, cache, "cpu", taps)
        cpath = os.path.join(_TMP.name, "calib.pt")
        PP.save_calib(calib, cpath)
        spec = {"calib": cpath, "edits": [{"unit": "jm.m05.hid", "frac": 0.75}]}
        narrowed, _ = JN.build_narrowed("e324_s_s0", spec)
        _FX.update(full=full, args=args, cache=cache, taps=taps, calib=calib, cpath=cpath, spec=narrowed.narrow_spec, narrowed=narrowed)
    return _FX


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    raise RuntimeError(f"{fn.__name__}{a} should have raised {exc.__name__}")


def listing(run):
    d = os.path.join(ROOT, "runs", run)
    return sorted((f, os.path.getmtime(os.path.join(d, f))) for f in os.listdir(d))


def same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def save_run(root, name, model, args):
    os.makedirs(os.path.join(root, "runs", name), exist_ok=True)
    torch.save(dict(state=model.state_dict(), args=args), os.path.join(root, "runs", name, "model.pt"))


class Root:
    """`jends_eval.ROOT` moved to a temporary folder (its dump writes into <ROOT>/runs/<run>, the stored runs' folders must not get a file)"""

    def __init__(self, root):
        self.root = root

    def __enter__(self):
        self.old, JN.JEV.ROOT = JN.JEV.ROOT, self.root

    def __exit__(self, *e):
        JN.JEV.ROOT = self.old


# ---------------------------------------------------------------- the choice of the groups
def test_groups_are_the_lowest_importance_groups():
    f = fx()
    s = f["calib"].s["jm.m05.2"].numpy().astype(np.float64)
    gone = sorted(np.argsort(s.reshape(8, 4).sum(1), kind="stable")[:6].tolist())                # numpy reference: 6 of the 8 groups of 4 channels go, the lowest summed importance
    t = f["spec"]["taps"]["jm.m05.2"]
    assert t["removed"] == [c for g in gone for c in range(4 * g, 4 * g + 4)] and t["groups"] == [2, 4], t
    assert f["spec"]["calib_sha256"] == JN.sha256_file(f["cpath"])
    probe = PP.spec_edits({"jm.m05.hid": 0.75}, PP.build_units(f["full"]), f["calib"])[0].idx          # the probe's per-channel set has the same size and is another set
    assert len(probe) == len(t["removed"]) and set(probe) != set(t["removed"])
    r = JN.resolve(f["full"], {"calib": f["cpath"], "edits": [{"unit": "jm.m05.hid", "frac": 0.25}, {"tap": "jm.blocks.1.f.2", "frac": 0.5}]})
    assert len(r["taps"]["jm.m05.2"]["removed"]) == 8 and r["taps"]["jm.m05.2"]["groups"] == [6, 4]
    assert len(r["taps"]["jm.blocks.1.f.2"]["removed"]) == 32 and r["taps"]["jm.blocks.1.f.2"]["groups"] == [4, 8]
    assert JN.resolve(f["full"], {})["taps"] == {} and JN.resolve(f["full"], r) is r
    for e, exc in (({"unit": "jm.stream", "frac": 0.25}, NotImplementedError), ({"unit": "pyr.d0", "frac": 0.25}, NotImplementedError), ({"unit": "jm.head05.c0", "frac": 0.5}, NotImplementedError), ({"unit": "je.pair.1", "frac": 0.25}, NotImplementedError),
                   ({"tap": "jm.blocks.0", "frac": 0.5}, NotImplementedError), ({"unit": "jm.m05.hid", "frac": 1.0}, AssertionError), ({"unit": "jm.m05.hid", "tap": "jm.m05.2", "frac": 0.5}, AssertionError)):
        raises(exc, JN.resolve, f["full"], {"calib": f["cpath"], "edits": [e]})
    raises(AssertionError, JN.resolve, f["full"], {"calib": f["cpath"], "edits": [{"unit": "jm.m05.hid", "frac": 0.5}, {"tap": "jm.m05.2", "frac": 0.25}]})      # a tensor edited twice


# ---------------------------------------------------------------- what the surgery changes
def test_equality_with_the_hook_ablated_model():
    f = fx()
    d = JN.difference(f["full"], f["narrowed"], [f["cache"][0]["x"].float()])
    print({k: v for k, v in d.items()})
    assert d["reader_interior_max"]["jm.m05.3"] < 1e-5, d                                         # the consumer conv, inside the map: float tolerance
    assert d["reader_border_max"]["jm.m05.3"] > 1e-2, d                                           # the zero-padding ring is NOT exact (the claim of the docstring)
    assert d["heat_max"] == 0.0 and d["kp_changed_frac"] == 0.0 and d["kp_pos_max_m"] == 0.0, d   # the 1 m heat and the keypoints do not read the 0.5 m branch
    assert d["logit_max"] < 1e-2 and d["logit_sign_flips"] == 0.0 and d["paint_inner_max"] < 0.2, d


def test_border_difference_is_the_padding_term():
    """at the corner pixel the ablated model has the removed constants on 4 of the 9 kernel taps (zero padding outside), the folded bias on all 9: narrowed - ablated = sum_c mean_c * (the 5 weights that read outside)"""
    f = fx()
    x, spec = f["cache"][0]["x"][:1].float(), f["spec"]
    ablated, _ = PP.ablate(f["full"], JN.edits_of(spec, f["calib"]), f["taps"])
    got, hs = {}, []
    for k, m in (("a", ablated), ("n", f["narrowed"])):
        hs.append(m.jmodel.jm.m05[3].register_forward_hook(lambda _m, _a, o, k=k: got.__setitem__(k, o.detach())))
    with torch.no_grad():
        ablated(x, gt=None)
        f["narrowed"](x, gt=None)
    for h in hs:
        h.remove()
    rem = spec["taps"]["jm.m05.2"]["removed"]
    W = f["full"].jmodel.jm.m05[3].weight.detach().double()[:, rem]                                # (32, 24, 3, 3)
    mean = f["calib"].mean["jm.m05.2"][rem].double()
    want = sum((W[:, :, kh, kw] * mean).sum(1) for kh in range(3) for kw in range(3) if kh == 0 or kw == 0)
    assert torch.allclose((got["n"] - got["a"])[0, :, 0, 0].double(), want, atol=1e-4), ((got["n"] - got["a"])[0, :, 0, 0], want)


def test_macs_and_parameters_equal_the_cost_table():
    f = fx()
    es = JN.expected_saving(f["full"], f["spec"])
    assert es["macs"] == 40000 * 9 * (96 + 32) * 24 == 1105920000, es                           # 200 x 200 positions; conv 1 loses 24 of its 32 outputs, conv 2 24 of its inputs
    mf, mn = JN.measured_macs(f["full"]), JN.measured_macs(f["narrowed"])
    assert abs((mf["maps"] - mn["maps"]) * 1e9 - es["macs_maps"]) <= 1e-6 * es["macs_maps"], (mf, mn, es)
    assert mf["pyr"] == mn["pyr"] and mf["pair"] == mn["pair"] and abs((mf["total"] - mn["total"]) * 1e9 - es["macs"]) <= 1e-6 * es["macs"]      # nothing else moved, the node head included
    assert abs(PC.Cost(copy.deepcopy(f["narrowed"])).all["macs_maps"] / 1e9 - mn["maps"]) < 1e-6                 # the layer table of the narrowed model is the counter's
    npar = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert npar(f["full"]) - npar(f["narrowed"]) == es["params"] - 32, "saved parameters of the table less the new bias of the consumer conv"
    assert sorted(set(f["narrowed"].state_dict()) - set(f["full"].state_dict())) == ["jmodel.jm.m05.3.bias"]


# ---------------------------------------------------------------- the spec in the checkpoint, the loaders, the dump wrapper
def test_spec_round_trip_and_loaders():
    f = fx()
    spec = json.loads(json.dumps(f["narrowed"].narrow_spec))
    spec["calib"] = "no/such/calib.pt"                                                             # a checkpoint is read without the calibration file
    x = f["cache"][0]["x"][:1].float()
    with tempfile.TemporaryDirectory() as tmp:
        save_run(tmp, "n1", f["narrowed"], dict(f["args"], narrow=spec))
        save_run(tmp, "e324_s_s0", f["full"], f["args"])
        with Root(tmp):
            m, a2 = JN.load_model("n1", "cpu")
            m0, _ = JN.load_model("e324_s_s0", "cpu")                                              # no spec: the stored loader
    assert same_state(m, f["narrowed"]) and a2["narrow"]["taps"] == spec["taps"] and same_state(m0, f["full"])
    with torch.no_grad():
        assert torch.equal(m(x, gt=None)[2]["logit"], f["narrowed"](x, gt=None)[2]["logit"])
    m2, _ = JN.build_narrowed("e324_s_s0", json.loads(json.dumps(f["narrowed"].narrow_spec)))      # the stored model + the spec again: the same folded biases
    assert same_state(m2, f["narrowed"])
    a0 = f["args"]
    assert {k: v.shape for k, v in JN.model_of_args(a0).state_dict().items()} == {k: v.shape for k, v in f["full"].state_dict().items()}, "model_of_args builds what jends_eval.load_model builds"
    raises(AssertionError, JN.narrow, JN.model_of_args(a0), dict(f["narrowed"].narrow_spec, taps={"jm.m05.2": dict(removed=[0, 1, 2, 3], groups=[2, 4])}), False)      # a spec whose group count is not the result of its channels


def test_dump_wrapper_and_score_check():
    import jnarrow_eval as JNE
    import qends_eval as QE
    f, keep = fx(), JN.JEV.load_model
    with tempfile.TemporaryDirectory() as ref, tempfile.TemporaryDirectory() as wrap:
        for root in (ref, wrap):                                                                   # the stored run's model through a link: the dump writes into the temporary folder
            os.makedirs(os.path.join(root, "runs", "e324_s_s0"))
            os.symlink(os.path.join(ROOT, "runs", "e324_s_s0", "model.pt"), os.path.join(root, "runs", "e324_s_s0", "model.pt"))
        os.symlink(os.path.join(ROOT, "runs", "e273_p_s0"), os.path.join(wrap, "runs", "e273_p_s0"))
        save_run(wrap, "n1", f["narrowed"], dict(f["args"], narrow=json.loads(json.dumps(f["spec"]))))
        with Root(ref):
            JN.JEV.dump(SimpleNamespace(run="e324_s_s0", thr=0.5, max=8))
        try:
            with Root(wrap):
                JNE.dump(["--run", "e324_s_s0", "--max", "8"])
                JNE.dump(["--run", "n1", "--thr", "0.1", "--max", "8"])
                seen = []
                saved, QE.cmd_score = QE.cmd_score, lambda ns: seen.append(ns)                     # `jends_eval.score` on the narrowed checkpoint: its checks pass, the scorer gets the run
                try:
                    JN.JEV.score(SimpleNamespace(run="n1", seed=0, thr=0.1))
                finally:
                    QE.cmd_score = saved
                assert len(seen) == 1 and seen[0].run == "n1" and seen[0].skip_base_check
        finally:
            JN.JEV.load_model = keep                                                               # the wrapper patches the loader in its own process: here it is undone
        a, b = (pickle.load(open(os.path.join(r, "runs", "e324_s_s0", "ends_val.pkl"), "rb")) for r in (ref, wrap))
        assert len(a) == len(b) == 8 and all(x.keys() == y.keys() and x["token"] == y["token"] and all(np.array_equal(x[k], y[k]) for k in x if isinstance(x[k], np.ndarray)) for x, y in zip(a, b)), "the unnarrowed dump through the wrapper is the stored dump"
        assert json.load(open(os.path.join(ref, "runs", "e324_s_s0", "ends_head.json"))) == json.load(open(os.path.join(wrap, "runs", "e324_s_s0", "ends_head.json")))
        n, _ = PP.dump_frames(f["narrowed"], f["cache"], "cpu", 0.1)
        got = pickle.load(open(os.path.join(wrap, "runs", "n1", "ends_val.pkl"), "rb"))
        assert len(got) == 8 and all(x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) for k in x if isinstance(x[k], np.ndarray)) for x, y in zip(n, got)), "the dump of a narrowed run is what its model computes"
        assert json.load(open(os.path.join(wrap, "runs", "n1", "ends_head.json")))["thr"] == 0.1


def test_timing_machinery_on_a_narrowed_toy():
    """`jtiming.py` loads its models through `jends_eval.load_model`: with the wrapper's loader a narrowed run gets the section markers (hooks around the pyramid / maps head / pair head, the markers inside JEnds.forward) and the same outputs as without"""
    import jends_train as JT
    import jnarrow_eval as JNE
    import jtiming as JTM
    torch.manual_seed(0)
    m = JT.JEndModel(dict(d=64, ctx="dil"), order=True, evidence=True, scales=True).eval()
    x = torch.randn(1, 39, 200, 200)
    cpath = os.path.join(_TMP.name, "toy_calib.pt")
    PP.save_calib(PP.calibrate(m, [dict(x=x)], "cpu", PP.tap_table(m)), cpath)
    JN.narrow(m, {"calib": cpath, "edits": [{"unit": "jm.m05.hid", "frac": 0.5}]})
    args = dict(maps_args=dict(d=64, ctx="dil"), no_order=False, no_evidence=False, scales=True, narrow=json.loads(json.dumps(m.narrow_spec)))
    keep = JN.JEV.load_model
    with tempfile.TemporaryDirectory() as tmp:
        save_run(tmp, "toy", m, args)
        try:
            with Root(tmp):
                JNE.install()
                import jends_eval as JE
                t, _ = JE.load_model("toy", "cpu")                                                 # what `jtiming.run` calls
        finally:
            JN.JEV.load_model = keep
    assert t.jmodel.jm.m05[0].out_channels == 16 and same_state(t, m)
    mk = JTM.Marks(cuda=False)
    JTM.hook(t.jmodel.pyr, "pyr", mk)
    JTM.hook(t.jmodel.jm, "jm", mk)
    JTM.hook(t.je, "je", mk)
    JTM.instrument(t.je, mk)
    with torch.no_grad():
        ref = m(x, gt=None)[2]
        mk.on = True
        out = t(x, gt=None)[2]
    assert all(torch.equal(ref[k], out[k]) for k in ref if torch.is_tensor(ref[k])), "the instrumented forward of the narrowed model changes an output"
    names = [n for n, _, _ in mk.items]
    assert names[:2] == ["pyr:in", "pyr:out"] and "jm:out" in names and [n for n in names if n.startswith("b")] == [b[1] for b in JTM.BOUNDARIES], names


# ---------------------------------------------------------------- the trainer
def test_frozen_set_and_checks_at_the_start():
    import jnarrow_train as JTR
    f = fx()
    m = copy.deepcopy(f["narrowed"])
    ps = JTR.set_trainable(m)
    train = {n for n, p in m.named_parameters() if p.requires_grad}
    assert train == {n for n, _ in m.named_parameters() if n.startswith(("je.", "jmodel.jm."))} and "jmodel.jm.m05.3.bias" in train, "the pair head and the maps head (the new bias included) train"
    assert all(not p.requires_grad for n, p in m.named_parameters() if n.startswith(("jmodel.pyr.", "nodes."))) and len(ps) == len(train), "the pyramid and B's node head are frozen"
    a = JTR.parse_args(["--name", "x", "--init", "runs/e324_s_s0/model.pt", "--seed", "0", "--e2e", "--scales", "--attach"])
    JTR.check_init(f["full"], f["args"], a)
    for change, what in ((dict(attach=False), "flags"), (dict(scales=False), "flags"), (dict(seed=1), "seed"), (dict(e2e=False), "e2e")):
        raises(AssertionError, JTR.check_init, f["full"], f["args"], SimpleNamespace(**dict(vars(a), **change)))
    os.environ["BFLC_SIG"] = "1.5"
    try:
        raises(AssertionError, JTR.check_init, f["full"], f["args"], a)                            # a launcher without BFLC_SIG: the heat targets would not match the maps run
    finally:
        os.environ["BFLC_SIG"] = "1.0"
    raises(AssertionError, JTR.check_init, f["narrowed"], dict(f["args"], narrow=f["spec"]), a)    # a narrowed init
    assert JTR.run_of("runs/e324_s_s0/model.pt") == "e324_s_s0"
    raises(AssertionError, JTR.run_of, "runs/e324_s_s0/ckpt.pt")


def test_fine_tune_records_resume_and_the_other_arms():
    import jnarrow_train as JTR
    import trainer as T
    f = fx()
    orig_run = T.Run
    with tempfile.TemporaryDirectory() as out:
        T.Run = lambda name: orig_run(name, root=os.path.join(out, "runs"))                        # the run folders go to a temporary folder
        spec_file = os.path.join(out, "spec.json")
        json.dump(dict(calib=f["cpath"], edits=[{"unit": "jm.m05.hid", "frac": 0.75}]), open(spec_file, "w"))
        base = ["--init", "runs/e324_s_s0/model.pt", "--seed", "0", "--bs", "2", "--workers", "0", "--e2e", "--scales", "--attach", "--lr", "3e-4"]
        try:
            JTR.main(["--name", "t_n1", "--narrow", spec_file, "--steps", "2"] + base)
            d = os.path.join(out, "runs", "t_n1")
            ck = torch.load(os.path.join(d, "model.pt"), map_location="cpu")
            assert ck["state"]["jmodel.jm.m05.0.weight"].shape == (8, 96, 3, 3) and ck["state"]["jmodel.jm.m05.3.weight"].shape == (32, 8, 3, 3) and "jmodel.jm.m05.3.bias" in ck["state"]
            assert ck["args"]["narrow"]["taps"] == f["spec"]["taps"] and ck["args"]["seed"] == 0 and ck["args"]["init_run"] == "e324_s_s0" and os.path.exists(os.path.join(d, ".done"))
            assert json.load(open(os.path.join(d, "narrow.json"))) == json.loads(json.dumps(ck["args"]["narrow"])) and JN.sha256_file(os.path.join(d, "calib.pt")) == f["spec"]["calib_sha256"]
            assert json.load(open(os.path.join(d, "config.json")))["narrow"]["taps"] == f["spec"]["taps"]
            with Root(out):                                                                            # the loader of the evaluation reads the trained run: the shapes from its spec, the state as it was saved
                m, a2 = JN.load_model("t_n1", "cpu")
            assert m.jmodel.jm.m05[0].out_channels == 8 and a2["narrow"]["taps"] == f["spec"]["taps"] and all(torch.equal(m.state_dict()[k], v) for k, v in ck["state"].items())
            JTR.main(["--name", "t_n1", "--narrow", spec_file, "--steps", "3"] + base)                 # the resume: step 2 -> 3
            log = open(os.path.join(d, "train.log")).read()
            assert "resumed from" in log and "at step 2" in log and json.load(open(os.path.join(d, "status.json")))["step"] == 3, log
            spec2 = os.path.join(out, "spec2.json")
            json.dump(dict(calib=f["cpath"], edits=[{"unit": "jm.m05.hid", "frac": 0.5}]), open(spec2, "w"))
            raises(AssertionError, JTR.main, ["--name", "t_n1", "--narrow", spec2, "--steps", "3"] + base)   # a folder with another spec is not resumed
            JTR.main(["--name", "t_z", "--narrow", spec_file, "--steps", "0"] + base)                  # n1z: the narrowed model before any training = the surgery on the stored state
            z = torch.load(os.path.join(out, "runs", "t_z", "model.pt"), map_location="cpu")
            assert all(torch.equal(z["state"][k], v) for k, v in f["narrowed"].state_dict().items()) and z["state"].keys() == f["narrowed"].state_dict().keys() and os.path.exists(os.path.join(out, "runs", "t_z", ".done"))
            JTR.main(["--name", "t_cp", "--steps", "0"] + base)                                        # cp: no spec, no surgery: the stored state
            c = torch.load(os.path.join(out, "runs", "t_cp", "model.pt"), map_location="cpu")
            assert all(torch.equal(c["state"][k], v) for k, v in f["full"].state_dict().items()) and c["args"]["narrow"]["taps"] == {} and json.load(open(os.path.join(out, "runs", "t_cp", "narrow.json")))["taps"] == {}
            assert not os.path.exists(os.path.join(out, "runs", "t_cp", "calib.pt"))
        finally:
            T.Run = orig_run


TESTS = [test_groups_are_the_lowest_importance_groups, test_equality_with_the_hook_ablated_model, test_border_difference_is_the_padding_term, test_macs_and_parameters_equal_the_cost_table, test_spec_round_trip_and_loaders, test_dump_wrapper_and_score_check,
         test_timing_machinery_on_a_narrowed_toy, test_frozen_set_and_checks_at_the_start, test_fine_tune_records_resume_and_the_other_arms]


if __name__ == "__main__":
    before = {r: listing(r) for r in ("e324_s_s0", "e273_p_s0")}
    want = sys.argv[1:]
    for t in TESTS:
        if not want or t.__name__ in want or t.__name__.replace("test_", "") in want:
            t0 = time.time()
            t()
            print(f"{t.__name__} ok ({time.time() - t0:.0f} s)")
    assert before == {r: listing(r) for r in before}, "a stored run folder changed"
    print("test_jnarrow: all ok; the stored runs' folders are unchanged")
