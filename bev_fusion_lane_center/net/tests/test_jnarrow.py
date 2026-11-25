"""E353 tests of the narrowing (`jnarrow.py`, `train/jnarrow_train.py`, `train/jnarrow_eval.py`), CPU only (the GPU is hidden), 2 threads, the stored E324 s model, 8 real VAL frames and real TRAIN frames; about 2 minutes on an idle box, 3 on a loaded one.

    wsl bash tools/env/_py14.sh net/tests/test_jnarrow.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

Known answers: the groups removed are the lowest summed-importance groups of a numpy reference on the calibration (not the probe's per-channel set); the refusals (also a calibration file that is not the one the spec was resolved on); the narrowed maps head against the hook-ablated model (`jprune_probe.ablate`) that removes the SAME channels:
the output of the consumer conv agrees inside the map to float tolerance, the border ring differs by exactly the zero-padding term (computed by hand from the weights), the heat and the keypoints do not depend on the branch at all (difference 0), the pair logits move by
less than 1e-2 and no sign flips; two edited tensors read by ONE layer (a toy: the offsets follow the current widths, either order); the MACs and parameters of the narrowed model against `jprune_cost.Cost` and torch FlopCounterMode (1.10592 GMACs by arithmetic); the spec round trip (a checkpoint is read without the calibration file;
one new bias tensor); the loader of a stored run is the stored loader; the dump wrapper: the UNNARROWED stored run through the wrapper (`--unnarrowed-ok`) equals `jends_eval.dump` frame for frame, without the flag a run that `jnarrow_train.py` did not write is refused and nothing is written, a narrowed run dumps what its model computes,
`jends_eval.score` accepts the narrowed checkpoint; the frozen set of the trainer, its checks at the start (flags, `--attach` needs `--geom-w`, seed, BFLC_SIG, narrowed init), the run-folder rule (never a stored run's folder; resume only with the same init run and seed); the loss is the closure of `jends_train.py` (AST, log texts apart); a fine-tune of 2 steps: the maps head
(0.5 m branch, new bias) and the pair head move, the pyramid and the node head do not, the saved loss weights and seed, the data flags reach the batch, the first batch is the same for two arms of a seed and another for another seed, the MACs line of the log, the record in the run folder (spec, calibration copy and sha256), the resume (step 2 -> 3),
the refusal of another spec, the n1z run (`--steps 0`) and the cp run (no spec); the timing machinery of `jtiming.py` on a narrowed toy model; `time paired`: the alternating blocks and the difference narrowed - control on a fake clock, the verdict on the host, and a whole CPU run on toy models (order of load / GPU samples / the 2.5 s pause, the file, never overwritten,
a busy GPU refused before anything is loaded). The stored runs' folders must stay as they were (checked at the end).
"""
import ast
import copy
import inspect
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
import torch.nn as nn  # noqa: E402

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, d) for d in ("model", "prep", "loss", os.path.join(os.pardir, "training", "train"))]
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


def raises(exc, fn, *a, msg=None, **k):
    """fn must raise exc (with `msg` in its text when given, so that it is THIS refusal and not another)"""
    try:
        fn(*a, **k)
    except exc as e:
        assert msg is None or msg in str(e), f"{fn.__name__} raised {exc.__name__} for another reason: {e}"
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


def test_two_edited_taps_of_one_reader():
    """`narrow` cuts the taps one after the other and reads the widths of the model AS IT IS: a layer that reads two edited tensors [a, b] has lost a's columns when b is cut, and b's offset follows (offsets from the unnarrowed model would slice
    the wrong columns). Toy: 1x1 reader, so the result is exact; the narrowed model equals the model whose removed channels are replaced by their constants, in either order"""
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.jmodel = nn.Module()
            self.jmodel.pyr = SimpleNamespace(c_in=4)
            self.jmodel.jm = nn.Module()
            self.jmodel.jm.blocks = nn.ModuleList()
            blk = lambda cout, g: nn.Sequential(nn.Conv2d(4, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.ReLU())  # noqa: E731
            self.jmodel.jm.a, self.jmodel.jm.b, self.jmodel.jm.r = blk(8, 2), blk(12, 3), nn.Conv2d(20, 5, 1)

        def forward(self, x, ca=None, cb=None):
            a, b = self.jmodel.jm.a(x), self.jmodel.jm.b(x)
            if ca is not None:                                                                                 # the hook-ablated reference: the removed channels are constants
                a, b = a.clone(), b.clone()
                a[:, ca[0]], b[:, cb[0]] = ca[1].view(1, -1, 1, 1), cb[1].view(1, -1, 1, 1)
            return self.jmodel.jm.r(torch.cat([a, b], 1))

    torch.manual_seed(0)
    W = {"jm.a.0": (["x"], "jm.a.2", None), "jm.b.0": (["x"], "jm.b.2", None), "jm.r": (["jm.a.2", "jm.b.2"], "jm.r.out", None)}
    x, rem_a, rem_b = torch.randn(2, 4, 9, 9), [0, 1, 2, 3], [4, 5, 6, 7]                                       # groups of 4 channels: a loses its group 0, b its group 1
    mean_a, mean_b = torch.rand(len(rem_a)), torch.rand(len(rem_b))
    full_a, full_b = torch.zeros(8), torch.zeros(12)
    full_a[rem_a], full_b[rem_b] = mean_a, mean_b
    calib = os.path.join(_TMP.name, "toy2_calib.pt")
    PP.save_calib(PP.Calib(mean={"jm.a.2": full_a, "jm.b.2": full_b}, s={}, head={}, frames=0), calib)
    taps = {"jm.a.2": dict(removed=rem_a, groups=[1, 4]), "jm.b.2": dict(removed=rem_b, groups=[2, 4])}        # a resolved spec
    base, keep = Toy().eval(), JN.PC.wiring
    JN.PC.wiring = lambda model: W                                                                             # the wiring of the toy: `narrow` itself cuts the taps
    try:
        with torch.no_grad():
            want = base(x, (rem_a, mean_a), (rem_b, mean_b))
            for order in (("jm.a.2", "jm.b.2"), ("jm.b.2", "jm.a.2")):
                m = JN.narrow(copy.deepcopy(base), dict(calib=calib, taps={t: taps[t] for t in order}))
                assert m.jmodel.jm.r.in_channels == 12 and m.jmodel.jm.a[0].out_channels == 4 and m.jmodel.jm.b[1].num_groups == 2
                assert torch.allclose(m(x), want, atol=1e-5), (order, float((m(x) - want).abs().max()))
    finally:
        JN.PC.wiring = keep


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
    other, c2 = os.path.join(_TMP.name, "calib_other.pt"), copy.deepcopy(f["calib"])                                                                                        # a calibration file that is not the one the spec was resolved on
    c2.mean["jm.m05.2"] = c2.mean["jm.m05.2"] + 1.0
    PP.save_calib(c2, other)
    raises(AssertionError, JN.narrow, JN.model_of_args(a0), dict(f["narrowed"].narrow_spec, calib=other), msg="not the one the spec was resolved on")


def test_dump_wrapper_and_score_check():
    import jnarrow_eval as JNE
    import qends_eval as QE
    f, keep = fx(), JN.JEV.load_model
    with tempfile.TemporaryDirectory() as ref, tempfile.TemporaryDirectory() as wrap:
        for root in (ref, wrap):                                                                   # the stored run's model through a link: the dump writes into the temporary folder
            os.makedirs(os.path.join(root, "runs", "e324_s_s0"))
            os.symlink(os.path.join(ROOT, "runs", "e324_s_s0", "model.pt"), os.path.join(root, "runs", "e324_s_s0", "model.pt"))
        os.symlink(os.path.join(ROOT, "runs", "e273_p_s0"), os.path.join(wrap, "runs", "e273_p_s0"))
        save_run(wrap, "n1", f["narrowed"], dict(f["args"], narrow=json.loads(json.dumps(f["spec"])), init_run="e324_s_s0"))
        save_run(wrap, "cp", f["full"], dict(f["args"], narrow={}, init_run="e324_s_s0"))                       # the cp control: an empty spec, still a run of jnarrow_train.py
        with Root(ref):
            JN.JEV.dump(SimpleNamespace(run="e324_s_s0", thr=0.5, max=8))
        try:
            with Root(wrap):
                folder = os.path.join(wrap, "runs", "e324_s_s0")
                assert os.listdir(folder) == ["model.pt"] and JNE.written_by_jnarrow("n1") and JNE.written_by_jnarrow("cp") and not JNE.written_by_jnarrow("e324_s_s0")
                raises(AssertionError, JNE.dump, ["--run", "e324_s_s0", "--max", "8"], msg="was not written by jnarrow_train.py")      # a stored run: refused, nothing written into its folder
                assert os.listdir(folder) == ["model.pt"]
                JNE.dump(["--run", "e324_s_s0", "--max", "8", "--unnarrowed-ok"])
                JNE.dump(["--run", "n1", "--thr", "0.1", "--max", "8"])
                seen = []
                saved, QE.cmd_score = QE.cmd_score, lambda ns: seen.append(ns)                     # `jends_eval.score` on the narrowed checkpoint: its checks pass, the scorer gets the run
                try:
                    JN.JEV.score(SimpleNamespace(run="n1", seed=0, thr=0.1))
                finally:
                    QE.cmd_score = saved
                assert len(seen) == 1 and seen[0].run == "n1" and seen[0].skip_base_check
            argv = sys.argv
            raises(SystemExit, JNE.score, ["--seed", "0"])                                         # the stored parser of `jends_eval.py` answers (--run is missing): the pass-through reaches it, and sys.argv is left as it was
            assert sys.argv == argv
        finally:
            JN.JEV.load_model = keep                                                               # the wrapper patches the loader in its own process: here it is undone
        a, b = (pickle.load(open(os.path.join(r, "runs", "e324_s_s0", "ends_val.pkl"), "rb")) for r in (ref, wrap))
        assert len(a) == len(b) == 8 and all(x.keys() == y.keys() and x["token"] == y["token"] and all(np.array_equal(x[k], y[k]) for k in x if isinstance(x[k], np.ndarray)) for x, y in zip(a, b)), "the unnarrowed dump through the wrapper is the stored dump"
        assert json.load(open(os.path.join(ref, "runs", "e324_s_s0", "ends_head.json"))) == json.load(open(os.path.join(wrap, "runs", "e324_s_s0", "ends_head.json")))
        n, _ = PP.dump_frames(f["narrowed"], f["cache"], "cpu", 0.1)
        got = pickle.load(open(os.path.join(wrap, "runs", "n1", "ends_val.pkl"), "rb"))
        assert len(got) == 8 and all(x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) for k in x if isinstance(x[k], np.ndarray)) for x, y in zip(n, got)), "the dump of a narrowed run is what its model computes"
        assert json.load(open(os.path.join(wrap, "runs", "n1", "ends_head.json")))["thr"] == 0.1


def toy_runs(tmp):
    """a random-init JEndModel (d=64) with the hidden tensor of its 0.5 m branch halved, saved as run `toy`, and an unnarrowed one of other weights as run `toy_c`, under tmp/runs -> (the narrowed model, a random input)"""
    import jends_train as JT
    torch.manual_seed(0)
    m = JT.JEndModel(dict(d=64, ctx="dil"), order=True, evidence=True, scales=True).eval()
    x = torch.randn(1, 39, 200, 200)
    cpath = os.path.join(_TMP.name, "toy_calib.pt")
    PP.save_calib(PP.calibrate(m, [dict(x=x)], "cpu", PP.tap_table(m)), cpath)
    base = dict(maps_args=dict(d=64, ctx="dil"), no_order=False, no_evidence=False, scales=True)
    save_run(tmp, "toy_c", JT.JEndModel(base["maps_args"], order=True, evidence=True, scales=True).eval(), base)
    JN.narrow(m, {"calib": cpath, "edits": [{"unit": "jm.m05.hid", "frac": 0.5}]})
    save_run(tmp, "toy", m, dict(base, narrow=json.loads(json.dumps(m.narrow_spec))))
    return m, x


def test_timing_machinery_on_a_narrowed_toy():
    """`jtiming.py` loads its models through `jends_eval.load_model`: with the wrapper's loader a narrowed run gets the section markers (hooks around the pyramid / maps head / pair head, the markers inside JEnds.forward) and the same outputs as without"""
    import jnarrow_eval as JNE
    import jtiming as JTM
    keep = JN.JEV.load_model
    with tempfile.TemporaryDirectory() as tmp:
        m, x = toy_runs(tmp)
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


def test_paired_timing():
    """`jnarrow_eval.py time paired`: (1) a fake clock (control 10.0 ms, narrowed 9.2 ms): the blocks alternate c n | n c, the difference is narrowed - control = -0.8 ms = -8 %; (2) the verdict follows the host (load, GPU, forced); (3) a whole CPU run on toy models (a narrowed one against an
    unnarrowed one through the surgery-aware loader): the order of the load average, the GPU samples and the 2.5 s pause, the file, the refusals (a file is never overwritten; a busy GPU is refused before anything is loaded; two different run names)"""
    import jnarrow_eval as JNE
    import jtiming as JTM
    clock, order = dict(c=10.0, n=9.2), []
    keep = JNE.one_ms
    JNE.one_ms = lambda fn, x, cuda: (order.append(fn), clock[fn])[1]
    try:
        blocks = JNE.paired_blocks({"c": "c", "n": "n"}, [None], 4, 3, False)
    finally:
        JNE.one_ms = keep
    assert len(order) == 24 and all(len(set(order[i:i + 3])) == 1 for i in range(0, 24, 3)) and order[::3] == ["c", "n", "n", "c", "c", "n", "n", "c"], order      # blocks of 3 forwards: A B | B A | A B | B A
    assert blocks == {"c": [10.0] * 4, "n": [9.2] * 4}, blocks
    idle, busy = dict(util_max=0.0, mem_mib=1000.0), dict(util_max=60.0, mem_mib=9000.0)
    host = dict(load_before=1.0, load_after=1.0, gpu_before=idle, gpu_after=idle, forced=False)
    r = JNE.paired_result(("c", "n"), blocks, host)
    mean, lo, hi = r["paired_narrowed_minus_control"]
    assert abs(mean + 0.8) < 1e-9 and abs(lo + 0.8) < 1e-9 and abs(hi + 0.8) < 1e-9 and abs(r["percent"] + 8.0) < 1e-9 and r["control_median_ms"] == 10.0 and r["narrowed_median_ms"] == 9.2, r
    assert r["label"] == "QUIET" and r["quiet"] and not r["forced"] and r["runs"] == ["c", "n"]
    for what, change in (("load after", dict(load_after=3.0)), ("load unreadable", dict(load_before=None)), ("GPU busy after", dict(gpu_after=busy)), ("GPU unreadable", dict(gpu_before=dict(error="no nvidia-smi")))):
        assert JNE.paired_result(("c", "n"), blocks, dict(host, **change))["label"] == "NOT QUIET", what
    f = JNE.paired_result(("c", "n"), blocks, dict(host, gpu_before=busy, gpu_after=busy, forced=True))
    assert f["label"] == "NOT QUIET" and f["forced"]
    # a whole run on the CPU (toy models, 2 real VAL frames; load average and nvidia-smi are replaced by recorders)
    events = []
    fake_time = SimpleNamespace(sleep=lambda s: events.append(("sleep", s)), perf_counter=time.perf_counter)
    gpu = [idle]
    saved = (JTM.load1, JTM.gpu_state, JNE.time)
    JTM.load1 = lambda: (events.append("load"), 1.0)[1]
    JTM.gpu_state = lambda: (events.append("gpu"), gpu[0])[1]
    JNE.time = fake_time
    try:
        with tempfile.TemporaryDirectory() as tmp, Root(tmp):
            toy_runs(tmp)
            out = os.path.join(tmp, "jt.json")
            a = SimpleNamespace(runs=["toy_c", "toy"], tag="x", out=out, rounds=2, iters=1, warm=1, frames=2, force=False)
            res = JNE.paired_run(a, "cpu")
            assert events == ["load", "gpu", ("sleep", 2.5), "load", "gpu"], events                  # the after-sample comes after the pause
            got = json.load(open(out))
            assert got == json.loads(json.dumps(res, default=float)) and got["runs"] == ["toy_c", "toy"] and got["narrowed_runs"] == {"toy_c": False, "toy": True} and got["label"] == "QUIET" and got["cudnn_benchmark"] is False, got
            bm = got["block_medians_ms"]
            assert len(bm["toy"]) == len(bm["toy_c"]) == 2 and abs(got["paired_narrowed_minus_control"][0] - float(np.mean(np.array(bm["toy"]) - np.array(bm["toy_c"])))) < 1e-9
            before = os.path.getmtime(out)
            raises(AssertionError, JNE.paired_run, a, "cpu", msg="never overwritten")
            assert os.path.getmtime(out) == before
            gpu[0] = busy
            busy_a = SimpleNamespace(**dict(vars(a), out=os.path.join(tmp, "busy.json"), runs=["no_such", "no_such_c"]))
            raises(SystemExit, JNE.paired_run, busy_a, "cpu")                                        # refused before a model is loaded (these runs do not exist)
            assert not os.path.exists(busy_a.out)
            raises(AssertionError, JNE.time_cmd, ["paired", "--runs", "toy", "--tag", "y"], msg="two different run names")      # the dispatch reaches `time paired`, not jtiming's parser
            raises(AssertionError, JNE.time_cmd, ["paired", "--runs", "toy,toy", "--tag", "y"], msg="two different run names")
    finally:
        JTM.load1, JTM.gpu_state, JNE.time = saved


# ---------------------------------------------------------------- the trainer
def test_frozen_set_and_checks_at_the_start():
    import jnarrow_train as JTR
    f = fx()
    m = copy.deepcopy(f["narrowed"])
    ps = JTR.set_trainable(m)
    train = {n for n, p in m.named_parameters() if p.requires_grad}
    assert train == {n for n, _ in m.named_parameters() if n.startswith(("je.", "jmodel.jm."))} and "jmodel.jm.m05.3.bias" in train, "the pair head and the maps head (the new bias included) train"
    assert all(not p.requires_grad for n, p in m.named_parameters() if n.startswith(("jmodel.pyr.", "nodes."))) and len(ps) == len(train), "the pyramid and B's node head are frozen"
    a = JTR.parse_args(["--name", "x", "--init", "runs/e324_s_s0/model.pt", "--seed", "0", "--e2e", "--scales", "--attach", "--geom-w", "0.5"])
    JTR.check_init(f["full"], f["args"], a)
    for change, what in ((dict(attach=False), "flags"), (dict(scales=False), "flags"), (dict(seed=1), "seed"), (dict(e2e=False), "e2e"), (dict(geom_w=0.0), "--attach needs --geom-w")):
        raises(AssertionError, JTR.check_init, f["full"], f["args"], SimpleNamespace(**dict(vars(a), **change)), msg=what if what.startswith("--attach") else None)
    os.environ["BFLC_SIG"] = "1.5"
    try:
        raises(AssertionError, JTR.check_init, f["full"], f["args"], a)                            # a launcher without BFLC_SIG: the heat targets would not match the maps run
    finally:
        os.environ["BFLC_SIG"] = "1.0"
    raises(AssertionError, JTR.check_init, f["narrowed"], dict(f["args"], narrow=f["spec"]), a)    # a narrowed init
    assert JTR.run_of("runs/e324_s_s0/model.pt") == "e324_s_s0"
    raises(AssertionError, JTR.run_of, "runs/e324_s_s0/ckpt.pt")


def test_run_folder_rule():
    """`check_name`: an arm never gets a stored run's folder (a typo `--name e324_s_s0` would resume the stored step-6000 checkpoint); a folder with a model.pt or ckpt.pt is only resumed for the same init run and seed"""
    import jnarrow_train as JTR
    runs = os.path.join(ROOT, "runs")
    with tempfile.TemporaryDirectory() as tmp:
        new = os.path.join(tmp, "new")
        os.makedirs(new)
        JTR.check_name("e353_n1_s0", new, "e324_s_s0", 0)                                          # an empty folder
        raises(AssertionError, JTR.check_name, "e324_s_s0", new, "e324_s_s0", 0, msg="is the init run")
        for stored, seed in (("e324_s_s0", 0), ("e324_s_s1", 1), ("e273_p_s0", 0)):                # the real stored folders (read only): model.pt without the markers of a jnarrow run
            raises(AssertionError, JTR.check_name, "e353_typo", os.path.join(runs, stored), "e324_s_s0", seed, msg="of another run")
        mine = os.path.join(tmp, "mine")                                                           # a finished run of jnarrow_train.py: resumable for ITS init run and seed only
        os.makedirs(mine)
        torch.save(dict(state={}, args=dict(init_run="e324_s_s0", seed=0, narrow={})), os.path.join(mine, "model.pt"))
        JTR.check_name("e353_n1_s0", mine, "e324_s_s0", 0)
        for run_init, seed in (("e324_s_s0", 1), ("e324_s_s1", 0)):
            raises(AssertionError, JTR.check_name, "e353_n1_s0", mine, run_init, seed, msg="of another run")
        cut = os.path.join(tmp, "cut")                                                             # a stopped run: ckpt.pt and config.json only
        os.makedirs(cut)
        open(os.path.join(cut, "ckpt.pt"), "w").close()
        raises(AssertionError, JTR.check_name, "e353_n1_s0", cut, "e324_s_s0", 0, msg="of another run")        # no args to compare: not resumed
        json.dump(dict(args=dict(init_run="e324_s_s0", seed=0)), open(os.path.join(cut, "config.json"), "w"))
        JTR.check_name("e353_n1_s0", cut, "e324_s_s0", 0)
        raises(AssertionError, JTR.check_name, "e353_n1_s0", cut, "e324_s_s0", 1, msg="of another run")


def test_loss_is_the_closure_of_jends_train():
    """`make_loss` is the `loss_fn` of `jends_train.main` statement by statement (AST), with two differences that are not code: the text of the log call, and the `if a.e2e:` block, which is unconditional here (`--e2e` is required)"""
    import jends_train as JT
    import jnarrow_train as JTR

    class Norm(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) == "run.log":
                node.args = []
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            return node.body if ast.unparse(node.test) == "a.e2e" else node

    def statements(tree):
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "loss_fn"]
        assert len(fn) == 1
        return [ast.dump(s) for s in Norm().visit(fn[0]).body]

    stored, ours = statements(ast.parse(inspect.getsource(JT))), statements(ast.parse(inspect.getsource(JTR.make_loss)))
    assert ours == stored and len(ours) == 11, (len(ours), len(stored))


def test_fine_tune_records_resume_and_the_other_arms():
    import jnarrow_train as JTR
    import lane_gt_data as LD
    import trainer as T
    f = fx()
    orig_run, orig_loss, first = T.Run, JTR.make_loss, {}

    def recording(a, run):                                                                         # the first batch of every arm: its tokens and its keys
        fn = orig_loss(a, run)

        def loss_fn(m, b):
            first.setdefault(a.name, (list(b["token"]), set(b)))
            return fn(m, b)
        return loss_fn
    with tempfile.TemporaryDirectory() as out:
        T.Run = lambda name: orig_run(name, root=os.path.join(out, "runs"))                        # the run folders go to a temporary folder
        JTR.make_loss = recording
        spec_file = os.path.join(out, "spec.json")
        json.dump(dict(calib=f["cpath"], edits=[{"unit": "jm.m05.hid", "frac": 0.75}]), open(spec_file, "w"))
        common = ["--bs", "2", "--workers", "0", "--e2e", "--scales", "--attach", "--lr", "3e-4"]
        base = ["--init", "runs/e324_s_s0/model.pt", "--seed", "0"] + common
        try:
            raises(AssertionError, JTR.main, ["--name", "e324_s_s0", "--narrow", spec_file, "--steps", "0"] + base, msg="is the init run")     # a typo for the stored run's name
            LD.QData.DEFAULT_JUNCTION = LD.QData.DEFAULT_PSCALE = LD.QData.DEFAULT_PPTS = False
            JTR.main(["--name", "t_n1", "--narrow", spec_file, "--steps", "2"] + base)
            assert LD.QData.DEFAULT_JUNCTION and LD.QData.DEFAULT_PSCALE and LD.QData.DEFAULT_PPTS and "ppts" in first["t_n1"][1], "the data flags of the recipe reach the batches"
            d = os.path.join(out, "runs", "t_n1")
            ck = torch.load(os.path.join(d, "model.pt"), map_location="cpu")
            assert ck["state"]["jmodel.jm.m05.0.weight"].shape == (8, 96, 3, 3) and ck["state"]["jmodel.jm.m05.3.weight"].shape == (32, 8, 3, 3) and "jmodel.jm.m05.3.bias" in ck["state"]
            n0 = f["narrowed"].state_dict()
            assert not torch.equal(ck["state"]["jmodel.jm.m05.0.weight"], n0["jmodel.jm.m05.0.weight"]) and not torch.equal(ck["state"]["jmodel.jm.m05.3.bias"], n0["jmodel.jm.m05.3.bias"]), "the 0.5 m branch (the new bias too) gets the fine-tune: --e2e"
            assert any(not torch.equal(v, n0[k]) for k, v in ck["state"].items() if k.startswith("je."))
            assert all(torch.equal(v, n0[k]) for k, v in ck["state"].items() if k.startswith(("jmodel.pyr.", "nodes."))), "the pyramid and the node head stay as they were"
            assert [ck["args"][k] for k in ("maps_w", "geom_w", "cross_w", "driven_w", "raster", "seed", "e2e")] == [1.0, 0.5, 1.0, 0.0, "legacy", 0, True], "the loss of the init run (E324 s) when the command line says nothing"
            assert ck["args"]["narrow"]["taps"] == f["spec"]["taps"] and ck["args"]["seed"] == 0 and ck["args"]["init_run"] == "e324_s_s0" and os.path.exists(os.path.join(d, ".done"))
            assert JTR.macs_line(JN.measured_macs(f["full"]), JN.measured_macs(f["narrowed"])) in open(os.path.join(d, "train.log")).read(), "the log carries the measured GMACs of the init and of the narrowed model"
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
            JTR.main(["--name", "t_c1", "--steps", "1"] + base)                                        # the data order: cp (one step) sees the first batch of n1 of the same seed ...
            JTR.main(["--name", "t_s1", "--steps", "1", "--init", "runs/e324_s_s1/model.pt", "--seed", "1"] + common)      # ... and another seed another one
            assert first["t_c1"][0] == first["t_n1"][0] != first["t_s1"][0], first
            assert json.load(open(os.path.join(out, "runs", "t_s1", "config.json")))["seed"] == 1
        finally:
            T.Run, JTR.make_loss = orig_run, orig_loss


TESTS = [test_groups_are_the_lowest_importance_groups, test_equality_with_the_hook_ablated_model, test_border_difference_is_the_padding_term, test_two_edited_taps_of_one_reader, test_macs_and_parameters_equal_the_cost_table, test_spec_round_trip_and_loaders,
         test_dump_wrapper_and_score_check, test_timing_machinery_on_a_narrowed_toy, test_paired_timing, test_frozen_set_and_checks_at_the_start, test_run_folder_rule, test_loss_is_the_closure_of_jends_train, test_fine_tune_records_resume_and_the_other_arms]


if __name__ == "__main__":
    before = {r: listing(r) for r in ("e324_s_s0", "e324_s_s1", "e273_p_s0")}
    want = sys.argv[1:]
    for t in TESTS:
        if not want or t.__name__ in want or t.__name__.replace("test_", "") in want:
            t0 = time.time()
            t()
            print(f"{t.__name__} ok ({time.time() - t0:.0f} s)")
    assert before == {r: listing(r) for r in before}, "a stored run folder changed"
    print("test_jnarrow: all ok; the stored runs' folders are unchanged")
