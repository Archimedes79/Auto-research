"""E345 tests of the pruning probe (`jprune_probe.py`, `jprune_cost.py`, `jprune_table.py`), CPU only (the GPU is hidden), 2 threads, the stored E312 / E324 s models and the first 16 VAL frames; about 6 minutes on a loaded machine.

    wsl bash tools/env/_py14.sh test_jprune_probe.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

Known answers: the unit table of the card (45 units, 84 points, 20 ladder units, the paint columns 145 / 146); `dump_frames` = `jends_eval.dump` frame for frame; the per-frame counts add up to the batch counts; the calibration (means, s_c, head means) against a numpy
reference; every applier against an independent reference (a skipped block = the module replaced by an identity, the stream channel set, the paint columns); the plumbing controls with their positive controls; `saved` MACs / parameters against physically narrowed copies
under torch FlopCounterMode (single and combined cases for the maps head and for the pair head, the attention formula by hand); the point folder and the scorer call; halves, classes, greedy curve, prefixes, sizes, plan and controls on synthetic sweeps; a stored card summed by halves.
The stored runs' folders must stay as they were (checked at the end).
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import jends_eval as JEV  # noqa: E402
import jprune_cost as PC  # noqa: E402
import jprune_probe as PP  # noqa: E402
import jprune_table as PT  # noqa: E402
import qendnet as QEN  # noqa: E402
from jprune_cost import Cost, Edit  # noqa: E402

torch.set_num_threads(2)
ROOT = PP.ROOT
_FX = {}


def fx():
    """the shared fixture: the E312 model on the CPU, the first 16 VAL frames (2 batches), a calibration on them, the taps, the units, the cost table"""
    if not _FX:
        m, _ = JEV.load_model("e312_p_s0", "cpu")
        cache = PP.build_cache("val", workers=0, max_frames=16)
        taps = PP.tap_table(m)
        _FX.update(model=m, cache=cache, taps=taps, calib=PP.calibrate(m, cache, "cpu", taps), units=PP.build_units(m), cost=Cost(copy.deepcopy(m)))
    return _FX


def listing(run):
    d = os.path.join(ROOT, "runs", run)
    return sorted((f, os.path.getmtime(os.path.join(d, f))) for f in os.listdir(d))


def forward(m, cache):
    return [o for o, _ in PP.forward_batches(m, cache, "cpu")]


# ---------------------------------------------------------------- units
def test_unit_table_both_models():
    for run, attach in (("e312_p_s0", False), ("e324_s_s0", True)):
        m, a = JEV.load_model(run, "cpu")
        assert bool(a.get("attach")) == attach
        units, c = PP.build_units(m), Cost(m)
        print(f"--- {run}\n" + PP.unit_table(units, c, expect=(45, 84, 20)))
        assert next(u for u in units if u.name == "je.paint").idx == (145, 146)
        assert next(u for u in units if u.name == "jm.stream").fracs == (0.25, 0.5)
        assert [sum(len(u.fracs) for u in units if u.group == g) for g in ("pyr", "maps", "pair")] == [24, 35, 25]
        assert abs(c.all["macs_pyr"] / 1e9 - 4.677) < 1e-3 and abs((c.all["macs_maps"] + c.all["macs_pair"]) / 1e9 - 3.278) < 1e-3, c.all            # the card's GMACs
        assert [n for n, _ in PC.uncovered(m)] == (["nodes", "je.attach"] if attach else [])


def test_paint_columns_known_answer():
    """columns 145 / 146 of the pair MLP input are the two paint maxima: a 0.5 m map of logit +4 / -4 gives sigmoid(4) / sigmoid(-4) there and not in the neighbouring columns"""
    import jends_train as JT
    torch.manual_seed(0)
    m = JT.JEndModel(dict(d=64, ctx="dil"), order=True, evidence=True, scales=True).eval()
    B = 2
    jout, P, gt = _toy_pair_inputs(B)
    jout["j05"] = torch.cat([torch.full((B, 1, 200, 200), 4.0), torch.full((B, 1, 200, 200), -4.0)], 1)
    got = []
    hs = PP.observe(m, PP.tap_table(m), lambda n, a: got.append(a) if n == "je.fcat" else None)
    with torch.no_grad():
        m.je(jout, P, gt=gt)
    for h in hs:
        h.remove()
    a = got[0]
    assert a.shape[1] == 152 and len(a) > 0
    assert abs(float(a[:, 145].mean()) - 0.982) < 0.01 and abs(float(a[:, 146].mean()) - 0.018) < 0.01 and float(a[:, 145].std()) < 1e-3
    assert not torch.allclose(a[:, 144], a[:, 145]) and not torch.allclose(a[:, 147], a[:, 146])


# ---------------------------------------------------------------- dump, per-frame numbers, halves
def test_dump_frames_equals_jends_eval_dump():
    f = fx()
    frames, _ = PP.dump_frames(f["model"], f["cache"], "cpu", PP.THR)
    with tempfile.TemporaryDirectory() as tmp:                               # jends_eval.dump writes into <ROOT>/runs/<run>: point its ROOT at a folder with a link to the model (the stored run's folder must not get a file)
        os.makedirs(os.path.join(tmp, "runs", "e312_p_s0"))
        os.symlink(os.path.join(ROOT, "runs", "e312_p_s0", "model.pt"), os.path.join(tmp, "runs", "e312_p_s0", "model.pt"))
        old, JEV.ROOT = JEV.ROOT, tmp
        try:
            JEV.dump(SimpleNamespace(run="e312_p_s0", thr=PP.THR, max=16))
        finally:
            JEV.ROOT = old
        ref = pickle.load(open(os.path.join(tmp, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))
    assert len(frames) == len(ref) == 16
    for a, b in zip(frames, ref):
        assert a.keys() == b.keys() and a["token"] == b["token"] and a["row"] == b["row"]
        assert all(np.array_equal(a[k], b[k]) for k in a if isinstance(a[k], np.ndarray)) and (a["scale"] is None) == (b["scale"] is None)
    print("dump_frames == jends_eval.dump on 16 frames, keys", sorted(frames[0]))


def test_frame_stats_and_attached_fields():
    f = fx()
    out, b = next(iter(PP.forward_batches(f["model"], f["cache"], "cpu")))
    st = PP.frame_stats(out, b)
    tot = QEN.end_counts(out, b, PP.THR)
    assert [sum(s["counts"][i] for s in st) for i in range(len(PP.COUNTS))] == [tot[k] for k in PP.COUNTS], "per-frame counts must add up to the batch counts"
    assert all(len(s["scores"]) == len(s["labels"]) for s in st) and sum(int(s["labels"].sum()) for s in st) >= tot["pairs_hit"]
    assert all(s["kp"][3] == 0 and s["kp"][4] == 0 for s in st), "E312 has no attached ends"
    m, _ = JEV.load_model("e324_s_s0", "cpu")                                # E324 s: the attached ends reach the frames and the error
    out, b = next(iter(PP.forward_batches(m, f["cache"][:1], "cpu")))
    st = PP.frame_stats(out, b)
    assert "att_pos" in PP.frames_of(out, b)[0] and sum(s["kp"][0] for s in st) > 0 and sum(s["kp"][3] for s in st) > 0 and sum(s["kp"][3] for s in st) != sum(s["kp"][1] for s in st)


def test_pair_curve_and_candidates_known_answers():
    c = PP.pair_curve([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0], 3)                  # AP = (1/1 + 2/3) / 3, precision at recall 0.5 (2 of the 3 true pairs) = 2/3
    assert abs(c["ap"] - (1 + 2 / 3) / 3) < 1e-12 and abs(c["p_at_r05"] - 2 / 3) < 1e-12
    assert PP.pair_curve([0.9], [1], 4)["p_at_r05"] is None and PP.pair_curve([], [], 3)["ap"] == 0.0
    K = 4
    kp, kd = torch.zeros(1, 2 * K, 2), torch.zeros(1, 2 * K, 2)
    kd[..., 1] = 1.0
    kp[0, 0], kp[0, 1], kp[0, 2] = torch.tensor([-10.0, 0.0]), torch.tensor([-10.0, 5.0]), torch.tensor([40.0, 40.0])      # entries: GT 0, GT 1, a stray one
    kp[0, K], kp[0, K + 1] = torch.tensor([10.0, 0.5]), torch.tensor([10.0, 5.5])                                       # exits: GT 0 (0.5 m off), GT 1 (0.5 m off)
    kd[0, K + 1] = torch.tensor([np.sin(np.radians(10.0)), np.cos(np.radians(10.0))])                                  # exit 1: heading 10 degrees off
    valid = torch.zeros(1, 2, K, dtype=torch.bool)
    valid[0, 0, :3], valid[0, 1, :2] = True, True
    p = torch.zeros(1, 2 * K)
    p[0, :3], p[0, K:K + 2] = 0.9, 0.9
    logit = torch.full((1, K, K), -3.0)
    logit[0, 0, 0], logit[0, 1, 1], logit[0, 0, 1], logit[0, 2, 0] = 2.0, 1.0, 0.5, 3.0
    out = dict(heat=torch.zeros(1, 2, 4, 4), idx=torch.zeros(1, 2, K, dtype=torch.long), valid=valid, kp_pos=kp, kp_dir=kd, kp_p=p, logit=logit, allowed=valid[:, 0, :, None] & valid[:, 1, None, :])
    b = dict(ent=torch.tensor([[[-10.0, 0.0, 0.0, 1.0], [-10.0, 5.0, 0.0, 1.0]]]), ext=torch.tensor([[[10.0, 0.0, 0.0, 1.0], [10.0, 5.0, 0.0, 1.0]]]), ent_m=torch.ones(1, 2, dtype=torch.bool),
             ext_m=torch.ones(1, 2, dtype=torch.bool), pairs=torch.tensor([[[0, 0], [1, 1]]]), token=["t"])
    sc, lab = PP.pair_candidates(out, b, 0)                                  # 3 entries x 2 exits; true: (0, 0) and (1, 1)
    assert len(sc) == 6 and sorted(sc[lab].tolist()) == [1.0, 2.0] and sorted(sc[~lab].tolist()) == [-3.0, -3.0, 0.5, 3.0], (sc, lab)
    e = PP.kp_error(out, b, 0)                                               # 4 matched GT ends, positions 0 / 0 / 0.5 / 0.5 m, one heading 10 degrees off
    assert e[0] == 4 and abs(e[1] - 1.0) < 1e-5 and abs(e[2] - 10.0) < 0.05 and e[3] == 0, e                 # (the heading from float32 unit vectors: arccos near 1 is good to about 0.01 degree)


def test_halves_are_the_cards():
    import qgraph_eval as QE
    rows = np.asarray(json.load(open(f"{QE.BP.ROOT}/data/splits.json"))["val"], int)
    h = PP.halves(rows)
    assert len(h) == 873 and int((h == 0).sum()) == 438 and int((h == 1).sum()) == 435, (int((h == 0).sum()), int((h == 1).sum()))
    sc = np.asarray(pickle.load(open(os.path.join(ROOT, "runs", "e312_p_s0", "ends_card_snap_thr0.1.counts.pkl"), "rb"))["scenes"])         # the scorer's own scene list gives the same assignment
    uniq = sorted(set(sc.tolist()))
    assert len(uniq) == 22 and np.array_equal(np.array([uniq.index(s) % 2 for s in sc]), h), "the halves of the probe and of the scored cards must be the same frames"
    try:
        PP.build_cache("test")
        raise RuntimeError("the TEST split must be refused")
    except AssertionError as ex:
        assert "TEST" in str(ex)
    assert len(range(0, 8414, PP.CAL_STRIDE)) == 201


# ---------------------------------------------------------------- calibration and appliers
def test_calibration_against_numpy_reference():
    f = fx()
    m, calib = f["model"], f["calib"]
    keys = ("pyr.e1", "jm.blocks.1.f.2", "je.blocks.0.ffn.1", "je.pair.1")
    cap, fcat, heads, outs = {k: [] for k in keys}, [], [], []
    hs = [PP.sub(m, k).register_forward_hook(lambda _m, _a, o, k=k: cap[k].append(o.clone())) for k in keys]
    hs += [PP.sub(m, "je.pair.0").register_forward_pre_hook(lambda _m, a: fcat.append(a[0].clone())), m.je.register_forward_hook(lambda _m, _a, o: outs.append(o)),
           PP.sub(m, "je.blocks.0.att").register_forward_pre_hook(lambda _m, a, k: heads.append((a[0].clone(), k["key_padding_mask"].clone())), with_kwargs=True)]
    with torch.no_grad():
        for b in f["cache"]:
            m(b["x"].float(), gt=None)
    for h in hs:
        h.remove()
    valid, allowed = (torch.cat([o[k] for o in outs]).numpy() for k in ("valid", "allowed"))
    for k, a in dict({k: torch.cat(v).double().numpy() for k, v in cap.items()}, **{"je.fcat": torch.cat(fcat).double().numpy()}).items():
        rows = a.transpose(1, 0, 2, 3).reshape(a.shape[1], -1).T if a.ndim == 4 else a[valid.reshape(len(a), -1)] if k.endswith("ffn.1") else a[allowed]       # numpy: conv = all positions, tokens = valid keypoints, pairs = allowed pairs
        mu = rows.mean(0)
        assert np.abs(calib.mean[k].numpy() - mu).max() < 2e-6 and np.abs(calib.s[k].numpy() - np.abs(rows - mu).mean(0)).max() < 2e-6, k
    h, pad = torch.cat([x[0] for x in heads]).double().numpy(), torch.cat([x[1] for x in heads]).numpy()
    att = m.je.blocks[0].att
    W, b_ = att.in_proj_weight.detach().double().numpy(), att.in_proj_bias.detach().double().numpy()
    acc, n = np.zeros((4, 16)), 0
    for i in range(len(h)):                                                  # the per-head attention output by hand, valid keys only, averaged over the valid queries
        if pad[i].all():
            continue
        qkv = h[i] @ W.T + b_
        q, k, v = (qkv[:, j * 64:(j + 1) * 64].reshape(-1, 4, 16).transpose(1, 0, 2) for j in range(3))
        s = q @ k.transpose(0, 2, 1) / 4.0
        s[:, :, pad[i]] = -np.inf
        w = np.exp(s - s.max(-1, keepdims=True))
        acc += ((w / w.sum(-1, keepdims=True)) @ v)[:, ~pad[i]].sum(1)
        n += int((~pad[i]).sum())
    assert all(np.abs(calib.head[("je.blocks.0", j)].numpy() - acc[j] / n).max() < 2e-6 for j in range(4))


def test_appliers_against_independent_references():
    f = fx()
    m, taps, calib, units, cache = f["model"], f["taps"], f["calib"], {u.name: u for u in f["units"]}, f["cache"]
    base = forward(m, cache)
    skip_block = type("Skip", (nn.Module,), {"forward": lambda self, x, pad: x})
    for name, parent, i, new in (("jm.blocks.2.skip", "jmodel.jm.blocks", 2, nn.Identity()), ("je.blocks.1.skip", "je.blocks", 1, skip_block())):       # a skipped block = the module replaced by an identity
        ref = copy.deepcopy(m)
        ref.get_submodule(parent)[i] = new
        mm, w = PP.ablate(m, PP.edits_for(units[name], 1.0, calib), taps)
        got = forward(mm, cache)
        w.check(len(cache))
        assert max(PP.max_diff(a, b) for a, b in zip(got, forward(ref, cache))) == 0.0 and max(PP.max_diff(a, b) for a, b in zip(got, base)) > 1e-6, name
    # the stream: one channel set in p2 and every block output, each tensor at its own calibration mean; the paint columns 145 / 146 at their means, their neighbour untouched
    mm, w = PP.ablate(m, PP.edits_for(units["jm.stream"], 0.5, calib) + PP.edits_for(units["je.paint"], 1.0, calib), taps)
    seen = {}
    hs = PP.observe(mm, {k: taps[k] for k in ("jm.p2", "jm.blocks.0", "jm.blocks.3", "je.fcat")}, lambda n, a: seen.setdefault(n, []).append(a.clone()))
    forward(mm, cache)
    for h in hs:
        h.remove()
    w.check(len(cache), calib)
    edits = {e.tap: e for e in PP.edits_for(units["jm.stream"], 0.5, calib)}
    assert len({e.idx for e in edits.values()}) == 1 and len(edits["jm.p2"].idx) == 32
    for k in ("jm.p2", "jm.blocks.0", "jm.blocks.3"):
        a = torch.cat(seen[k])[:, list(edits[k].idx)]
        assert float(a.std(dim=(0, 2, 3)).max()) < 1e-6 and torch.allclose(a.mean(dim=(0, 2, 3)), calib.mean[k][list(edits[k].idx)], atol=1e-6), k
    a = torch.cat(seen["je.fcat"])
    assert float((a[:, [145, 146]] - calib.mean["je.fcat"][[145, 146]]).abs().max()) < 1e-6 and float((a[:, 144] - calib.mean["je.fcat"][144]).abs().max()) > 1e-3
    # ranked / reverse / random: the lowest, the highest, a draw of the importance
    s = calib.s["jm.blocks.0.f.2"]
    r, v, d = (PP.edits_for(units["jm.blocks.0.hid"], 0.5, calib, h, np.random.RandomState(0))[0].idx for h in ("rank", "reverse", "random"))
    assert len(r) == len(v) == len(d) == 32 and s[list(r)].max() <= s[list(v)].min() and set(r) != set(d) and set(r).isdisjoint(set(v))
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), JEV.load_model("e312_p_s0", "cpu")[0].state_dict().values())), "an ablation must not touch the model"


def test_plumbing_controls():
    f = fx()
    res = PP.plumbing_controls(f["model"], f["calib"], f["cache"], "cpu", f["taps"])
    print("plumbing:", res)
    assert res["identity_max_diff"] == 0.0 and res["zeroed_copy_max_diff"] <= 1e-6 and res["zeroing_effect"] > 1e-4 and res["original_effect"] > 1e-6


def test_dead_hook_assert_catches():
    f = fx()
    m, taps, calib = f["model"], f["taps"], f["calib"]
    u = {u.name: u for u in f["units"]}["jm.blocks.0.hid"]
    mm, w = PP.ablate(m, PP.edits_for(u, 0.5, calib), taps)
    forward(mm, f["cache"][:1])
    w.check(1, calib)
    for bad in (0, 2):                                                       # a hook that did not fire / fired too often
        try:
            w.check(bad, calib)
            raise RuntimeError("a fired-count mismatch was not caught")
        except AssertionError as ex:
            assert "fired" in str(ex)
    dead = PP.Watch()                                                        # a hook that fired but changed nothing is dead unless the channels carry no variance
    dead.edit["k"], dead.sites["k"], dead.fired[("k", "x")], dead.changed["k"] = Edit("chan", tap="jm.blocks.0.f.2", idx=(1,), val=torch.zeros(1)), ("x",), 1, torch.zeros((), dtype=torch.bool)
    try:
        dead.check(1, calib)
        raise RuntimeError("a dead hook was not caught")
    except AssertionError as ex:
        assert "dead hook" in str(ex)
    flat = copy.deepcopy(calib)
    flat.s["jm.blocks.0.f.2"][1] = 0.0
    dead.check(1, flat)


# ---------------------------------------------------------------- saved cost against physically narrowed copies
def _slice(layer, out=None, inp=None):
    W = layer.weight.data
    W = W[list(out)] if out is not None else W
    W = W[:, list(inp)] if inp is not None else W
    n = nn.Conv2d(W.shape[1], W.shape[0], layer.kernel_size, layer.stride, layer.padding, layer.dilation, bias=layer.bias is not None) if isinstance(layer, nn.Conv2d) else nn.Linear(W.shape[1], W.shape[0], bias=layer.bias is not None)
    n.weight.data = W.clone()
    if layer.bias is not None:
        n.bias.data = (layer.bias.data[list(out)] if out is not None else layer.bias.data).clone()
    return n


def _gn(gn, keep):
    n = nn.GroupNorm(8, len(keep))
    n.weight.data, n.bias.data = gn.weight.data[list(keep)].clone(), gn.bias.data[list(keep)].clone()
    return n


def narrow_jm(jm, stream=None, hid=None, skip=(), m1=None, m05=None):
    """a physically narrowed copy of the maps head: stream channels kept (p2 and every block), hidden widths kept per block / m1 / m05, blocks skipped"""
    j = copy.deepcopy(jm)
    if stream is not None:
        j.p2[0], j.p2[1] = _slice(j.p2[0], out=stream), _gn(j.p2[1], stream)
        for b in j.blocks:
            b.f[0], b.f[3], b.f[4] = _slice(b.f[0], inp=stream), _slice(b.f[3], out=stream), _gn(b.f[4], stream)
        c1 = j.m1[0].in_channels - jm.p2[0].out_channels                         # m1 reads [feat1, stream]
        j.m1[0] = _slice(j.m1[0], inp=list(range(c1)) + [c1 + k for k in stream])
    for i, keep in (hid or {}).items():
        b = j.blocks[i]
        b.f[0], b.f[1], b.f[3] = _slice(b.f[0], out=keep), _gn(b.f[1], keep), _slice(b.f[3], inp=keep)
    for i in skip:
        j.blocks[i] = nn.Identity()
    for seq, keep in ((j.m1, m1), (getattr(j, "m05", None), m05)):
        if keep is not None:
            seq[0], seq[1], seq[3] = _slice(seq[0], out=keep), _gn(seq[1], keep), _slice(seq[3], inp=keep)
    return j


class _Skip(nn.Module):
    def forward(self, x, pad):
        return x


def narrow_je(je, pair1=None, pair3=None, scale1=None, ffn=None, skip=(), cols=()):
    """a physically narrowed copy of the pair head (`cols`: input columns of both MLPs dropped before the first linear)"""
    j = copy.deepcopy(je)
    if pair1 is not None:
        j.pair[0], j.pair[2] = _slice(j.pair[0], out=pair1), _slice(j.pair[2], inp=pair1)
    if pair3 is not None:
        j.pair[2], j.pair[4] = _slice(j.pair[2], out=pair3), _slice(j.pair[4], inp=pair3)
    if scale1 is not None:
        j.scale[0], j.scale[2] = _slice(j.scale[0], out=scale1), _slice(j.scale[2], inp=scale1)
    for i, keep in (ffn or {}).items():
        j.blocks[i].ffn[0], j.blocks[i].ffn[2] = _slice(j.blocks[i].ffn[0], out=keep), _slice(j.blocks[i].ffn[2], inp=keep)
    for i in skip:
        j.blocks[i] = _Skip()
    if cols:
        keep = [c for c in range(j.pair[0].in_features) if c not in cols]
        for seq in (j.pair, j.scale):
            seq[0] = _slice(seq[0], inp=keep)
            seq[0].register_forward_pre_hook(lambda _m, a, keep=keep: (a[0][..., keep],))
    return j


def _macs(fn):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 2


def _npar(m):
    return sum(p.numel() for p in m.parameters())


def _toy_pair_inputs(B=1, d=64):
    jout = dict(j1=torch.randn(B, 8, 100, 100) * 0.1 - 2.0, j05=torch.randn(B, 2, 200, 200) - 2.0, f1=torch.randn(B, d, 100, 100))
    jout["j1"][:, 4], jout["j1"][:, 6], jout["j1"][:, 5], jout["j1"][:, 7] = 0.0, 0.0, 5.0, 5.0
    ent = torch.tensor([[[-1.75, -8.0, 0.0, 1.0], [1.75, -8.0, 0.0, 1.0]]] * B)
    ext = torch.tensor([[[-1.75, 8.0, 0.0, 1.0], [1.75, 8.0, 0.0, 1.0]]] * B)
    return jout, dict(centre1=torch.randn(B, 12, 100, 100)), dict(ent=ent, ext=ext, ent_m=torch.ones(B, 2, dtype=torch.bool), ext_m=torch.ones(B, 2, dtype=torch.bool))


def test_saved_cost_against_narrowed_copies():
    f = fx()
    m, units, cost = f["model"].eval(), {u.name: u for u in f["units"]}, f["cost"]
    ed = lambda spec: [e for n, fr in spec.items() for e in PP.count_edits(units[n], fr, cost.C)]  # noqa: E731
    P = dict(feat2=torch.randn(1, 144, 50, 50), feat1=torch.randn(1, 48, 100, 100), feat05=torch.randn(1, 32, 200, 200), centre1=torch.randn(1, 12, 100, 100))
    jm = m.jmodel.jm
    full_m, full_p = _macs(lambda: jm(P)), _npar(jm)
    assert abs(full_m - cost.all["macs_maps"]) < 1, "the maps head table is the counter's"
    cases = {"single: block 1 hidden width 64 -> 32": (dict(hid={1: range(32, 64)}), {"jm.blocks.1.hid": 0.5}),
             "single: m05 hidden 32 -> 8": (dict(m05=range(8)), {"jm.m05.hid": 0.75}),
             "single: stream 64 -> 48": (dict(stream=range(16, 64)), {"jm.stream": 0.25}),
             "single: block 0 skipped": (dict(skip=(0,)), {"jm.blocks.0.skip": 1.0}),
             "combined: stream 64 -> 32, block 1 hidden 64 -> 48, block 2 skipped, m1 hidden 64 -> 16, m05 hidden 32 -> 16": (dict(stream=range(32, 64), hid={1: range(16, 64)}, skip=(2,), m1=range(48, 64), m05=range(16, 32)),
                                                                                                                            {"jm.stream": 0.5, "jm.blocks.1.hid": 0.25, "jm.blocks.2.skip": 1.0, "jm.m1.hid": 0.75, "jm.m05.hid": 0.5}),
             "combined: stream 64 -> 48 with block 1 skipped AND its hidden width cut (counted once)": (dict(stream=range(16, 64), skip=(1,)), {"jm.stream": 0.25, "jm.blocks.1.skip": 1.0, "jm.blocks.1.hid": 0.5})}
    for name, (plan, spec) in cases.items():
        nj = narrow_jm(jm, **{k: ({i: list(v) for i, v in x.items()} if k == "hid" else (list(x) if k != "skip" else x)) for k, x in plan.items()})
        got_m, got_p = full_m - _macs(lambda: nj(P)), full_p - _npar(nj)
        sv = cost.saved(ed(spec))
        assert abs(got_m - sv["macs_maps"]) < 1 and got_p == sv["params"], (name, got_m, sv["macs_maps"], got_p, sv["params"])
        print(f"saved cost == narrowed copy: {name}: {got_m / 1e9:.4f} GMACs, {got_p} parameters")
    jout, Pp, gt = _toy_pair_inputs()
    je = m.je
    full_m, full_p = _macs(lambda: je(jout, Pp, gt=gt)), _npar(je)
    assert abs(full_m - cost.all["macs_pair"]) < 1
    cases = {"single: pair MLP hidden 1, 64 -> 48": (dict(pair1=list(range(16, 64))), {"je.pair.1": 0.25}),
             "single: scale hidden 32 -> 8": (dict(scale1=list(range(8))), {"je.scale.1": 0.75}),
             "single: paint columns": (dict(cols=(145, 146)), {"je.paint": 1.0}),
             "single: set block 1 skipped": (dict(skip=(1,)), {"je.blocks.1.skip": 1.0}),
             "combined: pair 1 64 -> 32, pair 3 64 -> 16, scale 32 -> 16, ffn of block 1 128 -> 64, paint columns, block 0 skipped": (dict(pair1=list(range(32, 64)), pair3=list(range(48, 64)), scale1=list(range(16, 32)), ffn={1: list(range(64, 128))}, cols=(145, 146), skip=(0,)),
                                                                                                                                  {"je.pair.1": 0.5, "je.pair.3": 0.75, "je.scale.1": 0.5, "je.blocks.1.ffn": 0.5, "je.paint": 1.0, "je.blocks.0.skip": 1.0})}
    for name, (plan, spec) in cases.items():
        nj = narrow_je(je, **plan).eval()
        got_m, got_p = full_m - _macs(lambda: nj(jout, Pp, gt=gt)), full_p - _npar(nj)
        sv = cost.saved(ed(spec))
        assert abs(got_m - sv["macs_pair"]) < 1 and got_p == sv["params"], (name, got_m, sv["macs_pair"], got_p, sv["params"])
        print(f"saved cost == narrowed copy: {name}: {got_m / 1e6:.3f} MMACs, {got_p} parameters")
    # the attention modules: `macs_mha` of the table against a by-hand attention with the heads kept (the counter sees the linears and both matmuls there; in eval mode it does not see nn.MultiheadAttention)
    n, d, hd = 64, 64, 16

    def by_hand(hk):
        x, Wqkv, Wo = torch.randn(1, n, d), torch.randn(3 * hd * hk, d), torch.randn(d, hd * hk)

        def fn():
            q, k, v = (t.reshape(1, n, hk, hd).transpose(1, 2) for t in F.linear(x, Wqkv).chunk(3, -1))
            F.linear(((q @ k.transpose(-1, -2)).softmax(-1) @ v).transpose(1, 2).reshape(1, n, hk * hd), Wo)
        return _macs(fn)
    for hk in (4, 3, 1):
        assert abs(cost.totals([Edit("head", block="je.blocks.0", head=h) for h in range(hk, 4)])["macs_mha"] - by_hand(hk) - by_hand(4)) < 1, hk
    one = [Edit("head", block="je.blocks.0", head=2)]
    assert cost.saved(one)["macs"] == 0 and cost.saved(one, mha=True)["macs"] == cost.saved(one)["macs_mha"] > 0
    assert cost.saved([Edit("head", block="je.blocks.0", head=h) for h in range(4)])["params"] == 4 * (3 * 16 * 64 + 48 + 64 * 16), "four heads out: the projections of the block, not its output bias"


# ---------------------------------------------------------------- the sweep, the point folder, the scorer call
def test_sweep_passes_and_point_folder():
    f = fx()
    with tempfile.TemporaryDirectory() as tmp:
        sw = PP.Sweep.__new__(PP.Sweep)
        sw.__dict__.update(run="e312_p_s0", dev="cpu", out=tmp, log=lambda *a: None, runs_dir=os.path.join(tmp, "runs"), model=f["model"], args={}, units=f["units"], taps=f["taps"], cost=f["cost"], val=f["cache"], calib=f["calib"],
                           rows=[int(r) for b in f["cache"] for r in b["row"]], path=os.path.join(tmp, "sweep.json"))
        sw.half = PP.halves(sw.rows)
        sw.rec = dict(run="e312_p_s0", attach=False, rows=sw.rows, tokens=[], calib_frames=16, passes={}, points=[], controls={})
        os.makedirs(os.path.join(tmp, "runs", "e312_p_s0"))
        os.symlink(os.path.join(ROOT, "runs", "e312_p_s0", "model.pt"), os.path.join(tmp, "runs", "e312_p_s0", "model.pt"))
        sw.nulls(False)
        a, b = sw.rec["passes"]["null"], sw.rec["passes"]["null2"]
        assert a["counts"] == b["counts"] and a["kp"] == b["kp"] and a["pair"] == b["pair"] and sw.rec["controls"]["null_spread_ends_mean6"] == 0.0 and len(a["counts"]) == 16
        sw.do("p001", {"pyr.stem": 0.25})
        sec = sw.rec["passes"]["p001"]["seconds"]
        sw.do("p001", {"pyr.stem": 0.25})                                      # recorded: not run again
        assert sw.rec["passes"]["p001"]["seconds"] == sec and sw.rec["passes"]["p001"]["saved_macs"] == f["cost"].saved(PP.count_edits(f["units"][0], 0.25, f["cost"].C))["macs"]
        assert json.load(open(sw.path))["passes"].keys() == sw.rec["passes"].keys()
        spec = {"jm.blocks.2.skip": 1.0, "je.paint": 1.0}
        sw.dump_point(7, spec)
        d = os.path.join(tmp, "runs", "e345_p007_e312_p_s0")
        assert os.path.islink(os.path.join(d, "model.pt")) and os.path.realpath(os.path.join(d, "model.pt")) == os.path.realpath(os.path.join(ROOT, "runs", "e312_p_s0", "model.pt"))
        import hashlib
        meta = json.load(open(os.path.join(d, "point.json")))
        assert meta["sha256"] == hashlib.sha256(open(os.path.join(d, "ends_val.pkl"), "rb").read()).hexdigest() and meta["spec"] == spec and meta["frames"] == 16
        ref, _ = PP.dump_frames(PP.ablate(f["model"], PP.spec_edits(spec, f["units"], f["calib"]), f["taps"])[0], f["cache"], "cpu")
        null, _ = PP.dump_frames(f["model"], f["cache"], "cpu")
        got = pickle.load(open(os.path.join(d, "ends_val.pkl"), "rb"))
        assert all(np.array_equal(x["logit"], y["logit"]) for x, y in zip(got, ref)) and any(not np.array_equal(x["logit"], y["logit"]) for x, y in zip(got, null)), "the dumped point is the ablated model, not the null"
        try:
            sw.dump_point(7, spec)
            raise RuntimeError("a point folder must never be overwritten")
        except AssertionError as ex:
            assert "never overwritten" in str(ex)
        assert sorted(os.listdir(os.path.join(tmp, "runs", "e312_p_s0"))) == ["model.pt"], "the stored run's folder gets no file"


def test_score_point_calls_the_stored_scorer():
    calls, old = [], JEV.score
    JEV.score = lambda a: calls.append(a)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            PT.score_point("e312_p_s1", "e312", 12, tmp)
            PT.score_point("e324_s_s0", "e324", 101, tmp)
            os.makedirs(os.path.join(tmp, "e345_p012_e312_p_s1"))
            open(os.path.join(tmp, "e345_p012_e312_p_s1", "ends_card_e345012_snap_thr0.1.counts.pkl"), "w").write("x")      # scored already: skipped
            PT.score_point("e312_p_s1", "e312", 12, tmp)
    finally:
        JEV.score = old
    got = [(a.run, a.seed, a.thr, a.tag, bool(getattr(a, "snap", False)), bool(getattr(a, "attach", False)), bool(getattr(a, "snap_nogeom", False))) for a in calls]
    assert got == [("e345_p012_e312_p_s1", 1, 0.1, "e345012", True, False, False), ("e345_p012_e312_p_s1", 1, 0.1, "e345012", False, False, False),
                   ("e345_p101_e324_s_s0", 0, 0.1, "e345101", True, False, False), ("e345_p101_e324_s_s0", 0, 0.1, "e345101", True, True, True), ("e345_p101_e324_s_s0", 0, 0.1, "e345101", True, False, True),
                   ("e345_p012_e312_p_s1", 1, 0.1, "e345012", False, False, False)], got
    assert PT.card_stem(7, PT.READOUTS["e324"]["primary"]) == "ends_card_e345007_snap_nogeom_attach_thr0.1" and PT.card_stem(7, {}) == "ends_card_e345007_thr0.1"


# ---------------------------------------------------------------- the CPU side on synthetic sweeps and on a stored card
def _synthetic_run(units, cost, loss, extra=None, n=4):
    """a RunData without disk: n frames (half 0, 1, 0, 1); one pass per point whose pairs_hit is lowered so that ends_mean6 loses exactly `loss(unit, f)` (`extra`: {pass key: loss} for more passes)"""
    r = PT.RunData.__new__(PT.RunData)
    r.half, r.family, r.run, r.runs_dir = np.array([0, 1] * (n // 2)), "e312", "e312_p_s0", "/nonexistent"
    pts = PP.points(units)
    base = [100, 100, 80, 80, 100, 100, 80, 80, 1000, 1000, 800, 800]                                             # (n, kp, ok, hit) of entries, exits, pairs
    pair = {k: dict(ap=0.5, p_at_r05=0.5) for k in ("all", "sel", "jud")}
    r.rec = dict(run=r.run, attach=False, rows=list(range(n)), controls={}, points=[dict(id=i, unit=u.name, f=f, group=u.group, kind=u.kind) for i, u, f in pts])
    r.units = {u.name: [p for p in r.rec["points"] if p["unit"] == u.name] for u in units}

    def pas(L, sv):
        c = list(base)
        c[PP.COUNTS.index("pairs_hit")] -= 6 * L * base[PP.COUNTS.index("pairs_n")]                                # ends_mean6 is the mean of six numbers: -6 L x n_pairs hits = -L
        return dict(counts=[c] * n, kp=[[10, 20.0, 30.0, 0.0, 0.0]] * n, pair=pair, saved_macs=sv["macs"], saved_params=sv["params"])
    r.rec["passes"] = {"null": pas(0.0, dict(macs=0.0, params=0))}
    for i, u, f in pts:
        r.rec["passes"][f"p{i:03d}"] = pas(loss(u.name, f), cost.saved(PP.count_edits(u, f, cost.C)))
    for k, L in (extra or {}).items():
        r.rec["passes"][k] = pas(L, dict(macs=0.0, params=0))
    return r


def test_classes_greedy_prefixes_sizes():
    f = fx()
    units, cost = f["units"], f["cost"]
    assert [PT.classify(x, PT.FULL_LIMITS) for x in (0.005, 0.0051, 0.02, 0.0201, 0.05, 0.0501)] == ["FREE", "CHEAP", "CHEAP", "COSTLY", "COSTLY", "CRITICAL"]
    assert [PT.classify(x, PT.SWEEP_LIMITS) for x in (-0.1, 0.003, 0.0031, 0.03, 0.031)] == ["FREE", "FREE", "CHEAP", "COSTLY", "CRITICAL"]
    A, B, C = "jm.blocks.0.hid", "jm.blocks.1.skip", "je.blocks.0.att.h0"                                         # a ladder unit, a whole-block unit, a unit that saves no MACs (parameters only)
    L = {(A, 0.25): 0.006, (A, 0.5): 0.007, (A, 0.75): 0.030, (B, 1.0): 0.005, (C, 1.0): 0.0001}
    r = _synthetic_run(units, cost, lambda u, fr: L.get((u, fr), 0.5))
    mod = PT.Model([r, r], units, cost)
    d = r.delta(f"p{mod.pid[(A, 0.5)]:03d}", "sel")
    assert abs(d["d6"] - 0.007) < 1e-9 and d["dap"] == 0 and d["dpos"] == 0
    assert PT.sweep_free(dict(d6=0.003, dap=0.003, dpos=0.05)) and not PT.sweep_free(dict(d6=0.0031, dap=0, dpos=0)) and not PT.sweep_free(dict(d6=0, dap=0.0031, dpos=0)) and not PT.sweep_free(dict(d6=0, dap=0, dpos=0.051))
    k = mod.klass(mod.pid[(B, 1.0)])
    assert k[:2] == ("CHEAP", "sweep") and abs(k[2] - 0.005) < 1e-9 and mod.free_fraction(B) == 0.0 and mod.free_fraction(C) == 1.0 and mod.klass(mod.pid[(C, 1.0)])[0] == "FREE"
    order = mod.greedy(("maps", "pair"), r)
    names = [(i["unit"], i["f"]) for i in order]
    dm = lambda fr: mod.saved({A: fr})["macs"] / 1e9  # noqa: E731
    assert 0.006 / dm(0.25) > (0.007 - 0.006) / (dm(0.5) - dm(0.25)), "the test needs the second ladder step to look cheaper per GMAC than the first"
    assert names.index((A, 0.25)) < names.index((A, 0.5)) < names.index((A, 0.75)), "a unit's fraction comes only after its lower one"
    small = [k for k, i in enumerate(order) if i["dmacs"] < PT.SMALL_GMACS]
    assert small == list(range(len(order) - len(small), len(order))) and (C, 1.0) in names[-len(small):], "the increments that save less than 0.01 GMACs come last"
    # prefixes: the cut closest to a target share of the heads' MACs (joint savings); the end of the curve when the units run out
    tot = cost.all["macs_maps"] + cost.all["macs_pair"]
    cum, cands = {}, []
    for i in order:
        cum[i["unit"]] = i["f"]
        s = mod.saved(cum)
        cands.append(s["macs_maps"] + s["macs_pair"])
    for t, spec, sv in mod.prefixes(order, (0.10, 0.50, 5.0), tot, "heads"):
        assert abs(sv["macs_maps"] + sv["macs_pair"] - t * tot) == min(abs(c - t * tot) for c in cands)
    assert abs(mod.prefixes(order, (5.0,), tot, "heads")[0][2]["macs_maps"] + mod.prefixes(order, (5.0,), tot, "heads")[0][2]["macs_pair"] - cands[-1]) == 0, "units run out before the target: the end of the curve"
    s = PT.sizes([(0.3, 0.002), (0.6, 0.004), (0.9, 0.012), (1.2, 0.060)])                                         # slopes 0.0067, 0.0267, 0.16: the loss per GMAC more than doubles after the second prefix
    assert s["S_a"] == (0.6, 0.004) and s["S_b"] == (0.9, 0.012) and s["S_c"] == (0.6, 0.004), s
    assert PT.sizes([(0.3, 0.02)])["S_a"] is None


def test_plan_and_control_report():
    f = fx()
    units, cost = f["units"], f["cost"]
    r = _synthetic_run(units, cost, lambda u, fr: 0.0 if u.startswith(("jm.", "je.blocks")) else 0.5)             # the maps head and the set blocks are free, the pyramid and the pair MLPs are not
    mod = PT.Model([r, r], units, cost)
    pl = PT.plan(mod)
    ids = [p["id"] for p in pl]
    assert ids[0] == 0 and pl[0]["set"] == {} and len(set(ids)) == len(ids) and {101, 102, 103, 104, 111, 112} <= set(ids), ids
    top = [p for p in pl if p["name"].startswith("top-saver")]
    assert [round(mod.saved(p["set"])["macs"] / 1e9, 4) for p in top] == [1.1059, 0.7603, 0.5645, 0.1843], "the four units with the largest saved GMACs among the sweep-FREE ones: m05, m1, stream, a skipped block"
    assert not any(p["id"] == 120 for p in pl), "v needs the handle-scale points scored in full"
    assert len({tuple(sorted(p["set"].items())) for p in pl}) == len(pl), "a point set named twice is scored once"
    # controls: of the 4 non-tied ladder units the ranked removal beats the random one in 3 (mean of 3 draws), the reverse one is worse in 3; one unit is tied (0 and 0)
    lad = [u.name for u in units if len(u.fracs) > 1][:5]
    ranked, rand, rev = ([0.01, 0.01, 0.01, 0.01, 0.0], [0.02, 0.02, 0.02, 0.005, 0.0], [0.05, 0.05, 0.05, 0.001, 0.0])
    extra = {}
    for u, a, b, c in zip(lad, ranked, rand, rev):
        extra.update({f"rand{s}_{u}": b for s in range(3)}, **{f"rev_{u}": c})
    r2 = _synthetic_run(units, cost, lambda u, fr: dict(zip(lad, ranked)).get(u, 0.0) if fr == 0.5 else 0.0, extra)
    r2.units = {u: p for u, p in r2.units.items() if u in lad}
    rep = PT.control_report(r2)
    assert rep["units"] == 4 and rep["tied"] == 1 and abs(rep["ranked_beats_random"] - 0.75) < 1e-9 and abs(rep["reverse_worse"] - 0.75) < 1e-9, rep


def test_half_numbers_from_a_stored_card():
    """the stored counts.pkl of E312 summed by halves: the pooled whole is the stored card, the halves are the 438 (selection) / 435 (judgement) frames, and `RunData.full_loss` is the difference of two stored scored graphs"""
    import qgraph_eval as QE
    run = os.path.join(ROOT, "runs", "e312_p_s0")
    c = pickle.load(open(os.path.join(run, "ends_card_snap_thr0.1.counts.pkl"), "rb"))
    assert abs(QE.graph_numbers(c["head"])["topo_F1"] - json.load(open(os.path.join(run, "ends_card_snap_thr0.1.json")))["numbers"]["head"]["topo_F1"]) < 1e-9, "the pooled per-frame counts reproduce the stored card"
    with tempfile.TemporaryDirectory() as tmp:                               # a null folder (the snapped card of the run) and a point folder (its snap-nogeom card): other graphs of the same frames
        for pid, src in ((0, "ends_card_snap_thr0.1"), (5, "ends_card_snap_nogeom_thr0.1")):
            os.makedirs(os.path.join(tmp, PT.point_run("e312_p_s0", pid)))
            os.symlink(os.path.join(run, src + ".counts.pkl"), os.path.join(tmp, PT.point_run("e312_p_s0", pid), PT.card_stem(pid, dict(snap=True)) + ".counts.pkl"))
        sc = np.asarray(c["scenes"])
        uniq = sorted(set(sc.tolist()))
        r = PT.RunData.__new__(PT.RunData)
        r.half, r.family, r.run, r.runs_dir = np.array([uniq.index(s) % 2 for s in sc]), "e312", "e312_p_s0", tmp
        old, PT.JUDGED = PT.JUDGED, {"e312": ("snap",)}
        try:
            c5 = pickle.load(open(os.path.join(run, "ends_card_snap_nogeom_thr0.1.counts.pkl"), "rb"))
            for half, h in (("sel", 0), ("jud", 1)):
                idx = np.where(r.half == h)[0]
                want = QE.graph_numbers([c["head"][i] for i in idx])["topo_F1"] - QE.graph_numbers([c5["head"][i] for i in idx])["topo_F1"]
                assert abs(r.full_loss(5, half) - want) < 1e-12 and abs(r.full(0, "snap", half) - QE.graph_numbers([c["head"][i] for i in idx])["topo_F1"]) < 1e-12
            assert int((r.half == 0).sum()) == 438 and r.full(7, "snap", "sel") is None and r.full_loss(7, "sel") is None
        finally:
            PT.JUDGED = old


TESTS = [test_unit_table_both_models, test_paint_columns_known_answer, test_pair_curve_and_candidates_known_answers, test_halves_are_the_cards, test_classes_greedy_prefixes_sizes, test_plan_and_control_report, test_half_numbers_from_a_stored_card,
         test_score_point_calls_the_stored_scorer, test_dump_frames_equals_jends_eval_dump, test_frame_stats_and_attached_fields, test_calibration_against_numpy_reference, test_appliers_against_independent_references, test_plumbing_controls,
         test_dead_hook_assert_catches, test_saved_cost_against_narrowed_copies, test_sweep_passes_and_point_folder]


if __name__ == "__main__":
    before = {r: listing(r) for r in ("e312_p_s0", "e324_s_s0")}
    want = sys.argv[1:]
    for t in TESTS:
        if not want or t.__name__ in want or t.__name__.replace("test_", "") in want:
            t0 = time.time()
            t()
            print(f"{t.__name__} ok ({time.time() - t0:.0f} s)")
    assert before == {r: listing(r) for r in before}, "a stored run folder changed"
    print("test_jprune_probe: all ok; the stored runs' folders are unchanged")
