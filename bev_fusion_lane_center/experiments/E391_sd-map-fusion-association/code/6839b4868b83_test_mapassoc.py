"""E391 tests (`training/data/lane_mapassoc_data.py`, `training/grid/jmapassoc.py`, `evaluation/mapassoc_probe.py`); CPU.

    wsl bash tools/env/_py14.sh training/grid/test_mapassoc.py

perception untouched (D135): for every arm a map loss sends no gradient into the token tensors (they are detached in the head); the textual mutant without the detach is caught. labels: the association rule by hand
(inside an intersection -> the intersection holding the point, a split stretch -> unlabelled, a false token -> unassigned). data (needs the VAL label cache of seed 1, else skipped): the targets of an item (a missing
segment -> unassigned, a false token -> unassigned, the lane class only for present road segments), collate shapes, one forward and backward of every arm on a real batch, and the read-through of a null.
"""
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in (".", "../data", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jmapassoc as JM  # noqa: E402
import lane_mapassoc_data as MA  # noqa: E402

torch.set_num_threads(2)


def mutant(mod, *edits):
    src = open(mod.__file__, encoding="utf-8").read()
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    m = types.ModuleType(mod.__name__ + "_mutant")
    m.__file__ = mod.__file__
    exec(compile(src, mod.__file__, "exec"), m.__dict__)
    return m


def _batch(B=2, K=6, N=4):
    t = dict(tok=torch.randn(B, K, 64, requires_grad=True), pos=torch.randn(B, K, 2, requires_grad=True), dirv=torch.randn(B, K, 2), e=torch.randn(B, K), flags=torch.randn(B, K, 8), valid=torch.ones(B, K, dtype=torch.bool))
    return t, torch.randn(B, N, 64), torch.ones(B, N, dtype=torch.bool), torch.randn(B, K, N, JM.PA_F)


def grads_none(mod):
    for arm in mod.ARMS:
        t, h, sv, pa = _batch()
        o = mod.MapAssocHead(arm)(t, h, sv, pa)
        loss = o["lanes"].sum() + o["route"].sum() + (o["assoc"][..., 0].sum() if "assoc" in o else 0)
        loss.backward()
        assert t["tok"].grad is None and t["pos"].grad is None, f"{arm}: a map loss reached the token tensors"
    return True


def test_perception_untouched():
    assert grads_none(JM)
    M = mutant(JM, ('        tok = t["tok"].detach().float()', '        tok = t["tok"].float()'), ('        pos, dirv = t["pos"].detach().float(), t["dirv"].detach().float()', '        pos, dirv = t["pos"].float(), t["dirv"].float()'))
    try:
        grads_none(M)
    except AssertionError:
        print("perception untouched: no gradient into the token tensors for every arm; the no-detach mutant caught: ok")
        return
    raise AssertionError("the no-detach mutant was not caught")


def test_labels():
    MA.selftest()
    print("labels: the association rule by hand: ok")


def test_data():
    try:
        MA.load("val", 1)
    except FileNotFoundError:
        print("data: skipped (no VAL label cache of seed 1)")
        return
    import mapassoc_probe as MP
    data = MP.AssocData("val", 1, frames=list(range(4)), level="nd2")
    items = [data[j] for j in range(4)]
    for it in items:
        kd, a, lab = it["kind"], it["assoc"], it["lanes"]
        assert ((kd == 1) <= (a == -1)).all() and ((kd == 2) <= (a == -1)).all() and ((kd == 0) <= (a >= 0)).all(), "a missing segment or a false token must be 'unassigned'"
        assert ((lab >= 0) <= (kd == 0)).all(), "a lane class only for a present segment"
        pres = it["map"]["present"]
        assert (pres[a[a >= 0]]).all(), "a target segment must be present"
    b = MP.collate(items)
    N = b["map"]["x"].shape[1]
    assert b["pa"].shape == (4, 512, N, JM.PA_F) and b["assoc"].shape == (4, 512)
    enc, _ = MP.SP.load_models("g7", 1)
    h, sv = MP.encode(enc.eval(), b, "cpu")
    for arm in JM.ARMS:
        head = JM.MapAssocHead(arm)
        o = head(b, h, sv, b["pa"]) if arm != "z" else head(b)
        loss, parts = MP.losses(o, b)
        loss.backward()
        assert torch.isfinite(loss)
    lanes, route = MP.read_through(b, 0, b["assoc"][0].numpy())
    assert lanes.shape == (512,) and set(np.unique(route)) <= {0.0, 1.0}
    print("data: targets (missing segment / false token -> unassigned, lane class only for present road segments), collate shapes, every arm forward + backward on a real batch, the read-through: ok")


TESTS = [test_perception_untouched, test_labels, test_data]
if __name__ == "__main__":
    for t_ in TESTS:
        t_()
    print("test_mapassoc ok")
