"""Tests of evaluation/pyr_change.py (the E393 read-out of how much a stage-2 run moved B''s pyramid). CPU only, toy pyramids, no data.
Run (from the problem folder): wsl bash tools/env/_py14.sh test_pyr_change.py"""
import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pyr_change as PC  # noqa: E402
import pyramid  # noqa: E402  (training/grid, on the path through pyr_change)


def pair(seed=0):
    torch.manual_seed(seed)
    a, b = pyramid.PyramidHead(width=8), pyramid.PyramidHead(width=8)
    b.load_state_dict(a.state_dict())
    return a.eval(), b.eval(), [torch.randn(2, a.c_in, 32, 32), torch.randn(1, a.c_in, 32, 32)]


def test_no_change_is_zero():
    a, b, xs = pair()
    rel, fr = PC.output_change(a, b, xs)
    assert fr == 0.0 and set(rel) == set(PC.OUTS) and all(v == 0.0 for v in rel.values())
    assert PC.weight_change(a.state_dict(), b.state_dict()) == (0.0, {k: 0.0 for k in a.state_dict()})


def test_values_and_mutant():
    a, b, xs = pair()
    with torch.no_grad():
        next(b.m05.parameters()).mul_(1.1)                         # the first tensor of the 0.5 m block: +10 %
    rel, fr = PC.output_change(a, b, xs)
    assert rel["feat05"] > 0 and rel["lines05"] > 0 and rel["feat2"] == 0 and rel["feat1"] == 0 and rel["centre1"] == 0, rel    # only what follows m05 moves
    num = sum(float((b(x)[k] - a(x)[k]).double().pow(2).sum()) for x in xs for k in PC.FEATS)
    den = sum(float(a(x)[k].double().pow(2).sum()) for x in xs for k in PC.FEATS)
    assert abs(fr - (num / den) ** 0.5) < 1e-9, (fr, num, den)
    mutant = lambda s, e, bs: PC.output_change(s, s, bs)          # noqa: E731  (compares the start with itself)
    assert mutant(a, b, xs)[1] == 0.0 < fr, "a read-out that compares the start with itself must not pass for this one"
    wr, per = PC.weight_change(a.state_dict(), b.state_dict())
    moved = [k for k, v in per.items() if v > 0]
    assert len(moved) == 1 and moved[0].startswith("m05.") and abs(per[moved[0]] - 0.1) < 1e-6 and 0 < wr < 0.1, (moved, per, wr)


def test_frames_and_refusals():
    assert PC.frame_indices(873, 100) == sorted(set(PC.frame_indices(873, 100))) and len(PC.frame_indices(873, 100)) == 100
    assert PC.frame_indices(873, 100)[0] == 0 and PC.frame_indices(873, 100)[-1] == 872 and PC.frame_indices(3, 10) == [0, 1, 2]
    try:
        PC.weight_change({"a": torch.ones(1)}, {"b": torch.ones(1)})
        raise AssertionError("different tensors must be refused")
    except AssertionError as e:
        assert "differ" in str(e)
    with tempfile.TemporaryDirectory() as d:                        # a read-out that exists is never overwritten (checked before anything is loaded)
        os.makedirs(os.path.join(d, "r"))
        json.dump({}, open(os.path.join(d, "r", "pyr_change_vs_s_n100.json"), "w"))
        try:
            PC.measure("r", "s", 0, 100, "cpu", runs=d)
            raise AssertionError("an existing read-out must not be overwritten")
        except AssertionError as e:
            assert "never overwritten" in str(e), e


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_pyr_change ok: no change -> 0, the formula, a start-vs-start mutant, the one moved tensor, the frame choice, the refusals")
