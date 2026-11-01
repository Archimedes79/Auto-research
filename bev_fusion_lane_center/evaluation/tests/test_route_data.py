"""Spec test of training/pipeline/route_data.py (the one copy of resample_matrix / route_item for the joint trainers): the interpolation matrix times the tokens
equals p16_tok_resample.resample_item on every new point that has a source. CPU, toy item.
Usage (gcar): bash tools/env/_py14.sh test_route_data.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_tok_resample as TR  # noqa: E402
import route_data as RD  # noqa: E402


def toy():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 9.0], [25.0, 9.0]])
    return dict(xy=xy, tok=np.random.RandomState(0).normal(size=(4, 8)).astype(np.float32), has=np.array([True, True, False, True]), pieces=[[0, 1, 2, 3]],
                keep=np.array([0.9, 0.8, 0.5, 0.7], np.float32), cp=np.zeros(4, np.float32), edges=[(0, 1), (1, 2), (2, 3)])


def test_matrix_times_tokens_is_the_resampled_item():
    it = toy()
    for step in (4.0, 3.0):
        M = RD.resample_matrix(it, step)
        new = TR.resample_item(it, step)
        assert M.shape == (len(new["xy"]), len(it["xy"])), (M.shape, len(new["xy"]))
        has_src = M.sum(1) > 0
        assert has_src.any() and not has_src.all(), "the toy has points with and without a source"
        assert np.allclose((M @ it["tok"])[has_src], new["tok"][has_src], atol=2e-3)   # resample_item stores float16
        assert np.allclose(M[has_src].sum(1), 1.0), "interpolation weights add up to one"


def test_every_original_point_with_a_token_keeps_its_row():
    it = toy()
    M = RD.resample_matrix(it, 4.0)
    for i in range(4):
        assert (M[i].sum() == 1.0 and M[i, i] == 1.0) == bool(it["has"][i]), i


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
            n += 1
    print(f"test_route_data: {n} tests passed")
