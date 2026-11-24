"""Tests of evaluation/maptrust_gate.py (E394 v1 arm g): the label, the label-free frame features, the logistic and its threshold. CPU only, no data, no weights.
Run (from the problem folder): wsl bash tools/env/_py14.sh test_maptrust_gate.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import maptrust_gate as G  # noqa: E402


def test_label():
    assert G.gate_label(dict(wrong_scene=1)) and G.gate_label(dict(wrong_route=1, wrong_scene=1, pose=1.0))
    assert not G.gate_label(dict(pose=1.0, shift=5.0)) and not G.gate_label(dict(yaw=6.0)), "a shifted / turned map does not harm F1's lanes read (step 0): no gate label"
    assert not G.gate_label(dict(wrong_route=1, pose=1.0)) and not G.gate_label({}) and not G.gate_label(dict(pose=2.0, count_rate=0.1, outdated=0.3))


def test_features():
    K, N = 6, 3                                                    # N map segments + the unassigned slot
    lanes_in, seg_kind = np.array([1, 2, 0]), np.array([0, 0, 1])   # segment 2 is not a road
    sure = np.full((K, N + 1), -20.0)
    sure[np.arange(K), [0, 0, 1, 1, 2, 2]] = 20.0                  # every token sure of a segment
    z_lanes = np.array([1, 1, 2, 3, 0, 0])
    kept = np.array([1, 1, 1, 1, 1, 0], bool)
    x = G.frame_features(sure, np.array([0, -1, 1, 1, 2, 2]), z_lanes, kept, lanes_in, seg_kind)
    assert np.allclose(x[:2], 1.0, atol=1e-6) and abs(x[2] - 0.8) < 1e-12, x       # 4 of the 5 kept tokens assigned by p6
    assert abs(x[3] - (3 + 1) / (4 + 2)) < 1e-12, x                # known road tokens 0 - 3 (token 4 sits on a non-road segment, 5 is not kept); z agrees on 0, 1, 2
    none = np.full((K, N + 1), -20.0)
    none[:, -1] = 20.0                                             # every token unassigned: no known lane, the Laplace prior
    x = G.frame_features(none, np.full(K, -1), z_lanes, kept, lanes_in, seg_kind)
    assert np.allclose(x, [0.0, 0.0, 0.0, 0.5], atol=1e-6), x
    assert np.allclose(G.frame_features(sure, np.zeros(K, int), z_lanes, np.zeros(K, bool), lanes_in, seg_kind), [0, 0, 0, 0.5])
    full = G.frame_features(sure, np.array([0, -1, 1, 1, 2, 2]), lanes_in[np.clip([0, 0, 1, 1, 2, 2], 0, 2)], kept, lanes_in, seg_kind)
    assert abs(full[3] - (4 + 1) / (4 + 2)) < 1e-12 and full[3] > (3 + 1) / (4 + 2), "z agreeing with the map on every known token: 5/6 against 4/6"
    off = G.frame_features(sure, np.array([0, -1, 1, 1, 2, 2]), np.full(K, 3), kept, lanes_in, seg_kind)
    assert abs(off[3] - (0 + 1) / (4 + 2)) < 1e-12, "z never agreeing: 1/6 (a lane_agree that ignored z would not move)"


def test_combine():
    ap, l1, zl = np.array([2, -1, 0]), np.array([1, 2, 3]), np.array([0, 0, 1])
    a, l = G.combine(True, ap, l1, zl)
    assert (a == ap).all() and (l == l1).all(), "a trusted frame keeps F1's read"
    a, l = G.combine(False, ap, l1, zl)
    assert (a == -1).all() and (l == zl).all() and a.dtype == ap.dtype, "a distrusted frame: every token unassigned, the no-map lanes"


def test_fit_and_threshold():
    rng = np.random.default_rng(0)
    right = rng.normal([0.9, 0.8, 0.8, 0.9], 0.05, (300, 4))
    wrong = rng.normal([0.4, 0.3, 0.2, 0.4], 0.1, (150, 4))
    X, y = np.vstack([right, wrong]), np.r_[np.zeros(300), np.ones(150)].astype(int)
    f = G.fit_logistic(X, y)
    p = G.gate_prob(f, X)
    assert f["auroc"] > 0.99 and abs(f["trusted_right"] - G.TRUST_Q) < 0.01 and f["rejected_wrong"] > 0.95, f
    assert abs(float((p[y == 0] <= f["thr"]).mean()) - G.TRUST_Q) < 0.01
    assert f["feats"] == list(G.FEATS) and len(f["coef"]) == 4
    try:
        G.fit_logistic(right, np.zeros(300, int))
        raise AssertionError("a fit without wrong maps must be refused")
    except AssertionError as e:
        assert "right and wrong" in str(e), e


def test_card_levels():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE)))
    import maperror_probe as MEP
    names = dict(MEP.spec_list(["all"]))
    missing = [n for n in G.CARD_LEVELS if n not in names]
    assert not missing, missing


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_maptrust_gate ok: the label (the map of another scene), the four label-free features (lane_agree moves with z), the fall-back of a distrusted frame, the logistic and its TRAIN threshold, the card's levels exist in the probe")


if __name__ == "__main__":
    main()
