"""Tests of net/model/interface.py (its former self-test).

    python -P -m bev_fusion.tests.test_interface      (from the problem folder, with its module path)
"""
import torch

from bev_fusion.net.model.interface import RADAR_FIELDS, check_inputs, check_outputs


def _selftest():
    B, Fn = 2, 3
    z = torch.zeros
    radar = dict({f: z(5) for f in RADAR_FIELDS}, b=torch.tensor([0, 0, 1, 1, 1]))
    batch = {"K": z(B, Fn, 3, 3), "T_cam_car": z(B, Fn, 4, 4), "ida": None, "v": z(B), "omega": z(B),
             "steps": [[] for _ in range(Fn)], "radar": radar,
             "prob": z(B, Fn, 4, 2, 3), "ctx": z(B, Fn, 80, 2, 3), "hw": (16, 24), "tokens": ["a", "b"]}
    assert check_inputs(batch) == [], check_inputs(batch)
    e = check_inputs(dict(batch, radar={k: v for k, v in radar.items() if k != "vr"}))
    assert any("'vr'" in x for x in e), e
    assert any("lengths" in x for x in check_inputs(dict(batch, radar=dict(radar, rcs=z(4)))))
    assert check_inputs(dict(batch, radar_seq=[radar] * Fn)) == []                               # radar_per_frame: one set per camera frame
    e = check_inputs(dict(batch, radar_seq=[radar, {k: v for k, v in radar.items() if k != "vr"}]))
    assert any("2 radar sets != F = 3" in x for x in e) and any("radar_seq[1]: missing field 'vr'" in x for x in e), e
    bad = dict(batch, v=z(B + 1), extra=1)
    del bad["omega"]
    e = check_inputs(bad)
    assert any("omega" in x for x in e) and any("extra" in x for x in e) and any("v:" in x for x in e), e
    bad = dict(batch, ctx=torch.full((B, Fn, 80, 2, 3), float("nan")))
    assert any("not finite" in x for x in check_inputs(bad))
    assert check_inputs(dict(batch, lane_img_old=z(B, Fn - 1, 3, 4, 4), img_jitter={"u": z(4, B), "strength": 1.0, "noise_seed": 0})) == [], "t26 student_frames 'all': one older image per older frame"
    assert any("lane_img_old" in x for x in check_inputs(dict(batch, lane_img_old=z(B, Fn, 3, 4, 4)))) and any("lane_img_old" in x for x in check_inputs(dict(batch, lane_img_old=z(B + 1, Fn - 1, 3, 4, 4)))), \
        "lane_img_old: a wrong number of older frames or of samples is named"
    try:
        check_inputs(dict(batch, radar=dict({f: z(1) for f in RADAR_FIELDS}, b=torch.tensor([3]))), raise_on=True)
        raise AssertionError("no raise")
    except ValueError as ex:
        assert "sample index" in str(ex)
    p = {"sem": z(B, 12, 8, 6), "obj": {"hm": z(B, 3, 8, 6)}, "mass": z(B, 1, 8, 6)}
    res = {"out": {0.0: dict(p, v_hat=z(B, 2, 8, 6)), 0.5: p}, "state": {}, "features": z(B, 4, 8, 6)}
    assert check_outputs(res, (0.0, 0.5)) == [], check_outputs(res, (0.0, 0.5))
    assert any("times" in x for x in check_outputs(res, (0.0,)))
    res2 = {"out": {0.0: {"sem": z(B, 12, 8, 6), "obj": {}, "foo": 1}}, "state": {}, "features": None}
    e = check_outputs(res2, (0.0,))
    assert any("mass" in x for x in e) and any("foo" in x for x in e) and any("v_hat" in x for x in e), e
    print("interface selftest passed: input and output contracts checked (missing, unknown, shape, finite)")


if __name__ == "__main__":
    _selftest()
