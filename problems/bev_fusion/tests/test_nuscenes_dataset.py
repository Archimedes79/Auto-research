"""Tests of net/prep/nuscenes_dataset.py (its former self-test).

    python -P -m bev_fusion.tests.test_nuscenes_dataset      (from the problem folder, with its module path)
"""
import math
import os

import numpy as np
import torch

from GtGenerator import pose_matrix
from bev_fusion.net import netpath
from bev_fusion.net.prep.geometry import Grid
from bev_fusion.net.prep.nuscenes_dataset import FrontRadarDataset, RADARS, _PCD_KINDS, collate, parse_radar_pcd, read_files


def _selftest_per_frame(ds, root):
    """radar_per_frame (t25): one set per camera frame at that frame's own camera time; radar_window 0.09 = the newest sweep of each radar.
    ds: a mini dataset (its nusc is shared). -> the mean number of one-sweep detections inside the grid per keyframe"""
    kw = dict(grid=ds.grid, n_cam=3, cam_stride=2, with_gt=False, radar_window=0.09, nusc=ds.nusc)
    flat, per = FrontRadarDataset(root, "v1.0-mini", **kw), FrontRadarDataset(root, "v1.0-mini", radar_per_frame=True, **kw)
    sq = per[10]
    assert len(sq["radar_seq"]) == 3 and sq["radar_seq"][0] is sq["radar"], "frame 0 of radar_seq is the item's radar"
    assert all(torch.equal(sq["radar"][k], flat[10]["radar"][k]) for k in sq["radar"]), "radar_seq[0] = the one-sweep radar of the same sample"
    for k, r in enumerate(sq["radar_seq"]):
        t_k = sq["frames"][k]["timestamp"]
        assert 0 <= float(r["dt"].min()) and float(r["dt"].max()) <= 0.09 + 1e-6, (k, float(r["dt"].max()))
        for si, ch in enumerate(RADARS):                              # the newest sweep of this radar not after THIS frame's time
            sd = ds.nusc.get("sample_data", ds.nusc.get("sample", ds.samples[10])["data"][ch])
            while sd["next"] and ds.nusc.get("sample_data", sd["next"])["timestamp"] <= t_k:
                sd = ds.nusc.get("sample_data", sd["next"])
            while sd["timestamp"] > t_k:
                sd = ds.nusc.get("sample_data", sd["prev"])
            if (r["sensor"] == si).any():
                assert abs(float(r["dt"][r["sensor"] == si].min()) - (t_k - sd["timestamp"]) * 1e-6) < 1e-6, (k, ch)
    assert not torch.equal(sq["radar_seq"][0]["x"], sq["radar_seq"][2]["x"]), "the oldest frame's sweeps are other detections"
    g, n = ds.grid, []                                                # the points inside the grid per keyframe (54 on mini)
    for i in range(0, len(flat), 8):
        r = flat[i]["radar"]
        n.append(int(((r["x"] >= g.x_min) & (r["x"] < g.x_max) & (r["y"] >= g.y_min) & (r["y"] < g.y_max)).sum()))
    assert 40 < np.mean(n) < 70, f"{np.mean(n):.1f} points in the grid per keyframe with one sweep"
    bs = collate([per[10], per[11]])
    assert len(bs["radar_seq"]) == 3 and int(bs["radar_seq"][2]["b"].max()) == 1 and all(torch.equal(bs["radar_seq"][0][k], bs["radar"][k]) for k in bs["radar"])
    return float(np.mean(n))


def _selftest_gt_identity(ds):
    """the GT identity in cache names and records is the STORE key of the generator (the GT, after `same_gt_as`), never its code signature; and it is the key
    the files of the site's GT cache carry (a key that moved would leave 32,449 cached keyframes unread and fill the cache again) -> files that carry it"""
    sig = ds.gt_signature()
    # the key of the GT store of today (front v13): a new GT version changes this line together with the cache build
    assert sig == "0.5_4_a107e0b4e8" == f"{ds.grid.res}_{ds.gt_ss}_{ds.gt.store_signature()}", sig
    code, store = ds.gt.signature, ds.gt.store_signature
    ds.gt.signature, ds.gt.store_signature = (lambda: "c0de"), (lambda: "5t0re")
    try:
        assert ds.gt_signature() == f"{ds.grid.res}_{ds.gt_ss}_5t0re", ds.gt_signature()
    finally:
        ds.gt.signature, ds.gt.store_signature = code, store
    cache = os.path.join(netpath.SITE, "data", "cache", "gt")
    assert os.path.isdir(cache), f"the GT cache {cache} is missing (README: what the tests need beside the code)"
    names = [n for n in os.listdir(cache) if n.endswith(".npz")]
    mine = [n for n in names if n.endswith(f"_{sig}.npz")]
    assert len(mine) >= 32000, f"{len(mine)} of the {len(names)} files of {cache} carry the GT identity {sig} (e.g. {names[:1]}; 32,449 on 2026-11-28): the cache would be filled again"
    return len(mine)


def _selftest():
    root = netpath.nuscenes_root(mini=True)
    ds = FrontRadarDataset(root, "v1.0-mini", grid=Grid(), n_cam=3, cam_stride=2, with_gt=True)
    assert len(ds) > 300, len(ds)
    it = ds[10]
    fr = it["frames"]
    assert len(fr) == 3 and fr[0]["img"].shape == (3, 256, 704) and fr[0]["steps"] == []
    assert len(fr[1]["steps"]) == 2 and len(fr[2]["steps"]) == 4, [len(f["steps"]) for f in fr]
    dts = [sum(s[0] for s in f["steps"]) for f in fr]
    assert all(0.1 < d < 0.25 for d in [dts[1]]) and 0.25 < dts[2] < 0.45, dts
    assert fr[0]["timestamp"] > fr[1]["timestamp"] > fr[2]["timestamp"]
    assert fr[0]["cam_vec"].shape == (27,)
    rad = it["radar"]
    assert len(rad["x"]) > 50 and float(rad["dt"].max()) <= ds.radar_window + 1e-6 and float(rad["dt"].min()) >= 0
    # the vectorised .pcd parser gives the devkit's numbers (default filters) on real files of every radar, and for an empty cloud
    from nuscenes.utils.data_classes import RadarPointCloud
    n_pcd = 0
    for i in (10, 150):
        smp = ds.nusc.get("sample", ds.samples[i])
        for ch in RADARS:
            sd = ds.nusc.get("sample_data", smp["data"][ch])
            for _ in range(3):
                path = os.path.join(ds.root, sd["filename"])
                ref, mine = RadarPointCloud.from_file(path).points, parse_radar_pcd(read_files([path, path])[0])
                assert ref.shape == mine.shape and np.array_equal(ref, mine), (path, ref.shape, mine.shape)
                n_pcd += 1
                sd = ds.nusc.get("sample_data", sd["prev"])
    import tempfile
    raw = open(path, "rb").read()
    cut = raw.find(b"DATA binary\n")
    lines = [l for l in raw[:cut].decode().split("\n") if l]
    dt = np.dtype([(n, f"<{_PCD_KINDS[t]}{s}") for n, t, s in zip(lines[2].split(" ")[1:], lines[4].split(" ")[1:], lines[3].split(" ")[1:])])
    one = np.zeros(1, dt)
    one["x"] = np.nan
    empty = "\n".join(l.replace("WIDTH " + l.split(" ")[-1], "WIDTH 1").replace("POINTS " + l.split(" ")[-1], "POINTS 1") if l.startswith(("WIDTH", "POINTS")) else l
                      for l in lines).encode() + b"\nDATA binary\n" + one.tobytes() + b"\0"
    with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as tf:
        tf.write(empty)
    try:
        assert RadarPointCloud.from_file(tf.name).points.shape == parse_radar_pcd(empty).shape == (18, 0), "an empty cloud"
    finally:
        os.remove(tf.name)
    assert set(rad["sensor"].unique().tolist()) <= set(range(5)) and len(rad["sensor"].unique()) >= 3
    # static compensation is exact: a static radar point of an older sweep, transformed by the poses, is consistent
    # with the steps chain of splat.py (first-order translation per frame) to a few centimetres
    from bev_fusion.net.model.splat import chain_positions
    p = torch.tensor([[[30.0, 5.0]]], dtype=torch.float64)
    T_old = pose_matrix(ds.nusc.get("ego_pose", ds.nusc.get("sample_data", fr[2]["token"])["ego_pose_token"]))
    T_cur = pose_matrix(ds.nusc.get("ego_pose", ds.nusc.get("sample_data", fr[0]["token"])["ego_pose_token"]))
    q = np.linalg.inv(T_cur) @ T_old @ np.array([30.0, 5.0, 0.0, 1.0])
    c = chain_positions(p, fr[2]["steps"])[0, 0]
    assert math.hypot(c[0] - q[0], c[1] - q[1]) < 0.1, (c, q[:2])
    # GT present with the three keyframes and their steps
    g = it["gt"]
    assert set(g) == {"cur", "prev", "next"} and abs(g["next"]["meta"]["dt"] - 0.5) < 0.1
    assert all(it["gap_steps"][k] == g[k]["meta"]["steps"] and len(it["gap_steps"][k]) >= 5 for k in ("prev", "next")), \
        "the ego steps to the neighbouring keyframes are made from the poses here, equal to the ones the GT carries"
    assert it["radar_poses"].shape == (5, 4, 4) and bool((it["radar_poses"][:, 3, 3] == 1).all())
    assert g["cur"]["drivable_area"].shape == (120, 104)
    # collation of two samples
    b = collate([ds[10], ds[11]])
    assert b["img"].shape == (2, 3, 3, 256, 704) and b["K"].shape == (2, 3, 3, 3)
    assert len(b["steps"]) == 3 and len(b["steps"][2]) == 4 and b["steps"][2][0][0].shape == (2,)
    assert int(b["radar"]["b"].max()) == 1 and len(b["radar"]["b"]) == len(b["radar"]["x"]) and "radar_seq" not in b and "radar_seq" not in it
    n_grid = _selftest_per_frame(ds, root)
    n_cached = _selftest_gt_identity(ds)
    print(f"nuscenes_dataset selftest ok: {len(ds)} mini samples; 3 camera frames 0 / {dts[1]:.3f} / {dts[2]:.3f} s old "
          f"with 0 / 2 / 4 ego steps; {len(rad['x'])} radar detections from {len(rad['sensor'].unique())} radars, age <= "
          f"{float(rad['dt'].max()):.3f} s; pose transform = steps chain within {math.hypot(c[0] - q[0], c[1] - q[1]):.3f} m; "
          f"GT cur / prev / next; collation; radar_per_frame: one set per camera frame, frame 0 = the radar_window 0.09 set, "
          f"{n_grid:.1f} points in the grid per keyframe; the GT identity is the generator's store key ({ds.gt_signature()}), "
          f"{n_cached} files of the site's GT cache carry it")


if __name__ == "__main__":
    _selftest()
