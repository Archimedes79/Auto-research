"""The pixel-stage targets (`qraster`, E273 card sections 2.1 / 3.4) on REAL lane GT frames: the rasters against the vectors they are drawn from.

    wsl bash tools/env/_py14.sh test_qraster.py [--n 120]

Frames: a seeded sample of the stored lane GT (`test_qtargets.frames`). Checks:
  (a) the cell of every sample point (every 0.1 m) of every line has mask 1; mask and dist agree (dist < 0.25 m -> mask 1; mask 1 -> dist <= 0.36 m, the half diagonal of a cell + the sampling)
  (b) >= 99 % of the lane points (segments of kind lane) lie in cells with road = 1
  (c) the share of the stop / exit events with an intersection cell within 4 m of the position (reported, no gate; the same against the polygons)
  (d) time per frame (reported, target < 40 ms)
  and the centre points go through `losses.grid_loss` (format of `GridData`: float32, unit tangents, padded like `data.collate`).
"""
import argparse
import os
import sys

import numpy as np
import shapely

ROOT =os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import qraster as QR  # noqa: E402
from test_qtargets import frames  # noqa: E402


def cell(p, res):
    return np.clip(np.floor((p + 50.0) / res).astype(int), 0, int(100 / res) - 1)


def lane_points(g):
    out = []
    for p in g["pieces"]:
        q = np.asarray(p["pts"], float)
        for s in p["segments"]:
            if s["kind"] == "lane":
                out.append(q[s["i0"]: s["i1"] + 1])
    return np.concatenate(out) if out else np.zeros((0, 2))


def run(n=120):
    A = dict(pts=0, hit=0, sure=0, sure_miss=0, mask=0, far=0)
    B = dict(lane=0, road=0, frames_bad=[])
    C = []                                                       # per stop / exit event: (distance to the nearest intersection cell centre, distance to the polygons)
    T = []
    parts = {}
    zero_tan = 0
    batch = []
    for k, g in enumerate(frames(n)):
        t = QR.pixel_targets(g)
        T.append(t["ms"])
        for key, v in t["ms_parts"].items():
            parts.setdefault(key, []).append(v)
        assert t["lines_mask"].shape == t["lines_dist"].shape == (3, 200, 200) and t["events1"].shape == (6, 100, 100) and t["road2"].shape == (2, 50, 50)
        assert t["g05"].dtype == t["t05"].dtype == np.float32 and t["g05"].shape == t["t05"].shape and (np.abs(t["g05"]) < 48).all()
        norm = np.linalg.norm(t["t05"], axis=1)
        zero_tan += int((norm < 0.5).sum())
        # (a)
        for c, name in enumerate(QR.LINES):
            polys = QR._polylines(g["lines"][name])
            m, d = t["lines_mask"][c], t["lines_dist"][c]
            if polys:
                p = QR._samples(polys, 0.1)
                p = p[(np.abs(p) < 50).all(1)]
                i = cell(p, 0.5)
                A["pts"] += len(p)
                A["hit"] += int(m[i[:, 0], i[:, 1]].sum())
            A["sure"] += int((d < 0.25).sum())
            A["sure_miss"] += int(((d < 0.25) & (m == 0)).sum())
            A["mask"] += int(m.sum())
            A["far"] += int(((m == 1) & (d > 0.36)).sum())
        # (b)
        p = lane_points(g)
        p = p[(np.abs(p) < 50).all(1)]
        i = cell(p, 2.0)
        ok = int(t["road2"][0][i[:, 0], i[:, 1]].sum())
        B["lane"] += len(p)
        B["road"] += ok
        if ok < len(p):
            B["frames_bad"].append((k, len(p) - ok))
        # (c)
        ii, jj = np.nonzero(t["road2"][1])
        cen = np.stack([-50 + 2 * (ii + 0.5), -50 + 2 * (jj + 0.5)], 1)
        inter = QR._polygons(g["areas"]["intersection"], 0.0)
        for e in g["events"]:
            if e["type"] in ("stop", "exit"):
                pos = np.asarray(e["pos"], float)
                C.append((np.linalg.norm(cen - pos, axis=1).min() if len(cen) else np.inf, min([float(q.distance(shapely.Point(*pos))) for q in inter], default=np.inf)))
        if k < 2:
            batch.append(t)
    # the centre points through grid_loss
    import torch
    import losses as L
    G = max(len(t["g05"]) for t in batch)
    g05, t05, m05 = torch.full((len(batch), G, 2), 1e4), torch.zeros(len(batch), G, 2), torch.zeros(len(batch), G)
    for b, t in enumerate(batch):
        m = len(t["g05"])
        g05[b, :m], t05[b, :m], m05[b, :m] = torch.from_numpy(t["g05"]), torch.from_numpy(t["t05"]), 1.0
    out = torch.randn(len(batch), 6, 100, 100, requires_grad=True)
    loss, info = L.grid_loss(out, dict(g05=g05, t05=t05, m05=m05))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(out.grad).all() and out.grad.abs().sum() > 0, info
    C = np.array(C)
    r = dict(hit=A["hit"] / max(A["pts"], 1), sure_miss=A["sure_miss"], far=A["far"], road=B["road"] / max(B["lane"], 1), c_cell4=float((C[:, 0] <= 4).mean()), c_poly4=float((C[:, 1] <= 4).mean()), ms=float(np.mean(T)))
    print(f"frames {len(T)}: (a) line samples {A['pts']}, in a mask cell {r['hit']:.5f}; cells with dist < 0.25 and mask 0: {A['sure_miss']} of {A['sure']}; mask cells with dist > 0.36: {A['far']} of {A['mask']}")
    print(f"(b) lane points {B['lane']}, in a road cell {r['road']:.4f}; frames with a miss {len(B['frames_bad'])}: {B['frames_bad'][:8]}")
    print(f"(c) {len(C)} stop / exit events: intersection cell within 4 m {r['c_cell4']:.3f} (within 2 m {(C[:, 0] <= 2).mean():.3f}, median {np.median(C[:, 0]):.2f} m); "
          f"polygon within 4 m {r['c_poly4']:.3f} (within 1 m {(C[:, 1] <= 1).mean():.3f}, within 0.1 m {(C[:, 1] <= 0.1).mean():.3f})")
    print(f"(d) ms per frame: mean {r['ms']:.1f}, median {np.median(T):.1f}, p95 {np.percentile(T, 95):.1f}, max {np.max(T):.1f}; per target (mean): " + ", ".join(f"{k} {np.mean(v):.1f}" for k, v in parts.items())
          + f"   [target < 40: {'met' if r['ms'] < 40 else 'NOT met'}]")
    print(f"centre points: {sum(len(t['g05']) for t in batch)} in the first 2 frames through grid_loss ok (loss {float(loss):.3f}); tangents of norm < 0.5 in all frames: {zero_tan}")
    return r, A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    r, A, B = run(ap.parse_args().n)
    assert A["hit"] == A["pts"] > 0, (A["hit"], A["pts"])
    assert A["sure_miss"] == 0 and A["far"] == 0, A
    assert r["road"] >= 0.99, r
    print("test_qraster ok")


if __name__ == "__main__":
    main()
