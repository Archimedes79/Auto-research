"""The tokens of an SD map (D152 follow-up: moved unchanged from net/model/assign.py, which reads them from here): the roads and the route of a map as points with
direction, lane count and id, in the grid frame (x forward, y left)."""
import numpy as np


def _resample(p, ds):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / ds)), 1) + 1)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1), t


def _tangents(q):
    t = np.gradient(q, axis=0)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


def map_tokens(sd, ds=2.0, convert=True):
    """SD map dict(roads=[(poly, n)], route=[(poly, n)]) -> dict of numpy arrays pos (M, 2), dir (M, 2), n (M,), rid (M,), route (M,) bool, s (M,) arclength in the road;
    a road is a token id 0..R-1, the route pieces follow as ids R.. (flag route); convert: (x right, y forward) -> (x forward, y left)"""
    pos, dr, nn_, rid, rte, ss = [], [], [], [], [], []
    k = 0
    for key in ("roads", "route"):
        for poly, cnt in sd[key]:
            p = np.asarray(poly, np.float64)
            if len(p) < 2 or np.linalg.norm(np.diff(p, axis=0), axis=1).sum() < 1e-6:
                k += 1
                continue
            q, s = _resample(p, ds)
            c = np.interp(s, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]), np.asarray(cnt, np.float64))
            t = _tangents(q)
            if convert:
                q, t = q[:, ::-1] * np.array([1.0, -1.0]), t[:, ::-1] * np.array([1.0, -1.0])
            pos.append(q), dr.append(t), nn_.append(c), rid.append(np.full(len(q), k)), rte.append(np.full(len(q), key == "route")), ss.append(s)
            k += 1
    cat = lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros((0,), dt)  # noqa: E731
    return dict(pos=cat(pos, np.float32).reshape(-1, 2), dir=cat(dr, np.float32).reshape(-1, 2), n=cat(nn_, np.float32), rid=cat(rid, np.int64),
                route=cat(rte, bool), s=cat(ss, np.float32), n_ids=k)
