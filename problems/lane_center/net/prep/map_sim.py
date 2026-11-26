"""The map side of the lane head (E273): the lane GT pieces of a frame, with simulated map errors, as the tokens the assignment reads.

    mp = make_map(g, mode="exact" | "error" | "train", rng=None, token=None)

The map is what a lane-level navigation map would deliver around the ego: the pieces of the lane GT (ids that change only at splits and merges, successor / predecessor / left / right /
opposite ids, lane index and count, road) with errors (the correct correspondence is known by construction: a node's map piece is its own GT piece):
  exact   no error
  error   a FIXED perturbation per frame (seeded by the sample token, so every evaluation sees the same map): translation 1-2 m in a random direction, yaw N(0, 1 deg), shape points thinned by
          Douglas-Peucker (0.5-1.5 m), 5 % of the pieces dropped, 5 % of the lane counts off by one
  train   random: with p = 0.25 exact, else translation U(0, 2) m, yaw N(0, 1 deg), thinning 0.3-1.5 m, drop 5 %, lane count wrong 5 %
mp (dict): K pieces, T tokens (every 2 m along the pieces):
  pos (T, 2) float32, tan (T, 2), frac (T,) arclength fraction in the piece, pidx (T,) piece index, K, ids (K,) piece ids (list of str), part ids (K,), road (K,) int index of the road,
  pfeat (K, 12) piece features (kind one-hot lane / connector / mixed, lane count, lane index, number of successors / predecessors, length / 100, start_open, end_open, 0, 0),
  adj (5, K, K) bool: succ, pred, left, right, opposite among the kept pieces, dropped (list of ids), shift (2,), yaw (float, rad), n_roads
`piece_targets(g, mp)` -> for every GT piece of `g` the list of map piece indices with the same `piece` id (empty = dropped from the map: the dustbin).
Run the self-test: python map_sim.py
"""
import hashlib

import numpy as np

STEP = 2.0
KINDS = ("lane", "connector", "mixed")


def _resample(p, step=STEP):
    p = np.asarray(p, float)
    if len(p) < 2:
        return p, np.zeros(len(p))
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    t = np.linspace(0.0, L, max(int(np.ceil(L / step)), 1) + 1)
    q = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
    return q, (t / max(L, 1e-9))


def _simplify(p, eps):
    if len(p) < 3 or eps <= 0:
        return np.asarray(p, float)
    from shapely.geometry import LineString
    return np.asarray(LineString(p).simplify(eps, preserve_topology=False).coords, float)


def _rng_for(token):
    seed = int(hashlib.md5(str(token).encode()).hexdigest()[:8], 16)
    return np.random.RandomState(seed)


def make_map(g, mode="exact", rng=None, token=None):
    if rng is None:
        rng = _rng_for(token if token is not None else g.get("sample", "x"))
    pieces = g["pieces"]
    if mode == "exact" or (mode == "train" and rng.rand() < 0.25):
        shift, yaw, eps, p_drop, p_cnt = np.zeros(2), 0.0, 0.0, 0.0, 0.0
    elif mode == "error":
        ang = rng.uniform(0, 2 * np.pi)
        shift, yaw, eps, p_drop, p_cnt = rng.uniform(1.0, 2.0) * np.array([np.cos(ang), np.sin(ang)]), np.deg2rad(rng.randn()), rng.uniform(0.5, 1.5), 0.05, 0.05
    elif mode == "train":
        ang = rng.uniform(0, 2 * np.pi)
        shift, yaw, eps, p_drop, p_cnt = rng.uniform(0.0, 2.0) * np.array([np.cos(ang), np.sin(ang)]), np.deg2rad(rng.randn()), rng.uniform(0.3, 1.5), 0.05, 0.05
    else:
        raise ValueError(mode)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    keep = [i for i in range(len(pieces)) if not (p_drop > 0 and rng.rand() < p_drop)]
    kmap = {i: k for k, i in enumerate(keep)}
    dropped = [pieces[i]["id"] for i in range(len(pieces)) if i not in kmap]
    roads = sorted({str(pieces[i]["road"]) for i in keep})
    pos, tan, frac, pidx, pfeat, road = [], [], [], [], [], []
    for i in keep:
        pc = pieces[i]
        q = _simplify(np.asarray(pc["pts"], float) @ R.T + shift, eps) if eps > 0 else np.asarray(pc["pts"], float) @ R.T + shift
        q, fr = _resample(q)
        t = np.gradient(q, axis=0) if len(q) > 1 else np.tile([0.0, 1.0], (len(q), 1))
        t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        pos.append(q), tan.append(t), frac.append(fr), pidx.append(np.full(len(q), kmap[i]))
        lane_count = pc["lane_count"] + (rng.choice([-1, 1]) if (p_cnt > 0 and rng.rand() < p_cnt) else 0)
        kind = np.zeros(3)
        kind[KINDS.index(pc["kind"]) if pc["kind"] in KINDS else 0] = 1
        pfeat.append(np.concatenate([kind, [max(lane_count, 1) / 4.0, pc["lane_index"] / 4.0, len(pc["succ"]) / 3.0, len(pc["pred"]) / 3.0, pc["length"] / 100.0,
                                           float(pc["start_open"]), float(pc["end_open"]), 0.0, 0.0]]))
        road.append(roads.index(str(pc["road"])))
    K = len(keep)
    idx = {pieces[i]["id"]: kmap[i] for i in keep}
    adj = np.zeros((5, K, K), bool)
    for i in keep:
        for r, key in enumerate(("succ", "pred", "left", "right", "opposite")):
            for o in pieces[i][key]:
                if o in idx:
                    adj[r, kmap[i], idx[o]] = True
    cat = lambda a, dt: (np.concatenate(a).astype(dt) if a else np.zeros((0,) + ((2,) if a is pos or a is tan else ()), dt))  # noqa: E731
    return dict(pos=cat(pos, np.float32).reshape(-1, 2), tan=cat(tan, np.float32).reshape(-1, 2), frac=cat(frac, np.float32), pidx=cat(pidx, np.int64),
                K=K, ids=[pieces[i]["piece"] for i in keep], part_ids=[pieces[i]["id"] for i in keep], road=np.array(road, np.int64), pfeat=np.array(pfeat, np.float32).reshape(K, 12),
                adj=adj, dropped=dropped, shift=shift, yaw=yaw, n_roads=len(roads), roads=roads)


def piece_targets(g, mp):
    """for every piece of the GT graph: the indices of the map pieces with the same `piece` id ([] = not in the map: the dustbin)"""
    by = {}
    for k, pid in enumerate(mp["ids"]):
        by.setdefault(pid, []).append(k)
    return [by.get(p["piece"], []) for p in g["pieces"]]


def road_targets(g, mp):
    """for every piece of the GT graph: the index of its road among the map's roads (-1 = the road has no piece in the map)"""
    return [mp["roads"].index(str(p["road"])) if str(p["road"]) in mp["roads"] else -1 for p in g["pieces"]]


def _selftest():
    import qtargets as QT
    ys = lambda a, b: np.stack([np.zeros(int((b - a) / 0.5) + 1), np.arange(a, b + 1e-9, 0.5)], 1)
    A = QT._piece("A", ys(-40, 0), succ=["B"])
    B = QT._piece("B", ys(0, 40), pred=["A"])
    B["left"] = ["A"]
    g = dict(pieces=[A, B], events=[], sample="tok1")
    m0 = make_map(g, "exact")
    assert m0["K"] == 2 and m0["adj"][0, 0, 1] and m0["adj"][1, 1, 0] and abs(m0["pos"][:, 0]).max() < 1e-6 and (m0["frac"] >= 0).all() and (m0["frac"] <= 1 + 1e-6).all()
    m1, m2 = make_map(g, "error", token="t"), make_map(g, "error", token="t")
    assert np.allclose(m1["pos"], m2["pos"]), "the error map of a token must be reproducible"
    d = np.linalg.norm(m1["shift"])
    assert 1.0 <= d <= 2.0 or m1["K"] < 2, d
    assert piece_targets(g, m0) == [[0], [1]] and road_targets(g, m0) == [0, 0]
    big = [make_map(g, "train", rng=np.random.RandomState(i)) for i in range(200)]
    assert 0.02 < np.mean([b["K"] < 2 for b in big]) < 0.2, "about 5 % of the pieces dropped per piece"
    print("map_sim selftest ok")


if __name__ == "__main__":
    _selftest()
