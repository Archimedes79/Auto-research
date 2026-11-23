"""E394 step 3 of the SD-map line (card experiments/E394_*): the CATALOGUE of map errors, as calibrated, nested draws on the step-1 road graph (`lane_sdgraph_data.py`), and the TRAIN mix with wrong maps.

    Gin, T = apply(G, rng, spec, other=None)    # G = the exact graph of a frame; spec = {mode: parameter}; other = the exact graph of ANOTHER scene (for wrong_scene)
    spec = train_spec(rng)                      # one TRAIN draw of the mix (about a third of the draws are wrong maps)
    other_plan(scene, nodes)                    # the frame whose map frame i borrows in the mode wrong_scene (a fixed plan, as E389's `wrong`)
    is_wrong(spec)                              # the frame-level label of a trust gate: the map as a whole is wrong (not only noisy)
    wsl bash tools/env/_py14.sh lane_maperror_data.py selftest      # hand graph; tests with mutants: grid/test_maperror.py

What is new against E390's `perturb` (kept as it is, called with its own keys; nothing of lane_sdgraph_data.py / lane_mapassoc_data.py is edited): the BASE keys go to `perturb` unchanged (pose = shift AND turn, count_rate,
unknown_p, drop (successor links), hide (route flags), sigma (smooth deformation), eps (Douglas-Peucker), node_drop (missing segments)); the NEW keys are applied to the input graph after it:
  shift m   a pure translation of the map against the perceived scene, random direction (the coupled `pose` of E390 also turns by `pose` degrees)       yaw deg   a pure turn of the map about the ego, random sign
  count_big rate   a lane count off by +-2 (E390: +-1)                                                                                                 ghost rate  extra segments: rate x the true roads, a parallel road or a crossing road that does not exist
  link_extra rate  spurious successor links (exit -> entry within 25 m, not in the map)                                                                 link_swap rate  swapped connections (a -> b, c -> d become a -> d, c -> b)
  outdated rate    a stale map: each intersection missing w.p. rate (its approach roads linked straight to its exits) and each road's lane class one higher w.p. rate
  wrong_route 1    the route flags on another path of the same map                                                                                      wrong_scene 1   the map of another scene (every true segment is missing from it)
The labels never change with the errors: T["present"] (n original nodes) says which ORIGINAL segments the input map still holds (False: missing: the association target is 'unassigned'); extra nodes are appended
to the INPUT graph only (indices >= n), no token has them as a target. Draws are NESTED: every random choice is drawn once per frame, a higher rate / size only adds to the lower one, and each new mode has its own
generator (seeded before `perturb` draws), so adding a mode never changes the draws of another mode or of E390's modes.
"""
import numpy as np

import lane_sdgraph_data as SG

BASE_KEYS = ("pose", "count_rate", "unknown_p", "drop", "hide", "sigma", "eps", "node_drop")        # E390's `perturb` keywords (rigid is the frame transform of step 1: not a map error here)
NEW_KEYS = ("shift", "yaw", "count_big", "ghost", "link_extra", "link_swap", "outdated", "wrong_route", "wrong_scene")
LINK_GAP = 25.0                                                                                  # m: a spurious link joins an exit and an entry at most this far apart
GHOST_SHIFT = (6.0, 20.0)                                                                        # m: the sideways shift of a parallel ghost road
GHOST_LEN = (30.0, 70.0)                                                                         # m: the length of a crossing ghost road
WRONG_SHIFT, WRONG_YAW = 4.0, 4.0                                                                # a map shifted by >= 4 m (E389's gross draws 4 - 8 m) or turned by >= 4 degrees counts as a wrong map
OTHER = 97                                                                                       # frames: E389's offset of the other scene (lane_route_data.OTHER)

# the catalogue: mode -> (spec key, the grid, what 'realistic' means and where it comes from). 'realistic' = the range of the simulated maps of E374 / E389 / E390 (sd_lib.degrade_sd: Douglas-Peucker 0.5 - 2 m, lane
# count errors 10 %, missing roads 0 - 10 %; map_sim: translation 1 - 2 m, yaw N(0, 1 degree), 5 % pieces dropped, 5 % counts off; E389: gross draws 4 - 8 m, the route of another scene); the new modes have NO
# protocol: their grids are STRESS levels (named so in the report), the level where a mechanism breaks is the result.
CATALOGUE = [
    dict(mode="pose", key="pose", grid=[0.5, 1.0, 2.0, 4.0], realistic=(0.5, 2.0), source="E390 / E374: shift 1 - 2 m and yaw N(0, 1 degree) (coupled: `pose` m and `pose` degrees)", new=False),
    dict(mode="shift", key="shift", grid=[1.0, 2.0, 4.0, 6.0, 8.0], realistic=(0.0, 2.0), source="E374 / map_sim 1 - 2 m; E389 gross 4 - 8 m", new=True),
    dict(mode="yaw", key="yaw", grid=[0.5, 1.0, 2.0, 4.0, 8.0], realistic=(0.0, 2.0), source="E374 / map_sim yaw N(0, 1 degree)", new=True),
    dict(mode="sigma", key="sigma", grid=[1.0, 2.0, 4.0], realistic=(0.0, 2.0), source="E390: a smooth deformation of the whole map (m)", new=False),
    dict(mode="eps", key="eps", grid=[0.5, 1.0, 2.0, 4.0], realistic=(0.5, 2.0), source="sd_lib.degrade_sd: Douglas-Peucker 0.5 - 2 m", new=False),
    dict(mode="count_rate", key="count_rate", grid=[0.05, 0.1, 0.2, 0.4], realistic=(0.05, 0.1), source="map_sim 5 %, sd_lib.degrade_sd 10 % of the roads off by one lane", new=False),
    dict(mode="unknown_p", key="unknown_p", grid=[0.1, 0.3, 0.6, 1.0], realistic=(0.0, 0.1), source="E389: lane counts unknown for 8 % of the draws", new=False),
    dict(mode="count_big", key="count_big", grid=[0.1, 0.3], realistic=(0.0, 0.0), source="none: a count off by 2 is a stress level", new=True),
    dict(mode="drop", key="drop", grid=[0.1, 0.2, 0.4], realistic=(0.0, 0.1), source="E390: successor links missing (E391 dec 0.1)", new=False),
    dict(mode="node_drop", key="node_drop", grid=[0.05, 0.1, 0.2, 0.4], realistic=(0.05, 0.1), source="map_sim 5 %, sd_lib.degrade_sd 0 - 10 % missing roads", new=False),
    dict(mode="hide", key="hide", grid=[0.3, 0.6, 1.0], realistic=(0.0, 1.0), source="E390: the route flag of a segment removed (the navigation route is missing)", new=False),
    dict(mode="ghost", key="ghost", grid=[0.1, 0.3, 0.6], realistic=(0.0, 0.0), source="none: extra segments that do not exist (stress)", new=True),
    dict(mode="link_extra", key="link_extra", grid=[0.1, 0.3, 0.6], realistic=(0.0, 0.0), source="none: extra connections (stress)", new=True),
    dict(mode="link_swap", key="link_swap", grid=[0.1, 0.3], realistic=(0.0, 0.0), source="none: swapped connections (stress)", new=True),
    dict(mode="outdated", key="outdated", grid=[0.3, 0.6, 1.0], realistic=(0.0, 0.0), source="none: a map older than the road (junctions missing, lane counts one higher) (stress)", new=True),
    dict(mode="wrong_route", key="wrong_route", grid=[1], realistic=(0.0, 0.0), source="E389: a wrong route; here another path of the same map", new=True),
    dict(mode="wrong_scene", key="wrong_scene", grid=[1], realistic=(0.0, 0.0), source="E389 `wrong`: the map of another scene", new=True),
]
DECIDING = dict(wrong=dict(wrong_scene=1), right=dict(pose=1.0, count_rate=0.1, drop=0.1, hide=0.3, node_drop=0.1))      # the card's two deciding levels: a wrong map; the E391 deciding level 'dec' (the right map with the realistic errors)


def is_wrong(spec):
    """True when the map as a whole is wrong (not only noisy): the map of another scene, a shift >= 4 m, a turn >= 4 degrees, or the route on another path: the label of a trust gate"""
    return bool(spec.get("wrong_scene") or spec.get("wrong_route") or spec.get("shift", 0.0) >= WRONG_SHIFT or spec.get("yaw", 0.0) >= WRONG_YAW or spec.get("pose", 0.0) >= WRONG_SHIFT)


def other_plan(scene, nodes, shift=OTHER):
    """(N,) int: the frame whose map frame i borrows (wrong_scene): (i + shift) mod N, moved on frame by frame until it lies in another scene and has at least 3 map nodes. A fixed plan (E389's `other_scene_plan`)"""
    scene, nodes = np.asarray(scene), np.asarray(nodes)
    n = len(scene)
    assert len(set(scene.tolist())) >= 2, "one scene only: no map of another scene"
    out = np.zeros(n, np.int64)
    for i in range(n):
        j = (i + shift) % n
        for _ in range(n):
            if scene[j % n] != scene[i] and nodes[j % n] >= 3:
                break
            j += 1
        out[i] = j % n
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the new modes (each on the INPUT graph, its own generator)
def _extend(Gin, kind, geo, lanes_in):
    """append extra nodes (kind (m,), geo [m lists of polylines], lanes_in (m,)) to the input graph: no route flag, present, never a target"""
    m = len(kind)
    Gin["kind"] = np.concatenate([Gin["kind"], np.asarray(kind, int)])
    Gin["geo"] = list(Gin["geo"]) + list(geo)
    for k in ("lanes", "lanes_in"):
        Gin[k] = np.concatenate([Gin[k], np.asarray(lanes_in, int)])
    for k in ("route", "route_in"):
        Gin[k] = np.concatenate([Gin[k], np.full(m, -1, int)])
    Gin["present"] = np.concatenate([Gin["present"], np.ones(m, bool)])


def _line(p0, heading, length, step=2.0):
    s = np.arange(0.0, length + 1e-9, step)
    return np.stack([p0[0] + s * np.cos(heading), p0[1] + s * np.sin(heading)], 1)


def _ghost(Gin, rng, rate):
    """extra segments: n = int(rate x the present roads + u) ghost roads, each a parallel copy of a true road moved sideways by 6 - 20 m or a straight road of 30 - 70 m through a random point of the +-40 m box. Every choice is
    drawn for the first MAX_GHOST ghosts up front (nested in the rate)"""
    roads = [i for i in range(len(Gin["kind"])) if Gin["kind"][i] == 0 and Gin["present"][i] and len(Gin["geo"][i][0]) >= 2]
    draws = [(rng.random(), rng.uniform(*GHOST_SHIFT), rng.choice([-1.0, 1.0]), rng.uniform(-40, 40, 2), rng.uniform(0, 2 * np.pi), rng.uniform(*GHOST_LEN), rng.integers(0, 3), rng.integers(1 << 30)) for _ in range(16)]
    n = min(int(rate * max(len(roads), 1) + rng.random()), len(draws))
    kind, geo, lanes = [], [], []
    for u, sh, sg, c, hd, ln, lc, pick in draws[:n]:
        if u < 0.5 and roads:
            p = np.asarray(Gin["geo"][roads[pick % len(roads)]][0], float)
            t = np.gradient(p, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            line = p + sg * sh * np.stack([-t[:, 1], t[:, 0]], 1)
        else:
            line = _line(c - 0.5 * ln * np.array([np.cos(hd), np.sin(hd)]), hd, ln)
        kind.append(0), geo.append([line]), lanes.append(int(lc))
    if kind:
        _extend(Gin, kind, geo, lanes)
    return len(kind)


def _link_extra(Gin, rng, rate):
    """spurious successor links: candidates = present ordered pairs (a, b) not linked whose exit -> entry gap is at most 25 m; n = int(rate x the links + u) are taken from one fixed random order (nested)"""
    N = len(Gin["kind"])
    have = set(map(tuple, Gin["succ"]))
    ex = [SG.ends(Gin, i) for i in range(N)]
    cand = []
    for a in range(N):
        for b in range(N):
            if a != b and Gin["present"][a] and Gin["present"][b] and (a, b) not in have:
                if np.linalg.norm(ex[a][2][:, None] - ex[b][0][None], axis=-1).min() <= LINK_GAP:
                    cand.append((a, b))
    order = rng.permutation(len(cand))
    n = min(int(rate * max(len(Gin["succ"]), 1) + rng.random()), len(cand))
    Gin["succ"] = list(Gin["succ"]) + [cand[k] for k in order[:n]]
    return n


def _link_swap(Gin, rng, rate):
    """swapped connections: n = int(rate x the links / 2 + u) disjoint pairs of links (a -> b, c -> d) become (a -> d, c -> b), taken from one fixed random order of the pairs (nested); a pair is skipped when it would make
    a self link or a link that exists"""
    S = [tuple(e) for e in Gin["succ"]]
    pairs = [(i, j) for i in range(len(S)) for j in range(i + 1, len(S)) if S[i][0] != S[j][0] and S[i][1] != S[j][1]]
    order = rng.permutation(len(pairs))
    n = min(int(rate * len(S) / 2.0 + rng.random()), len(pairs))
    used, done = set(), 0
    for k in order:
        i, j = pairs[k]
        (a, b), (c, d) = S[i], S[j]
        if done >= n:
            break
        if i in used or j in used or a == d or c == b or (a, d) in S or (c, b) in S:
            continue
        S[i], S[j] = (a, d), (c, b)
        used |= {i, j}
        done += 1
    Gin["succ"] = S
    return done


def _outdated(Gin, T, rng, rate):
    """a stale map: each present intersection is missing w.p. `rate` (u drawn once per node: nested) and its approach roads are linked straight to its exit roads; each road's known lane class is one higher w.p. `rate`
    (clipped). The missing intersections are also absent in T['present'] (their tokens' target is 'unassigned')"""
    N = len(Gin["kind"])
    u_i, u_c = rng.random(N), rng.random(N)
    gone = (Gin["kind"] == 1) & Gin["present"] & (u_i < rate)
    new = []
    for k in np.flatnonzero(gone):
        pre = [a for a, b in Gin["succ"] if b == k and Gin["present"][a] and not gone[a]]
        suc = [b for a, b in Gin["succ"] if a == k and Gin["present"][b] and not gone[b]]
        new += [(a, b) for a in pre for b in suc if a != b]
    Gin["present"] = Gin["present"] & ~gone
    T["present"] = T["present"] & ~gone[: len(T["present"])]
    Gin["succ"] = [e for e in Gin["succ"] if Gin["present"][e[0]] and Gin["present"][e[1]]] + [e for e in dict.fromkeys(new) if e not in set(map(tuple, Gin["succ"]))]
    Gin["oppl"] = [e for e in Gin["oppl"] if Gin["present"][e[0]] and Gin["present"][e[1]]]
    up = (Gin["kind"] == 0) & (u_c < rate) & (Gin["lanes_in"] < SG.N_LANE - 1)
    Gin["lanes_in"] = np.where(up, Gin["lanes_in"] + 1, Gin["lanes_in"])
    Gin["route_in"] = SG.route_ranks(np.where(Gin["present"], Gin["route_in"], -1))
    return int(gone.sum()), int(up.sum())


def _wrong_route(Gin, rng):
    """the route flags on ANOTHER path of the same map: a random present road other than the first route node, followed along random successors for as many nodes as the true route has (at least 2), the flags of the true
    route removed; up to 6 starts are tried until the flagged set differs from the true one"""
    true = set(np.flatnonzero(Gin["route_in"] >= 0).tolist())
    roads = [i for i in range(len(Gin["kind"])) if Gin["present"][i] and Gin["kind"][i] == 0]
    want = max(len(true), 2)
    starts = rng.permutation(roads) if roads else []
    for s in list(starts)[:6]:
        path, cur = [int(s)], int(s)
        while len(path) < want:
            nxt = [b for a, b in Gin["succ"] if a == cur and b not in path]
            if not nxt:
                break
            cur = int(nxt[rng.integers(len(nxt))])
            path.append(cur)
        if set(path) != true:
            Gin["route_in"] = SG.route_ranks(_order(Gin, path))
            return True
    return False


def _order(Gin, path):
    """route order values: the position of a node in `path`, -1 elsewhere"""
    o = np.full(len(Gin["kind"]), -1, int)
    for r, n in enumerate(path):
        o[n] = r
    return o


def apply(G, rng, spec, other=None):
    """-> (Gin, T): the input map and the targets of the exact graph G under the errors of `spec` (see the module docstring). `rng` is a numpy Generator; the same (G, rng state, spec) gives the same map. T is E390's
    (lanes, succ, dropped, kind, route, hidden, present, pseg) with `present` (n original nodes) updated by the modes that remove a node, plus T['spec'] and T['ghosts'] (the number of extra nodes)"""
    unknown = set(spec) - set(BASE_KEYS) - set(NEW_KEYS)
    assert not unknown, f"unknown map error keys {sorted(unknown)}"
    sub = [np.random.default_rng(int(s)) for s in rng.integers(1 << 62, size=len(NEW_KEYS))]       # one generator per new mode, drawn BEFORE perturb: E390's draws are what they were
    g = dict(zip(NEW_KEYS, sub))
    Gin, T = SG.perturb(G, rng, **{k: v for k, v in spec.items() if k in BASE_KEYS})
    Gin = dict(Gin, geo=[[p.copy() for p in gs] for gs in Gin["geo"]], succ=list(Gin["succ"]), oppl=list(Gin["oppl"]), lanes_in=Gin["lanes_in"].copy(), route_in=Gin["route_in"].copy(), present=Gin["present"].copy())
    T = dict(T, present=T["present"].copy(), spec=dict(spec), ghosts=0)
    if spec.get("shift", 0.0) > 0:
        r = g["shift"]
        ang = r.uniform(0, 2 * np.pi)
        t = float(spec["shift"]) * np.array([np.cos(ang), np.sin(ang)])
        Gin["geo"] = [[p + t for p in gs] for gs in Gin["geo"]]
    if spec.get("yaw", 0.0) > 0:
        r = g["yaw"]
        R = SG.rot(np.deg2rad((1.0 if r.random() < 0.5 else -1.0) * float(spec["yaw"])))
        Gin["geo"] = [[p @ R.T for p in gs] for gs in Gin["geo"]]
    if spec.get("count_big", 0.0) > 0:
        r = g["count_big"]
        N = len(Gin["kind"])
        flip = (Gin["kind"] == 0) & (r.random(N) < spec["count_big"]) & (Gin["lanes_in"] < SG.N_LANE)
        sgn = np.where(r.random(N) < 0.5, -2, 2)
        moved = np.clip(Gin["lanes_in"] + sgn, 0, SG.N_LANE - 1)
        moved = np.where(moved == Gin["lanes_in"], np.clip(Gin["lanes_in"] - sgn, 0, SG.N_LANE - 1), moved)
        Gin["lanes_in"] = np.where(flip, moved, Gin["lanes_in"])
    if spec.get("outdated", 0.0) > 0:
        _outdated(Gin, T, g["outdated"], spec["outdated"])
    if spec.get("link_swap", 0.0) > 0:
        _link_swap(Gin, g["link_swap"], spec["link_swap"])
    if spec.get("link_extra", 0.0) > 0:
        _link_extra(Gin, g["link_extra"], spec["link_extra"])
    if spec.get("wrong_route"):
        _wrong_route(Gin, g["wrong_route"])
    if spec.get("ghost", 0.0) > 0:
        T["ghosts"] = _ghost(Gin, g["ghost"], spec["ghost"])
    if spec.get("wrong_scene"):
        assert other is not None, "wrong_scene needs the exact graph of another scene (`other`)"
        Go, _ = SG.perturb(other, g["wrong_scene"], pose=1.0)
        Gin = dict(Go, prox=Gin["prox"], pdir=Gin["pdir"], count_rate=Gin.get("count_rate", 0.0))
        Gin["geo"] = [[p.copy() for p in gs] for gs in Gin["geo"]]
        T["present"] = np.zeros(len(T["present"]), bool)
    return Gin, T


# ------------------------------------------------------------------------------------------------------------------------------ the TRAIN mix
def train_spec(rng):
    """one TRAIN draw: exact 12 %, the realistic simple mix of E391 (pose {0, 1, 2}, count {0, 0.1}, links {0, 0.1}, route hide {0, 0.3}, missing segments {0, 0.1}) 28 %, the realistic mix plus ONE moderate new
    error (ghost 0.3, link_extra 0.3, link_swap 0.1, outdated 0.3, count_big 0.1, yaw 2, sigma 2) 25 %, and 35 % WRONG maps (E389's lesson: 5 % wrong maps were too few): wrong_scene 15 %, a shift of 4 - 8 m 10 %,
    the route on another path 5 %, a turn of 4 - 8 degrees 5 %. Returns the spec (the frame-level label is `is_wrong(spec)`)"""
    u = rng.random()
    simple = dict(pose=float(rng.choice([0.0, 1.0, 2.0])), count_rate=float(rng.choice([0.0, 0.1])), drop=float(rng.choice([0.0, 0.1])), hide=float(rng.choice([0.0, 0.3])), node_drop=float(rng.choice([0.0, 0.1])))
    extra = [dict(ghost=0.3), dict(link_extra=0.3), dict(link_swap=0.1), dict(outdated=0.3), dict(count_big=0.1), dict(yaw=2.0), dict(sigma=2.0)]
    pick = extra[int(rng.integers(len(extra)))]
    if u < 0.12:
        return {}
    if u < 0.40:
        return simple
    if u < 0.65:
        return {**simple, **pick}
    if u < 0.80:
        return {**simple, "wrong_scene": 1}
    if u < 0.90:
        return {**simple, "shift": float(rng.uniform(4.0, 8.0))}
    if u < 0.95:
        return {**simple, "wrong_route": 1}
    return {**simple, "yaw": float(rng.uniform(4.0, 8.0))}


def selftest():
    G = SG.toy()
    Gin, T = apply(G, np.random.default_rng(0), {})
    assert (Gin["lanes_in"] == G["lanes"]).all() and Gin["succ"] == G["succ"] and T["present"].all() and T["ghosts"] == 0
    for spec in (dict(shift=5.0), dict(yaw=3.0), dict(count_big=1.0), dict(ghost=0.6), dict(link_extra=0.6), dict(link_swap=1.0), dict(outdated=1.0), dict(wrong_route=1), dict(wrong_scene=1)):
        Go, To = apply(G, np.random.default_rng(1), spec, other=G)
        assert len(Go["kind"]) == len(Go["geo"]) == len(Go["lanes_in"]) == len(Go["route_in"]) == len(Go["present"]), spec
        SG.featurize(Go)                                                       # the encoder's features can be built from every error
    print("lane_maperror_data selftest ok")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        print(__doc__)
