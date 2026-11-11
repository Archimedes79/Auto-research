"""Junction ground truth of the lane head, made on the lane side from the lane GT of GtGenerator (E285, D82; no edit of GtGenerator, no new store: derived data of this project).

    jg = junction_gt(g)           # g = the lane GT graph of a frame (GtGenerator lane_gt.load); None when the frame has no junction
    jg["zones"]                   list of dict(poly (n, 2) exterior points, area, runs): the junction zone = the union of the connector runs grown by `BUF` (closing `CLOSE`), not the nuScenes road segments
    jg["ent"], jg["ext"]          (E, 4) / (X, 4) entry / exit points (x, y, tx, ty) as in `qends.end_table`
    jg["ent_arm"], jg["ext_arm"]  arm index of every end; jg["ent_rank"], jg["ext_rank"] lane rank in the arm group (0 = leftmost looking along the travel direction)
    jg["arms"]                    list of dict(c (2,) centre, t (2,) heading into the junction, n_in, n_out, spread_in, spread_out (longitudinal std of the ends, metres), zone)
    jg["pairs"]                   (P, 10) float: entry, exit, arm_in, arm_out, rank_in, rank_out, k_in, k_out (entries / exits of that arm pair in use), turn (degrees, + left), driven (the ego drove it, tier 1)
    jg["core"]                    shapely geometry or None: the conflict core, where connectors of different pairs cross, merge or diverge (`_core`)
    jg["ent_vis"], jg["ext_vis"]  (E,) / (X,) bool: the end is seen by a camera and not hidden by a box (the l4 label `vis` at the start / end of the connector run; True without the label)
    jg["order"]                   dict(maneuvers, multi, monotone, identity): the order-preserving rule of D82 counted on the true pairs
    fields(jg, n=100, res=1.0)    the 1 m maps of the head: zone and conflict core (2 channels), entry heat, exit heat (Gaussian, sigma `SIG`), entry / exit heading (2 each)

Rules (written before the numbers, card E285): an entry is the first point of a connector run, an exit its last (`qends`); ends are clustered (1.5 m, 40 degrees); an ARM is a single-linkage group of
ends with heading within 25 degrees, no more than 6 m apart along and 6 m across the heading, entries and exits separately; an entry arm and an exit arm with opposite headings (at least 155 degrees)
whose centres lie within 14 m are the same arm (the same road, the two directions). Pairs inside one (entry arm, exit arm) maneuver must keep their order: entry rank i before i' implies exit rank j <= j'
(non-crossing); where k entries meet k exits the solution is entry i to exit i (identity). The zone and the arms use nothing but the lane GT.
Run the self-test: python jgt.py
"""
import numpy as np

import qends as QE
import qtargets as QT

BUF, CLOSE, SIG = 1.8, 2.0, 1.5
CORE_BUF = 0.9                                                         # conflict core: connectors of DIFFERENT pairs closer than 2 x CORE_BUF (parallel lanes of one maneuver stay apart)
ARM_ANG, ARM_ALONG, ARM_ACROSS, ARM_JOIN_DIST, ARM_JOIN_ANG = 25.0, 6.0, 6.0, 14.0, 155.0


def _runs(g):
    runs = []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        tan = QT._tangents(q)
        conn = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
        tier = np.asarray(pc.get("tier", np.full(len(q), 3)))
        vis = np.asarray(pc.get("vis", np.ones(len(q), bool))).astype(bool)
        k = 0
        while k < len(q):
            if conn[k]:
                j = k
                while j + 1 < len(q) and conn[j + 1]:
                    j += 1
                if j - k + 1 >= 3:
                    runs.append(dict(pts=q[k: j + 1], a=(q[k], tan[k]), b=(q[j], tan[j]), driven=bool((tier[k: j + 1] == 1).mean() > 0.5), piece=pc["piece"], vis_a=bool(vis[k]), vis_b=bool(vis[j])))
                k = j + 1
            else:
                k += 1
    return runs


def _zones(runs):
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    u = unary_union([LineString(r["pts"]).buffer(BUF, cap_style=2) for r in runs])
    u = u.buffer(CLOSE).buffer(-CLOSE)
    polys = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
    out = []
    for p in polys:
        if p.is_empty:
            continue
        out.append(dict(poly=np.asarray(p.exterior.coords, np.float32), area=float(p.area), shp=p))
    return out


def _core(runs, eid, xid):
    """the conflict core: where connector paths of different (entry, exit) pairs cross, merge or diverge; the union of the pairwise overlaps of the connectors grown by CORE_BUF, grown by 1 m. None when there is none"""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    bufs = [LineString(r["pts"]).buffer(CORE_BUF, cap_style=2) for r in runs]
    keys = list(zip(eid, xid))
    parts = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            if keys[i] != keys[j] and bufs[i].intersects(bufs[j]):
                parts.append(bufs[i].intersection(bufs[j]))
    if not parts:
        return None
    u = unary_union(parts).buffer(1.0)
    return u if not u.is_empty else None


def _angle(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def _groups(E):
    """single-linkage groups of ends (E, 4): heading within ARM_ANG, along <= ARM_ALONG, across <= ARM_ACROSS"""
    n = len(E)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = E[i, 2:], E[j, 2:]
            if _angle(ti, tj) > ARM_ANG:
                continue
            t = (ti + tj) / np.linalg.norm(ti + tj + 1e-9)
            d = E[j, :2] - E[i, :2]
            if abs(d @ t) <= ARM_ALONG and abs(d @ np.array([-t[1], t[0]])) <= ARM_ACROSS:
                par[find(i)] = find(j)
    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in range(n)], int)


def _ranks(E, grp):
    """lane rank inside a group, 0 = leftmost looking along the heading"""
    rk = np.zeros(len(E), int)
    for gi in set(grp.tolist()):
        idx = np.where(grp == gi)[0]
        t = E[idx, 2:].mean(0)
        t /= np.linalg.norm(t) + 1e-9
        lat = (E[idx, :2] - E[idx, :2].mean(0)) @ np.array([-t[1], t[0]])
        for r, i in enumerate(idx[np.argsort(-lat)]):
            rk[i] = r
    return rk


def _spread(E, grp, gi):
    idx = np.where(grp == gi)[0]
    t = E[idx, 2:].mean(0)
    t /= np.linalg.norm(t) + 1e-9
    return float(np.std((E[idx, :2] - E[idx, :2].mean(0)) @ t)) if len(idx) > 1 else 0.0


def junction_gt(g):
    runs = _runs(g)
    if not runs:
        return None
    ent, eid = QE.cluster([(r["a"][0], r["a"][1]) for r in runs])
    ext, xid = QE.cluster([(r["b"][0], r["b"][1]) for r in runs])
    ENT = np.array([np.r_[e[0], e[1]] for e in ent], float)
    EXT = np.array([np.r_[x[0], x[1]] for x in ext], float)
    ga, gb = _groups(ENT), _groups(EXT)
    ra, rb = _ranks(ENT, ga), _ranks(EXT, gb)
    # arms: entry groups and exit groups joined when they are the two directions of one road
    na, nb = ga.max() + 1, gb.max() + 1
    cin = [ENT[ga == i, :2].mean(0) for i in range(na)]
    tin = [ENT[ga == i, 2:].mean(0) / np.linalg.norm(ENT[ga == i, 2:].mean(0)) for i in range(na)]
    cout = [EXT[gb == i, :2].mean(0) for i in range(nb)]
    tout = [EXT[gb == i, 2:].mean(0) / np.linalg.norm(EXT[gb == i, 2:].mean(0)) for i in range(nb)]
    arm_of_in, arm_of_out, arms = {}, {}, []
    used_out = set()
    for i in range(na):
        best = None
        for j in range(nb):
            if j in used_out:
                continue
            d = np.linalg.norm(cin[i] - cout[j])
            if d <= ARM_JOIN_DIST and _angle(tin[i], tout[j]) >= ARM_JOIN_ANG and (best is None or d < best[0]):
                best = (d, j)
        a = len(arms)
        arm_of_in[i] = a
        if best:
            used_out.add(best[1])
            arm_of_out[best[1]] = a
        arms.append(dict(c=(cin[i] + (cout[best[1]] if best else cin[i])) / 2, t=tin[i], n_in=int((ga == i).sum()), n_out=int((gb == best[1]).sum()) if best else 0,
                         spread_in=_spread(ENT, ga, i), spread_out=_spread(EXT, gb, best[1]) if best else 0.0))
    for j in range(nb):
        if j not in arm_of_out:
            arm_of_out[j] = len(arms)
            arms.append(dict(c=cout[j], t=-tout[j], n_in=0, n_out=int((gb == j).sum()), spread_in=0.0, spread_out=_spread(EXT, gb, j)))
    ent_arm = np.array([arm_of_in[i] for i in ga], int)
    ext_arm = np.array([arm_of_out[j] for j in gb], int)
    ent_vis, ext_vis = np.zeros(len(ENT), bool), np.zeros(len(EXT), bool)
    for r, a_, b_ in zip(runs, eid, xid):                                              # a GT end counts as VISIBLE when a camera sees its point and no box hides it (the l4 label `vis`) for at least one of its runs
        ent_vis[a_] |= r["vis_a"]
        ext_vis[b_] |= r["vis_b"]
    zones = _zones(runs)
    core = _core(runs, eid, xid)
    for a in arms:
        a["zone"] = next((k for k, z in enumerate(zones) if z["shp"].buffer(8.0).contains(__import__("shapely.geometry", fromlist=["Point"]).Point(*a["c"]))), -1)
    # pairs: one row per (entry cluster, exit cluster) that a connector run joins
    keys = sorted(set(zip(eid, xid)))
    drv = {}
    for r, a, b in zip(runs, eid, xid):
        drv[(a, b)] = drv.get((a, b), False) or r["driven"]
    cnt_in, cnt_out = {}, {}
    for a, b in keys:
        cnt_in.setdefault((ent_arm[a], ext_arm[b]), set()).add(a)
        cnt_out.setdefault((ent_arm[a], ext_arm[b]), set()).add(b)
    rows = []
    for a, b in keys:
        m = (ent_arm[a], ext_arm[b])
        rows.append([a, b, ent_arm[a], ext_arm[b], ra[a], rb[b], len(cnt_in[m]), len(cnt_out[m]), QE.sang(ENT[a, 2:], EXT[b, 2:]), float(drv[(a, b)])])
    pairs = np.array(rows, float)
    first = {}
    for r, a_, b_ in zip(runs, eid, xid):
        first.setdefault((a_, b_), r["pts"])
    pair_pts = [first[(int(r[0]), int(r[1]))] for r in rows]                                                         # E303: the GT connector points of every pair (first run), in the order of `pairs`
    # the order-preserving rule per maneuver
    man = {}
    for r in rows:
        man.setdefault((int(r[2]), int(r[3])), []).append(r)
    n_multi = n_mono = n_ident = 0
    for m, rr in man.items():
        if len(rr) < 2:
            continue
        n_multi += 1
        rr = sorted(rr, key=lambda r: (r[4], r[5]))
        mono = all(rr[i][5] <= rr[i + 1][5] for i in range(len(rr) - 1)) and all(rr[i][4] <= rr[i + 1][4] for i in range(len(rr) - 1))
        n_mono += int(mono)
        ki, ko = int(rr[0][6]), int(rr[0][7])
        n_ident += int(mono and ki == ko and all(int(r[4]) == int(r[5]) for r in rr))
    return dict(zones=zones, ent=ENT, ext=EXT, ent_arm=ent_arm, ext_arm=ext_arm, ent_rank=ra, ext_rank=rb, arms=arms, pairs=pairs,
                ent_vis=ent_vis, ext_vis=ext_vis, pair_pts=pair_pts, core=core, order=dict(maneuvers=len(man), multi=n_multi, monotone=n_mono, identity=n_ident))


def fields(jg, n=100, res=1.0, half=50.0):
    """the 1 m maps of the head, row = x, column = y as `qtargets`: zone + conflict core (2, n, n), entry / exit heat (2, n, n) Gaussians, entry / exit heading (4, n, n) at the cells within 2 sigma of an end"""
    import cv2
    zone = np.zeros((2, n, n), np.uint8)
    heat = np.zeros((2, n, n), np.float32)
    head = np.zeros((4, n, n), np.float32)
    if jg is None:
        return zone.astype(np.float32), heat, head
    for z in jg["zones"]:
        pts = ((z["poly"] + half) / res)[:, ::-1]
        cv2.fillPoly(zone[0], [np.round(pts).astype(np.int32)], 1)
    if jg.get("core") is not None:
        for q in (jg["core"].geoms if hasattr(jg["core"], "geoms") else [jg["core"]]):
            if q.geom_type == "Polygon" and not q.is_empty:
                cv2.fillPoly(zone[1], [np.round(((np.asarray(q.exterior.coords, float) + half) / res)[:, ::-1]).astype(np.int32)], 1)
    ii, jj = np.meshgrid(-half + (np.arange(n) + 0.5) * res, -half + (np.arange(n) + 0.5) * res, indexing="ij")
    for k, E in enumerate((jg["ent"], jg["ext"])):
        for e in E:
            d2 = (ii - e[0]) ** 2 + (jj - e[1]) ** 2
            h = np.exp(-d2 / (2 * SIG ** 2)).astype(np.float32)
            upd = h > heat[k]
            heat[k][upd] = h[upd]
            near = (d2 <= (2 * SIG) ** 2) & upd
            head[2 * k][near], head[2 * k + 1][near] = e[2], e[3]
    return zone.astype(np.float32), heat, head


def _selftest():
    # a plus-shaped junction: two lanes come from the south (x = -1.75 and 1.75 would be one lane each way; here two entries side by side), each goes straight and one turns left
    def run(a, b, n=12):
        return np.stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)], 1)
    south = [run((-1.75, -30), (-1.75, -8)), run((1.75, -30), (1.75, -8))]
    north = [run((-1.75, 8), (-1.75, 30)), run((1.75, 8), (1.75, 30))]
    pcs = []
    for i, (s, nth) in enumerate(zip(south, north)):
        conn = run(s[-1], nth[0], 16)
        pts = np.vstack([s, conn, nth])
        pcs.append(dict(piece=f"p{i}", pts=pts, tier=np.full(len(pts), 3), segments=[dict(kind="lane", i0=0, i1=len(s) - 1), dict(kind="connector", i0=len(s), i1=len(s) + len(conn) - 1),
                                                                            dict(kind="lane", i0=len(s) + len(conn), i1=len(pts) - 1)]))
    jg = junction_gt(dict(pieces=pcs))
    assert jg is not None and len(jg["ent"]) == 2 and len(jg["ext"]) == 2 and len(jg["zones"]) == 1 and len(jg["arms"]) == 2, (len(jg["ent"]), len(jg["zones"]), len(jg["arms"]))
    assert jg["order"] == dict(maneuvers=1, multi=1, monotone=1, identity=1), jg["order"]
    assert jg["ent_rank"].tolist() == [1, 0] or jg["ent_rank"].tolist() == [0, 1]
    # the leftmost lane looking north is x = -1.75 -> rank 0, and it goes to the leftmost exit
    for r in jg["pairs"]:
        assert r[4] == r[5], jg["pairs"]
    zn, heat, head = fields(jg)
    assert zn[0].sum() > 50 and zn.shape[0] == 2 and heat.max() > 0.85 and abs(np.hypot(head[0], head[1]).max() - 1.0) < 1e-5
    assert junction_gt(dict(pieces=[])) is None
    print("jgt selftest ok")


if __name__ == "__main__":
    _selftest()
