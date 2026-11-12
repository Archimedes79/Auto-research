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
import os

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


def raster_exact():
    """BFLC_RASTER=exact: the polygon masks cover a cell when MORE THAN HALF of its area is inside the polygon (8 x 8 supersampling, no half-cell shift); default: the legacy `np.round` + `cv2.fillPoly` of E285 / E286, kept so the stored numbers stay reproducible"""
    return os.environ.get("BFLC_RASTER", "") == "exact"


SS = 8


def fill_poly(mask, pts_cells, line=False):
    """fill the polygon `pts_cells` ((m, 2) in CELL units, column = y first as cv2 wants: x = column) into `mask` (uint8, row = x). Legacy: integer vertices (cv2 puts integer coordinates on pixel CENTRES while our cell k spans [k, k + 1): the mask is
    shifted by half a cell, critic 2026-11-11); `line`: the outline is drawn too (thin stop lines stay visible). Exact (BFLC_RASTER=exact): a cell is in when more than half of it is inside (`line`: more than a quarter for polygons thinner than one cell, so a thin stop line stays visible; the others use the half rule). 8 x 8 supersampling: about 3 % of the cells differ from the analytic area rule (the name 'exact' is relative to the legacy fill)"""
    import cv2
    pts = np.asarray(pts_cells, float)
    if raster_exact():
        c0, c1 = max(int(np.floor(pts[:, 0].min())), 0), min(int(np.ceil(pts[:, 0].max())) + 1, mask.shape[1])
        r0, r1 = max(int(np.floor(pts[:, 1].min())), 0), min(int(np.ceil(pts[:, 1].max())) + 1, mask.shape[0])
        if c1 <= c0 or r1 <= r0:
            return
        big = np.zeros(((r1 - r0) * SS, (c1 - c0) * SS), np.uint8)
        q = np.round(((pts - np.array([c0, r0])) * SS - 0.5) * 16).astype(np.int32)                     # sub-cell vertices (1/16 of a supersample), supersample centres at integer coordinates
        cv2.fillPoly(big, [q], 1, lineType=cv2.LINE_8, shift=4)
        cov = big.reshape(r1 - r0, SS, c1 - c0, SS).mean((1, 3))
        per = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum()
        thick = 2.0 * abs(0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])) / max(per, 1e-9)         # 2 A / perimeter ~ the width of a strip
        mask[r0:r1, c0:c1] |= (cov > (0.25 if (line and thick < 1.0) else 0.5)).astype(mask.dtype)
    else:
        q = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [q], 1)
        if line:
            cv2.polylines(mask, [q], True, 1, 1)


def fields(jg, n=100, res=1.0, half=50.0):
    """the 1 m maps of the head, row = x, column = y as `qtargets`: zone + conflict core (2, n, n), entry / exit heat (2, n, n) Gaussians, entry / exit heading (4, n, n) at the cells within 2 sigma of an end"""
    zone = np.zeros((2, n, n), np.uint8)
    heat = np.zeros((2, n, n), np.float32)
    head = np.zeros((4, n, n), np.float32)
    if jg is None:
        return zone.astype(np.float32), heat, head
    for z in jg["zones"]:
        fill_poly(zone[0], ((z["poly"] + half) / res)[:, ::-1])
    if jg.get("core") is not None:
        for q in (jg["core"].geoms if hasattr(jg["core"], "geoms") else [jg["core"]]):
            if q.geom_type == "Polygon" and not q.is_empty:
                fill_poly(zone[1], ((np.asarray(q.exterior.coords, float) + half) / res)[:, ::-1])
    sig = float(os.environ.get("BFLC_SIG", SIG))                                                          # E307: the width of the end Gaussians (default 1.5 m; BFLC_SIG=1.0 for the crowded ends), read at the call, inherited by the loader workers
    ii, jj = np.meshgrid(-half + (np.arange(n) + 0.5) * res, -half + (np.arange(n) + 0.5) * res, indexing="ij")
    for k, E in enumerate((jg["ent"], jg["ext"])):
        for e in E:
            d2 = (ii - e[0]) ** 2 + (jj - e[1]) ** 2
            h = np.exp(-d2 / (2 * sig ** 2)).astype(np.float32)
            upd = h > heat[k]
            heat[k][upd] = h[upd]
            near = (d2 <= (2 * sig) ** 2) & upd
            head[2 * k][near], head[2 * k + 1][near] = e[2], e[3]
    return zone.astype(np.float32), heat, head


def _selftest_raster():
    """the exact raster puts a cell in the mask when its CENTRE is inside the polygon; the legacy one is shifted by half a cell (kept for the stored numbers)"""
    box = np.array([[-1.2, 0.3], [3.4, 0.3], [3.4, 5.7], [-1.2, 5.7]])                           # x in [-1.2, 3.4], y in [0.3, 5.7] metres
    cells = ((box + 50.0) / 1.0)[:, ::-1]
    old = os.environ.pop("BFLC_RASTER", None)
    try:
        m0 = np.zeros((100, 100), np.uint8)
        fill_poly(m0, cells)
        os.environ["BFLC_RASTER"] = "exact"
        m1 = np.zeros((100, 100), np.uint8)
        fill_poly(m1, cells)
        xc = -49.5 + np.arange(100)
        want = ((xc[:, None] >= -1.2) & (xc[:, None] <= 3.4) & (xc[None, :] >= 0.3) & (xc[None, :] <= 5.7)).astype(np.uint8)
        assert (m1 == want).all(), (int(m1.sum()), int(want.sum()))
        assert m0.sum() > 0 and not (m0 == want).all(), "the legacy raster is the half-cell shifted one"
        from shapely.geometry import Polygon, box as sbox
        rng = np.random.RandomState(0)
        bad_exact = bad_legacy = n_ref = 0
        for _ in range(20):                                                                           # random convex polygons: the cells with more than half their area inside agree with shapely
            ang = np.sort(rng.uniform(0, 2 * np.pi, 6))
            poly = np.stack([rng.uniform(-20, 20) + rng.uniform(2, 9) * np.cos(ang), rng.uniform(-20, 20) + rng.uniform(2, 9) * np.sin(ang)], 1)
            sp = Polygon(poly)
            if not sp.is_valid or sp.area < 2:
                continue
            mm = np.zeros((100, 100), np.uint8)
            fill_poly(mm, ((poly + 50.0) / 1.0)[:, ::-1])
            os.environ.pop("BFLC_RASTER", None)
            ml = np.zeros((100, 100), np.uint8)
            fill_poly(ml, ((poly + 50.0) / 1.0)[:, ::-1])
            os.environ["BFLC_RASTER"] = "exact"
            lo = np.array([sp.bounds[0] + 50 - 2, sp.bounds[1] + 50 - 2]).astype(int).clip(0, 99)
            hi = np.array([sp.bounds[2] + 50 + 2, sp.bounds[3] + 50 + 2]).astype(int).clip(0, 99)
            ref = np.zeros((100, 100), bool)
            for i in range(lo[0], hi[0] + 1):
                for j in range(lo[1], hi[1] + 1):
                    ref[i, j] = sp.intersection(sbox(i - 50, j - 50, i - 49, j - 49)).area > 0.5
            bad_exact += int((mm.astype(bool) != ref).sum())
            bad_legacy += int((ml.astype(bool) != ref).sum())
            n_ref += int(ref.sum())
        assert bad_exact <= 0.06 * n_ref and bad_exact < 0.5 * bad_legacy, (bad_exact, bad_legacy, n_ref)       # the exact mode follows the area rule, the legacy one is off by the half cell
    finally:
        os.environ.pop("BFLC_RASTER", None)
        if old is not None:
            os.environ["BFLC_RASTER"] = old


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
    for e, r in zip(jg["ent"], jg["ent_rank"]):                                           # looking north (+y) the left is -x: rank 0 is the lane at x = -1.75
        assert (r == 0) == (e[0] < 0), (e, r)
    # crossed connectors (the left entry goes to the right exit and the right entry to the left exit): the order rule is violated on this maneuver and a conflict core exists
    pcs2 = []
    for i, (s, nth) in enumerate(zip(south, north[::-1])):
        conn = run(s[-1], nth[0], 16)
        pts = np.vstack([s, conn, nth])
        pcs2.append(dict(piece=f"q{i}", pts=pts, tier=np.full(len(pts), 3), segments=[dict(kind="lane", i0=0, i1=len(s) - 1), dict(kind="connector", i0=len(s), i1=len(s) + len(conn) - 1),
                                                                            dict(kind="lane", i0=len(s) + len(conn), i1=len(pts) - 1)]))
    jx = junction_gt(dict(pieces=pcs2))
    assert jx["order"]["multi"] == 1 and jx["order"]["monotone"] == 0 and jx["order"]["identity"] == 0, jx["order"]
    assert jx["core"] is not None and not jx["core"].is_empty and jx["core"].area > 1.0
    zn, heat, head = fields(jg)
    os.environ["BFLC_SIG"] = "1.0"                                                                              # E307: a narrower Gaussian is lower one metre from the end and equal at the end
    try:
        _, heat_n, _ = fields(jg)
    finally:
        os.environ.pop("BFLC_SIG", None)
    e0 = jg["ent"][0]
    ci, cj = int((e0[0] + 50.0) // 1.0), int((e0[1] + 50.0) // 1.0)
    assert heat_n[0][ci, cj] <= heat[0][ci, cj] + 1e-6 and heat_n[0][ci, min(cj + 2, 99)] < heat[0][ci, min(cj + 2, 99)], (heat_n[0][ci, cj], heat[0][ci, cj])
    for k, EE in enumerate((jg["ent"], jg["ext"])):                                       # the heat peaks at the cell that holds the end (row = x, column = y, origin -50 m)
        for e in EE:
            i, j = int((e[0] + 50.0) // 1.0), int((e[1] + 50.0) // 1.0)
            assert heat[k][i, j] > 0.7, (k, e, heat[k][i, j])
    assert zn[0].sum() > 50 and zn.shape[0] == 2 and heat.max() > 0.85 and abs(np.hypot(head[0], head[1]).max() - 1.0) < 1e-5
    assert junction_gt(dict(pieces=[])) is None
    print("jgt selftest ok")


if __name__ == "__main__":
    _selftest_raster()
    _selftest()
