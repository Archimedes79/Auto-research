"""The lines of a keyframe window: lane dividers, road dividers and the road boundary, as polylines with their corners.

Source: the nuScenes map. `lane_divider` and `road_divider` are the map's own line records; the BOUNDARY is the outline of the union of the polygons of
`boundary_layers` (default: the drivable area; add "carpark_area" and / or "walkway" to count them as part of the road, i.e. no boundary between them
and the road). nuScenes has no boundary line record, this is a definition (open with Mirko: which layers).

CORNERS. Along every polyline the turning angle is measured between the chord of the last `corner_base` metres (default 2) and the chord of the next
`corner_base` metres; a vertex whose angle exceeds `corner_deg` (default 30 degrees) and is the largest within 2 m is a corner (a kerb corner,
a sharp end of a divider). A corner is {"pos", "angle" (degrees), "idx" (the nearest point of the polyline), "dir_in", "dir_out"}.

AREAS. The ROAD area is the union of the polygons of `boundary_layers` (the area inside the boundary), the INTERSECTION areas are the connected areas of the road
segments flagged is_intersection (road_graph.py gives them an id that matches the road nodes); `crop_areas` cuts them to the window as polygons
{"id", "exterior", "holes", "area"} (exterior counter-clockwise, holes clockwise). They are NOT dilated: a head that wants intersection pixels dilated by a few
metres does that itself; rasterising the polygon gives the intersection pixels as they are.

The result of `crop_lines` is {"lane_divider": [line, ...], "road_divider": [...], "boundary": [...]} in the frame of `lane_graph.py` (lane frame by
default, x right, y forward), cut to the square window; a line is {"pts" (float32, travel / outline order), "start_open", "end_open", "corners"}.
"""
import numpy as np

from .lane_graph import clip_rect, reach, to_frame, window_of

CHUNK = 200                                                  # vertices per chunk of a long outline (a keyframe takes the chunks near the ego)


def _chunks(P, kind):
    """a long polyline in overlapping chunks with their bounding boxes (a chunk shares its last vertex with the next one); `closed`: a ring"""
    out = []
    closed = bool(len(P) > 3 and np.allclose(P[0], P[-1]))
    for i in range(0, max(len(P) - 1, 1), CHUNK):
        Q = P[i:i + CHUNK + 1]
        if len(Q) >= 2:
            out.append(dict(kind=kind, pts=Q, bbox=np.r_[Q.min(0), Q.max(0)], first=i == 0, last=i + CHUNK + 1 >= len(P), closed=closed))
    return out


def build_line_graph(nmap, boundary_layers=("drivable_area",), simplify=0.05):
    """the polylines of a map in global coordinates, cut into chunks"""
    import shapely
    chunks = []
    ring_id = 0
    for layer in ("lane_divider", "road_divider"):
        for rec in getattr(nmap, layer):
            ln = nmap.extract_line(rec["line_token"])
            if not ln.is_empty:
                for c in _chunks(np.asarray(ln.coords)[:, :2], layer):
                    c["line"] = ring_id
                    chunks.append(c)
                ring_id += 1
    polys = []
    for layer in boundary_layers:
        for rec in getattr(nmap, layer):
            for ptok in (rec["polygon_tokens"] if "polygon_tokens" in rec else [rec["polygon_token"]]):   # a token of None is real (see nuscenes_gt)
                poly = nmap.extract_polygon(ptok)
                if not poly.is_empty:
                    polys.append(poly)
    union = shapely.union_all(polys) if polys else None
    if union is not None and simplify:
        union = union.simplify(simplify, preserve_topology=True)
    for g in ([] if union is None else (union.geoms if hasattr(union, "geoms") else [union])):
        for ring in [g.exterior] + list(g.interiors):
            for c in _chunks(np.asarray(ring.coords)[:, :2], "boundary"):
                c["line"] = ring_id
                chunks.append(c)
            ring_id += 1
    return dict(chunks=chunks, road=polygon_chunks([(None, union)]) if union is not None else [])


def polygon_chunks(items, tile=100.0, min_area=0.0):
    """[(id, shapely geometry)] -> chunks {"id", "poly", "bbox"}: every polygon cut into tiles of `tile` metres where it is bigger (the road area of a map is one
    polygon with 100,000 vertices; a keyframe takes the tiles near the ego)"""
    import shapely
    chunks = []
    for ident, geom in items:
        for poly in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            x0, y0, x1, y1 = poly.bounds
            if max(x1 - x0, y1 - y0) <= 1.5 * tile:
                chunks.append(dict(id=ident, poly=poly, bbox=np.array(poly.bounds)))
                continue
            for ix in range(int(np.floor(x0 / tile)), int(np.floor(x1 / tile)) + 1):
                for iy in range(int(np.floor(y0 / tile)), int(np.floor(y1 / tile)) + 1):
                    part = shapely.clip_by_rect(poly, ix * tile, iy * tile, (ix + 1) * tile, (iy + 1) * tile)
                    for q in (part.geoms if hasattr(part, "geoms") else [part]):
                        if not q.is_empty and q.geom_type == "Polygon" and q.area > min_area:
                            chunks.append(dict(id=ident, poly=q, bbox=np.array(q.bounds)))
    return chunks


def _valid_polygons(g):
    """the polygons of a shapely geometry, each one OGC valid: `clip_by_rect` and the union of the map leave rings that touch each other (shapely.make_valid splits such a polygon)"""
    import shapely
    out = []
    for part in (g.geoms if hasattr(g, "geoms") else [g]):
        if part.is_empty:
            continue
        if part.geom_type == "Polygon" and part.is_valid:
            out.append(part)
        elif part.geom_type == "Polygon":
            out += _valid_polygons(shapely.make_valid(part))
        elif hasattr(part, "geoms"):
            out += _valid_polygons(part)
    return out


def crop_areas(chunks, T_e2g, half=50.0, frame="lane", merge=False, min_area=0.5, window=None):
    """the polygons of `chunks` cut to the window of one keyframe: [{"id", "exterior" (n, 2) float32 counter-clockwise, "holes" [(m, 2)] clockwise, "area"}];
    `merge`: the pieces that touch (the tiles of a big polygon) are united first; the window is the square |x|, |y| <= half or the rectangle `window` = (x0, x1, y0, y1)"""
    import shapely
    from shapely import affinity
    from shapely.geometry.polygon import orient
    t, Re = T_e2g[:2, 3], T_e2g[:2, :2]
    A = Re if frame == "ego" else Re @ np.array([[0.0, 1.0], [-1.0, 0.0]])              # frame point = (global point - t) @ A
    mat = [A[0, 0], A[1, 0], A[0, 1], A[1, 1], -(t[0] * A[0, 0] + t[1] * A[1, 0]), -(t[0] * A[0, 1] + t[1] * A[1, 1])]
    W = window_of(half, window)
    r = reach(half, window)
    cut = {}
    for c in chunks:
        b = c["bbox"]
        if b[2] < t[0] - r or b[0] > t[0] + r or b[3] < t[1] - r or b[1] > t[1] + r:
            continue
        g = shapely.clip_by_rect(affinity.affine_transform(c["poly"], mat), W[0], W[2], W[1], W[3])           # shapely: xmin, ymin, xmax, ymax
        if not g.is_empty:
            cut.setdefault(c["id"], []).append(g)
    out = []
    for ident in sorted(cut, key=lambda x: (x is None, str(x))):
        geoms = cut[ident]
        if merge:
            geoms = [shapely.union_all(geoms)]
        for g in geoms:
            for poly in _valid_polygons(g):
                if poly.area < min_area:
                    continue
                poly = orient(poly, 1.0)
                out.append(dict(id=ident, exterior=np.asarray(poly.exterior.coords, np.float32), holes=[np.asarray(h.coords, np.float32) for h in poly.interiors],
                                area=float(poly.area)))
    out.sort(key=lambda d: (str(d["id"]), -d["area"]))
    return out


def find_corners(P, deg=30.0, base=2.0):
    """the corners of a polyline (n, 2): [{"pos", "angle", "idx", "dir_in", "dir_out"}]"""
    P = np.asarray(P, float)
    if len(P) < 3:
        return []
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 2 * base:
        return []
    u = np.arange(0, s[-1] + 1e-9, 0.25)
    R = np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])]
    k = max(1, int(round(base / 0.25)))
    if len(R) <= 2 * k:
        return []
    c = np.arange(k, len(R) - k)
    d_in, d_out = R[c] - R[c - k], R[c + k] - R[c]
    ang = np.degrees(np.abs(np.arctan2(d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0], (d_in * d_out).sum(1))))
    out, i = [], 0
    window = int(round(2.0 / 0.25))
    while i < len(c):                                          # the largest angle within 2 m
        if ang[i] > deg:
            j = min(i + window, len(c))
            m = i + int(np.argmax(ang[i:j]))
            out.append(dict(pos=R[c[m]].astype(np.float32), angle=float(ang[m]), idx=int(np.argmin(np.linalg.norm(P - R[c[m]], axis=1))),
                            dir_in=(d_in[m] / max(np.linalg.norm(d_in[m]), 1e-9)).astype(np.float32),
                            dir_out=(d_out[m] / max(np.linalg.norm(d_out[m]), 1e-9)).astype(np.float32)))
            i = m + window
        else:
            i += 1
    return out


def crop_lines(lg, T_e2g, half=50.0, frame="lane", corner_deg=30.0, corner_base=2.0, window=None):
    """the lines of one keyframe window (see the module docstring); `lg` = build_line_graph(...); the window is the square |x|, |y| <= half or the rectangle `window`"""
    t = T_e2g[:2, 3]
    W = window_of(half, window)
    r = reach(half, window)
    found = {"lane_divider": [], "road_divider": [], "boundary": []}
    for c in lg["chunks"]:
        b = c["bbox"]
        if b[2] < t[0] - r or b[0] > t[0] + r or b[3] < t[1] - r or b[1] > t[1] + r:
            continue
        for pts, cut0, cut1, first, last in clip_rect(to_frame(c["pts"], T_e2g, frame), *W):
            at_start, at_end = first == 0 and not cut0, last == len(c["pts"]) - 2 and not cut1      # the part runs up to the chunk's first / last vertex
            found[c["kind"]].append(dict(line=c["line"], chunk_first=c["first"], closed=c["closed"], pts=pts, joint_start=at_start, joint_end=at_end,
                                         start_open=bool(cut0) or (at_start and not c["first"]),      # the line goes on in the chunk before / after
                                         end_open=bool(cut1) or (at_end and not c["last"])))
    out = {}
    for kind, items in found.items():
        # a line that was cut into chunks: join the pieces that meet at a chunk boundary
        items.sort(key=lambda x: (x["line"], not x["chunk_first"]))
        merged = []
        for it in items:
            if merged and merged[-1]["line"] == it["line"] and merged[-1]["joint_end"] and it["joint_start"] and \
                    np.linalg.norm(merged[-1]["pts"][-1] - it["pts"][0]) < 1e-6:
                m = merged[-1]
                m["pts"] = np.concatenate([m["pts"], it["pts"][1:]])
                m["end_open"], m["joint_end"] = it["end_open"], it["joint_end"]
            else:
                merged.append(dict(it))
        by_line = {}
        for m in merged:
            by_line.setdefault(m["line"], []).append(m)
        joined = []
        for ms in by_line.values():                                    # a ring that leaves the window and has its first vertex inside: its last and first parts are one line
            if len(ms) >= 2 and ms[0]["closed"] and ms[0]["joint_start"] and ms[-1]["joint_end"] and np.linalg.norm(ms[-1]["pts"][-1] - ms[0]["pts"][0]) < 1e-6:
                last = ms.pop()
                ms[0]["pts"] = np.concatenate([last["pts"], ms[0]["pts"][1:]])
                ms[0]["start_open"], ms[0]["joint_start"] = last["start_open"], False
            joined += ms
        out[kind] = []
        for m in joined:
            P = m["pts"]
            if np.linalg.norm(np.diff(P, axis=0), axis=1).sum() < 0.5:     # a line of under half a metre is a clipping crumb
                continue
            out[kind].append(dict(pts=P.astype(np.float32), start_open=m["start_open"], end_open=m["end_open"], corners=find_corners(P, corner_deg, corner_base)))
    return out
