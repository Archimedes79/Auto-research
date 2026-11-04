"""The lines of a keyframe window: lane dividers, road dividers and the road boundary, as polylines with their corners.

Source: the nuScenes map. `lane_divider` and `road_divider` are the map's own line records; the BOUNDARY is the outline of the union of the polygons of
`boundary_layers` (default: the drivable area; add "carpark_area" and / or "walkway" to count them as part of the road, i.e. no boundary between them
and the road). nuScenes has no boundary line record, this is a definition (open with Mirko: which layers).

CORNERS. Along every polyline the turning angle is measured between the chord of the last `corner_base` metres (default 2) and the chord of the next
`corner_base` metres; a vertex whose angle exceeds `corner_deg` (default 30 degrees) and is the largest within 2 m is a corner (a kerb corner,
a sharp end of a divider). A corner is {"pos", "angle" (degrees), "idx" (the nearest point of the polyline), "dir_in", "dir_out"}.

The result of `crop_lines` is {"lane_divider": [line, ...], "road_divider": [...], "boundary": [...]} in the frame of `autogt.lane_graph` (lane frame by
default, x right, y forward), cut to the square window; a line is {"pts" (float32, travel / outline order), "start_open", "end_open", "corners"}.
"""
import numpy as np

from .lane_graph import clip_square, dirs_to_frame, to_frame

DEFAULTS = dict(boundary_layers=("drivable_area",), corner_deg=30.0, corner_base=2.0, simplify=0.05)
CHUNK = 200                                                  # vertices per chunk of a long outline (a keyframe takes the chunks near the ego)


def _chunks(P, kind):
    """a long polyline in overlapping chunks with their bounding boxes (a chunk shares its last vertex with the next one)"""
    out = []
    for i in range(0, max(len(P) - 1, 1), CHUNK):
        Q = P[i:i + CHUNK + 1]
        if len(Q) >= 2:
            out.append(dict(kind=kind, pts=Q, bbox=np.r_[Q.min(0), Q.max(0)], first=i == 0, last=i + CHUNK + 1 >= len(P)))
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
    return dict(chunks=chunks, params=dict(boundary_layers=tuple(boundary_layers), simplify=simplify))


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


def crop_lines(lg, T_e2g, half=50.0, frame="lane", corner_deg=30.0, corner_base=2.0, margin=25.0):
    """the lines of one keyframe window (see the module docstring); `lg` = build_line_graph(...)"""
    t = T_e2g[:2, 3]
    r = half * np.sqrt(2.0) + margin
    found = {"lane_divider": [], "road_divider": [], "boundary": []}
    for c in lg["chunks"]:
        b = c["bbox"]
        if b[2] < t[0] - r or b[0] > t[0] + r or b[3] < t[1] - r or b[1] > t[1] + r:
            continue
        for pts, cut0, cut1, first, last in clip_square(to_frame(c["pts"], T_e2g, frame), half):
            at_start, at_end = first == 0 and not cut0, last == len(c["pts"]) - 2 and not cut1      # the part runs up to the chunk's first / last vertex
            found[c["kind"]].append(dict(line=c["line"], chunk_first=c["first"], pts=pts, joint_start=at_start, joint_end=at_end,
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
        out[kind] = []
        for m in merged:
            P = m["pts"]
            cs = find_corners(P, corner_deg, corner_base)
            for c in cs:
                c["dir_in"], c["dir_out"] = c["dir_in"], c["dir_out"]
            out[kind].append(dict(pts=P.astype(np.float32), start_open=m["start_open"], end_open=m["end_open"], corners=cs))
    return out
