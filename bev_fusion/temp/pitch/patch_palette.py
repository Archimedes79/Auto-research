p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''COLORS = {"drivable_area": (0.80, 0.80, 0.80), "carpark_area": (0.70, 0.78, 0.90), "walkway": (0.62, 0.85, 0.62),
          "ped_crossing": (1.00, 0.70, 0.35), "stop_line": (0.80, 0.68, 0.92),
          "vehicle": (0.15, 0.35, 0.90), "pedestrian": (0.85, 0.15, 0.75), "two_wheeler": (0.10, 0.75, 0.80)}
LINE_COLORS = {"lane_divider": (0.15, 0.15, 0.15), "road_divider": (0.85, 0.10, 0.10)}''',
    '''# The colour scheme of the bird's-eye pictures (Mirko's proposal of 2026-11-03): black plot background, dark-grey road with a black boundary, white lane markings, light-grey
# crossings, dark-red objects (vehicles, pedestrians, two-wheelers) and the same red, translucent, where they lie outside the observed area, a calmer green for walkways.
BG = (0.0, 0.0, 0.0)
ROAD_EDGE = (0.0, 0.0, 0.0)
DARK_RED, OUT_RED, OUT_ALPHA = (0.62, 0.04, 0.07), (0.95, 0.16, 0.16), 0.45
OBJECT_LAYERS = ("vehicle", "pedestrian", "two_wheeler")
GT_BOX_COLOR, GT_BOX_OUT = (0.80, 0.12, 0.14), (0.95, 0.30, 0.30, 0.55)     # outline of a ground-truth box inside / outside the observed area (RGBA: translucent)
GT_OVERLAY, DET_COLOR = (0.93, 0.93, 0.93), (1.0, 0.72, 0.15)                 # dashed GT outline on a prediction panel; predicted boxes
COLORS = {"drivable_area": (0.23, 0.23, 0.25), "carpark_area": (0.31, 0.33, 0.40), "walkway": (0.31, 0.60, 0.43),
          "ped_crossing": (0.74, 0.74, 0.74), "stop_line": (0.60, 0.60, 0.64),
          "vehicle": DARK_RED, "pedestrian": DARK_RED, "two_wheeler": DARK_RED}
LINE_COLORS = {"lane_divider": (0.97, 0.97, 0.97), "road_divider": (0.98, 0.82, 0.30)}''')
rep('''MAP_LEGEND = [(n, c) for n, c in list(COLORS.items()) + list(LINE_COLORS.items())]''',
    '''MAP_LEGEND = [(n, c) for n, c in list(COLORS.items()) + list(LINE_COLORS.items())] + [("object outside the observed area", OUT_RED)]''')

# box overlays: per-box colour by the observed area
rep('''def box_shapes(boxes, color, ls="-", lw=1.2, score_min=None) -> List[Shape]:
    """box dicts (cx, cy, length, width, yaw | radius) -> overlay shapes in bird's-eye plot coordinates (horizontal = y)"""
    out = []
    for b in boxes:
        if score_min is not None and b.get("score", 1.0) < score_min:
            continue''',
    '''def observed_area(g) -> np.ndarray:
    """cells the rig observes in this GT frame (the fusion area of D37: camera cells not hidden behind a box, inside the radar fan); all True for a GT without these layers"""
    cam = np.asarray(g["camera"], np.float32) > 0 if "camera" in g else None
    if cam is None:
        return np.ones_like(np.asarray(g["undefined"]), bool) if "undefined" in g else np.ones((120, 104), bool)
    area = cam
    if "occluded" in g:
        area = area & ~(np.asarray(g["occluded"], np.float32) > 0)
    if "radar" in g:
        area = area & (np.asarray(g["radar"], np.float32) > 0)
    return area


def box_shapes(boxes, color, ls="-", lw=1.2, score_min=None, observed=None) -> List[Shape]:
    """box dicts (cx, cy, length, width, yaw | radius) -> overlay shapes in bird's-eye plot coordinates (horizontal = y). observed: a bool grid (observed_area) -> `color` for a
    box with at least half of its footprint inside, translucent GT_BOX_OUT for the others (the colour argument is then the one of the observed boxes)"""
    out = []
    if observed is not None:
        from evaluation.bev_fusion_area import footprint_share
    for b0 in boxes:
        b = b0
        if score_min is not None and b.get("score", 1.0) < score_min:
            continue
        color_b = color if observed is None or footprint_share(b, observed) >= 0.5 else GT_BOX_OUT''')
a = s.index("        color_b = color if observed is None")
seg_end = s.index("    return out\n", a)
seg = s[a:seg_end]
seg = seg.replace("color, ls, lw", "color_b, ls, lw")
s = s[:a] + seg + s[seg_end:]

rep('''    return [Shape("line", [(0, 0), (sg * 60 * math.tan(FRONT_HALF_FOV), 60)], "0.4", ":", 0.6) for sg in (-1, 1)]''',
    '''    return [Shape("line", [(0, 0), (sg * 60 * math.tan(FRONT_HALF_FOV), 60)], "0.62", ":", 0.7) for sg in (-1, 1)]''')

# composite
a = s.index("def composite(")
b = s.index("def tolerant_error_rgb")
s = s[:a] + '''def _edge(mask: np.ndarray) -> np.ndarray:
    """the cells of a boolean map that touch a cell outside it (4-neighbourhood, the map border counts as inside)"""
    pad = np.pad(mask, 1, mode="edge")
    inner = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return mask & ~inner


def composite(prob: Dict[str, np.ndarray], dist: Dict[str, np.ndarray], line_width: float = 0.35, observed: Optional[np.ndarray] = None) -> np.ndarray:
    """RGB (X, Y, 3) on the black background: the areas (alpha = probability, in the order of COLORS), a black boundary along the drivable area, then crossings, the object
    layers (dark red; outside `observed`, a bool grid, the same red translucent) and the lines (from the distance field)"""
    shape = next(iter(prob.values())).shape
    img = np.zeros(shape + (3,), np.float32) + np.asarray(BG, np.float32)

    def paint(c, a):
        nonlocal img
        img = img * (1 - a[..., None]) + np.asarray(c, np.float32) * a[..., None]
    for name in ("drivable_area", "carpark_area", "walkway"):
        if name in prob:
            paint(COLORS[name], np.clip(prob[name], 0, 1))
    if "drivable_area" in prob:                                                 # the road boundary, black, over the neighbouring walkway / car park
        paint(ROAD_EDGE, _edge(prob["drivable_area"] > 0.5).astype(np.float32))
    for name, c in COLORS.items():
        if name in ("drivable_area", "carpark_area", "walkway") or name not in prob:
            continue
        a = np.clip(prob[name], 0, 1)
        if name in OBJECT_LAYERS and observed is not None:
            paint(OUT_RED, a * np.where(observed, 0.0, OUT_ALPHA))
            paint(c, a * np.where(observed, 1.0, 0.0))
        else:
            paint(c, a)
    for name, c in LINE_COLORS.items():
        if name in dist:
            paint(c, np.clip(1 - dist[name] / line_width, 0, 1))
    return img


''' + s[b:]

# call sites: GT panels get the observed area, colours of the overlays
rep('''composite(gt_prob, gt_dist), box_shapes(gtb, "tab:blue")''', '''composite(gt_prob, gt_dist, observed=observed_area(g0)), box_shapes(gtb, GT_BOX_COLOR, observed=observed_area(g0))''', 3)
s = s.replace('box_shapes(gtb, "k", "--", 0.7)', 'box_shapes(gtb, GT_OVERLAY, "--", 0.7)')
s = s.replace('"k", "--", 0.7)', 'GT_OVERLAY, "--", 0.7)')
s = s.replace('"tab:red")', "DET_COLOR)")
s = s.replace("np.ones((2, 2, 3))", "np.zeros((2, 2, 3))")
open(p, "w", encoding="utf-8").write(s)
print("ok")
