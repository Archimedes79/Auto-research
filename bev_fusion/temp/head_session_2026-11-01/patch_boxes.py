"""One-off patch: ONE box extractor. extract.py gets _params (the box parameters of any set of cells), dense_boxes built on it, a vectorised
select_boxes and boxes_from_maps (gathers only the K peak cells); object_head.decode becomes a wrapper; model.boxes calls boxes_from_maps.
Exact-string / marker replacements; each anchor must match once."""
import os

T = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training"))


def rd(n):
    return open(os.path.join(T, n), encoding="utf-8").read()


def wr(n, s):
    open(os.path.join(T, n), "w", encoding="utf-8", newline="\n").write(s)


# ------------------------------------------------------------------ extract.py
src = rd("extract.py")
a = src.index("def _centres(grid, X, Y, off):")
b = src.index("def soft_line_map(")
NEW = '''def _params(obj, ix, iy, grid, soft):
    """the box parameters of a set of cells from the head maps indexed on those cells: obj fields (B, C, *cells), ix / iy (B, *cells) the cell
    indices (same dtype as the maps). One code for the dense output (every cell: cells = (X, Y)) and for the selection (the K peak
    cells only: cells = (K,)). Returns cx, cy (metres, vehicle frame), length, width, radius, yaw, vx, vy and, soft, the heading vector."""
    out = {"cx": grid.x_min + (ix + obj["off"][:, 0] + 0.5) * grid.res, "cy": grid.y_min + (iy + obj["off"][:, 1] + 0.5) * grid.res}
    if "size" in obj:
        out["length"], out["width"] = torch.exp(obj["size"][:, 0]), torch.exp(obj["size"][:, 1])
    if "rad" in obj:
        out["radius"] = torch.exp(obj["rad"][:, 0])
    if "rot" in obj:
        yaw = torch.atan2(obj["rot"][:, 0], obj["rot"][:, 1])
        if "dir" in obj:                                   # the box axis in [off, off + pi), then the half from the class
            axis = torch.remainder(yaw - DIR_OFFSET, math.pi) + DIR_OFFSET
            back = obj["dir"][:, 0] < 0
            if soft:                                       # heading vector: the axis times tanh(dir) - its length is the front / back
                t = torch.tanh(obj["dir"][:, 0])           # confidence, so the gradient reaches the dir logit (an angle cannot carry it)
                out["hx"], out["hy"] = t * torch.cos(axis), t * torch.sin(axis)
            yaw = torch.where(back, torch.remainder(axis + 2 * math.pi, 2 * math.pi) - math.pi, axis)
        elif soft:
            out["hx"], out["hy"] = torch.cos(yaw), torch.sin(yaw)
        out["yaw"] = yaw
    v = obj["vel"]
    if "mov" in obj:                                       # static (moving logit < 0): velocity 0 (the AVE-optimal answer)
        gate = torch.sigmoid(obj["mov"]) if soft else (obj["mov"] >= 0).to(v.dtype)
        v = v * gate
    out["vx"], out["vy"] = v[:, 0], v[:, 1]
    return out


def dense_boxes(obj, grid, groups, beta=10.0, soft=True, from_probabilities=False):
    """obj: the object head maps (hm logits (B, G, X, Y), off, size, rot, rad, vel; optional dir, mov) -> dict of dense maps:
    hm (heatmap in [0, 1]), w (peak weight), cx, cy, length, width, radius, yaw, vx, vy (each (B, X, Y); w and hm (B, G, X, Y))."""
    hm = obj["hm"] if from_probabilities else torch.sigmoid(obj["hm"])
    B, G, X, Y = hm.shape
    w = hm * torch.exp(-beta * (F.max_pool2d(hm, 3, 1, 1) - hm))
    ix = torch.arange(X, device=hm.device, dtype=hm.dtype).view(1, X, 1)
    iy = torch.arange(Y, device=hm.device, dtype=hm.dtype).view(1, 1, Y)
    return {"hm": hm, "w": w, **_params(obj, ix, iy, grid, soft)}


def _peaks(hm, k):
    """hard local maxima (3 x 3) of the heatmap (B, G, X, Y) -> top-k scores and the group / flat cell index of each, both (B, K)"""
    B, G, X, Y = hm.shape
    peaks = hm * (hm == F.max_pool2d(hm, 3, 1, 1)).to(hm.dtype)
    scores, idx = peaks.reshape(B, -1).topk(min(k, G * X * Y))
    return scores, torch.div(idx, X * Y, rounding_mode="floor"), idx % (X * Y)


_FIELDS = ("cx", "cy", "vx", "vy", "length", "width", "yaw", "radius")


def _to_dicts(scores, group, vals, fields, groups, threshold):
    """(B, K) scores / group index and (B, K, F) values -> list (B) of box dicts (object_head.decode's format), scores >= threshold; one
    transfer to the host for the whole batch, no per-box tensor access"""
    sc, gr, vl = scores.tolist(), group.tolist(), vals.tolist()
    col = {f: i for i, f in enumerate(fields)}
    res = []
    for b in range(len(sc)):
        boxes = []
        for s, g, v in zip(sc[b], gr[b], vl[b]):
            if s < threshold:
                break                                      # sorted: the rest is below as well
            bx = {"group": groups[g][0], "score": s, "cx": v[col["cx"]], "cy": v[col["cy"]], "vx": v[col["vx"]], "vy": v[col["vy"]]}
            if groups[g][1] == "box":
                bx.update(length=v[col["length"]], width=v[col["width"]], yaw=v[col["yaw"]])
            else:
                bx["radius"] = v[col["radius"]]
            boxes.append(bx)
        res.append(boxes)
    return res


def select_boxes(dense, groups, k=50, threshold=0.2):
    """the post-processing of a dense box output (planner side): hard local maximum of the heatmap, top-k, threshold -> list (B) of box
    dicts (object_head.decode's format); the K cells are gathered from the dense fields in one go"""
    scores, group, cell = _peaks(dense["hm"], k)
    fields = [f for f in _FIELDS if f in dense]
    vals = torch.stack([dense[f].reshape(dense[f].shape[0], -1).gather(1, cell) for f in fields], -1)
    return _to_dicts(scores, group, vals, fields, groups, threshold)


def boxes_from_maps(obj, grid, groups, k=50, threshold=0.2, from_probabilities=False):
    """THE box extractor: head maps -> list (B) of box dicts. Only the K peak cells are gathered from the raw maps (on the device of the
    maps) and parametrised, in float64, by the same `_params` as the dense output; object_head.decode, BEVFusionModel.boxes, the evaluation
    adapter and the prototype all come here. from_probabilities: hm already in [0, 1]."""
    hm = obj["hm"] if from_probabilities else torch.sigmoid(obj["hm"])
    scores, group, cell = _peaks(hm, k)
    Y = hm.shape[3]
    sub = {n: m.flatten(2).gather(2, cell[:, None, :].expand(-1, m.shape[1], -1)).double() for n, m in obj.items() if n != "hm"}
    p = _params(sub, torch.div(cell, Y, rounding_mode="floor").double(), (cell % Y).double(), grid, soft=False)
    fields = [f for f in _FIELDS if f in p]
    return _to_dicts(scores, group, torch.stack([p[f] for f in fields], -1), fields, groups, threshold)


'''
src = src[:a] + NEW + src[b:]

# the self-test: an independent reference (the plain loop of the old decode) for steps 1 and 1b
old_imp = '''    try:
        from .object_head import decode
    except ImportError:
        from object_head import decode
    import sys'''
assert src.count(old_imp) == 1
src = src.replace(old_imp, '''    import sys''', 1)
REF = '''def _decode_reference(out, grid, groups, k=50, threshold=0.2):
    """the plain per-box loop of the old object_head.decode: the independent oracle of the self-test (python floats, one box at a time)"""
    hm = torch.sigmoid(out["hm"])
    hm = hm * (hm == F.max_pool2d(hm, 3, 1, 1)).float()
    B, G, X, Y = hm.shape
    scores, idx = hm.reshape(B, -1).topk(min(k, G * X * Y))
    res = []
    for b in range(B):
        boxes = []
        for s, i in zip(scores[b].tolist(), idx[b].tolist()):
            if s < threshold:
                continue
            g, cell = divmod(i, X * Y)
            ix, iy = divmod(cell, Y)
            ox, oy = out["off"][b, :, ix, iy].tolist()
            bx = {"group": groups[g][0], "score": s, "cx": grid.x_min + (ix + ox + 0.5) * grid.res, "cy": grid.y_min + (iy + oy + 0.5) * grid.res,
                  "vx": out["vel"][b, 0, ix, iy].item(), "vy": out["vel"][b, 1, ix, iy].item()}
            if "mov" in out and out["mov"][b, 0, ix, iy].item() < 0:
                bx["vx"] = bx["vy"] = 0.0
            if groups[g][1] == "box":
                ll, lw = out["size"][b, :, ix, iy].tolist()
                sn, cs = out["rot"][b, :, ix, iy].tolist()
                yaw = math.atan2(sn, cs)
                if "dir" in out:
                    yaw = (yaw - DIR_OFFSET) % math.pi + DIR_OFFSET
                    if out["dir"][b, 0, ix, iy].item() < 0:
                        yaw = math.remainder(yaw + math.pi, 2 * math.pi)
                bx.update(length=math.exp(ll), width=math.exp(lw), yaw=yaw)
            else:
                bx["radius"] = math.exp(out["rad"][b, 0, ix, iy].item())
            boxes.append(bx)
        res.append(boxes)
    return res


# ----------------------------------------------------------------------------------------------- self-test
'''
mark = "# ----------------------------------------------------------------------------------------------- self-test\n"
assert src.count(mark) == 1
src = src.replace(mark, REF, 1)
old1 = '''    # 1. the hard selection reproduces object_head.decode box by box
    ref = decode(obj, Grid, groups, k=60, threshold=0.2)
    got = select_boxes(dense_boxes(obj, Grid, groups, soft=False), groups, k=60, threshold=0.2)
'''
assert src.count(old1) == 1
src = src.replace(old1, '''    # 1. the vectorised selection (from the dense output AND straight from the maps, = object_head.decode) reproduces the plain loop box by box
    ref = _decode_reference(obj, Grid, groups, k=60, threshold=0.2)
    got = select_boxes(dense_boxes(obj, Grid, groups, soft=False), groups, k=60, threshold=0.2)
    direct = boxes_from_maps(obj, Grid, groups, k=60, threshold=0.2)
    assert [len(a) for a in direct] == [len(b) for b in got] == [len(r) for r in ref], "the two entry points select the same boxes"
    for db, gb in zip(direct, got):
        for d_, g_ in zip(db, gb):
            assert d_.keys() == g_.keys() and all(abs(d_[kk] - g_[kk]) < 1e-9 for kk in d_ if kk != "group") and d_["group"] == g_["group"]
''', 1)
old2 = '''    r32 = decode(o32, Grid, groups, k=60, threshold=0.2)
    g32 = select_boxes(dense_boxes({kk: v.double() for kk, v in o32.items()}, Grid, groups, soft=False), groups, k=60, threshold=0.2)
'''
assert src.count(old2) == 1
src = src.replace(old2, '''    r32 = _decode_reference(o32, Grid, groups, k=60, threshold=0.2)
    g32 = boxes_from_maps(o32, Grid, groups, k=60, threshold=0.2)
''', 1)
old3 = '''    print("extract selftest passed: hard selection = object_head.decode on", sum(len(r) for r in ref),'''
assert src.count(old3) == 1
src = src.replace(old3, '''    print("extract selftest passed: the vectorised selection = the plain per-box loop on", sum(len(r) for r in ref),''', 1)
wr("extract.py", src)

# ------------------------------------------------------------------ object_head.py: decode is a wrapper
oh = rd("object_head.py")
a = oh.index("@torch.no_grad()\ndef decode(out, grid, groups=GROUPS, k=50, threshold=0.2, from_probabilities=False):")
b = oh.index("class TimeObjectHead(ObjectHead):")
assert a < b
WRAP = '''@torch.no_grad()
def decode(out, grid, groups=GROUPS, k=50, threshold=0.2, from_probabilities=False):
    """out maps -> list (B) of box dicts with 'score': the local maxima of the heatmap (3 x 3 max-pool), top-k, score threshold. With
    out['mov'] the velocity is 0 where sigmoid(mov) < 0.5 (static: the AVE-optimal answer of a two-point mixture; inference only).
    There is ONE extractor, extract.boxes_from_maps (vectorised, only the K peak cells are read); this is its name in the object head's
    module. extract.py's self-test checks it box by box against the plain per-box loop. from_probabilities: hm already in [0, 1]."""
    try:
        from .extract import boxes_from_maps               # imported here: extract imports this module (DIR_OFFSET)
    except ImportError:
        from extract import boxes_from_maps
    return boxes_from_maps(out, grid, groups, k, threshold, from_probabilities)


'''
oh = oh[:a] + WRAP + oh[b:]
wr("object_head.py", oh)

# ------------------------------------------------------------------ model.py: boxes() calls the one extractor
m = rd("model.py")
old = '''        dense = dense_boxes({kk: v.double().cpu() for kk, v in p["obj"].items()}, self.grid, NUSC_GROUPS, soft=False)   # double: = decode
        dets = select_boxes(dense, NUSC_GROUPS, k=k, threshold=threshold)
'''
assert m.count(old) == 1, m.count(old)
m = m.replace(old, '''        dets = boxes_from_maps({kk: v.detach() for kk, v in p["obj"].items()}, self.grid, NUSC_GROUPS, k=k, threshold=threshold)
''', 1)
for imp_old, imp_new in (("    from .extract import dense_boxes, select_boxes, soft_line_map", "    from .extract import boxes_from_maps, dense_boxes, soft_line_map"),
                         ("    from extract import dense_boxes, select_boxes, soft_line_map", "    from extract import boxes_from_maps, dense_boxes, soft_line_map")):
    assert m.count(imp_old) == 1, imp_old
    m = m.replace(imp_old, imp_new, 1)
old = "(boxes(): extract.select_boxes = object_head.decode)."
assert m.count(old) == 1
m = m.replace(old, "(boxes(): extract.boxes_from_maps, the one box extractor; object_head.decode is the same function).", 1)
wr("model.py", m)
print("patched extract.py, object_head.py, model.py")
