p = "comparison/maps.py"
s = open(p, encoding="utf-8").read()
old_start = s.index("def dilate(mask, radius_cells: float)")
old_end = s.index("def tolerant_confusion")
new = '''try:                                                       # OpenCV makes the dilation about 8x faster; the plain loop below gives the same map
    import cv2
except ImportError:                                         # pragma: no cover
    cv2 = None


def _disc(radius_cells: float) -> np.ndarray:
    """the structuring element of a dilation: (2r + 1)^2 booleans, set where the offset (dx, dy) lies within radius_cells"""
    r = int(math.floor(radius_cells + 1e-9))
    d = np.arange(-r, r + 1)
    return (d[:, None] ** 2 + d[None, :] ** 2) <= radius_cells * radius_cells + 1e-9


def dilate_loop(m: np.ndarray, radius_cells: float) -> np.ndarray:
    """the dilation written out: OR of the map shifted to every offset of the disc (the reference of `dilate`)"""
    r = int(math.floor(radius_cells + 1e-9))
    rows, cols = m.shape
    padded = np.pad(m, r)
    out = np.zeros_like(m)
    for dx, dy in zip(*np.nonzero(_disc(radius_cells))):
        out |= padded[dx:dx + rows, dy:dy + cols]
    return out


def dilate(mask, radius_cells: float) -> np.ndarray:
    """boolean dilation of a 2-D mask by a disc: a cell is set if a set cell's centre lies within radius_cells of its centre."""
    m = _as_mask(mask, "mask")
    if m.ndim != 2:
        raise ValueError("dilate works on 2-D maps")
    if int(math.floor(radius_cells + 1e-9)) <= 0:
        return m.copy()
    if cv2 is None:
        return dilate_loop(m, radius_cells)
    return cv2.dilate(np.ascontiguousarray(m, np.uint8), _disc(radius_cells).astype(np.uint8)).astype(bool)      # outside the map counts as empty, as the zero padding of the loop


'''
s = s[:old_start] + new + s[old_end:]
old = "def tolerant_scores(counts)"
add = '''def tolerant_counts_many(pred, gt, near_gt, near_pred, valids) -> np.ndarray:
    """`tolerant_counts` for S selections at once: valids = S boolean maps of the same shape -> int64 (S, 4), row i = tolerant_counts(..., valids[i]).
    A count is a sum of 0 / 1 products over the cells, i.e. a dot product of the selection with an indicator map, so all S x 4 counts are ONE matrix
    product (S x cells) @ (cells x 4); float32 holds the counts exactly (up to 2^24 cells)."""
    p, g = _as_mask(pred, "pred"), _as_mask(gt, "gt")
    near_g, near_p = _as_mask(near_gt, "near_gt"), _as_mask(near_pred, "near_pred")
    V = np.stack([_as_mask(v, "valid").ravel() for v in valids]).astype(np.float32)
    _same_shape(p, g, near_g, near_p, np.asarray(valids[0], bool))
    if p.size >= 1 << 24:
        raise ValueError("more than 2^24 cells: float32 counts would no longer be exact")
    X = np.stack([(p & near_g).ravel(), (p & ~near_g).ravel(), (g & near_p).ravel(), (g & ~near_p).ravel()], 1).astype(np.float32)
    return np.rint(V @ X).astype(np.int64)


'''
s = s.replace(old, add + old, 1)
open(p, "w", encoding="utf-8").write(s)
