p = "../autoresearch/evalsuite/comparison/maps.py"
s = open(p, encoding="utf-8").read()
a = s.index("def tolerant_counts_many")
b = s.index("def tolerant_scores")
new = '''def tolerant_counts_many(pred, gt, nears, valids) -> np.ndarray:
    """`tolerant_counts` for several tolerances and several selections at once: nears = T pairs (near_gt, near_pred), valids = S boolean maps of the same
    shape -> int64 (T, S, 4), [t, i] = tolerant_counts(pred, gt, *nears[t], valids[i]).
    A count is a sum of 0 / 1 products over the cells, i.e. a dot product of a selection with an indicator map, so all T x S x 4 counts are ONE matrix
    product (S x cells) @ (cells x 4T); float32 holds the counts exactly (up to 2^24 cells)."""
    p, g = _as_mask(pred, "pred"), _as_mask(gt, "gt")
    V = np.stack([_as_mask(v, "valid").ravel() for v in valids]).astype(np.float32)
    if p.size >= 1 << 24:
        raise ValueError("more than 2^24 cells: float32 counts would no longer be exact")
    cols = []
    for near_gt, near_pred in nears:
        near_g, near_p = _as_mask(near_gt, "near_gt"), _as_mask(near_pred, "near_pred")
        _same_shape(p, g, near_g, near_p, np.asarray(valids[0], bool))
        cols += [(p & near_g).ravel(), (p & ~near_g).ravel(), (g & near_p).ravel(), (g & ~near_p).ravel()]
    out = np.rint(V @ np.stack(cols, 1).astype(np.float32)).astype(np.int64)         # (S, 4T)
    return out.reshape(len(valids), len(nears), 4).transpose(1, 0, 2)


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/metrics.py"
s = open(p, encoding="utf-8").read()
old = '''        cnt = [_maps.tolerant_counts_many(P, G, DG, DP, cms) for DG, DP in dil]   # per tolerance (selections, 4): [p_hit, p_miss, g_hit, g_miss], one matrix product'''
new = '''        cnt = _maps.tolerant_counts_many(P, G, dil, cms)                        # (tolerances, selections, 4): [p_hit, p_miss, g_hit, g_miss], one matrix product'''
assert old in s
s = s.replace(old, new)
s = s.replace("                acc[\"cnt\"][i] += cnt[i][j]", "                acc[\"cnt\"][i] += cnt[i, j]")
old = "        bins = cell_bins(P.shape, self.grid, self.bins)\n        dil = "
assert old in s
s = s.replace(old, "        if P.shape not in self._bins:\n            self._bins[P.shape] = cell_bins(P.shape, self.grid, self.bins)       # the same for every layer and frame of a grid\n        bins = self._bins[P.shape]\n        dil = ")
s = s.replace("        self.c = {}                                       # (dt, name, subset, bin) -> counts", "        self.c = {}                                       # (dt, name, subset, bin) -> counts\n        self._bins = {}                                   # map shape -> range bin of every cell")
open(p, "w", encoding="utf-8").write(s)
