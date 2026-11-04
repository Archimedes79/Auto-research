p = "autoresearch/evalsuite/comparison/README.md"
s = open(p, encoding="utf-8").read()
old = "- `dilate(mask, radius_cells)`: the disc dilation both use (2-D; a radius that is an integer up to float noise, like 0.3 / 0.1, is that integer)."
assert old in s
new = old + """ OpenCV does it when installed (8x faster), `dilate_loop` is the same dilation written out (the reference; the selftest compares the two, borders included).
- `tolerant_counts_many(pred, gt, nears, valids)` -> int64 `(tolerances, selections, 4)`: `tolerant_counts` for several tolerances and many `valid` selections in ONE matrix product. A count is the dot product of a selection (0 / 1 over the cells) with an indicator map (`pred & near_gt`, ...), so `(selections x cells) @ (cells x 4 per tolerance)` gives all counts at once (about 16x faster than one call per selection; float32 is exact up to 2^24 cells, larger maps are refused)."""
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
