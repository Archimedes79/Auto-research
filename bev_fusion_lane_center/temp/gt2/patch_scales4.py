def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:70])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))


sub("training/grid/qjunction.py", "def fit_pair_scales(jg, scales=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85)):", "def fit_pair_scales(jg, scales=(0.05, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25)):")
sub("training/grid/qjunction.py", "np.isfinite(sc).all() and (sc > 0.05).all(), sc", "np.isfinite(sc).all() and (sc >= 0.05).all(), sc")
sub("training/grid/jends.py", "out[\"scale\"] = 0.1 + 0.75 * self.scale(fcat).sigmoid()", "out[\"scale\"] = 0.05 + 1.2 * self.scale(fcat).sigmoid()")
sub("training/grid/jends.py", "assert out[\"scale\"].shape == (B, K, K, 2) and float(out[\"scale\"].min()) >= 0.1 and float(out[\"scale\"].max()) <= 0.85", "assert out[\"scale\"].shape == (B, K, K, 2) and float(out[\"scale\"].min()) >= 0.05 and float(out[\"scale\"].max()) <= 1.25")
sub("training/grid/jends.py", "in chords (0.1 .. 0.85 like the fit grid)", "in chords (0.05 .. 1.25 like the fit grid)")
print("ok")
