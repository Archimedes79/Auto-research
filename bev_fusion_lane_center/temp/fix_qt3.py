p = "training/grid/qtargets.py"
s = open(p, encoding="utf-8").read()
old = s[s.index("def decode(nt, fit_long=True"): s.index("def coverage(")]
new = '''def fit_long_scales(nt, scales=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85)):
    """the two Hermite handle scales (a, b) of every LONG link, fitted to the true path between its two nodes (grid search): (L, 2), NaN for the short links.
    This is what the learned scales of the link head are trained towards"""
    from scipy.spatial import cKDTree
    out = np.full((len(nt["src"]), 2), np.nan)
    for l, (a, b, lg, path) in enumerate(zip(nt["src"], nt["dst"], nt["long"], nt["path"])):
        if not lg:
            continue
        p0, h0, p1, h1 = nt["pos"][a], nt["dirv"][a], nt["pos"][b], nt["dirv"][b]
        tgt = nt["pts"][path]
        tt = cKDTree(tgt)
        best = 1e9
        for sa in scales:
            for sb in scales:
                cur = hermite(p0, h0, p1, h1, sa, sb)
                m = tt.query(cur)[0].mean() + 0.2 * cKDTree(cur).query(tgt)[0].mean()
                if m < best:
                    best, out[l] = m, (sa, sb)
    return out


def decode(nt, fit_long=True, fit=None):
    """Hermite curve of every link -> list of (K, 2) arrays; long links: the handle scales are the fitted ones (`fit` = `fit_long_scales(nt)`, computed here if not given), the oracle of what
    the learned scales can reach"""
    if fit is None and fit_long:
        fit = fit_long_scales(nt)
    out = []
    for l, (a, b, lg) in enumerate(zip(nt["src"], nt["dst"], nt["long"])):
        p0, h0, p1, h1 = nt["pos"][a], nt["dirv"][a], nt["pos"][b], nt["dirv"][b]
        if lg and fit is not None and not np.isnan(fit[l, 0]):
            out.append(hermite(p0, h0, p1, h1, fit[l, 0], fit[l, 1]))
        else:
            out.append(hermite(p0, h0, p1, h1))
    return out


'''
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
