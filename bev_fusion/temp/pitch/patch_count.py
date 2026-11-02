p = "evaluation/metrics.py"
s = open(p, encoding="utf-8").read()
a = s.index("    def _count(self, key_name, P, G, subsets, dt, extra=None):")
b = s.index("    def add(self, pred, gt, mask=None")
new = '''    def _count(self, key_name, P, G, subsets, dt, extra=None):
        bins = cell_bins(P.shape, self.grid, self.bins)
        dil = [(dilate(G, t / self.grid.res), dilate(P, t / self.grid.res)) for t in self.tol]
        keys, cms = [], []                                                      # every selection of cells: (subset, range bin)
        for sname, sm in subsets.items():
            for b in list(range(len(self.bins))) + ["all"]:
                cms.append(sm & ((bins >= 0) if b == "all" else (bins == b)))
                keys.append((round(float(dt), 3), key_name, sname, "all" if b == "all" else bin_label(b)))
        cnt = [_maps.tolerant_counts_many(P, G, DG, DP, cms) for DG, DP in dil]   # per tolerance (selections, 4): [p_hit, p_miss, g_hit, g_miss], one matrix product
        for j, (key, cm) in enumerate(zip(keys, cms)):
            acc = self.c.setdefault(key, {"cnt": np.zeros((len(self.tol), 4)), "n_gt": 0, "err": 0.0, "n_err": 0})
            for i in range(len(self.tol)):
                acc["cnt"][i] += cnt[i][j]
            acc["n_gt"] += int((G & cm).sum())
            if extra is not None:
                err, valid = extra
                acc["err"] += float(err[valid & cm].sum())
                acc["n_err"] += int((valid & cm).sum())

'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
