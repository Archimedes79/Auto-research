p = "training/data/lane_ptok_gpu.py"
t = open(p, encoding="utf-8").read()
reps = [
('''        self.ypair, self.ypl, self.yps = t("ypair").long(), t("ypair_long"), t("ypair_scale")
        self.P = int((self.ypair[..., 0] >= 0).sum(1).max())                  # the widest pair list of a frame: the fixed width of `sp_*` (the stored width is larger)
        self.n = len(self.rows)
''','''        yp = np.array(f["ypair"])
        used = np.flatnonzero((yp[..., 0] >= 0).any(0))
        self.P = int(used.max()) + 1 if len(used) else 1                      # the pair columns in use (one past the last used slot, front-packed or not): the fixed width of `sp_*`; the stored width (4096) is larger
        self.ypair = torch.from_numpy(np.ascontiguousarray(yp[:, :self.P])).to(dev)         # int16 on the device, cast per batch
        self.ypl = torch.from_numpy(np.ascontiguousarray(np.array(f["ypair_long"])[:, :self.P])).to(dev)
        self.yps = torch.from_numpy(np.ascontiguousarray(np.array(f["ypair_scale"])[:, :self.P])).to(dev)
        self.rows_np = np.asarray(self.rows)
        self.n = len(self.rows)
'''),
('''        idx = torch.as_tensor(idx, dtype=torch.long, device=self.dev)
        B, K, P = len(idx), self.K, self.P''','''        idx_cpu = np.asarray(idx.cpu() if torch.is_tensor(idx) else idx, dtype=np.int64)       # a list (the loader) costs no device sync
        idx = torch.as_tensor(idx_cpu, device=self.dev)
        B, K, P = len(idx), self.K, self.P'''),
('''        yp, ypl, yps = self.ypair[idx][:, :P], self.ypl[idx][:, :P], self.yps[idx][:, :P]''','''        yp, ypl, yps = self.ypair[idx].long(), self.ypl[idx], self.yps[idx]'''),
('''out["gt_out"], out["token"], out["row"] = self.gt_out[idx], [self.tokens[int(k)] for k in idx.tolist()], torch.as_tensor(np.asarray(self.rows)[idx.cpu().numpy()], device=self.dev)''','''out["gt_out"], out["token"], out["row"] = self.gt_out[idx], [self.tokens[int(k)] for k in idx_cpu], torch.as_tensor(self.rows_np[idx_cpu], device=self.dev)'''),
('''The device holds the TRAIN fields only (about 1.3 GB for the 8414 frames of K = 512)''','''The device holds the TRAIN fields only (about 0.9 GB for the 8414 frames of K = 512: the pair lists are cut to the columns in use)'''),
]
for a, b in reps:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
open(p, "w", encoding="utf-8").write(t)
p = "training/data/lane_ptok_data.py"
t = open(p, encoding="utf-8").read()
a = '''    sig = os.path.basename(LT.LD.lane_store()[1])
    pat = os.path.join(cache, f"b{ident}'''
assert t.count(a) == 1
t = t.replace(a, '''    sig = os.path.basename(LT.LD.lane_store()[1])
    cache = cache if os.path.isabs(cache) else os.path.join(ROOT, cache)                  # a cache root given relative to the problem folder (a second K lives in its own root)
    pat = os.path.join(cache, f"b{ident}''')
open(p, "w", encoding="utf-8").write(t)
print("ok")
