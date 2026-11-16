p = "training/data/lane_gt_data.py"
s = open(p, encoding="utf-8").read()
s = s.replace("rng=np.random.RandomState((hash(token) + int(np.random.randint(1 << 30))) % (1 << 31)) if self.map_mode == \"train\" else None, token=token)", "rng=np.random.RandomState() if self.map_mode == \"train\" else None, token=token)")
open(p, "w", encoding="utf-8").write(s)
