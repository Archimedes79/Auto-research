import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_host.py")
s = open(p, encoding="utf-8").read()
old = '''        args, wins = [], []
        for k in range(n):
            x0, x1, y0, y1 = max(0, ix[k] - r[k]), min(X, ix[k] + r[k] + 1), max(0, iy[k] - r[k]), min(Y, iy[k] + r[k] + 1)
            gx = np.arange(x0, x1, dtype=np.float32)[:, None]
            gy = np.arange(y0, y1, dtype=np.float32)[None, :]
            args.append((-((gx - ix[k]) ** 2 + (gy - iy[k]) ** 2) / np.float32(2 * sigma[k] * sigma[k])).reshape(-1))
            wins.append((x0, x1, y0, y1))'''
assert s.count(old) == 1
new = '''        args, wins = [], []
        for cx_, cy_, r_, s_ in zip(ix.tolist(), iy.tolist(), r.tolist(), sigma.tolist()):      # python numbers: float32 stays float32 (numpy 2 would promote an int64 scalar)
            x0, x1, y0, y1 = max(0, cx_ - r_), min(X, cx_ + r_ + 1), max(0, cy_ - r_), min(Y, cy_ + r_ + 1)
            gx = np.arange(x0, x1, dtype=np.float32)[:, None]
            gy = np.arange(y0, y1, dtype=np.float32)[None, :]
            args.append((-((gx - cx_) ** 2 + (gy - cy_) ** 2) / np.float32(2 * s_ * s_)).reshape(-1))
            wins.append((x0, x1, y0, y1))'''
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
