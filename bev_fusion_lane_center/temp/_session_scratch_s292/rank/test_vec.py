import os, sys, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import vec
JG = vec.JG
ROOT = os.environ["BFLC_ROOT"]
tot = bad = 0
t_old = t_new = 0.0
nvals = []
for run in ("e312_p_s0", "e312_p_s1"):
    fr = pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))
    if run == "e312_p_s0":
        f = fr[0]
        print({k: (np.asarray(v).shape, np.asarray(v).dtype) for k, v in f.items() if k in ("kp_pos", "kp_dir", "valid")})
    for f in fr:
        pos, D, val = np.asarray(f["kp_pos"], np.float32), np.asarray(f["kp_dir"], np.float32), np.asarray(f["valid"])
        K = pos.shape[0] // 2
        val = val.reshape(2, K)
        for kind in (0, 1):
            ids = np.where(val[kind])[0]
            if len(ids) < 2:
                continue
            E = np.concatenate([pos[kind * K + ids], D[kind * K + ids]], 1)
            nvals.append(len(ids))
            t0 = time.perf_counter(); g0 = JG._groups(E); t1 = time.perf_counter(); g1 = vec.groups_vec(E); t2 = time.perf_counter()
            t_old += t1 - t0; t_new += t2 - t1
            tot += 1
            if not np.array_equal(g0, g1):
                bad += 1
                if bad <= 3:
                    print("MISMATCH", run, f["token"], kind, g0, g1)
print(f"real frames: {tot} (kind, frame) sets, mismatches {bad}; valid keypoints per set mean {np.mean(nvals):.1f}, max {max(nvals)}; old {t_old / tot * 1e3:.3f} ms, new {t_new / tot * 1e3:.3f} ms per set")
# random sets, clustered like lanes (many near-threshold cases)
rng = np.random.default_rng(1)
tot = bad = 0
for it in range(20000):
    n = int(rng.integers(2, 33))
    base = rng.uniform(-30, 30, (n, 2)).astype(np.float32)
    ang = rng.normal(0, 0.3, n) + rng.integers(0, 4, n) * np.pi / 2
    k = rng.integers(1, 5)
    base = np.resize(np.repeat(base[: max(1, n // k)], k, 0), (n, 2)) + rng.normal(0, 2.5, (n, 2)).astype(np.float32)
    D = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    E = np.concatenate([base.astype(np.float32), D], 1)
    g0, g1 = JG._groups(E), vec.groups_vec(E)
    tot += 1
    if not np.array_equal(g0, g1):
        bad += 1
        if bad <= 3:
            print("MISMATCH random", it, n)
print(f"random sets: {tot}, mismatches {bad}")
