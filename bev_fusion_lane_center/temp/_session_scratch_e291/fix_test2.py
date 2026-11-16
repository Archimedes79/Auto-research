p = "temp/e291/test_e291_staged.py"
s = open(p, encoding="utf-8").read()
a = s.index("def t_features_real():")
b = s.index("def fake_main(")
new = '''def ref_features(pos, D, kind, ok, tb):
    """the six features of the keypoints of ONE frame by plain loops (numpy): the reference of `jprior.features` (n = count of the keypoint's direction, flag, axis distance, heading agreement, rank, found - n) and the arm index"""
    N, A = len(pos), len(tb["n_in"])
    ref = [[tb["t"][a] if kind[i] == 0 else -tb["t"][a] for a in range(A)] for i in range(N)]
    rel = [[pos[i] - tb["c"][a] for a in range(A)] for i in range(N)]
    lat = np.array([[rel[i][a] @ np.array([-ref[i][a][1], ref[i][a][0]]) for a in range(A)] for i in range(N)]).reshape(N, A)
    cos = np.array([[D[i] @ ref[i][a] for a in range(A)] for i in range(N)]).reshape(N, A)
    elig = np.array([[np.linalg.norm(rel[i][a]) <= JP.RADIUS and cos[i][a] >= JP.COS_MIN for a in range(A)] for i in range(N)]).reshape(N, A)
    f, arm = np.zeros((N, JP.NF)), -np.ones(N, int)
    for i in range(N):
        cand = [a for a in range(A) if elig[i][a]]
        if not cand:
            f[i, 1] = 1.0
            continue
        a = min(cand, key=lambda x: (abs(lat[i][x]), x))
        arm[i] = a
        n = int((tb["n_in"] if kind[i] == 0 else tb["n_out"])[a])
        f[i, 2], f[i, 3] = abs(lat[i][a]) / 20.0, cos[i][a]
        if n == JP.UNK:
            f[i, 1] = 1.0
            continue
        mem = [k for k in range(N) if ok[k] and kind[k] == kind[i] and elig[k][a]]
        # the lateral offset of k to the axis of arm a: the signed one of THAT arm (heading of the kind)
        rank = sum(1 for k in mem if lat[k][a] > lat[i][a] or (lat[k][a] == lat[i][a] and k < i))
        f[i, 0], f[i, 4], f[i, 5] = n / 4.0, rank / max(n, 1), (len(mem) - n) / 4.0
    return f, arm, elig


def t_features_real():
    """`jprior.features` against plain loops (numpy) on the real stored E312 dumps and the real clean tables, and the arm-side shares of keypoints fewer / equal / more than the count against the independent loop of the step 0 (temp/e291/step0.py, loose setting: peaks >= 0.1, 10 m, 60 degrees: the same quantity)"""
    import json
    import pickle
    sys.path.insert(0, os.path.join(ROOT, "tools", "viz"))
    import qgraph_view as QV
    st0 = json.load(open(os.path.join(ROOT, "temp", "e291", "step0_thr0.1_r10.json")))["stats"]["e312_p_s0"]
    frames = {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))}
    cnt = {k: 0 for k in ("in_def", "in_eq", "in_sur", "out_def", "out_eq", "out_sur")}
    n_frames = n_kp = n_seen = 0
    for sid, (name, toks) in QV.val_scenes().items():
        for t in toks:
            jg = QJ.JG.junction_gt(QV.load_gt(t))
            if jg is None or t not in frames:
                continue
            f = frames[t]
            tb = JP.table(jg, t, JP.Spec("clean"))
            kind = np.r_[np.zeros(32, int), np.ones(32, int)]
            ok = f["valid"].reshape(-1)
            fe, arm = JP.features(torch.from_numpy(f["kp_pos"])[None].float(), torch.from_numpy(f["kp_dir"])[None].float(), torch.from_numpy(kind)[None], torch.from_numpy(ok)[None], JP.collate([tb]))
            fr, ar, elig = ref_features(f["kp_pos"].astype(np.float64), f["kp_dir"].astype(np.float64), kind, ok, tb)
            assert np.array_equal(arm[0].numpy(), ar) and np.abs(fe[0].numpy() - fr).max() < 1e-4, (t, np.abs(fe[0].numpy() - fr).max())
            n_frames += 1
            n_kp += int(ok.sum())
            n_seen += int(((arm[0].numpy() >= 0) & ok).sum())
            for a_i in range(len(tb["n_in"])):
                for k, nn_ in ((0, tb["n_in"][a_i]), (1, tb["n_out"][a_i])):
                    if nn_ <= 0:
                        continue
                    found = int((elig[:, a_i] & ok & (kind == k)).sum())
                    cnt[("in_" if k == 0 else "out_") + ("def" if found < nn_ else ("eq" if found == nn_ else "sur"))] += 1
    for d in ("in", "out"):
        tot = sum(cnt[f"{d}_{x}"] for x in ("def", "eq", "sur"))
        ref = {x: st0[f"{x}_{d}"] / st0[f"arms_with_{d}"] for x in ("def", "eq", "sur")}
        got = {x: cnt[f"{d}_{x}"] / tot for x in ("def", "eq", "sur")}
        print(f"  {d}: deficit / exact / surplus  here {got['def']:.3f} {got['eq']:.3f} {got['sur']:.3f}   step 0 {ref['def']:.3f} {ref['eq']:.3f} {ref['sur']:.3f}")
        assert all(abs(got[x] - ref[x]) < 0.004 for x in got), (d, got, ref)
    print(f"features_real ok: {n_frames} frames with junction GT, {n_kp} real keypoints, {n_seen} of them in reach of an arm ({n_seen / n_kp:.3f})")


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
