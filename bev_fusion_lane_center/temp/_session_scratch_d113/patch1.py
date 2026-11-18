import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''def label_frame(tpos, tvalid, table, r, ks=KS):
    """the labels of one frame: tokens (K, 2) / (K,) and the GT table of `lane_tok_data` (pos, flags, sstart, src, dst, long, conn, scale) -> dict of the label fields (module docstring), per-frame fields as 0-d / 1-d arrays"""
    K, M = len(tpos), len(table["pos"])
    gt, dist = assign_unique(tpos, tvalid, table["pos"], r)''', '''def label_frame(tpos, tvalid, table, r, K=None, ks=KS):
    """the labels of one frame: tokens (T, 2) / (T,) in rank order and the GT table of `lane_tok_data` (pos, flags, sstart, src, dst, long, conn, scale) -> dict of the label fields (module docstring) for the first K tokens (default all);
    the ceiling statistics `match_at` / `lost_at` are made for the first ks[j] of ALL the tokens given (so T may be larger than K)"""
    K, M = (len(tpos) if K is None else K), len(table["pos"])
    tpos_all, tvalid_all = tpos, tvalid
    tpos, tvalid = tpos[:K], tvalid[:K]
    gt, dist = assign_unique(tpos, tvalid, table["pos"], r)''')
rep('''        g2, _ = assign_unique(tpos[:k], tvalid[:k], table["pos"], r)
        _, _, ok2 = links(np.concatenate([g2, -np.ones(K - len(g2), int)]) if len(g2) < K else g2)''', '''        g2, _ = assign_unique(tpos_all[:k], tvalid_all[:k], table["pos"], r)
        _, _, ok2 = links(g2)''')
rep('''    assert t["cell"][0, 0] == 3 * 50 + 3 and t["cell"][0, 1] == 5, "tie order"
    assert t["n_e50"].tolist() == [int((e[b] >= 0).sum()) for b in range(B)] or True''', '''    assert t["cell"][0, 0] == 5 and t["cell"][0, 1] == 3 * 50 + 3, "tie order: the lower flat cell first"
    assert t["n_e50"].tolist() == [int((e[b].sigmoid() >= 0.5).sum()) for b in range(B)]''')
rep('''lab = label_frame(tpos, np.array([True, True, True, False]), table, 1.5, ks=(1, 2, 4))''', '''lab = label_frame(tpos, np.array([True, True, True, False]), table, 1.5, ks=(1, 2, 4))''')
# build: tokens up to Kfull, store K
rep('''            tk = select_tokens(no, P["feat2"] if a.f2 else None, a.K)
            tk512 = select_tokens({k: v for k, v in no.items() if k in ("e", "pos")} | {k: no[k][:, :1] for k in ("off", "dir_raw", "flags", "s", "uout", "uin", "tok")}, None, max(KS)) if a.K < max(KS) else tk
        tk = {k: v.cpu().numpy() for k, v in tk.items()}
        pos_all, valid_all = tk512["pos"].cpu().numpy() if a.K < max(KS) else tk["pos"], tk512["valid"].cpu().numpy() if a.K < max(KS) else tk["valid"]''', '''            tk = select_tokens(no, P["feat2"] if a.f2 else None, max(a.K, max(KS)))                              # all the tokens the ceiling statistics need, K of them are stored
        tk = {k: v.cpu().numpy() for k, v in tk.items()}
        pos_all, valid_all = tk["pos"], tk["valid"]''')
rep('''            lab = label_frame(pos_all[j], valid_all[j], b["table"][j], a.r)
            assert len(lab["gt_id"]) >= a.K or True
            for n_, v in lab.items():
                arrays[n_][i + j] = v[:a.K] if n_ in ("gt_id", "gt_dist", "gt_type", "gt_flags", "gt_aux") else v
            for n_, v in tk.items():
                arrays[n_][i + j] = v[j]''', '''            lab = label_frame(pos_all[j], valid_all[j], b["table"][j], a.r, K=a.K)
            for n_, v in lab.items():
                arrays[n_][i + j] = v
            for n_, v in tk.items():
                arrays[n_][i + j] = v[j] if n_ in ("n_e50", "n_e20") else v[j, :a.K]''')
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
