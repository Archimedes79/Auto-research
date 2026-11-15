import re
p = "temp/e299/heat_replay.py"
s = open(p, encoding="utf-8").read()
a = s.index("def pair_stats(")
b = s.index("def selftest():")
new = '''def pair_stats(maps0, maps1, f0, f1, g1, ws=WS, rng=None):
    """maps = (p (2, N, N), dirv (2, 2, N, N)) of t-1 and t; -> {arm: counts} over both kinds. Arms: 'w<w>' the noisy-OR with the warped previous heat; 'N1' the heat of t with ITSELF (no time information);
    'N2' the previous heat NOT warped. Also the heat-visible ceiling (section 5b amendment): of the GT ends the baseline misses, those with a warped-previous-heat value >= 0.05 within 4 m, against random places."""
    p0, _ = maps0
    p1, d1 = maps1
    pw = warp_heat(p0, f0, f1)
    pos_b, dir_b, val_b = extract(p1, d1)
    k = len(pos_b) // 2
    arms = {f"w{w}": combine(p1, pw, w) for w in ws}
    arms["N1"] = combine(p1, p1, 1.0)
    arms["N2"] = combine(p1, p0, 1.0)
    res = {}
    for name, pc in arms.items():
        pos_c, dir_c, val_c = extract(pc, d1)
        r = dict(gt=0, base_tp=0, comb_tp=0, gained=0, lost=0, base_delivered=int(val_b.sum()), comb_delivered=int(val_c.sum()), new=0, true_new=0, false_new=0, false_new_near=0, false_new_isolated=0,
                 vanished=0, vanished_matched=0)
        for kind, (G, m) in enumerate(((g1["ent"], g1["ent_m"]), (g1["ext"], g1["ext_m"]))):
            sl = slice(kind * k, (kind + 1) * k)
            mb = JW.match(pos_b[sl], dir_b[sl], val_b[sl], G, m)
            mc = JW.match(pos_c[sl], dir_c[sl], val_c[sl], G, m)
            sb, sc = _sets(mb), _sets(mc)
            missed = {int(x) for x in np.nonzero(m)[0]} - sb
            r["gt"] += int(m.sum())
            r["base_tp"] += len(sb)
            r["comb_tp"] += len(sc)
            r["gained"] += len(sc - sb)
            r["lost"] += len(sb - sc)
            Eb = np.concatenate([pos_b[sl], dir_b[sl]], 1)
            Ec = np.concatenate([pos_c[sl], dir_c[sl]], 1)
            cp = JTM.counterpart(Ec, Eb, val_c[sl], val_b[sl])                       # comb as "previous", base as "current": cp < 0 = a keypoint of p_c without a baseline counterpart
            vb_pos = pos_b[sl][val_b[sl]]
            for i in np.nonzero(val_c[sl])[0]:
                if cp[i] < 0:
                    r["new"] += 1
                    if mc[i] >= 0 and int(mc[i]) in missed:
                        r["true_new"] += 1
                    else:
                        r["false_new"] += 1
                        near = len(vb_pos) and float(np.linalg.norm(vb_pos - pos_c[sl][i], axis=1).min()) <= 4.0
                        r["false_new_near" if near else "false_new_isolated"] += 1
            cpb = JTM.counterpart(Eb, Ec, val_b[sl], val_c[sl])                      # baseline keypoints without a counterpart in p_c
            for i in np.nonzero(val_b[sl])[0]:
                if cpb[i] < 0:
                    r["vanished"] += 1
                    r["vanished_matched"] += int(mb[i] >= 0)
        assert r["comb_tp"] - r["base_tp"] == r["gained"] - r["lost"], r
        res[name] = r
    # heat-visible ceiling: baseline-missed GT ends with warped previous heat >= 0.05 within 4 m (any heading) vs the same number of random places in the same frame
    c = JE._centres("cpu").numpy()
    X, Y = np.meshgrid(c, c, indexing="ij")
    ceil = dict(missed=0, visible=0, random_visible=0)
    for kind, (G, m) in enumerate(((g1["ent"], g1["ent_m"]), (g1["ext"], g1["ext_m"]))):
        sl = slice(kind * k, (kind + 1) * k)
        mb = JW.match(pos_b[sl], dir_b[sl], val_b[sl], G, m)
        missed = sorted({int(x) for x in np.nonzero(m)[0]} - _sets(mb))
        h = pw[kind].numpy()

        def vis(xy):
            return bool(h[((X - xy[0]) ** 2 + (Y - xy[1]) ** 2) <= 16.0].max(initial=0.0) >= 0.05)
        for gi in missed:
            ceil["missed"] += 1
            ceil["visible"] += int(vis(G[gi, :2]))
            ceil["random_visible"] += int(vis(rng.uniform(-HALF + 1, HALF - 1, 2)))
    res["ceiling"] = ceil
    return res


'''
s = s[:a] + new + s[b:]
# the selftest call and the loop
s = s.replace("r = pair_stats((p0, d1), (p1, d1), I, I, g1, ws=(1.0, 0.0))", "r = pair_stats((p0, d1), (p1, d1), I, I, g1, ws=(1.0, 0.0), rng=np.random.RandomState(0))")
s = s.replace('assert r[1.0]["base_tp"] == 1 and r[1.0]["comb_tp"] == 2 and r[1.0]["gained"] == 1 and r[1.0]["lost"] == 0 and r[1.0]["true_new"] == 1 and r[1.0]["false_new"] == 0, r[1.0]',
              'assert r["w1.0"]["base_tp"] == 1 and r["w1.0"]["comb_tp"] == 2 and r["w1.0"]["gained"] == 1 and r["w1.0"]["lost"] == 0 and r["w1.0"]["true_new"] == 1 and r["w1.0"]["false_new"] == 0, r["w1.0"]\n    assert r["N2"]["gained"] == 1, r["N2"]                                           # identity pose here: the unwarped heat equals the warped one\n    assert r["ceiling"]["missed"] == 1 and r["ceiling"]["visible"] == 1, r["ceiling"]')
s = s.replace('assert r[0.0]["gained"] == 0 and r[0.0]["new"] == 0, r[0.0]', 'assert r["w0.0"]["gained"] == 0 and r["w0.0"]["new"] == 0, r["w0.0"]\n    assert r["N1"]["gained"] == 0, r["N1"]                                          # 0.05 with itself is 0.0975 < 0.1: the self-OR alone does not find the end')
open(p, "w", encoding="utf-8").write(s)
