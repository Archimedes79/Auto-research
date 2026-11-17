p = "evaluation/jdup.py"
s = open(p, encoding="utf-8").read()
i = s.index("def _load(a):"); j = s.index("def selftest():")
new = '''# the read-outs that are attributed: name, which run's dump, attached ends, snap-nogeom wiring (all snapped, connectors at p >= 0.1, B's delivered graph with the hard step)
VARIANTS = (("c", "c_run", False, False), ("u", "c_run", False, True), ("k", "a_run", False, False), ("kn", "a_run", False, True), ("a", "a_run", True, True), ("l", "l_run", False, False))
NAMES = dict(c="c  = E312 snapped (keypoint ends, snap geometry)", u="u' = E312 snap-nogeom (the floor, same curves as c)", k="k  = E320 own keypoints, snapped", kn="kn = E320 own keypoints, snap-nogeom",
             a="a  = E320 PRIMARY (attached ends, snap-nogeom)", l="l  = the third run, snapped (--l-run, e.g. E316)")


def _load(a):
    import qgraph_eval as QE
    import qgraph_readout as QR
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{a.seed}.pkl"), "rb"))
    runs = dict(c_run=a.c_run or f"e312_p_s{a.seed}", a_run=a.a_run or f"e320_a_s{a.seed}", l_run=a.l_run)
    ends = {r: {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for r in set(v for v in runs.values() if v)}
    return QE, QR, dump, ends, runs


def score(a):
    import jwiring as JW
    from qends_eval import COVER_POS, add_connectors, make_pairs
    QE, QR, dump, ends, runs = _load(a)
    gt = pickle.load(open(JW.GT_PATH, "rb"))
    frames, spec = QE.val_frames(), QR.parse("close:1.5@0.5")
    use = [(n, runs[r], att, ng) for n, r, att, ng in VARIANTS if runs.get(r)]
    keys = ("points", "dup", "CC", "CB", "BC", "BB", "conn_points", "arc_4_8", "arc_8_16", "arc_16p", "partner_section", "partner_link", "conn_dup_with_c_partner")
    bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]
    tot = {n: dict({k: 0 for k in keys}, run=run, curves=0, truth={tr: dict(curves=0, points=0, dup=0) for tr in ("true", "false")}, bins={str(b): dict(curves=0, points=0, dup=0, true=0) for b in bins}) for n, run, _, _ in use}
    t0 = time.time()
    for b, (f, d) in enumerate(zip(frames, dump)):
        if a.max and b >= a.max:
            break
        pr = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                           links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
        graph, bk = QE.readout(pr, 0.5, 0.5)
        lm = np.asarray(pr["links"]["long"], bool)
        g = gt[f.row]
        gtp = {(int(x), int(y)) for x, y in g["pairs"] if x >= 0}
        for v, run, att, ng in use:
            e = ends[run][f.row]
            k = e["kp_pos"].shape[0] // 2
            pm = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
            ia, ib = np.nonzero(pm >= 0.1)
            valid = np.asarray(e["valid"], bool)
            m_in = JW.match(e["kp_pos"][:k], e["kp_dir"][:k], valid[0], g["ent"], g["ent_m"])
            m_out = JW.match(e["kp_pos"][k:], e["kp_dir"][k:], valid[1], g["ext"], g["ext_m"])
            pairs = make_pairs(e, ia, ib, k, attach=att)
            raw, n_c, _ = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], lm, pairs, snap=True, snap_nogeom=ng)
            kept = [kd for i, kd in enumerate(bk["kind"]) if not (kd == "link" and bool(lm[bk["piece_links"][i][0]]) and any(np.linalg.norm(np.asarray(graph.lines[i], float)[0] - q[0]) <= COVER_POS
                                                                                                                             and np.linalg.norm(np.asarray(graph.lines[i], float)[-1] - q[2]) <= COVER_POS for q in pairs))]
            assert len(kept) == len(raw["pts"]) - n_c, (len(kept), len(raw["pts"]), n_c)
            t, (dups, pts) = attribute(raw, n_c, kept)
            T = tot[v]
            for kk in keys:
                T[kk] += t[kk]
            T["curves"] += n_c
            for n_, (i, j) in enumerate(zip(ia, ib)):
                tr = "true" if m_in[i] >= 0 and m_out[j] >= 0 and (int(m_in[i]), int(m_out[j])) in gtp else "false"
                T["truth"][tr]["curves"] += 1
                T["truth"][tr]["points"] += int(pts[n_])
                T["truth"][tr]["dup"] += int(dups[n_])
                for bn in bins:
                    if bn[0] <= float(pm[i, j]) < bn[1]:
                        bb = T["bins"][str(bn)]
                        bb["curves"] += 1
                        bb["points"] += int(pts[n_])
                        bb["dup"] += int(dups[n_])
                        bb["true"] += int(tr == "true")
        f._lane = f._ref = None
        if (b + 1) % 200 == 0:
            print(f"  {b + 1}/{len(frames)} {time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.join(ROOT, "temp", "jdup"), exist_ok=True)
    json.dump(tot, open(os.path.join(ROOT, "temp", "jdup", f"s{a.seed}{'_max' + str(a.max) if a.max else ''}.json"), "w"), indent=1)
    for v, run, _, _ in use:
        T = tot[v]
        print(f"{NAMES[v]} [{run}]: points {T['points']} dup {T['dup']} ({T['dup'] / max(T['points'], 1):.4f}); CC (no B partner) {T['CC']} CB {T['CB']} (of which {T['conn_dup_with_c_partner']} connector duplicate points also have a connector partner) BC {T['BC']} BB {T['BB']}; "
              f"curves {T['curves']} (points {T['conn_points']}); CB by arc 4-8 / 8-16 / 16+: {T['arc_4_8']} / {T['arc_8_16']} / {T['arc_16p']}, partner section / link {T['partner_section']} / {T['partner_link']}")
        print(f"   curves true / false {T['truth']['true']['curves']} / {T['truth']['false']['curves']}; duplicate points of connectors true / false {T['truth']['true']['dup']} / {T['truth']['false']['dup']}; per curve {T['truth']['true']['dup'] / max(T['truth']['true']['curves'], 1):.1f} / {T['truth']['false']['dup'] / max(T['truth']['false']['curves'], 1):.1f}")
        for bn, bb in T["bins"].items():
            print(f"   p in {bn}: {bb['curves']} curves ({100 * bb['true'] / max(bb['curves'], 1):.0f} % true), duplicate share {100 * bb['dup'] / max(bb['points'], 1):.1f} %")


'''
s = s[:i] + new + s[j:]
s = s.replace('s.add_argument("--a-run", default=None)\n', 's.add_argument("--a-run", default=None)\n    s.add_argument("--l-run", default=None, help="a third run (e.g. e316_p_s0), read out snapped like c")\n', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
