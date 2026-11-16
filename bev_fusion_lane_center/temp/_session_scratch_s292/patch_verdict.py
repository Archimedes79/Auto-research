import io
p = "temp/e360/verdict_e360.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:60], s.count(a))
    s = s.replace(a, b)
rep("SYM_TOL = 0.02", "SYM_TOL = 0.01")
# (i) the whole base block against the stored graph-level card, ck / ak against c / a
rep('''    stop = []
    for arm, card, ref in (("c", c, BASE_REF[seed]["c"]), ("a", a, BASE_REF[seed]["a"])):                # (i) the base block = the stored graph-level card; (ii) the curves
        b = card["numbers"]["base"]["topo_F1"]
        if abs(b - ref) > 1e-3:
            stop.append(f"(i) {arm}: base topo F1 {b:.4f} != stored {ref:.4f}")
        if card["curves"] != CURVES[seed]:
            stop.append(f"(ii) {arm}: curves {card['curves']} != {CURVES[seed]}")''',
'''    stop = []
    for arm, card, ref in (("c", c, BASE_REF[seed]["c"]), ("a", a, BASE_REF[seed]["a"])):                # (i) the WHOLE base block = the stored graph-level card; (ii) the curves
        b = card["numbers"]["base"]
        if abs(b["topo_F1"] - ref) > 1e-3:
            stop.append(f"(i) {arm}: base topo F1 {b['topo_F1']:.4f} != stored {ref:.4f}")
        if base_cards is not None:
            g = base_cards[arm]
            bad = [k for k in b if k in g and abs(float(b[k]) - float(g[k])) > 1e-3]
            if bad or not set(b) & set(g):
                stop.append(f"(i) {arm}: base block differs from the stored graph-level card in {bad or 'all keys'}")
        if card["curves"] != CURVES[seed]:
            stop.append(f"(ii) {arm}: curves {card['curves']} != {CURVES[seed]}")
    for arm, w in (("ck", "c"), ("ak", "a")):                                                              # the witnesses read the same graph and the same pair run
        wc = cards.get(arm)
        if wc is not None and (wc["numbers"]["base"] != cards[w]["numbers"]["base"] or wc["curves"] != CURVES[seed]):
            stop.append(f"(i/ii) {arm}: base block or curves differ from {w}'s")''')
rep('''    if t_ok is False:
        stop.append("(v) arm t does not reproduce E338a")''', '''    if t_ok is not True and t_ok != "n/a":
        stop.append("(v) arm t missing or does not reproduce E338a")''')
# main: t check incl. the counts file, base cards loaded
rep('''        t_ok = None
        if s == 0 and os.path.exists(f"{R}e324_s_s0/ends_card_e360t_snap_nogeom_attach_thr0.1.json"):
            x, y = load(f"{R}e324_s_s0/ends_card_e360t_snap_nogeom_attach_thr0.1.json"), load(f"{R}e324_s_s0/ends_card_e338a_snap_nogeom_attach_thr0.1.json")
            t_ok = all(x[k] == y[k] for k in ("numbers", "curves", "wired_both_ends", "long_links_dropped", "ends_within_1m_of_a_delivered_piece_end"))''',
'''        t_ok = "n/a"                                                                                         # arm t exists on seed 0 only
        if s == 0:
            pt = f"{R}e324_s_s0/ends_card_e360t_snap_nogeom_attach_thr0.1"
            t_ok = False                                                                                    # a missing t card is a failed stop rule, never a silent pass
            if os.path.exists(pt + ".json") and os.path.exists(pt + ".counts.pkl"):
                x, y = load(pt + ".json"), load(f"{R}e324_s_s0/ends_card_e338a_snap_nogeom_attach_thr0.1.json")
                import pickle
                cx, cy = pickle.load(open(pt + ".counts.pkl", "rb")), pickle.load(open(f"{R}e324_s_s0/ends_card_e338a_snap_nogeom_attach_thr0.1.counts.pkl", "rb"))
                t_ok = all(x[k] == y[k] for k in ("numbers", "curves", "wired_both_ends", "long_links_dropped", "ends_within_1m_of_a_delivered_piece_end")) and repr(cx) == repr(cy)
        base_cards = {arm: load(f"{R}e292/e292_{g}_s{s}.plain_t0.5/card.json")["p"]["graph"] for arm, g in (("c", "c2"), ("a", "a"))}''')
rep("tok, lines = judge(s, cards, None, boots, t_ok)", "tok, lines = judge(s, cards, base_cards, boots, t_ok)")
rep("tok, lines = judge(s, cards, None, boots, t_ok)", "tok, lines = judge(s, cards, base_cards, boots, t_ok)", 0) if False else None
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
