import io
p = "temp/e363/verdict_e363.py"
s = io.open(p, encoding="utf-8").read()


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:90], s.count(a))
    s = s.replace(a, b)


rep('PRED = {0: dict(topo_F1=(0.016, 0.008), dup_frac=(0.015, 0.006), junction_F1=(0.085, 0.030), lanes_found=(-0.002, 0.010)),\n        1: dict(topo_F1=(0.024, 0.008), dup_frac=(0.010, 0.006), junction_F1=(0.100, 0.030), lanes_found=(-0.004, 0.010))}',
    'PRED = {0: dict(topo_F1=(0.016, 0.008), dup_frac=(0.018, 0.006), junction_F1=(0.085, 0.030), lanes_found=(-0.002, 0.010)),\n        1: dict(topo_F1=(0.024, 0.008), dup_frac=(0.011, 0.006), junction_F1=(0.100, 0.030), lanes_found=(-0.004, 0.010))}\nWITNESS = {0: 0.0162, 1: 0.0239}                                   # E360: ak - ck topo F1 (the own-keypoint witness, independent of the node head)\nSYM_TOL = 0.01\nSTACK_WINDOW = (0.0005, 0.0035)')
rep("def judge(seed, cards, base_cards, boots, fixture_ok):", "def judge(seed, cards, base_cards, boots, surgery_ok, nodes_ok):")
rep('''    if fixture_ok is not True and fixture_ok != "n/a":
        stop.append("(iii) the fixture (dump with B's own node head against the stored dump) is missing or failed")''',
    '''    if surgery_ok is not True and surgery_ok != "n/a":
        stop.append("(ii'/iii/iv) the surgery check (arrays of the dumps, the fixture on both seeds, the node tensors, the provenance) is missing or failed")
    if nodes_ok is not True and nodes_ok != "n/a":
        stop.append("(vi) the node check (the pair model's node outputs against QGraph.node_stage of B') is missing or failed for an arm of this seed")''')
rep('''    out.append(f"  attached - keypoint share within 1 m (REPORTED, not a rule here): c entry {sc[0]:+.4f} exit {sc[1]:+.4f} | a entry {sa[0]:+.4f} exit {sa[1]:+.4f}")''',
    '''    out.append(f"  attached - keypoint share within 1 m (STOP RULE vii, tolerance {SYM_TOL}): c entry {sc[0]:+.4f} exit {sc[1]:+.4f} | a entry {sa[0]:+.4f} exit {sa[1]:+.4f}")
    for i_, side in enumerate(("entry", "exit")):
        if abs(sa[i_] - sc[i_]) > SYM_TOL:
            stop.append(f"(vii) symmetry {side}: a {sa[i_]:+.4f} vs c {sc[i_]:+.4f} (> {SYM_TOL})")''')
rep('''    rec = (na["topo_F1"] - nc["topo_F1"]) / (E338A[seed] - nc["topo_F1"]) if E338A[seed] != nc["topo_F1"] else float("nan")
    out.append(f"  recovery share (a - c) / (E338a - c) = {rec:+.2f}   (E338a {E338A[seed]:.4f}: c2 WITH the hard step, attached ends from B's nodes)")''',
    '''    ct = cards.get("t")
    if ct is not None:
        nt = ct["numbers"]["head"]
        rec = (na["topo_F1"] - nc["topo_F1"]) / (nt["topo_F1"] - nc["topo_F1"]) if nt["topo_F1"] != nc["topo_F1"] else float("nan")
        out.append(f"  rule-version arm t (c2 WITH the hard step, faithful): {nt['topo_F1']:.4f}; recovery share (a - c) / (t - c) = {rec:+.2f}; a - t {na['topo_F1'] - nt['topo_F1']:+.4f}   (stale E338a {E338A[seed]:.4f}: history only)")
    else:
        out.append("  rule-version arm t: not scored")''')
rep('''    keep = all(guards.values()) and b is not None and b[1] > 0
    return "KEEP" if keep else "TIE", out''',
    '''    witness_gap = d["topo_F1"] - WITNESS[seed]
    out.append(f"  KEEP condition (b): (a - c) {d['topo_F1']:+.4f} against E360's own-keypoint witness {WITNESS[seed]:+.4f}: difference {witness_gap:+.4f} ({'within' if abs(witness_gap) <= 0.01 else 'OUTSIDE'} 0.01)")
    keep = all(guards.values()) and b is not None and b[1] > 0 and abs(witness_gap) <= 0.01
    tok = "KEEP" if keep else "TIE"
    if abs(witness_gap) > 0.01:
        tok += " (attachment-dominated: the gain differs from the own-keypoint witness by more than 0.01)"
    return tok, out''')
rep('''        out.append(f"  STACK aw (layer + head) {nw['topo_F1']:.4f}: aw - a {ds['topo_F1']:+.4f} (pred {STACK_PRED[seed]:+.4f} +-0.003: {'HIT' if abs(ds['topo_F1'] - STACK_PRED[seed]) <= 0.003 else 'miss'}), guards against a: {'all held' if all(gs.values()) else 'BROKEN ' + str([k for k, v in gs.items() if not v])}, "''',
    '''        win = STACK_WINDOW[0] <= ds["topo_F1"] <= STACK_WINDOW[1] and bs is not None and bs[1] > 0
        out.append(f"  STACK aw (layer + head) {nw['topo_F1']:.4f}: aw - a {ds['topo_F1']:+.4f} (pred {STACK_PRED[seed]:+.4f}; HIT needs the window [{STACK_WINDOW[0]:+.4f}, {STACK_WINDOW[1]:+.4f}] and a bootstrap lower bound above 0: {'HIT' if win else 'miss'}), guards against a: {'all held' if all(gs.values()) else 'BROKEN ' + str([k for k, v in gs.items() if not v])}, "''')
rep('''    fx = f"{ROOT}/temp/e363/fixture_result.json"
    fixture_ok = load(fx).get("pass") if os.path.exists(fx) else False''', '''    sx = f"{ROOT}/temp/e363/surgery_check.json"
    surgery_ok = bool(load(sx).get("pass")) if os.path.exists(sx) else False''')
rep('''        cards = dict(c=card("c2", "e363c"), a=card("a", "e363a"), cw=card("c2", "e363cw"), aw=card("a", "e363aw"))''', '''        cards = dict(c=card("c2", "e363c"), a=card("a", "e363a"), cw=card("c2", "e363cw"), aw=card("a", "e363aw"), t=card("c2", "e363t"))
        nn = []
        for g in ("a", "c2"):
            npth = f"{ROOT}/temp/e363/nodecheck_{g}_s{s}.json"
            nn.append(os.path.exists(npth) and bool(load(npth).get("passed")))
        nodes_ok = all(nn)''')
rep("tok, lines = judge(s, cards, base_cards, boots, fixture_ok)", "tok, lines = judge(s, cards, base_cards, boots, surgery_ok, nodes_ok)")
# the selftest: adapt calls (the planted calls pass 'n/a' for both)
head, tail = s.split("def selftest():")
tail = tail.replace('"n/a")', '"n/a", "n/a")')
tail = tail.replace("for fx in (False, None):\n        tok, lines = judge(0, good, None, {\"a\": b_pos}, fx)\n        assert tok == \"INVALID\" and \"(iii)\" in lines[-1], (fx, tok)\n    tok, _ = judge(0, good, None, {\"a\": b_pos}, True)\n    assert tok == \"KEEP\"",
                    "for fx in (False, None):\n        tok, lines = judge(0, good, None, {\"a\": b_pos}, fx, \"n/a\")\n        assert tok == \"INVALID\" and \"(ii'/iii/iv)\" in lines[-1], (fx, tok)\n        tok, lines = judge(0, good, None, {\"a\": b_pos}, \"n/a\", fx)\n        assert tok == \"INVALID\" and \"(vi)\" in lines[-1], (fx, tok)\n    tok, _ = judge(0, good, None, {\"a\": b_pos}, True, True)\n    assert tok == \"KEEP\"")
s = head + "def selftest():" + tail
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
