p = "evaluation/bev_errors.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
rep('''def main(argv=None):''', '''def check_against_card(run, n, tag="", groups=("vehicle", "pedestrian", "two_wheeler"), thr=2.0):
    """the proof that this table is the metric's: with every detection counted (min-score 0) the true positives, the missed boxes and the false positives per class equal
    the card's n_tp, n_gt - n_tp and n_det - n_tp (dt 0, the scored window). -> list of (class, what, here, card) that differ (empty = identical)"""
    card = os.path.join(run, "eval", f"card{tag}.json")
    if not os.path.exists(card):
        raise SystemExit(f"no {card} to check against")
    with open(card) as f:
        m = json.load(f)["metrics"]
    _, tot, _ = error_rows(run, n, tag, groups, thr, 0.0)
    bad = []
    for c in groups:
        t, k = tot.get(c, {}), f"det/dt+0.00/{c}/all/"
        for what, here, want in (("tp", t.get("tp", 0), m.get(k + "n_tp")), ("fn", t.get("fn", 0), None if m.get(k + "n_gt") is None else m[k + "n_gt"] - m.get(k + "n_tp", 0)),
                                 ("fp", t.get("fp", 0), None if m.get(k + "n_det") is None else m[k + "n_det"] - m.get(k + "n_tp", 0))):
            if want is None or int(want) != here:
                bad.append((c, what, here, want))
    return bad


def main(argv=None):''')
rep('''    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")''', '''    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="also prove the table against the card: with every detection counted, tp / fn / fp per class equal the card's numbers")
    ap.add_argument("--selftest", action="store_true")''')
rep('''    groups = (a.cls,) if a.cls else ("vehicle", "pedestrian", "two_wheeler")
''', '''    groups = (a.cls,) if a.cls else ("vehicle", "pedestrian", "two_wheeler")
    if a.check:
        bad = check_against_card(a.run.rstrip("/"), a.n, a.tag, groups, a.thr)
        print("check against the card (every detection counted): " + ("IDENTICAL for tp / fn / fp of every class" if not bad else f"DIFFERENT: {bad}"))
        if bad:
            return 1
''')
open(p, "w", encoding="utf-8").write(s)
