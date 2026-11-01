import sys
dc, ex, cd, sc, st, ev, sm = sys.argv[1:8]


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)


patch(dc, [
('''def card_lower(metric, cards=()):
    """the direction the cards themselves declare (their evaluator's "lower_is_better" list, card contract): True / False, or None when no
    card declares one. Cards that disagree are refused (two evaluators with other conventions answer different questions)"""
    said = {metric in c["lower_is_better"] for _, c, _ in cards if isinstance(c, dict) and isinstance(c.get("lower_is_better"), list)}
    if len(said) > 1:
        refs = ", ".join(str(r) for _, _, r in cards)
        raise ValueError(f"the cards disagree whether {metric} is lower-is-better ({refs})")
    return said.pop() if said else None''',
'''def card_lower(metric, cards=(), root=None):
    """the direction the cards declare for a metric: True (lower is better) / False, or None when no card declares it. cards: (metrics,
    card, ref) tuples; only cards that HOLD the metric count. A card declares through its "better" map ({metric: "lower" | "higher"},
    written by its evaluator for every metric it writes, card contract); a card that holds the metric without declaring it (written
    before the field, or a metric a script added) is undeclared and is read by the fallback lower_names. Refused (ValueError): declaring
    cards that disagree, and a declaration that contradicts how the undeclared cards of the same comparison would be read"""
    said, undeclared = {}, []
    for m, c, ref in cards:
        if not isinstance(m, dict) or metric not in m:
            continue
        b = c.get("better") if isinstance(c, dict) else None
        if isinstance(b, dict) and b.get(metric) in ("lower", "higher"):
            said.setdefault(b[metric] == "lower", []).append(str(ref))
        else:
            undeclared.append(str(ref))
    if len(said) > 1:
        raise ValueError(f"the cards disagree whether {metric} is lower-is-better (lower: {', '.join(said[True])}; higher: {', '.join(said[False])})")
    if not said:
        return None
    low = next(iter(said))
    if undeclared and low != (metric in lower_names(root)):
        raise ValueError(f"{metric}: {', '.join(said[low])} declare it {'lower' if low else 'higher'}-is-better, but the older cards "
                         f"{', '.join(undeclared)} would be read the other way (fallback: project.json lower_is_better); fix the fallback")
    return low'''),
('''    declared = card_lower(metric, cards)''', '''    declared = card_lower(metric, cards, root)'''),
('''    headline metric, else the cards' own "lower_is_better" list (the evaluator declares it; cards that disagree are refused), else for older
    cards the generic cost names (LOWER) plus the problem's project.json "lower_is_better" list''',
 '''    headline metric, else the cards' own "better" map (the evaluator declares every metric it writes; cards that disagree are refused), else
    for older cards the generic cost names (LOWER) plus the problem's project.json "lower_is_better" list'''),
('''        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path))))
        d = decide_experiment(exp, root)''', '''        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path))))
        try:
            d = decide_experiment(exp, root)
        except ValueError as e:  # e.g. cards that disagree about a metric's direction
            sys.exit(f"{exp.get('id')}: not decided: {e}")'''),
('''LEGACY_LANE_LOWER = {"vert100", "false_len_frac", "dup_frac", "asm_ms", "map_err", "path_err", "ece"}''',
 '''LEGACY_LANE_LOWER = {"vert100", "verts_per_frame", "false_len_frac", "dup_frac", "asm_ms", "map_err", "path_err", "ece"}'''),
])

patch(ex, [
('''from decide import lower_names  # noqa: E402''', '''from decide import card_lower, lower_names  # noqa: E402'''),
('''    declared = set()  # what the arms' cards say about the headline's direction (their evaluator's lower_is_better list)
    for a in arms:''', '''    every = []  # the cards of all arms: the directions they declare must agree (decide.card_lower)
    for a in arms:'''),
('''            for m, full, ref in lst:
                if isinstance(full, dict) and isinstance(full.get("lower_is_better"), list):
                    declared.add(exp["metric"] in full["lower_is_better"])
''', '''            for m, full, ref in lst:
                every.append((m, full, ref))
'''),
('''    if len(declared) > 1:
        bad.append(f"the cards disagree whether {exp['metric']} is lower-is-better")
    elif declared and declared.pop() != (exp["direction"] == "lower"):
        bad.append(f"direction {exp['direction']} contradicts the cards: their evaluator declares {exp['metric']} "
                   f"{'lower' if exp['direction'] != 'lower' else 'higher'}-is-better")
    elif not declared and exp["metric"] in lower_names(root) and exp["direction"] != "lower":
        bad.append(f"metric {exp['metric']} is lower-is-better (project.json lower_is_better) but direction is {exp['direction']}")''',
'''    for metric in dict.fromkeys([exp["metric"]] + [c["metric"] for c in exp["rule"] if isinstance(c, dict) and c.get("metric")]):
        try:
            said = card_lower(metric, every, root)
        except ValueError as e:
            bad.append(str(e))
            continue
        if metric != exp["metric"]:
            continue
        if said is not None and said != (exp["direction"] == "lower"):
            bad.append(f"direction {exp['direction']} contradicts the cards: their evaluator declares {metric} {'lower' if said else 'higher'}-is-better")
        elif said is None and metric in lower_names(root) and exp["direction"] != "lower":
            bad.append(f"metric {metric} is lower-is-better (project.json lower_is_better) but direction is {exp['direction']}")'''),
])

patch(cd, [
('''    lib = card.get("lower_is_better")
    if lib is not None and not all(isinstance(x, str) and x for x in lib):
        bad.append("lower_is_better must be a list of metric names")''', '''    for where, blk in (("", card), ("compare.", card.get("compare") if isinstance(card.get("compare"), dict) else {})):
        bt = blk.get("better")
        if bt is None:
            continue
        if not isinstance(bt, dict) or any(v not in ("lower", "higher") for v in bt.values()):
            bad.append(f"{where}better must map metric names to 'lower' or 'higher'")
        elif any(k not in blk.get("metrics", {}) for k in bt):
            bad.append(f"{where}better names a metric the card does not hold: " + ", ".join(k for k in bt if k not in blk.get("metrics", {}))[:120])'''),
])

patch(sc, [
('"settings": "dict", "lower_is_better": "list"}', '"settings": "dict", "better": "dict"}'),
('''lower-is-better metrics: see lower_is_better",
    "lower_is_better": "the metric names (of metrics and compare.metrics) where a smaller number is better, declared by the evaluator that defines them; every other metric of the card is higher-is-better. decide.py takes a metric's direction from here (the experiment's direction wins for its headline metric; cards that disagree are refused); for cards written without this field it falls back to project.json lower_is_better",''',
 '''the direction of each is in better",
    "better": "{metric: 'lower' | 'higher'} for the metrics of this card (the compare block carries its own map for compare.metrics), declared by the evaluator that defines them; a metric without an entry (e.g. a count) is undeclared. decide.py takes a metric's direction from here (the experiment's direction wins for its headline metric; cards that disagree are refused); for undeclared metrics and cards written without this field it falls back to project.json lower_is_better and the generic names",'''),
])

patch(ev, [
('''                info=info, commit=commit(), date=time.strftime("%Y-%m-%d %H:%M"), metrics=m,
                lower_is_better=list(SM.LOWER_IS_BETTER))  # the direction of each metric, declared by its evaluator (card contract)''',
 '''                info=info, commit=commit(), date=time.strftime("%Y-%m-%d %H:%M"), metrics=m, better=SM.better(m))  # each metric's direction (card contract)'''),
('''        card["compare"] = dict(name=compare, metrics=SM.card_any(g2, gts, extra=True), paired=pr)''',
 '''        m2 = SM.card_any(g2, gts, extra=True)
        card["compare"] = dict(name=compare, metrics=m2, better=SM.better(m2), paired=pr)'''),
])

patch(sm, [
('''# the metrics of card_any() where a smaller number is better; every other number of a card is higher-is-better (n_samples and len_ratio are
# counts / ratios no rule judges). evaluate.py writes this list into every metric card ("lower_is_better", the evaluator contract), so
# ../autoresearch/tools/decide.py reads the direction from the card instead of knowing lane metric names (D24)
LOWER_IS_BETTER = ("false_len_frac", "vert100", "verts_per_frame", "dup_frac")''',
 '''# the direction of every metric of card_any(): lower is better for these, higher for the others except the counts / ratios no rule judges
# (NO_DIRECTION). evaluate.py writes better(m) into every metric card ("better", the evaluator contract), so ../autoresearch/tools/decide.py
# reads each metric's direction from the card instead of knowing lane metric names (D24)
LOWER_IS_BETTER = ("false_len_frac", "vert100", "verts_per_frame", "dup_frac")
NO_DIRECTION = ("n_samples", "len_ratio")


def better(metrics):
    """{metric: 'lower' | 'higher'} of a card's metrics (counts and ratios without a direction are left out)"""
    return {k: ("lower" if k in LOWER_IS_BETTER else "higher") for k in metrics if k not in NO_DIRECTION}'''),
])

s = open(st, encoding="utf-8").read()
start = "    # the evaluator declares directions in its cards"
end = '''    assert CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), lower_is_better=["err"])) == []'''
a = s.index(start)
b = s.index(end) + len(end)
new = '''    # the evaluator declares each metric's direction in its cards ("better"): decide.py reads it there, not from names (D24)
    def fcard(q, err, bt):
        return dict(full, name="x", metrics=dict(Q=q, err=err), **({"better": bt} if bt is not None else {}))
    LOW, HIGH = dict(Q="higher", err="lower"), dict(Q="higher", err="higher")
    json.dump(dict(a=fcard(0.76, 0.20, LOW), b=fcard(0.75, 0.21, LOW), ca=fcard(0.73, 0.10, LOW), cb=fcard(0.72, 0.11, LOW),
                   na=fcard(0.76, 0.20, None), nb=fcard(0.75, 0.21, None), nca=fcard(0.73, 0.10, None), ncb=fcard(0.72, 0.11, None),
                   xa=fcard(0.76, 0.20, HIGH), xb=fcard(0.75, 0.21, HIGH), qa=fcard(0.76, 0.20, dict(Q="higher")),
                   ta=dict(full, name="t", metrics=dict(ms=10.0), better=dict(ms="lower")), tc=dict(full, name="t", metrics=dict(ms=12.0), better=dict(ms="lower"))),
              open(f"{T}/dir.json", "w"))
    arm = lambda *ks: dict(cards=[f"dir.json#{k}" for k in ks])  # noqa: E731
    errguard = [dict(metric="Q", min_effect=0.01), dict(metric="err", max_loss=0.02)]
    assert verdict(arms=dict(a=arm("a", "b"), c=arm("ca", "cb")), rule=errguard) == "CUT", "err declared lower by the cards: +0.10 breaks the guard"
    assert verdict(arms=dict(a=arm("na", "nb"), c=arm("nca", "ncb")), rule=errguard) == "KEEP", "no declaration, err not a known name: read higher"
    assert verdict(arms=dict(a=arm("qa", "b"), c=arm("ca", "cb")), rule=errguard) == "CUT", "a card that does not declare err is undeclared, not 'higher'"
    for arms_, why in ((dict(a=arm("xa", "xb"), c=arm("ca", "cb")), "disagree"), (dict(a=arm("a", "b"), c=arm("nca", "ncb")), "other way")):
        try:
            verdict(arms=arms_, rule=errguard)
            raise AssertionError(f"must be refused: {why}")
        except ValueError as e:
            assert why in str(e), str(e)
    import experiment as X
    ef = f"{T}/experiments/E1_x/experiment.json"
    contra = dict(base, arms=dict(a=arm("a", "b"), c=arm("ca", "cb")), metric="err", direction="higher", rule=[dict(metric="err", min_effect=0.01)])
    assert any("contradicts the cards" in b for b in X.check(contra, ef)), X.check(contra, ef)
    assert not any("contradicts" in b or "disagree" in b for b in X.check(dict(contra, direction="lower"), ef))
    assert any("disagree" in b for b in X.check(dict(base, arms=dict(a=arm("xa", "xb"), c=arm("ca", "cb")), rule=errguard), ef)), "a guard's disagreement is found by check"
    timing = dict(base, metric="err", direction="lower", rule=[dict(metric="err", min_effect=0.01), dict(metric="ms", max_loss=1.0, subsets=["t"])],
                  arms=dict(a=dict(cards={"": ["dir.json#a", "dir.json#b"], "t": ["dir.json#ta"]}), c=dict(cards={"": ["dir.json#ca", "dir.json#cb"], "t": ["dir.json#tc"]})))
    assert not any("disagree" in b or "contradicts" in b for b in X.check(timing, ef)), ("cards without the metric do not count", X.check(timing, ef))
    import card as CARD
    assert any("better" in b for b in CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), better=dict(Q="up"))))
    assert any("does not hold" in b for b in CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), better=dict(err="lower"))))
    assert CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), better=dict(Q="higher"))) == []'''
s = s[:a] + new + s[b:]
open(st, "w", encoding="utf-8", newline="").write(s)
print("ok")
