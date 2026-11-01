import sys
dc, ex, cd, sc = sys.argv[1:5]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
patch(dc, [
('''def is_lower(metric, exp=None, root=None):
    """lower is better: the experiment's direction for its headline metric, else the problem's names (lower_names)"""
    if exp is not None and metric == exp.get("metric"):
        return exp.get("direction") == "lower"
    return metric in lower_names(root)''',
'''def card_lower(metric, cards=()):
    """the direction the cards themselves declare (their evaluator's "lower_is_better" list, card contract): True / False, or None when no
    card declares one. Cards that disagree are refused (two evaluators with other conventions answer different questions)"""
    said = {metric in c["lower_is_better"] for _, c, _ in cards if isinstance(c, dict) and isinstance(c.get("lower_is_better"), list)}
    if len(said) > 1:
        refs = ", ".join(str(r) for _, _, r in cards)
        raise ValueError(f"the cards disagree whether {metric} is lower-is-better ({refs})")
    return said.pop() if said else None


def is_lower(metric, exp=None, root=None, cards=()):
    """lower is better: the experiment's direction for its headline metric, else what the cards declare (card_lower), else the
    problem's names (lower_names: project.json, the generic LOWER, the temporary lane list) for cards written before the field existed"""
    if exp is not None and metric == exp.get("metric"):
        return exp.get("direction") == "lower"
    declared = card_lower(metric, cards)
    return declared if declared is not None else metric in lower_names(root)'''),
('''        effect, diffs, interval = paired_effect(metric, vc, cc, is_lower(metric, exp, root))''',
 '''        effect, diffs, interval = paired_effect(metric, vc, cc, is_lower(metric, exp, root, list(vc) + list(cc)))'''),
('''  * bands: the experiment's band for its headline metric, a clause's own band, else 0.010; lower-is-better: the experiment's direction for its
    headline metric, else the generic cost names (LOWER) plus the problem's project.json "lower_is_better" list''',
 '''  * bands: the experiment's band for its headline metric, a clause's own band, else 0.010; lower-is-better: the experiment's direction for its
    headline metric, else the cards' own "lower_is_better" list (the evaluator declares it; cards that disagree are refused), else for older
    cards the generic cost names (LOWER) plus the problem's project.json "lower_is_better" list'''),
])
patch(ex, [
('''    if exp["metric"] in lower_names(root) and exp["direction"] != "lower":
        bad.append(f"metric {exp['metric']} is lower-is-better (project.json lower_is_better) but direction is {exp['direction']}")
''', ''),
('''    for a in arms:
        try:
            cards = arm_cards(exp, a, root)''', '''    declared = set()  # what the arms' cards say about the headline's direction (their evaluator's lower_is_better list)
    for a in arms:
        try:
            cards = arm_cards(exp, a, root)'''),
('''            for m, full, ref in lst:
                if full is not None:  # a full card follows the evaluator contract (legacy cards without the schema field pass); a bare metrics dict is accepted as is''',
 '''            for m, full, ref in lst:
                if isinstance(full, dict) and isinstance(full.get("lower_is_better"), list):
                    declared.add(exp["metric"] in full["lower_is_better"])
                if full is not None:  # a full card follows the evaluator contract (legacy cards without the schema field pass); a bare metrics dict is accepted as is'''),
('''    return bad


def main():''', '''    if len(declared) > 1:
        bad.append(f"the cards disagree whether {exp['metric']} is lower-is-better")
    elif declared and declared.pop() != (exp["direction"] == "lower"):
        bad.append(f"direction {exp['direction']} contradicts the cards: their evaluator declares {exp['metric']} "
                   f"{'lower' if exp['direction'] != 'lower' else 'higher'}-is-better")
    elif not declared and exp["metric"] in lower_names(root) and exp["direction"] != "lower":
        bad.append(f"metric {exp['metric']} is lower-is-better (project.json lower_is_better) but direction is {exp['direction']}")
    return bad


def main():'''),
])
patch(cd, [
('''    cmp = card.get("compare")''', '''    lib = card.get("lower_is_better")
    if lib is not None and not all(isinstance(x, str) and x for x in lib):
        bad.append("lower_is_better must be a list of metric names")
    cmp = card.get("compare")'''),
])
patch(sc, [
('''"compare": "dict", "settings": "dict"}''', '''"compare": "dict", "settings": "dict", "lower_is_better": "list"}'''),
('''lower-is-better metrics are known by the experiment file's direction, else by name (project.json lower_is_better, plus the generic ms names of decide.py LOWER)",''',
 '''lower-is-better metrics: see lower_is_better",
    "lower_is_better": "the metric names (of metrics and compare.metrics) where a smaller number is better, declared by the evaluator that defines them; every other metric of the card is higher-is-better. decide.py takes a metric's direction from here (the experiment's direction wins for its headline metric; cards that disagree are refused); for cards written without this field it falls back to project.json lower_is_better",'''),
])
print("ok")
