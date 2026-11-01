import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:60], s.count(old))
    s = s.replace(old, new)
rep("""  * bands: the experiment's band for its headline metric, a clause's own band, else 0.010; lower-is-better metrics are known by name (LOWER)""",
    """  * bands: the experiment's band for its headline metric, a clause's own band, else 0.010; lower-is-better: the experiment's direction for its
    headline metric, else the problem's project.json "lower_is_better" list, else the generic cost names (LOWER)
  * the headline is judged for its evidence (one seed, a disagreeing seed, an interval containing 0) also when its clause is a guard (max_loss)
    or it has no min_effect clause: a better headline on one seed is INCONCLUSIVE, not KEEP""")
rep("""LOWER = {"vert100", "false_len_frac", "dup_frac", "ms", "ms_per_frame", "asm_ms", "map_err", "path_err", "ece"}""",
    """LOWER = {"ms", "ms_per_frame"}  # generic cost names; problem metrics are listed in project.json "lower_is_better\"""")
rep('''def is_lower(metric, exp=None):
    """lower is better: the experiment's direction for its headline metric, else the known names (LOWER)"""
    if exp is not None and metric == exp.get("metric"):
        return exp.get("direction") == "lower"
    return metric in LOWER''',
'''def lower_names(root=None):
    """the lower-is-better metric names of a problem: project.json "lower_is_better" plus the generic LOWER"""
    import project as P
    try:
        return LOWER | set(P.cfg_of(root or P.root()).get("lower_is_better", []))
    except SystemExit:  # no problem folder (typed interface outside a project)
        return set(LOWER)


def is_lower(metric, exp=None, root=None):
    """lower is better: the experiment's direction for its headline metric, else the problem's names (lower_names)"""
    if exp is not None and metric == exp.get("metric"):
        return exp.get("direction") == "lower"
    return metric in lower_names(root)''')
rep('''    sgn = -1.0 if (metric in LOWER if lower is None else lower) else 1.0''',
    '''    sgn = -1.0 if (is_lower(metric) if lower is None else lower) else 1.0''')
rep('''def judge_clause(clause, exp, cards):''', '''def judge_clause(clause, exp, cards, root=None):''')
rep('''        effect, diffs, interval = paired_effect(metric, vc, cc, is_lower(metric, exp))''',
    '''        effect, diffs, interval = paired_effect(metric, vc, cc, is_lower(metric, exp, root))''')
rep('''    effect, diffs, interval = paired_effect(head_metric, head(variant), head(exp["control"]), is_lower(head_metric, exp))
    effect *= sgn
    clauses = [judge_clause(c, exp, cards) for c in exp["rule"]]''',
'''    effect, diffs, interval = paired_effect(head_metric, head(variant), head(exp["control"]), is_lower(head_metric, exp, root))
    effect *= sgn
    clauses = [judge_clause(c, exp, cards, root) for c in exp["rule"]]''')
# T1: evidence flags of the headline when no min_effect clause judges it
rep('''    flags = sorted({f for c in clauses if c["ok"] for f in c["flags"]})
    if guards:''',
'''    flags = sorted({f for c in clauses if c["ok"] for f in c["flags"]})
    if hc is None and effect > band:  # no min_effect clause on the headline (e.g. only a guard): its evidence is judged here (review 2026-10-31, T1)
        if len(diffs) < 2 and not (exp.get("deterministic") and interval is not None and interval[0] > 0):
            flags.append("one seed")
        if len(diffs) >= 2 and not all(d > 0 for d in diffs):
            flags.append("a seed disagrees")
        if interval is not None and interval[0] <= 0:
            flags.append(f"interval [{interval[0]:+.3f}, {interval[1]:+.3f}] contains 0")
        flags = sorted(set(flags))
    if guards:''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
