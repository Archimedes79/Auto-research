p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
old = "links_matched / _only_in_ref / _only_in_new (among the links of matched nodes; p_max_one_sided: their probability sits at the threshold p >= 0.05), links (max abs of p / a / b), long_equal)."
new = "links_matched / _only_in_ref / _only_in_new (among the links of matched nodes: one-sided ones are mostly at the threshold p >= 0.05, a few belong to nodes next to an unmatched node; p_max_one_sided = the largest p among them, 0.26 / 0.53 on 24 frames of seeds 0 / 1), links (max abs of p / a / b), long_equal)."
assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
