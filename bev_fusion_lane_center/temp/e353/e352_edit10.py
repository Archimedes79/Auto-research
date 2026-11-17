p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    r = dict(frames=len(ref), keypoints_matched=0, keypoints_only_in_ref=0, keypoints_only_in_new=0, p_max_unmatched=0.0, max_abs={}, allowed_equal=True)
''', '''    r = dict(frames=len(ref), keypoints_matched=0, keypoints_only_in_ref=0, keypoints_only_in_new=0, frames_with_unmatched=0, p_max_unmatched=0.0, max_abs={}, allowed_equal=True)
''')
s = s.replace('''        m = []
        for kind in (0, 1):''', '''        m, lone = [], 0
        for kind in (0, 1):''')
s = s.replace('''            r["keypoints_only_in_new"] += len(ub)
            r["p_max_unmatched"]''', '''            r["keypoints_only_in_new"] += len(ub)
            lone += len(ua) + len(ub)
            r["p_max_unmatched"]''')
s = s.replace('''        (ea, eb), (xa, xb) = m
        if len(ea) and len(xa):
            for k in ("logit", "scale"):''', '''        (ea, eb), (xa, xb) = m
        r["frames_with_unmatched"] += int(lone > 0)
        if len(ea) and len(xa) and not lone:                                                                    # a keypoint in one dump only shifts the rank features of the pair head (group sizes, order): the logits are compared in the frames without one
            for k in ("logit", "scale"):''')
s = s.replace('''max_abs per key over the matched keypoints and over the logit /
    scale entries of the matched pairs, allowed_equal)"""''', '''max_abs per key over the matched keypoints and over the logit /
    scale entries of the matched pairs of the frames without an unmatched keypoint, allowed_equal)"""''')
open(p, "w", encoding="utf-8").write(s)
