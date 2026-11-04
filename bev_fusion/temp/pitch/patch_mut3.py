def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, (p, old[:60])
        s = s.replace(old, new, 1)
    open(p, "w", encoding="utf-8").write(s)
s = open("evaluation/bev_mutants.py", encoding="utf-8").read()
a = s.index("def m_neutral_rule_off")
b = s.index("def m_fusion_area_ignores_hidden")
new = '''def _near_neutral(d, gts_all, scored):
    return any(b["group"] == d["group"] and FA._dist(d, b) <= FA.NEAR for b in gts_all if b not in scored)


def m_neutral_rule_off():                                            # a detection on a neutral (partial / don't-care) box that lies inside the area counts again
    real = FA.prepare

    def prepare(dets, gts_all, shares, area, include_partial):
        keep, scored = real(dets, gts_all, shares, area, include_partial)
        return keep + [d for d in dets if d not in keep and orig["share"](d, area) >= 0.5], scored
    FA.prepare = prepare


def m_area_rule_off():                                               # detections outside the fusion area are kept (those on a neutral box stay out: only the area rule is removed)
    real = FA.prepare

    def prepare(dets, gts_all, shares, area, include_partial):
        keep, scored = real(dets, gts_all, shares, area, include_partial)
        return keep + [d for d in dets if d not in keep and not _near_neutral(d, gts_all, scored)], scored
    FA.prepare = prepare


'''
s = s[:a] + new + s[b:]
open("evaluation/bev_mutants.py", "w", encoding="utf-8").write(s)
edit("evaluation/bev_response.py", [(
'''    assert Nc > 0 and Np > 0 and cnt["dont_care"] > 0, f"too few frames for the fusion rows (care {Nc}, partial {Np}, don't care {cnt['dont_care']}): use --n 24 or more"''',
'''    assert Nc > 0, f"too few frames for the fusion rows (no care box): use --n 24 or more"
    rows += [("sanity: the fusion rows rest on all three classes being present", f"{c} boxes > 0 (1 = yes)", float(cnt[c] > 0), 1.0, 0.0) for c in ("care", "partial", "dont_care")]''')])
