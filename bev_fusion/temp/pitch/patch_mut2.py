p = "evaluation/bev_mutants.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''def m_neutral_radius_zero():
    FA.NEAR = 0.0
''', '''def m_neutral_rule_off():                                             # a detection on a neutral (partial / don't-care) box that lies inside the area counts again
    real = FA.prepare

    def prepare(dets, gts_all, area, include_partial):
        keep, scored = real(dets, gts_all, area, include_partial)
        again = [d for d in dets if d not in keep and orig["share"](d, area) >= 0.5]
        return keep + again, scored
    FA.prepare = prepare
''')
s = s.replace('("neutral radius 0 (detections on neutral boxes count)", m_neutral_radius_zero)', '("neutral rule off (detections on neutral boxes count)", m_neutral_rule_off)')
s = s.replace('''orig = {"care": FA.CARE, "near": FA.NEAR,''', '''orig = {"care": FA.CARE, "near": FA.NEAR, "prepare": FA.prepare,''')
s = s.replace('''    FA.CARE, FA.NEAR, FA.footprint_share, FA.fusion_area = orig["care"], orig["near"], orig["share"], orig["farea"]''', '''    FA.CARE, FA.NEAR, FA.footprint_share, FA.fusion_area, FA.prepare = orig["care"], orig["near"], orig["share"], orig["farea"], orig["prepare"]''')
open(p, "w", encoding="utf-8").write(s)
