p = "evaluation/bev_fusion_area.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''def selftest():
    area = np.zeros((120, 104), bool)''', '''def selftest():
    cam, rad, occ = (np.zeros((4, 4), np.float32) for _ in range(3))
    cam[:2], rad[1:3], occ[1, 1] = 1, 1, 1                                    # camera rows 0-1, radar rows 1-2, hidden cell (1, 1)
    fa, ca = fusion_area({"camera": cam, "radar": rad, "occluded": occ}), camera_area({"camera": cam, "radar": rad, "occluded": occ})
    assert fa.sum() == 3 and fa[1, 0] and fa[1, 2] and fa[1, 3] and not fa[1, 1] and not fa[0].any() and not fa[2].any(), "fusion area = camera x radar x not hidden"
    assert ca.sum() == 7 and ca[0].all() and not ca[1, 1], "camera area = camera x not hidden"
    area = np.zeros((120, 104), bool)''')
open(p, "w", encoding="utf-8").write(s)

p = "evaluation/bev_mutants.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''import evaluation.metrics as M  # noqa: E402''', '''import evaluation.metrics as M  # noqa: E402
import evaluation.bev_fusion_area as FA  # noqa: E402''')
s = s.replace('''orig = {"cls": C.classification_numbers,''', '''orig = {"care": FA.CARE, "near": FA.NEAR, "share": FA.footprint_share, "farea": FA.fusion_area, "cls": C.classification_numbers,''')
s = s.replace('''MUTANTS = [(''', '''def m_care_threshold_03():                                           # the response rows count the classes with the same function: the module's own selftest catches this one
    FA.CARE = 0.3


def m_neutral_radius_zero():
    FA.NEAR = 0.0


def m_area_rule_off():                                               # detections outside the fusion area are kept (the GT boxes keep the real share: they carry no score)
    FA.footprint_share = lambda b, a: 1.0 if "score" in b else orig["share"](b, a)


def m_fusion_area_ignores_hidden():
    FA.fusion_area = lambda g: (np.asarray(g["camera"], np.float32) > 0) & (np.asarray(g["radar"], np.float32) > 0)


MUTANTS = [(''')
s = s.replace('''("camera mask = every cell (code side only)", m_fov_mask_all)]''', '''("camera mask = every cell (code side only)", m_fov_mask_all),
           ("care threshold 0.3 instead of 0.5", m_care_threshold_03), ("neutral radius 0 (detections on neutral boxes count)", m_neutral_radius_zero),
           ("detections outside the fusion area are kept", m_area_rule_off), ("fusion area ignores hidden cells", m_fusion_area_ignores_hidden)]''')
s = s.replace('''    C.classification_numbers, C.regression_numbers, B.tp_errors, M.in_front, M.TP_THRESHOLD, C.velocity_sums = (
        orig["cls"], orig["reg"], orig["tp"], orig["front"], orig["thr"], orig["vs"])''', '''    C.classification_numbers, C.regression_numbers, B.tp_errors, M.in_front, M.TP_THRESHOLD, C.velocity_sums = (
        orig["cls"], orig["reg"], orig["tp"], orig["front"], orig["thr"], orig["vs"])
    FA.CARE, FA.NEAR, FA.footprint_share, FA.fusion_area = orig["care"], orig["near"], orig["share"], orig["farea"]''')
s = s.replace('''            bad = off(rows)
''', '''            bad = off(rows)
            if not bad:                                                     # the unit selftest of the fusion-area module is the second net
                try:
                    FA.selftest()
                except AssertionError as e:
                    bad = [("bev_fusion_area selftest", str(e)[:40], 0.0, 1.0, 0.0)]
''')
open(p, "w", encoding="utf-8").write(s)

p = "tests/run_gates.sh"
s = open(p, encoding="utf-8").read()
s = s.replace('''run "curve metrics"     "$PY" evaluation/bev_curves.py --selftest''', '''run "fusion area"       "$PY" evaluation/bev_fusion_area.py --selftest   # footprint shares, GT classes, neutral detections, coverage classes (D37 / D38); no data
run "curve metrics"     "$PY" evaluation/bev_curves.py --selftest''', 1)
open(p, "w", encoding="utf-8").write(s)
