import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_data.py", [
("""    # 6. the pilot's share within 0.5 m counts the ends with an oracle label: the nearest piece end 0.2 m away (near), 1.02 m away (far), no GT end (not counted)
    near_rec = dict(toy_record(1, 1), pairs=np.array([np.r_[[0.0, -0.8], up, [5.0, 20.0], up]]))
    nogt = dict(toy_record(3, 3), gts=np.full((1, 2, 4), np.nan))
    assert pilot_stats([near_rec, toy_record(2, 2), nogt])["near05"] == 0.5 and pilot_stats([near_rec])["near05"] == 1.0 and pilot_stats([toy_record(2, 2)])["near05"] == 0.0
""", """    # 6. the pilot's share within 0.5 m counts the ends with an oracle label: the nearest piece end 0.2 / 0.48 m away (near), 0.52 / 1.02 m away (far), no GT end (not counted)
    at_m = lambda d, row: dict(toy_record(row, row), pairs=np.array([np.r_[[0.0, -1.0 + d], up, [5.0, 20.0], up]]))  # noqa: E731  (the entry d metres from the end of the first piece)
    nogt = dict(toy_record(5, 5), gts=np.full((1, 2, 4), np.nan))
    assert pilot_stats([at_m(0.2, 1), at_m(0.48, 2), at_m(0.52, 3), toy_record(4, 4), nogt])["near05"] == 0.5 and pilot_stats([at_m(0.2, 1)])["near05"] == 1.0 and pilot_stats([toy_record(4, 4)])["near05"] == 0.0
"""),
])
print("ok")
