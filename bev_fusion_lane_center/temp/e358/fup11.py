import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
apply(R + "training/train/jwire_train.py", [
    ("""    return float(JD.is_near(feat[orc][:, 0, 0]).float().mean()) if bool(orc.any()) else float("nan")""", """    return float(JD.is_near(feat[orc][:, 0, 0]).float().mean())                                          # the mean of no end is NaN"""),
])
apply(R + "temp/_session_scratch_e358/mutate.py", [
    ("""        ("near_share: no NaN without labelled ends", "if bool(orc.any()) else float(\\"nan\\")", "if True else float(\\"nan\\")"),
""", ""),
])
