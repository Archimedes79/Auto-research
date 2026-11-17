import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
S = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e358/"
apply(S + "mutate.py", [
    ("""float(i["feat"][0, 0]) * JH.REACH <= NEAR_M""", """is_near(float(i["feat"][0, 0]))"""),
])
