import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e358/test_staged.py", [
    ("""    JH.check_head(legacy, "e312_p_s0", 0, False)                                                          # E325's head: no field = kp, E312's run
""", """    JH.check_head(legacy, "e312_p_s0", 0, False)                                                          # E325's head: no field = kp, E312's run
    JH.check_head(legacy, "e312_p_s1", 1, False)                                                          # ... of the seed of the read-out (no seed field: the seed asked for, and E312's run of that seed)
"""),
])
