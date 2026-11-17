import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_verdict.py", [
("""read_seed(dict_ns := NS(**dict(vars(a), read_out="snap")), 0)["c"] is not None""", """read_seed(NS(**dict(vars(a), read_out="snap")), 0)["c"] is not None"""),
("""and "primary" not in text.replace("(the primary", ""), text""", """and "primary" not in text, text"""),
])
print("ok")
