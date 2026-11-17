import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply

R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
# (3) record_of: the kept piece ends no longer rebind the end-kind parameter; (1) the one rule of 'near' shared by pilot_stats and jwire_train
apply(R + "evaluation/jwire_data.py", [
    ("""    keep, ends = JW.kept_ends(graph.lines,""", """    keep, kept = JW.kept_ends(graph.lines,"""),
    ("""np.r_[en[0], en[1], en[2], en[3]] for en in ends], float).reshape(-1, 8), attrs=attrs,""", """np.r_[en[0], en[1], en[2], en[3]] for en in kept], float).reshape(-1, 8), attrs=attrs,"""),
    ("""def pilot_stats(records):""", """def is_near(d_feat):
    \"\"\"True where the nearest candidate piece end of an end is NEAR: `d_feat` = feature 0 of its place 0 (distance / reach; a float, an array or a tensor) within `NEAR_M` metres: the one rule of `pilot_stats` (from the records) and of the
    jittered share `jwire_train` logs (from the features the head trains on)\"\"\"
    return d_feat * JH.REACH <= NEAR_M


def pilot_stats(records):"""),
    ("""        near += [float(i["feat"][0, 0]) * JH.REACH <= NEAR_M for i in inst if i["kind"] == "oracle"]       # place 0 = the nearest candidate; feature 0 = its distance / reach""", """        near += [is_near(float(i["feat"][0, 0])) for i in inst if i["kind"] == "oracle"]                   # place 0 = the nearest candidate; feature 0 = its distance / reach"""),
])
