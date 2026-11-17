import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e358/stage/qends_eval.py", [
("""    for ok_opt in (NS2(), NS2(wire="twin", snap=True), NS2(wire="head", snap=True, wire_run="e325_w_s0")):""", """    for ok_opt in (NS2(), NS2(wire="twin", snap=True), NS2(wire="head", snap=True, wire_run="e325_w_s0"), NS2(wire="twin", snap=True, attach=True, snap_nogeom=True), NS2(wire="head", snap=True, wire_run="e358_w_s0", attach=True, snap_nogeom=True)):     # E358 staged: --attach is allowed"""),
("""NS2(wire="twin", snap=True, wire_run="x"), NS2(wire="head", snap=True, wire_run="x", attach=True), NS2(wire="head", snap=True, wire_run="x", attach_r=1.5),""", """NS2(wire="twin", snap=True, wire_run="x"), NS2(wire="twin", attach=True), NS2(wire="head", wire_run="x", attach=True), NS2(wire="head", snap=True, wire_run="x", attach_r=1.5), NS2(wire="head", snap=True, wire_run="x", attach=True, attach_r=1.5),"""),
])
print("ok")
