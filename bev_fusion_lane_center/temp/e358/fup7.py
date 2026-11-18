import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply

R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
S = R + "temp/_session_scratch_e358/"
apply(S + "test_staged.py", [
    ("""NS(wire="twin", attach=True), NS(wire="twin", snap=True, attach=True),""", """NS(wire="twin", attach=True, snap_nogeom=True), NS(wire="twin", snap=True, attach=True),"""),
    ("""NS(wire="head", attach=True, wire_run="x"), NS(wire="head", snap=True, attach=True), NS(wire="twin", snap=True, attach=True, wire_run="x"), NS(wire="twin", snap=True, attach=True, attach_r=1.5),""", """NS(wire="head", attach=True, snap_nogeom=True, wire_run="x"), NS(wire="head", snap=True, attach=True, snap_nogeom=True), NS(wire="twin", snap=True, attach=True, snap_nogeom=True, wire_run="x"), NS(wire="twin", snap=True, attach=True, snap_nogeom=True, attach_r=1.5),"""),
    ("""NS(wire="twin", snap=True, attach=True, snap_all=True), NS(wire="head", snap=True, attach=True, snap_guard=True, wire_run="x"), NS(wire="soft", snap=True), NS(snap=True, attach=True, wire_run="x")]""", """NS(wire="twin", snap=True, attach=True, snap_nogeom=True, snap_all=True), NS(wire="head", snap=True, attach=True, snap_nogeom=True, snap_guard=True, wire_run="x"), NS(wire="soft", snap=True), NS(snap=True, attach=True, snap_nogeom=True, wire_run="x")]"""),
])
apply(S + "mutate.py", [
    ("""    run_tests(name, load(name, src))
    sys.modules[name] = real""", """    run_tests(name, load(name, src))
    sys.modules[name] = real
    TS.QE, TS.JH = real_qe, real_jh"""),
    ("""        finally:
            sys.modules[name] = real
""", """        finally:
            sys.modules[name] = real
            TS.QE, TS.JH = real_qe, real_jh
"""),
])
