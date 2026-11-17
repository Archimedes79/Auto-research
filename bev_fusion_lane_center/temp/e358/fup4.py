import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply

R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
S = R + "temp/_session_scratch_e358/"
apply(S + "stage/qends_eval.py", [
    ("""    \"\"\"-> (wire, wire_run) after the option checks of `--wire` / `--wire-run` (E325): twin / head need --snap and none of --attach, --attach-r, --snap-all, --snap-guard (the head was not trained on attached ends, the others
    override a choice); `--wire head` needs `--wire-run` and only it takes it\"\"\"""", """    \"\"\"-> (wire, wire_run) after the option checks of `--wire` / `--wire-run` (E325): twin / head need --snap and none of --attach-r, --snap-all, --snap-guard (they override a choice); E358 staged copy: --attach is allowed
    but only WITH --snap-nogeom (the primary read-out of E324 s: a card named `..._snap_attach_...` is not the card the verdict reads, and the head must have been trained on the attached ends: `jwire_head.check_head`);
    `--wire head` needs `--wire-run` and only it takes it\"\"\""""),
    ("""and getattr(a, "attach_r", None) is None), "--wire twin / head needs --snap and none of --attach-r, --snap-all, --snap-guard (E358 staged copy: --attach is allowed, the head must have been trained on the attached ends)\"""", """and getattr(a, "attach_r", None) is None), "--wire twin / head needs --snap and none of --attach-r, --snap-all, --snap-guard (E358 staged copy: --attach is allowed, the head must have been trained on the attached ends)"
    assert wire is None or not getattr(a, "attach", False) or getattr(a, "snap_nogeom", False), "--wire twin / head with --attach needs --snap-nogeom (E358 staged copy: the primary read-out; without it the card has another name and is not read by the verdict)\""""),
    ("""NS2(wire="twin", attach=True), NS2(wire="head", wire_run="x", attach=True),""", """NS2(wire="twin", attach=True), NS2(wire="twin", snap=True, attach=True), NS2(wire="head", snap=True, attach=True, wire_run="x"), NS2(wire="head", wire_run="x", attach=True),"""),
])
apply(S + "test_staged.py", [
    ("""    bad = [NS(wire="twin"), NS(wire="twin", attach=True),""", """    bad = [NS(wire="twin"), NS(wire="twin", attach=True), NS(wire="twin", snap=True, attach=True), NS(wire="head", snap=True, attach=True, wire_run="x"), NS(wire="head", snap=True, attach=True, snap_nogeom=False, wire_run="x"),"""),
    ("""        def score(head, attach, wire="head", seed=0, run="e324_s_s0", tag="t"):
            try:
                QE.cmd_score(NS(**dict(opts, seed=seed, run=run, tag=tag, wire=wire, wire_run=head, attach=attach)))""", """        def score(head, attach, wire="head", seed=0, run="e324_s_s0", tag="t", **more):
            try:
                QE.cmd_score(NS(**dict(opts, seed=seed, run=run, tag=tag, wire=wire, wire_run=head, attach=attach, **more)))"""),
    ("""        assert "seed" in score("h_att", True, seed=1)""", """        assert "--snap-nogeom" in score("h_att", True, snap_nogeom=False), "an attached read-out without --snap-nogeom is refused before the head is read"
        assert score("h_kp350", False, snap_nogeom=False) == "passed: model.pt", "the keypoint read-out does not need it"
        assert "seed" in score("h_att", True, seed=1)"""),
])
