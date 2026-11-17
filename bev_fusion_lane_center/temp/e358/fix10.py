import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_verdict.py", [
("""("w0", dict(attach=True)), ("c", dict(thr=0.5)),""", """("w0", dict(attach=True)), ("c", dict(snap_options=["snap_nogeom"])), ("w1", dict(snap_options=["snap_nogeom"])), ("c", dict(thr=0.5)),"""),
("""    print("jwire_verdict selftest ok")""", """    # E358, the PRIMARY read-out (attached ends, snap wiring without the snap geometry): the same rule on the cards `ends_card_[<tag>_]snap_nogeom_attach_thr0.1.json`; the head must have been trained on attached ends
    assert stem("snap") == "snap_" and f"ends_card_{stem('primary')}thr0.1.json" == "ends_card_snap_nogeom_attach_thr0.1.json" and f"ends_card_e358w1_{stem('primary')}thr0.1.json" == "ends_card_e358w1_snap_nogeom_attach_thr0.1.json"   # the file names of E324 s (`jwire_check` pins them to `qends_eval.card_name`)
    gp = dict(good, wa=agree(ends="att", near=(0.45, 0.44), nj=True))
    ptok = lambda spec, **k: tok(spec, ro="primary", **k)  # noqa: E731
    pwhy = lambda spec, **k: why(spec, ro="primary", **k)  # noqa: E731
    pseeds = lambda bad: seeds(bad, gp)  # noqa: E731
    assert ptok({0: gp, 1: gp}) == "KEEP+"
    for sp in pseeds(dict(gp, iv=(-0.001, 0.004))):
        assert ptok(sp) == "KEEP=", sp
    for sp in pseeds(dict(gp, d=dict(topo_F1=-0.003), iv=(-0.006, -0.001))):
        assert ptok(sp) == "KEEP-", sp
    for sp in pseeds(dict(gp, d=dict(topo_F1=0.002, dup_frac=0.011))) + pseeds(dict(gp, d=dict(topo_F1=-0.0051))):
        assert ptok(sp) == "TIE", sp
    for sp in pseeds(dict(gp, wa=agree(360, ends="att"))) + pseeds(dict(gp, wa=agree(float("nan"), ends="att"))):
        assert ptok(sp) == "CUT", sp
    assert ptok({0: gp}) == "INCOMPLETE" and ptok({0: gp, 1: dict(gp, d=None)}) == "INCOMPLETE" and ptok({0: gp, 1: dict(gp, iv=None)}) == "INCOMPLETE" and ptok({0: gp, 1: dict(gp, wa=None)}) == "INCOMPLETE"
    with tempfile.TemporaryDirectory() as t:                                                           # the primary read-out reads the primary names and not the snapped ones (and the other way round)
        world(t, {0: good, 1: good}, "cs", "snap")
        a = NS(root=t, card="E000", control_prefix="cs", head_prefix="hs", tag="tw", tag_x=None, x_head_prefix="xs", boot="temp/e000/boot_w_vs_c_s{seed}.log", read_out="primary")
        assert read_seed(a, 0)["c"] is None and read_seed(a, 0)["w"] == [None] * 3 and read_seed(dict_ns := NS(**dict(vars(a), read_out="snap")), 0)["c"] is not None
    # INVALID: a snapped card under the primary name (attach false, no snap_nogeom, snap false), a head of the other kind (kp: key missing or 'kp'; and an att head under the snapped read-out), the bootstrap log of the snapped cards
    for name, over in (("c", dict(attach=False)), ("w0", dict(attach=False)), ("w2", dict(attach=False)), ("c", dict(snap_options=[])), ("w1", dict(snap_options=[])), ("w0", dict(snap_options=None)), ("c", dict(snap=False)), ("w0", dict(thr=0.5)),
                       ("c", dict(run="other")), ("w1", dict(seed="@other")), ("c", dict(wire="head")), ("w0", dict(wire_run="hs9"))):
        for s in SEEDS:
            t, _, (_, reason, _) = run_case({s: dict(gp, over={name: over}), 1 - s: gp}, ro="primary")
            assert t == "INVALID" and f"seed {s}" in reason, (name, over, s, t, reason)
    for s in SEEDS:
        for wa in (agree(), agree(ends="kp")):
            assert ptok({s: dict(gp, wa=wa), 1 - s: gp}) == "INVALID" and "needs a head trained on 'att' ends" in pwhy({s: dict(gp, wa=wa), 1 - s: gp}), (s, wa.get("ends"))
        assert tok({s: dict(good, wa=agree(ends="att")), 1 - s: good}) == "INVALID" and "needs a head trained on 'kp' ends" in why({s: dict(good, wa=agree(ends="att")), 1 - s: good}), "an att head under the snapped read-out"
        assert tok({s: dict(good, wa=agree(ends="kp")), 1 - s: good}) == "KEEP+", "a head that names its kind kp is E325's"
        ok_p = boot_text(f"cs{s}", 0.002, 0.001, 0.004, f"cs{s} ends_card_tw_snap_nogeom_attach_thr0.1", f"cs{s} ends_card_snap_nogeom_attach_thr0.1")
        assert ptok({s: dict(gp, boot=ok_p), 1 - s: gp}) == "KEEP+" and ptok({s: dict(gp, boot=boot_text(f"cs{s}", 0.002, 0.001, 0.004)), 1 - s: gp}) == "INVALID", "the bootstrap log must name the primary cards"
    # the pilot's second statistic: the share of labelled ends within 0.5 m differing by more than 0.10 labels the result shift-dominated (a note, the token stays); a file without it (E325's) says nothing
    for near, flagged in (((0.45, 0.56), True), ((0.56, 0.45), True), ((0.45, 0.54), False), (None, False)):
        t, _, (_, _, notes) = run_case({0: gp, 1: dict(gp, wa=agree(ends="att", near=near))}, ro="primary")
        assert t == "KEEP+" and any("SHIFT-DOMINATED (TRAIN share" in n for n in notes) == flagged, (near, notes)
    # the report names the read-out, the head's kind, the share within 0.5 m and the no-jitter head
    t, lines, _ = run_case({0: gp, 1: gp}, ro="primary")
    text = "\n".join(lines)
    assert t == "KEEP+" and "read-out primary" in text and "head 0 primary:" in text and "ends att" in text and "no-jitter head (wn, reported only)" in text and "TRAIN 0.450, VAL 0.440" in text and "snapped:" not in text, text
    t, lines, _ = run_case({0: good, 1: good})
    assert "read-out snap" in "\n".join(lines) and "head 0 snapped:" in "\n".join(lines) and "no-jitter head" not in "\n".join(lines) and "primary" not in "\n".join(lines).replace("the primary", ""), "\n".join(lines)
    print("jwire_verdict selftest ok")"""),
])
print("ok")
