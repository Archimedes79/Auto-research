import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply

R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
apply(R + "training/train/jwire_train.py", [
    # (1) the share of the jittered draw
    ("""@torch.no_grad()
def calibration(w, feat, mask, label, orc, bins=10):""", """def near_share(feat, orc):
    \"\"\"the share of the oracle-labelled ends of the tensors of `tensors` whose nearest candidate piece end is NEAR (`JD.is_near`, within 0.5 m; NaN without such an end): the number `JD.pilot_stats` reports as `near05` from the
    records, here from the (jittered) features the head trains on, which the pilot on the unperturbed records does not see\"\"\"
    return float(JD.is_near(feat[orc][:, 0, 0]).float().mean()) if bool(orc.any()) else float("nan")


@torch.no_grad()
def calibration(w, feat, mask, label, orc, bins=10):"""),
    ("""    feat, mask, label, orc, npl = (torch.cat([p[i] for p in parts]) for i in range(5))
    vf, vm, vl, vo, vn, _ = tensors(val)""", """    feat, mask, label, orc, npl = (torch.cat([p[i] for p in parts]) for i in range(5))
    info["pilot_train_jittered"] = dict(near05=near_share(feat, orc), jitter_scale=scale, draws=draws)    # what the head trains on: the guard of the pilot (unperturbed TRAIN records against VAL) does not see the jitter
    run.log(f"pilot TRAIN jittered (scale {scale:.3f} m, {draws} draws): labelled ends within 0.5 m of a piece end {info['pilot_train_jittered']['near05']:.3f} (unperturbed TRAIN {pt['near05']:.3f}, VAL {pv['near05']:.3f})")
    vf, vm, vl, vo, vn, _ = tensors(val)"""),
    # (5) the test: the fit calls record all hyperparameters, wn trains like head 0
    ("""--heads", "1", "--epochs", "1", "--max-train", "30"]
        try:
            # att without""", """--heads", "1", "--epochs", "2", "--max-train", "30"]
        try:
            # att without"""),
    ("""calls["fit"].append((len(label), seed, cols)) or real_fit(feat, mask, label, epochs, lr, bs, seed, cols, wd)""", """calls["fit"].append((len(label), seed, cols, epochs, lr, bs, wd)) or real_fit(feat, mask, label, epochs, lr, bs, seed, cols, wd)"""),
    ("""            assert [c[1:] for c in calls["fit"]] == [(0, None), (1, DIST_ONLY), (0, None)] and calls["fit"][2][0] == 3 * n_unjit, calls["fit"]      # head 0, the ablation, wn: head seed 0, all columns, the unperturbed rows once per draw
""", """            assert [c[1:3] for c in calls["fit"]] == [(0, None), (1, DIST_ONLY), (0, None)] and calls["fit"][2][0] == 3 * n_unjit, calls["fit"]      # head 0, the ablation, wn: head seed 0, all columns, the unperturbed rows once per draw
            assert [c[3] for c in calls["fit"]] == [2, 2, 2] and calls["fit"][2][3:] == calls["fit"][0][3:], calls["fit"]                       # wn trains like head 0: the run's epochs (2, not the default 60 or 1), the same lr / batch size / weight decay
            # the jittered share the head trains on (known answer, and from the draws of this run); the unperturbed share is the one `pilot_stats` gives
            feat_ = torch.zeros(4, JH.P_MAX, JH.FEAT)
            feat_[:, 0, 0] = torch.tensor([0.05, 0.12, 0.13, 0.5])                                                  # the nearest candidate 0.2 / 0.48 / 0.52 / 2.0 m away (reach 4 m)
            assert abs(near_share(feat_, torch.tensor([True, True, True, False])) - 2 / 3) < 1e-6 and abs(near_share(feat_, torch.tensor([True, True, True, True])) - 0.5) < 1e-6 and np.isnan(near_share(feat_, torch.zeros(4, dtype=torch.bool)))
            jit = [real_tensors([JD.toy_record(i, i % 22) for i in range(30)], JD.make_jitter(wa["jitter_scale"], d)) for d in range(3)]
            share, pj = near_share(torch.cat([p[0] for p in jit]), torch.cat([p[3] for p in jit])), wa["pilot_train_jittered"]
            assert pj == dict(near05=share, jitter_scale=wa["jitter_scale"], draws=3), (pj, share)
            assert abs(wa["pilot_train"]["near05"] - near_share(unjit[0], unjit[3])) < 1e-12 and share != wa["pilot_train"]["near05"], (share, wa["pilot_train"]["near05"])
            assert "pilot TRAIN jittered" in open(os.path.join(rd, "log.txt")).read()
"""),
])
