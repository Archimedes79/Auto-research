import pickle, numpy as np
R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
for s in (0, 1):
    fr = pickle.load(open(R + f"runs/e324_s_s{s}/ends_val.pkl", "rb"))
    disp, ap, valid_ap = [], [], []
    npair = 0
    dirang = []
    for e in fr:
        k = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= 0.1)
        npair += len(ia)
        for idx in list(ia) + [k + j for j in ib]:
            disp.append(np.linalg.norm(e["att_pos"][idx] - e["kp_pos"][idx]))
            ap.append(e["att_p"][idx])
            c = np.clip(e["att_dir"][idx] @ e["kp_dir"][idx], -1, 1)
            dirang.append(np.degrees(np.arccos(c)))
    disp, ap, dirang = map(np.array, (disp, ap, dirang))
    print(f"seed {s}: pairs {npair}, pair-ends {len(disp)}")
    print("  |att-kp| m: mean %.3f median %.3f p75 %.3f p90 %.3f ; share<0.1m %.3f <0.5 %.3f <1 %.3f" % (disp.mean(), np.median(disp), np.percentile(disp, 75), np.percentile(disp, 90), (disp < .1).mean(), (disp < .5).mean(), (disp < 1).mean()))
    print("  att_p (=1-w0): mean %.3f median %.3f ; share>0.8 %.3f  >0.5 %.3f  <0.2 %.3f ; share exactly 0 %.3f" % (ap.mean(), np.median(ap), (ap > .8).mean(), (ap > .5).mean(), (ap < .2).mean(), (ap == 0).mean()))
    print("  heading change att-kp deg: median %.2f p90 %.2f" % (np.median(dirang), np.percentile(dirang, 90)))
    # does displacement correlate with att_p
    m = ap > .8
    print("  disp where att_p>0.8: median %.3f (n=%d); where att_p<0.2: median %.3f (n=%d)" % (np.median(disp[m]) if m.any() else float('nan'), m.sum(), np.median(disp[ap < .2]) if (ap<.2).any() else float('nan'), (ap < .2).sum()))
