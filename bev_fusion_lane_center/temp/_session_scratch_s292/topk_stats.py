import os, pickle, numpy as np
ROOT = os.environ["BFLC_ROOT"]
for run in ("e312_p_s0", "e312_p_s1"):
    fr = pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))
    n = np.array([np.asarray(f["valid"]).reshape(2, 32).sum(1) for f in fr])          # (frames, 2): valid keypoints per kind
    full = (n >= 32).mean(0)
    print(run, "frames", len(fr), "valid keypoints per kind: mean %.1f / %.1f, p90 %d / %d, max %d / %d; share of frames at K = 32: entry %.3f, exit %.3f" % (n[:, 0].mean(), n[:, 1].mean(), np.percentile(n[:, 0], 90), np.percentile(n[:, 1], 90), n[:, 0].max(), n[:, 1].max(), full[0], full[1]))
