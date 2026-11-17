"""E357 T0: threshold sweep of the LINK tables only (no graph scorer; precision / recall / F1 of the short lane links and of the junction links) from the stored dump of a run.
Labelled IN-SAMPLE on VAL (a threshold chosen on VAL): a calibration-free reading of whether the head ranks the links, never the verdict (the verdict is the card's, at p >= 0.5).
usage: python sweep_links.py <run> <seed>   (run from the problem folder through tools/env/_py14.sh; this file lives in temp/e357, nothing imports it)"""
import os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
import jtok_eval as JE  # noqa: E402

run, seed = sys.argv[1], int(sys.argv[2])
ROOT = JE.ROOT
with open(os.path.join(ROOT, "runs", run, "tok_val.pkl"), "rb") as fh:
    preds = pickle.load(fh)
frames, table = JE.val_tables(2.5)
assert len(preds) == len(frames) == 873
THR = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
rows = {t: [] for t in THR}
with JE.JT.lattice(2.5):
    for k, p in enumerate(preds):
        t = table(k)
        lk = p["links"]
        for thr in THR:
            keep = np.asarray(lk["p"]) >= thr
            rows[thr].append(JE.link_counts(t, np.asarray(lk["src"], int)[keep], np.asarray(lk["dst"], int)[keep]))
print(f"run {run}: link tables by threshold (IN-SAMPLE on VAL, not the verdict)")
print("thr     short P / R / F1          junction P / R / F1       junction n_pred / n_gt")
for thr in THR:
    m = JE.link_numbers(rows[thr])
    print(f"{thr:<6g}  {m['link_short_P']:.3f} / {m['link_short_R']:.3f} / {m['link_short_F1']:.3f}      {m['link_junction_P']:.3f} / {m['link_junction_R']:.3f} / {m['link_junction_F1']:.3f}      {m['link_junction_n_pred']:.0f} / {m['link_junction_n_gt']:.0f}")
