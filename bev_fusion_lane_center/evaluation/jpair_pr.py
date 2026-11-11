"""Pair precision at a fixed pair recall (E302's headline, also a fair read of E287 / E298 / E303 / E305): the precision-recall curve of a pair head from its dump, and the precision of one run at the recall of another.

    wsl bash tools/env/_py14.sh jpair_pr.py score --run e302_k_s0 [--control e287_p2_s0] [--kp-thr 0.5]       CPU, VAL 873, from runs/<run>/ends_val.pkl (`jends_eval.py dump`); -> runs/e302/pair_pr_<run>.json
    wsl bash tools/env/_py14.sh jpair_pr.py selftest

A candidate pair (i, j) of a dump is a pair of strong keypoints (p >= kp-thr, valid, allowed); it is TRUE when the keypoints are matched (strict one-to-one, 4 m / 30 degrees, `jrule_control._match`) to the two GT ends of a GT pair. Sweeping the logit
threshold gives recall (true pairs delivered / GT pairs of the frames, 11083 on VAL) against precision (true / delivered); the keypoints do not change with the threshold, so the curve is the pair decision only.
With `--control` the precision of the run is read at the recall of the control's own operating point (logit >= 0), the card of E302: "precision at the recall of c".
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jgt as J  # noqa: E402
import jrule_control as JR  # noqa: E402

THRS = np.round(np.arange(-4.0, 4.01, 0.5), 2)


def curve(scores, labels, n_gt, thrs=THRS):
    """[(thr, recall, precision, delivered)] for the candidate pairs (scores, bool labels) against n_gt true pairs"""
    rows = []
    for t in thrs:
        m = scores >= t
        d, tp = int(m.sum()), int((m & labels).sum())
        rows.append((float(t), tp / max(n_gt, 1), tp / max(d, 1), d))
    return rows


def precision_at_recall(rows, recall):
    """the best precision over the thresholds whose recall is at least `recall` (None when the curve never reaches it)"""
    p = [r[2] for r in rows if r[1] >= recall - 1e-12]
    return max(p) if p else None


def candidates(run, kp_thr):
    """(logits, labels, n_gt) of all candidate pairs of a dump (VAL 873)"""
    import qgraph_view as QV
    frames = {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
    sc, lab, n_gt = [], [], 0
    for sid, (name, toks) in QV.val_scenes().items():
        for t in toks:
            if t not in frames:
                continue
            jg = J.junction_gt(QV.load_gt(t))
            f = frames[t]
            K = f["logit"].shape[0]
            gt_pairs = set() if jg is None else {(int(p[0]), int(p[1])) for p in jg["pairs"]}
            n_gt += len(gt_pairs)
            ve = f["valid"][0] & (f["kp_p"][:K] >= kp_thr)
            vx = f["valid"][1] & (f["kp_p"][K:] >= kp_thr)
            kp, kd = f["kp_pos"], f["kp_dir"]
            if jg is not None:
                ma, mb = JR._match(kp[:K], kd[:K], ve, jg["ent"]), JR._match(kp[K:], kd[K:], vx, jg["ext"])
                inv_a, inv_b = {int(k): g for g, k in enumerate(ma) if k >= 0}, {int(k): g for g, k in enumerate(mb) if k >= 0}
            else:
                inv_a = inv_b = {}
            ii, jj = np.where(ve[:, None] & vx[None, :] & f["allowed"].astype(bool))
            for i, j in zip(ii, jj):
                sc.append(float(f["logit"][i, j]))
                lab.append((inv_a.get(int(i), -1), inv_b.get(int(j), -1)) in gt_pairs)
    return np.array(sc), np.array(lab, bool), n_gt


def score(a):
    out = {}
    cands = {r: candidates(r, a.kp_thr) for r in ([a.run] + ([a.control] if a.control else []))}
    for r, (sc, lab, n_gt) in cands.items():
        rows = curve(sc, lab, n_gt)
        out[r] = dict(n_gt=n_gt, candidates=len(sc), curve=[dict(thr=x[0], recall=x[1], precision=x[2], delivered=x[3]) for x in rows], at_zero=[dict(thr=x[0], recall=x[1], precision=x[2]) for x in rows if x[0] == 0.0][0])
        print(r, "logit >= 0: recall %.3f precision %.3f" % (out[r]["at_zero"]["recall"], out[r]["at_zero"]["precision"]), f"({n_gt} true pairs, {len(sc)} candidates)")
    if a.control:
        rc = out[a.control]["at_zero"]["recall"]
        pr = precision_at_recall(curve(*cands[a.run][:2], cands[a.run][2]), rc)
        out["precision_at_control_recall"] = dict(control_recall=rc, control_precision=out[a.control]["at_zero"]["precision"], run_precision=pr, gain=None if pr is None else pr - out[a.control]["at_zero"]["precision"])
        print("precision of", a.run, "at the recall of", a.control, "(%.3f):" % rc, None if pr is None else "%.3f (control %.3f, gain %+.3f)" % (pr, out[a.control]["at_zero"]["precision"], out["precision_at_control_recall"]["gain"]))
    os.makedirs(os.path.join(ROOT, "runs", "e302"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "runs", "e302", f"pair_pr_{a.run}{'_vs_' + a.control if a.control else ''}.json"), "w"), indent=1)


def selftest():
    sc = np.array([3.0, 2.0, 1.0, 0.5, -1.0, -3.0])
    lab = np.array([True, True, False, True, False, True])
    rows = curve(sc, lab, n_gt=5, thrs=np.array([0.0, 1.5, 4.0]))
    assert rows[0] == (0.0, 3 / 5, 3 / 4, 4), rows[0]                                       # four delivered, three true, five true pairs exist
    assert rows[1][3] == 2 and rows[1][2] == 1.0 and abs(rows[1][1] - 0.4) < 1e-12
    assert rows[2][3] == 0 and rows[2][1] == 0.0
    assert precision_at_recall(rows, 0.4) == 1.0 and precision_at_recall(rows, 0.6) == 0.75 and precision_at_recall(rows, 0.9) is None
    print("jpair_pr selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--control", default=None)
    s.add_argument("--kp-thr", type=float, default=0.5)
    sub.add_parser("selftest")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else score(a)
