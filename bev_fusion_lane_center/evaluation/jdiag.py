"""Diagnostics of the junction thread that the cards of E307 / E308 / E310 / E313 cite as how a number was made (promoted from scratch scripts on 2026-11-12: the git rule of this repository promotes a script that a record names).

    wsl bash tools/env/_py14.sh jdiag.py common e284_h2_s0 e287_p2_s0 e308_t_s0    the pair heads on the SAME ends: AUC and recall / precision of the pair logit among the candidates whose two ends are GT ends that ALL runs found
    wsl bash tools/env/_py14.sh jdiag.py curve  e308_t_s0                           the connector drawn from the keypoints / the true ends against the GT connector of its pair (chord tangent and fitted handle scales)
    wsl bash tools/env/_py14.sh jdiag.py kperr  e284_h2_s0 e287_p2_s0               position and heading error of the matched keypoints on the GT ends that all runs found
    wsl bash tools/env/_py14.sh jdiag.py maps   e313_s e307_n                       keypoint F1 / crowded / isolated / far entry recall / zone IoU of one maps arm against another, both seeds (runs/<arm>_s<S>/jmaps_val.json)
    wsl bash tools/env/_py14.sh jdiag.py snaprule e287_p2 e305_e                    the rule of the E311 card on runs/<net>_s<S>/ends_card_snap_thr<t>.json (`jends_eval.py score --snap --thr t`): topo F1 over the plain graph > +0.02 with F1@1.0 >= -0.01, dup_frac not higher, vertices <= +1.0, junction F1 >= -0.03; both seeds of every net
    wsl bash tools/env/_py14.sh jdiag.py boot   e320_a_s0 ends_card_snap_nogeom_attach_thr0.1 e312_p_s0 ends_card_snap_thr0.1
                                                                                    paired scene bootstrap (1000 resamples of the VAL scenes) of the difference of two read-outs on the same frames: topo F1, junction F1, F1@1.0, lanes found (E320: a - c per seed; reads runs/<run>/<card stem>.counts.pkl that `score` writes)
    wsl bash tools/env/_py14.sh jdiag.py attlog e320_a_s0                           the attachment statistics of the training log (att_acc, att_acc_nearest, att_ce, att_bin, att_kp_d, att_d), the mean of the last 10 rows of runs/<run>/log.jsonl weighted by att_n (E320 kill criterion; one row = one batch of 8 frames, a single row is noise)
    wsl bash tools/env/_py14.sh jdiag.py selftest

`common`, `curve` and `kperr` read runs/<run>/ends_val.pkl (`jends_eval.py dump` / `qends_eval.py dump`) and the VAL lane GT (matching: strict one-to-one, 4 m, 30 degrees, `jrule_control._match`); VAL only. CPU.
"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))


def auc(s, y):
    """area under the ROC curve of scores s for the boolean labels y (rank formula; nan without both classes)"""
    s, y = np.asarray(s), np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def _frames(runs):
    return {r: {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for r in runs}


def _matches(e, jg):
    """(entry matches, exit matches): per GT end the index of the matched keypoint or -1 (strict one-to-one, keypoints with p >= 0.5)"""
    import jrule_control as JR
    K = e["logit"].shape[0]
    ve = e["valid"][0] & (e["kp_p"][:K] >= 0.5)
    vx = e["valid"][1] & (e["kp_p"][K:] >= 0.5)
    return JR._match(e["kp_pos"][:K], e["kp_dir"][:K], ve, jg["ent"]), JR._match(e["kp_pos"][K:], e["kp_dir"][K:], vx, jg["ext"]), ve, vx


def common(runs):
    import jgt as J
    import qgraph_view as QV
    frames = _frames(runs)
    S = {r: [] for r in runs}
    Y = {r: [] for r in runs}
    n_common = n_gt = 0
    for sid, (name, toks) in QV.val_scenes().items():
        for t in toks:
            if not all(t in frames[r] for r in runs):
                continue
            jg = J.junction_gt(QV.load_gt(t))
            if jg is None:
                continue
            gt = {(int(p[0]), int(p[1])) for p in jg["pairs"]}
            n_gt += len(gt)
            inv = {}
            for r in runs:
                ma, mb, ve, vx = _matches(frames[r][t], jg)
                inv[r] = ({int(k): g for g, k in enumerate(ma) if k >= 0}, {int(k): g for g, k in enumerate(mb) if k >= 0}, ve, vx)
            ents = set.intersection(*[set(inv[r][0].values()) for r in runs])
            exts = set.intersection(*[set(inv[r][1].values()) for r in runs])
            n_common += sum(1 for a, b in gt if a in ents and b in exts)
            for r in runs:
                f = frames[r][t]
                ia, ib, ve, vx = inv[r]
                ii, jj = np.where(ve[:, None] & vx[None, :] & f["allowed"].astype(bool))
                for i, j in zip(ii, jj):
                    a, b = ia.get(int(i), -1), ib.get(int(j), -1)
                    if a in ents and b in exts:
                        S[r].append(float(f["logit"][i, j]))
                        Y[r].append((a, b) in gt)
    print(f"GT pairs {n_gt}, with both ends found by all {len(runs)} systems: {n_common}")
    ref = None
    for r in runs:
        s, y = np.array(S[r]), np.array(Y[r], bool)
        rec0 = ((s >= 0) & y).sum() / n_common
        prec0 = ((s >= 0) & y).sum() / max(1, (s >= 0).sum())
        ref = rec0 if ref is None else ref
        o = np.argsort(-s)
        tp = np.cumsum(y[o])
        k = min(int(np.searchsorted(tp, ref * n_common)) + 1, len(s))
        print(f"{r:12s} candidates {len(s):6d}, true {int(y.sum()):5d}, AUC {auc(s, y):.3f}, at logit >= 0: recall {rec0:.3f} precision {prec0:.3f}; precision at the recall of {runs[0]} ({ref:.3f}): {tp[k - 1] / k:.3f}")


def curve(run):
    from scipy.spatial import cKDTree
    import jgt as J
    import qends as QENDS
    import qjunction as QJ
    import qgraph_view as QV
    frames = _frames([run])[run]
    acc = {k: [] for k in ("kp_chord", "kp_fit", "true_chord", "true_fit")}
    dpos = []
    n = 0
    for t, e in frames.items():
        jg = J.junction_gt(QV.load_gt(t))
        if jg is None:
            continue
        ma, mb, _, _ = _matches(e, jg)
        K = e["logit"].shape[0]
        sc = QJ.fit_pair_scales(jg)
        for ip, (r, pts) in enumerate(zip(jg["pairs"], jg["pair_pts"])):
            a, b = int(r[0]), int(r[1])
            tgt = np.asarray(pts, float)
            if len(tgt) < 3 or ma[a] < 0 or mb[b] < 0 or not np.isfinite(sc[ip]).all():
                continue
            pa, ta = e["kp_pos"][ma[a]], e["kp_dir"][ma[a]]
            pb, tb = e["kp_pos"][K + mb[b]], e["kp_dir"][K + mb[b]]
            dpos += [float(np.linalg.norm(pa - jg["ent"][a, :2])), float(np.linalg.norm(pb - jg["ext"][b, :2]))]
            for name, (p0, t0, p1, t1) in (("kp", (pa, ta, pb, tb)), ("true", (jg["ent"][a, :2], jg["ent"][a, 2:], jg["ext"][b, :2], jg["ext"][b, 2:]))):
                for cn, c in (("chord", QENDS.curve(p0, t0, p1, t1)), ("fit", QENDS.curve_s(np.asarray(p0, float), np.asarray(t0, float), np.asarray(p1, float), np.asarray(t1, float), float(sc[ip][0]), float(sc[ip][1])))):
                    d = cKDTree(c).query(tgt)[0]
                    acc[f"{name}_{cn}"].append((float((d <= 1.0).mean()), float((d <= 1.5).mean()), float(d.max())))
            n += 1
    print(f"{run}: {n} GT pairs with both ends matched and a connector; keypoint position error mean {np.mean(dpos):.2f} m")
    for k, v in acc.items():
        a = np.array(v)
        print(f"{k:12s} GT connector points within 1.0 m: {a[:, 0].mean():.3f}, within 1.5 m: {a[:, 1].mean():.3f}; pairs with >= 90 % within 1.5 m: {(a[:, 1] >= 0.9).mean():.3f}; median max deviation {np.median(a[:, 2]):.2f} m")


def kperr(runs):
    import jgt as J
    import qgraph_view as QV
    frames = _frames(runs)
    E = {r: dict(pos=[], ang=[], apos=[], aang=[]) for r in runs}
    n = 0
    for sid, (name, toks) in QV.val_scenes().items():
        for t in toks:
            if not all(t in frames[r] for r in runs):
                continue
            jg = J.junction_gt(QV.load_gt(t))
            if jg is None:
                continue
            inv = {}
            for r in runs:
                ma, mb, _, _ = _matches(frames[r][t], jg)
                inv[r] = ({g: int(k) for g, k in enumerate(ma) if k >= 0}, {g: int(k) for g, k in enumerate(mb) if k >= 0}, frames[r][t]["logit"].shape[0])
            ents = set.intersection(*[set(inv[r][0]) for r in runs])
            exts = set.intersection(*[set(inv[r][1]) for r in runs])
            n += len(ents) + len(exts)
            for r in runs:
                f = frames[r][t]
                ia, ib, K = inv[r]
                for kind, gs, m, gt in (("ent", ents, ia, jg["ent"]), ("ext", exts, ib, jg["ext"])):
                    for g in gs:
                        k = m[g] + (0 if kind == "ent" else K)
                        c = float(np.clip(np.dot(f["kp_dir"][k], gt[g][2:4]) / (np.linalg.norm(f["kp_dir"][k]) * np.linalg.norm(gt[g][2:4]) + 1e-9), -1, 1))
                        E[r]["pos"].append(float(np.linalg.norm(f["kp_pos"][k] - gt[g][:2])))
                        E[r]["ang"].append(float(np.degrees(np.arccos(c))))
                        if "att_pos" in f:                                                                                     # E320: the ATTACHED end of the same keypoint against the same GT end
                            ca = float(np.clip(np.dot(f["att_dir"][k], gt[g][2:4]) / (np.linalg.norm(f["att_dir"][k]) * np.linalg.norm(gt[g][2:4]) + 1e-9), -1, 1))
                            E[r]["apos"].append(float(np.linalg.norm(f["att_pos"][k] - gt[g][:2])))
                            E[r]["aang"].append(float(np.degrees(np.arccos(ca))))
    print("common GT ends:", n)
    for r in runs:
        p, a = np.array(E[r]["pos"]), np.array(E[r]["ang"])
        print(f"{r:12s} position error: mean {p.mean():.2f} m, median {np.median(p):.2f}, 90% {np.percentile(p, 90):.2f}; heading error: mean {a.mean():.1f} deg, median {np.median(a):.1f}, 90% {np.percentile(a, 90):.1f}")
        if E[r]["apos"]:                                                                                                         # E320: the attached ends of the same matched keypoints
            p2, a2 = np.array(E[r]["apos"]), np.array(E[r]["aang"])
            print(f"{'  attached':12s} position error: mean {p2.mean():.2f} m, median {np.median(p2):.2f}, 90% {np.percentile(p2, 90):.2f}; heading error: mean {a2.mean():.1f} deg, median {np.median(a2):.1f}, 90% {np.percentile(a2, 90):.1f}")


def boot(ra, ca, rb, cb, n=1000, seed=0):
    """paired scene bootstrap of (read-out A - read-out B) on the same VAL frames: the pooled topo F1, junction F1, F1@1.0 and lanes found over resampled scenes (frames of a scene stay together), from the per-frame counts of the two score cards
    (`runs/<run>/<card stem>.counts.pkl`, written by `qends_eval.cmd_score`); prints the point difference, the bootstrap mean and the 95 % interval. The rule of a card stays the point estimate; this is the interval next to it."""
    import qgraph_eval as QE
    A, B = (pickle.load(open(os.path.join(ROOT, "runs", r, c + ".counts.pkl"), "rb")) for r, c in ((ra, ca), (rb, cb)))
    assert A["scenes"] == B["scenes"], "the two read-outs must be scored on the same frames"
    sc = np.asarray(A["scenes"])
    uniq = sorted(set(sc.tolist()))
    where = {s: np.where(sc == s)[0] for s in uniq}
    keys = ("topo_F1", "junction_F1", "F1@1.0", "lanes_found")
    pa, pb = QE.graph_numbers(A["head"]), QE.graph_numbers(B["head"])
    rng = np.random.RandomState(seed)
    d = {k: [] for k in keys}
    for _ in range(n):
        pick = np.concatenate([where[uniq[i]] for i in rng.randint(0, len(uniq), len(uniq))])
        na, nb = QE.graph_numbers([A["head"][i] for i in pick]), QE.graph_numbers([B["head"][i] for i in pick])
        for k in keys:
            d[k].append(na[k] - nb[k])
    print(f"A = {ra} {ca}   B = {rb} {cb}   {len(uniq)} scenes, {n} resamples")
    for k in keys:
        x = np.asarray(d[k])
        print(f"  {k:12s} A - B: point {pa[k] - pb[k]:+.4f}, bootstrap mean {x.mean():+.4f}, 95 % [{np.percentile(x, 2.5):+.4f}, {np.percentile(x, 97.5):+.4f}]")


def attlog(run, last=10, rows=None):
    """E320: the attachment statistics of the training log, the mean over the last `last` rows that carry att_n of runs/<run>/log.jsonl (`rows`: a list of dicts instead, for the test) weighted by att_n; one row is one batch of 8 frames, so a single row is noise (the standard error of att_acc - att_acc_nearest is about 0.05 there). Returns the dict and prints it,
    with att_acc - att_acc_nearest: the kill criterion of the card reads this difference against its margin."""
    if rows is None:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs", run, "log.jsonl")) if l.strip()]
    rows = [r for r in rows if r.get("att_n", 0) > 0][-last:]
    assert rows, f"{run}: no row of log.jsonl carries att_n (a run without --attach)"
    w = np.array([r["att_n"] for r in rows], float)
    out = {k: float(np.average([r[k] for r in rows], weights=w)) for k in ("att_acc", "att_acc_nearest", "att_ce", "att_bin", "att_kp_d", "att_d")}
    out.update(rows=len(rows), att_n=float(w.sum()), first_step=rows[0].get("step"), last_step=rows[-1].get("step"), att_acc_minus_nearest=out["att_acc"] - out["att_acc_nearest"])
    print(f"{run}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items()))
    return out


def maps_summary(d):
    """the numbers of one jmaps_val.json that the maps cards compare: keypoint F1 (mean of entry and exit, 3 m, thr 0.3, distance only), zone IoU, crowded / isolated entry recall, entry recall beyond 35 m, best-threshold F1"""
    k = d["keypoints"]
    return dict(kp_f1=(k["kp_ours_tol3.0_noang_ent"]["f1"] + k["kp_ours_tol3.0_noang_ext"]["f1"]) / 2, zone_iou=d["zone_iou"], core_iou=d["core_iou"], crowded=d["crowded_split"]["ours_ent"]["crowded_recall"],
                isolated=d["crowded_split"]["ours_ent"]["isolated_recall"], far=d["range_city_split"]["range_ours_ent_gt35"]["recall"], best=d["kp_f1_best_threshold"]["ours"])


def maps(arm, ctl):
    for s in (0, 1):
        pa, pc = (os.path.join(ROOT, "runs", f"{x}_s{s}", "jmaps_val.json") for x in (arm, ctl))
        if not (os.path.exists(pa) and os.path.exists(pc)):
            print(f"s{s}: pending")
            continue
        a, c = maps_summary(json.load(open(pa))), maps_summary(json.load(open(pc)))
        print(f"s{s}: " + "; ".join(f"{k} {a[k]:.3f} vs {c[k]:.3f} ({a[k] - c[k]:+.3f})" for k in a))


def rule311(diff):
    """the rule of the E311 card applied to the differences (head - plain graph) of one scored read-out"""
    return diff["topo_F1"] > 0.02 and diff["F1@1.0"] >= -0.01 and diff["dup_frac"] <= 0 and diff["vert100_dp"] <= 1.0 and diff["junction_F1"] >= -0.03


def snaprule(nets, thrs=(0.5, 0.2, 0.1)):
    cells = {}
    for net in nets:
        for thr in thrs:
            for s in (0, 1):
                p = os.path.join(ROOT, "runs", f"{net}_s{s}", f"ends_card_snap_thr{thr}.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))["difference"]
                cells[(net, thr, s)] = (rule311(d), d)
                print(f"{net:9s} thr {thr} s{s}: {'HOLDS' if cells[(net, thr, s)][0] else 'fails'} topo {d['topo_F1']:+.4f} F1@1.0 {d['F1@1.0']:+.4f} dup {d['dup_frac']:+.4f} vert {d['vert100_dp']:+.2f} jF1 {d['junction_F1']:+.4f}")
    for thr in thrs:
        got = [cells.get((net, thr, s)) for net in nets for s in (0, 1)]
        print(f"thr {thr}: {sum(1 for c in got if c)}/{len(got)} cells scored; all hold: {all(c and c[0] for c in got)}")


def selftest():
    t = attlog("x", last=2, rows=[dict(att_n=10, att_acc=0.1, att_acc_nearest=0.1, att_ce=1.0, att_bin=0.0, att_kp_d=1.0, att_d=1.0, step=100), dict(att_n=10, att_acc=0.5, att_acc_nearest=0.4, att_ce=1.0, att_bin=0.0, att_kp_d=1.0, att_d=0.8, step=200),
                                  dict(att_n=30, att_acc=0.8, att_acc_nearest=0.5, att_ce=0.5, att_bin=0.0, att_kp_d=1.0, att_d=0.6, step=300), dict(step=400)])         # E320: the last 2 rows with att_n, weighted by att_n
    assert abs(t["att_acc"] - (0.5 * 10 + 0.8 * 30) / 40) < 1e-9 and abs(t["att_acc_minus_nearest"] - ((0.5 - 0.4) * 10 + (0.8 - 0.5) * 30) / 40) < 1e-9 and t["last_step"] == 300 and t["rows"] == 2, t
    assert auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0 and auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert np.isnan(auc([0.1, 0.2], [1, 1]))
    d = dict(keypoints={"kp_ours_tol3.0_noang_ent": {"f1": 0.7}, "kp_ours_tol3.0_noang_ext": {"f1": 0.5}}, zone_iou=0.76, core_iou=0.58, crowded_split={"ours_ent": {"crowded_recall": 0.8, "isolated_recall": 0.7}},
             range_city_split={"range_ours_ent_gt35": {"recall": 0.6}}, kp_f1_best_threshold={"ours": 0.65})
    ok = dict(topo_F1=0.025, **{"F1@1.0": 0.01}, dup_frac=-0.03, vert100_dp=0.2, junction_F1=0.02)
    assert rule311(ok) and not rule311(dict(ok, topo_F1=0.02)) and not rule311(dict(ok, dup_frac=0.001)) and not rule311(dict(ok, vert100_dp=1.1)) and not rule311(dict(ok, junction_F1=-0.031)) and not rule311(dict(ok, **{"F1@1.0": -0.011}))
    m = maps_summary(d)
    assert abs(m["kp_f1"] - 0.6) < 1e-9 and m["crowded"] == 0.8 and m["far"] == 0.6 and m["best"] == 0.65
    print("jdiag selftest ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "common" and len(sys.argv) > 2:
        common(sys.argv[2:])
    elif cmd == "curve" and len(sys.argv) == 3:
        curve(sys.argv[2])
    elif cmd == "kperr" and len(sys.argv) > 2:
        kperr(sys.argv[2:])
    elif cmd == "boot" and len(sys.argv) == 6:
        boot(*sys.argv[2:6])
    elif cmd == "attlog" and len(sys.argv) == 3:
        attlog(sys.argv[2])
    elif cmd == "snaprule" and len(sys.argv) > 2:
        snaprule(sys.argv[2:])
    elif cmd == "maps" and len(sys.argv) == 4:
        maps(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
