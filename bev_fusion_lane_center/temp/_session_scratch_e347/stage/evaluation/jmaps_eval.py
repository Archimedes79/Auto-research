"""Score of the junction maps head (E286): maps and entry / exit keypoints on the VAL frames, next to the baseline B's own event heat maps read the same way.

    wsl bash tools/env/_py14.sh jmaps_eval.py score --run e286_j0_s0 [--base runs/e273_p_s0/model.pt] [--max 0]     -> runs/<run>/jmaps_val.json + a printed table (`--raster exact|legacy`: the zone / core / paint targets rasterised that way, written to jmaps_val_<raster>.json)
    wsl bash tools/env/_py14.sh jmaps_eval.py selftest

Numbers (VAL 873 frames or the first --max, scene-disjoint from TRAIN, TEST never; every frame counts, frames without a junction carry the false alarms):
  zone               IoU and F1 of the junction zone at p > 0.5 (1 m cells)
  stop / crossing    precision and recall of the painted stop line / pedestrian crossing masks with a tolerance of one 0.5 m cell
  keypoints          entry and exit points: a prediction is a 3 x 3 local maximum of the heat above `thr` (soft-argmax position); it matches a GT end within `tol` metres and, for the head with a heading,
                     within 30 degrees (greedy by distance); recall = matched GT ends / GT ends, precision = matched predictions / predictions; thr 0.3 and 0.5, tol 1.5 and 3.0 m (D79 used 3 m, 30 degrees)
  baseline B         the stop / exit event heat maps of the baseline's pixel stage (centre1 channels 8 / 9), the same extraction, distance only (B has no heading)
  heading            median angle error (degrees) at the matched predictions
  painted split      recall of the entries / exits that are PAINTED (a stop line of another type than the turn stop within 3 m, or a crossing within 4 m: `qjunction.painted_flags`) and of the rest (thr 0.3, 3 m)
E347 (the 2 m head, ctx '+r2'): the head's output is 50 x 50 (zone / core IoU on 2 m cells, keypoints at cell centre + offset). Every head is read in BOTH read-outs: `ours` = the head's own rule (3 x 3 peak; '+onehot': threshold alone), `ours_alt` = the other one.
The threshold sweep carries per read-out, kind and threshold: recall / precision / F1 at 3 m and F1 at 1.5 m (distance only), the crowded GT ends' recall (heading-matched, 3 m), and the SURPLUS DUPLICATES (a keypoint that no GT end claims in the one-to-one
`match` at 3 m and that has another keypoint of its kind within `DUP_R` = 2 m): `thr_sweep[who][typ]["rows"]`, and `gate` = the stage-1 numbers of card E347 at each kind's own best 3 m threshold (as `kp_f1_best_threshold`) and at the fixed 0.3.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jmaps as JM  # noqa: E402

ANG = 30.0


def match(pred, gt, tol, use_ang):
    """greedy matching by distance; pred (n, 5) x, y, tx, ty, p; gt (m, 4) x, y, tx, ty -> (matched pairs [(i, j, angle deg)], n_pred, n_gt)"""
    pairs = []
    if len(pred) and len(gt):
        P, G = np.asarray(pred, float), np.asarray(gt, float)
        d = np.linalg.norm(P[:, None, :2] - G[None, :, :2], axis=-1)
        ang = np.degrees(np.arccos(np.clip(P[:, None, 2] * G[None, :, 2] + P[:, None, 3] * G[None, :, 3], -1, 1)))
        ok = (d <= tol) & ((ang <= ANG) if use_ang else True)
        cand = sorted(((d[i, j], i, j) for i, j in zip(*np.where(ok))), key=lambda t: t[0])
        ui, uj = set(), set()
        for _, i, j in cand:
            if i in ui or j in uj:
                continue
            ui.add(i)
            uj.add(j)
            pairs.append((i, j, float(ang[i, j])))
    return pairs, len(pred), len(gt)


def painted_of(token, n_ent, n_ext):
    """(painted flags of the GT entries, of the GT exits, city) of a frame, aligned with the batch's GT ends; (None, None, city) when the paint sidecar is missing or the junction GT does not line up with the batch (counted by the caller)"""
    import jgt as JG
    import junction_paint as JP
    import qjunction as QJ
    from GtGenerator import lane_gt
    g = lane_gt.load(token)
    city = "boston" if "boston" in str(g.get("location", "")) else "singapore"
    jg = JG.junction_gt(g)
    try:
        paint = JP.load(token)
    except FileNotFoundError:
        return None, None, city
    pe, px = QJ.painted_flags(jg, paint)
    return (pe, px, city) if len(pe) == n_ent and len(px) == n_ext else (None, None, city)


def crowded_of(G, tol=4.0, ang=30.0):
    """(n,) bool: a GT end with another end of the same kind within tol metres and ang degrees of heading (one keypoint can serve only one of them: the strict matching is harder there)"""
    G = np.asarray(G, float)
    out = np.zeros(len(G), bool)
    for j in range(len(G)):
        d = np.linalg.norm(G[:, :2] - G[j, :2], axis=1)
        a = np.degrees(np.arccos(np.clip(G[:, 2] * G[j, 2] + G[:, 3] * G[j, 3], -1, 1)))
        m = (d <= tol) & (a <= ang)
        m[j] = False
        out[j] = bool(m.any())
    return out


THR_SWEEP = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
DUP_R = 2.0                                                                                          # E347: a keypoint within this distance of another keypoint of its kind, that no GT end claims, is a surplus duplicate


def surplus_duplicates(pred, pairs, r=DUP_R):
    """number of keypoints (pred (n, >= 2)) that are not in the one-to-one `pairs` of `match` (no GT end claims them) and have another keypoint within r metres (matched or not): the surplus copies of an end"""
    if len(pred) < 2:
        return 0
    P = np.asarray(pred, float)[:, :2]
    d = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(P)) * 1e9
    claimed = np.zeros(len(P), bool)
    claimed[[i for i, _, _ in pairs]] = True
    return int(((d <= r).any(1) & ~claimed).sum())


def sweep_counts(pred, gt, crowded, use_ang):
    """the counts of one frame, one kind, one threshold of one read-out: tp / pred / gt at 3 m and tp15 at 1.5 m (distance only: the F1 of the sweep), gt_c / hit_c = the crowded GT ends and how many of them a keypoint within 3 m (and 30 degrees when
    `use_ang`: the head has a heading) claims, dup = the surplus duplicates among the pred keypoints (matched at 3 m, distance only)"""
    pairs, npred, ngt = match(pred, gt, 3.0, False)
    hit = np.zeros(len(gt), bool)
    for _, j, _ in match(pred, gt, 3.0, use_ang)[0]:
        hit[j] = True
    return dict(tp=len(pairs), pred=npred, gt=ngt, tp15=len(match(pred, gt, 1.5, False)[0]), gt_c=int(crowded.sum()), hit_c=int(hit[crowded].sum()), dup=surplus_duplicates(pred, pairs))


def sweep_rows(acc, who, typ):
    """the sweep of one read-out and kind (rows over THR_SWEEP, the best 3 m F1 and its threshold) from the accumulated counts"""
    rows = []
    for t in THR_SWEEP:
        d = acc.k[f"sweep_{who}_{t}_{typ}"]
        r_, p_ = d["tp"] / max(d["gt"], 1), d["tp"] / max(d["pred"], 1)
        r15, p15 = d["tp15"] / max(d["gt"], 1), d["tp15"] / max(d["pred"], 1)
        rows.append(dict(thr=t, recall=r_, precision=p_, f1=2 * r_ * p_ / max(r_ + p_, 1e-9), f1_tol1_5=2 * r15 * p15 / max(r15 + p15, 1e-9), crowded_recall=d["hit_c"] / max(d["gt_c"], 1), crowded_n=int(d["gt_c"]),
                         n_pred=int(d["pred"]), n_dup=int(d["dup"])))
    return dict(rows=rows, best_f1=max(r["f1"] for r in rows), best_thr=max(rows, key=lambda r: r["f1"])["thr"])


def gate_numbers(sw, thr=None):
    """the stage-1 numbers of card E347 of one read-out from its sweep `sw` (`sweep_rows` of ent and ext): the mean of the two kinds of the 3 m F1 and of the 1.5 m F1, the crowded recall of each kind, the surplus duplicates among all keypoints; every kind at its
    own best 3 m threshold (`thr` None: the threshold that maximises that run's own F1, as `kp_f1_best_threshold`) or at the fixed threshold `thr`"""
    pick = {typ: next(r for r in sw[typ]["rows"] if r["thr"] == (sw[typ]["best_thr"] if thr is None else thr)) for typ in ("ent", "ext")}
    n_pred = sum(r["n_pred"] for r in pick.values())
    return dict(thr={typ: r["thr"] for typ, r in pick.items()}, f1_tol3=sum(r["f1"] for r in pick.values()) / 2, f1_tol1_5=sum(r["f1_tol1_5"] for r in pick.values()) / 2,
                crowded_recall={typ: r["crowded_recall"] for typ, r in pick.items()}, dup_share=sum(r["n_dup"] for r in pick.values()) / max(n_pred, 1))


def _range_class(E):
    d = np.linalg.norm(np.asarray(E)[:, :2], axis=1)
    return np.where(d < 20, 0, np.where(d < 35, 1, 2))


def _dil(m):
    return F.max_pool2d(m, 3, 1, 1)


class Acc:
    def __init__(self):
        self.k = {}

    def add(self, key, **kw):
        d = self.k.setdefault(key, {})
        for n, v in kw.items():
            d[n] = d.get(n, 0.0) + v

    def lst(self, key, name, v):
        self.k.setdefault(key, {}).setdefault(name, []).extend(v)


def score(a):
    import lane_gt_data as LD
    LD.QData.DEFAULT_JUNCTION = True
    run_dir = os.path.join(ROOT, "runs", a.run)
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    args = ck["args"]
    LD.QData.DEFAULT_R2 = "r2" in JM.ctx_mods(args.get("ctx", "dil"))                                      # E347: a 2 m head is scored against the targets on its own grid
    model = JM.JModel(width=48, d=args.get("d", 64), ctx=args.get("ctx", "dil"))
    model.load_state_dict(ck["state"])
    base_path = a.base or args.get("base") or os.path.join(ROOT, "runs", f"e273_p_s{args.get('seed', 0)}", "model.pt")
    basem = JM.JModel(width=48, d=64, ctx="none")
    basem.load_base(base_path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":                                                                                                                          # D100: a dump / score next to two trainings crashed one of them (CUDA unknown error, 2026-11-14)
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jmaps_eval score {a.run}")
    model.to(dev).eval()
    basem.to(dev).eval()
    raster = a.raster or args.get("raster", "legacy")                                                      # the targets of the zone / core / paint IoU are rasterised as the run was trained (or as asked)
    if raster == "exact":
        os.environ["BFLC_RASTER"] = "exact"
    else:
        os.environ.pop("BFLC_RASTER", None)                                                               # an inherited exact setting must not score a legacy run
    ds, dl = LD.loader("val_clean", bs=8, workers=a.workers, shuffle=False, map_mode="exact", pixels=False)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    rule_peak = not model.jm.onehot                                                                         # E347: the head's own read-out (3 x 3 peak; the one-to-one target: threshold alone) and the other one
    acc = Acc()
    n = 0
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(dev).float()
            P, out = model(x, joint=False)
            Pb = basem.pyr(x)
            jz, jp = b["jz"].to(dev).float(), b["jpaint"].to(dev).float()
            if "zone" not in out.get("skip", ()):                                                          # E332: a head without the zone supervision has no zone / core numbers
                pz_all = (out["j1"][:, :2].float().sigmoid() > 0.5).float()
                for c, name in enumerate(("zone", "core")):
                    pz, tz = pz_all[:, c:c + 1], jz[:, c:c + 1]
                    acc.add(name, tp=float((pz * tz).sum()), fp=float((pz * (1 - tz)).sum()), fn=float(((1 - pz) * tz).sum()))
            if "j05" in out:                                                                               # E332: ... and one without the paint branch no stop / crossing numbers
                pp = (out["j05"].float().sigmoid() > 0.5).float()
                for c, name in enumerate(("stop", "crossing")):
                    pr, tg = pp[:, c:c + 1], jp[:, c:c + 1]
                    acc.add(name, pred=float(pr.sum()), pred_ok=float((pr * _dil(tg)).sum()), gt=float(tg.sum()), gt_ok=float((tg * _dil(pr)).sum()))
            # baseline B: event heat stop / exit of the pyramid, distance only
            ev = Pb["centre1"][:, 6:12].float()
            fake = dict(j1=torch.cat([torch.zeros_like(ev[:, :2]), ev[:, 2:3], ev[:, 3:4], torch.zeros_like(ev[:, :1]).repeat(1, 4, 1, 1)], 1))
            kps = {"ours": JM.keypoints(out, thr=0.3), "ours5": JM.keypoints(out, thr=0.5), "B": JM.keypoints(fake, thr=0.3), "B5": JM.keypoints(fake, thr=0.5)}
            sweep ={t: dict(ours=JM.keypoints(out, thr=t), ours_alt=JM.keypoints(out, thr=t, peak=not rule_peak), B=JM.keypoints(fake, thr=t)) for t in THR_SWEEP}   # critic 2026-11-11: both maps at the same fixed threshold is not a fair comparison of differently calibrated maps
            for i in range(x.shape[0]):
                ent_g = b["ent"][i][b["ent_m"][i]].numpy()
                ext_g = b["ext"][i][b["ext_m"][i]].numpy()
                pe, px, city = painted_of(b["token"][i], len(ent_g), len(ext_g))
                if pe is None and len(ent_g):
                    acc.add("painted_unavailable", frames=1)
                cr_e, cr_x = crowded_of(ent_g) if len(ent_g) else np.zeros(0, bool), crowded_of(ext_g) if len(ext_g) else np.zeros(0, bool)
                for t, kws in sweep.items():
                    for typ, gt, cr in (("ent", ent_g, cr_e), ("ext", ext_g, cr_x)):
                        for who, kk in kws.items():
                            acc.add(f"sweep_{who}_{t}_{typ}", **sweep_counts(kk[i][typ].numpy(), gt, cr, who != "B"))
                rc_e, rc_x = _range_class(ent_g) if len(ent_g) else np.zeros(0, int), _range_class(ext_g) if len(ext_g) else np.zeros(0, int)
                for who, kk in kps.items():
                    for tol in (1.5, 3.0):
                        for use_ang in ((True, False) if who.startswith("ours") else (False,)):
                            key = f"kp_{who}_tol{tol}_{'ang' if use_ang else 'noang'}"
                            for typ, gt in (("ent", ent_g), ("ext", ext_g)):
                                pairs, npred, ngt = match(kk[i][typ].numpy(), gt, tol, use_ang)
                                acc.add(key + "_" + typ, tp=len(pairs), pred=npred, gt=ngt)
                                if use_ang and who == "ours" and tol == 3.0:
                                    acc.lst(key + "_" + typ, "ang", [p[2] for p in pairs])
                                if tol == 3.0 and who in ("ours", "B") and use_ang == (who == "ours"):                       # recall on painted / unpainted ends (head with heading, B distance only, thr 0.3)
                                    fl = pe if typ == "ent" else px
                                    hit = np.zeros(len(gt), bool)
                                    for _, j, _ in pairs:
                                        hit[j] = True
                                    if fl is not None:
                                        acc.add(f"split_{who}_{typ}", gt_p=int(fl.sum()), hit_p=int(hit[fl].sum()), gt_u=int((~fl).sum()), hit_u=int(hit[~fl].sum()))
                                    cr = cr_e if typ == "ent" else cr_x
                                    acc.add(f"crowd_{who}_{typ}", gt_c=int(cr.sum()), hit_c=int(hit[cr].sum()), gt_n=int((~cr).sum()), hit_n=int(hit[~cr].sum()))
                                    rc = rc_e if typ == "ent" else rc_x
                                    for k, nm in enumerate(("lt20", "20to35", "gt35")):                                  # range from the ego, E300: the far ends are the hole
                                        acc.add(f"range_{who}_{typ}_{nm}", n=int((rc == k).sum()), hit=int(hit[rc == k].sum()))
                                    if city:
                                        acc.add(f"city_{who}_{typ}_{city}", n=len(gt), hit=int(hit.sum()))
            n += x.shape[0]
            if a.max and n >= a.max:
                break
    from GtGenerator import lane_gt
    res = dict(run=a.run, base=base_path, frames=n, args=args, raster=raster, lane_gt_signature=lane_gt.signature(), painted_unavailable_frames=int(acc.k.get("painted_unavailable", {}).get("frames", 0)))
    for name in ("zone", "core"):
        z = acc.k.get(name)
        res[f"{name}_iou"] = None if z is None else z["tp"] / max(z["tp"] + z["fp"] + z["fn"], 1)
        res[f"{name}_f1"] = None if z is None else 2 * z["tp"] / max(2 * z["tp"] + z["fp"] + z["fn"], 1)
    for name in ("stop", "crossing"):
        d = acc.k.get(name)
        res[f"{name}_prec"], res[f"{name}_rec"] = (None, None) if d is None else (d["pred_ok"] / max(d["pred"], 1), d["gt_ok"] / max(d["gt"], 1))
    kp = {}
    for key, d in acc.k.items():
        if not key.startswith("kp_"):
            continue
        r = d["tp"] / max(d["gt"], 1)
        p = d["tp"] / max(d["pred"], 1)
        kp[key] = dict(recall=r, precision=p, f1=2 * r * p / max(r + p, 1e-9), gt=int(d["gt"]), pred=int(d["pred"]), tp=int(d["tp"]), ang_median=float(np.median(d["ang"])) if d.get("ang") else None)
    res["keypoints"] = kp
    split = {}
    for key, d in acc.k.items():
        if key.startswith("split_"):
            split[key[6:]] = dict(painted_recall=d["hit_p"] / max(d["gt_p"], 1), unpainted_recall=d["hit_u"] / max(d["gt_u"], 1), painted_gt=int(d["gt_p"]), unpainted_gt=int(d["gt_u"]))
    res["painted_split"] = split
    other = {}
    for key, d in acc.k.items():
        if key.startswith("range_") or key.startswith("city_"):
            other[key] = dict(n=int(d["n"]), recall=d["hit"] / max(d["n"], 1))
    res["range_city_split"] = other
    crowd = {}
    for key, d in acc.k.items():
        if key.startswith("crowd_"):
            crowd[key[6:]] = dict(crowded_recall=d["hit_c"] / max(d["gt_c"], 1), crowded_n=int(d["gt_c"]), isolated_recall=d["hit_n"] / max(d["gt_n"], 1), isolated_n=int(d["gt_n"]))
    res["crowded_split"] = crowd
    sweep_res = {who: {typ: sweep_rows(acc, who, typ) for typ in ("ent", "ext")} for who in ("ours", "ours_alt", "B")}
    res["thr_sweep"] = sweep_res
    res["kp_f1_best_threshold"] = {who: (sweep_res[who]["ent"]["best_f1"] + sweep_res[who]["ext"]["best_f1"]) / 2 for who in sweep_res}
    res["readout"] = dict(ours="peak" if rule_peak else "threshold", ours_alt="threshold" if rule_peak else "peak")             # E347: the head's own rule and the other read-out
    res["gate"] = {who: dict(at_best=gate_numbers(sweep_res[who]), at_thr0_3=gate_numbers(sweep_res[who], 0.3)) for who in ("ours", "ours_alt")}
    out_path = os.path.join(run_dir, "jmaps_val.json" if a.raster is None else f"jmaps_val_{a.raster}.json")                  # a score on re-rasterised targets never overwrites the run's own file
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    na = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731                                          # E332: n/a for the maps a head does not have
    print(f"{a.run}: VAL {n} frames; zone IoU {na(res['zone_iou'])} F1 {na(res['zone_f1'])}; core IoU {na(res['core_iou'])} F1 {na(res['core_f1'])}; stop line P/R {na(res['stop_prec'])}/{na(res['stop_rec'])}; crossing P/R {na(res['crossing_prec'])}/{na(res['crossing_rec'])}")
    print("keypoints (tol 3.0 m): head with heading | head distance only | baseline B events (distance only)")
    for typ in ("ent", "ext"):
        for thr, tag in (("ours", "thr .3"), ("ours5", "thr .5")):
            h = kp[f"kp_{thr}_tol3.0_ang_{typ}"]
            hn = kp[f"kp_{thr}_tol3.0_noang_{typ}"]
            bkey = "B" if thr == "ours" else "B5"
            bb = kp[f"kp_{bkey}_tol3.0_noang_{typ}"]
            print(f"  {typ} {tag}: R/P {h['recall']:.3f}/{h['precision']:.3f} (angle median {h['ang_median']}) | {hn['recall']:.3f}/{hn['precision']:.3f} | B {bb['recall']:.3f}/{bb['precision']:.3f}")
    for k, v in split.items():
        print(f"  recall at {k}: painted {v['painted_recall']:.3f} (n {v['painted_gt']}), unpainted {v['unpainted_recall']:.3f} (n {v['unpainted_gt']})")
    print("  best F1 over the thresholds (distance only, 3 m): ours %.3f (entry thr %s, exit thr %s), B %.3f (%s, %s)" % (res["kp_f1_best_threshold"]["ours"], sweep_res["ours"]["ent"]["best_thr"], sweep_res["ours"]["ext"]["best_thr"], res["kp_f1_best_threshold"]["B"], sweep_res["B"]["ent"]["best_thr"], sweep_res["B"]["ext"]["best_thr"]))
    for who in ("ours", "ours_alt"):
        g = res["gate"][who]["at_best"]
        print(f"  gate (E347) {who} [{res['readout'][who]} read-out, each kind at its best 3 m threshold {g['thr']}]: F1@3 {g['f1_tol3']:.3f}, F1@1.5 {g['f1_tol1_5']:.3f}, crowded recall ent/ext {g['crowded_recall']['ent']:.3f}/{g['crowded_recall']['ext']:.3f}, surplus duplicates {g['dup_share']:.4f} of the keypoints")
    for k, v in sorted(crowd.items()):
        print(f"  recall {k}: crowded {v['crowded_recall']:.3f} (n {v['crowded_n']}), isolated {v['isolated_recall']:.3f} (n {v['isolated_n']})")
    for k, v in sorted(other.items()):
        print(f"  recall {k}: {v['recall']:.3f} (n {v['n']})")
    print("written", out_path)


def selftest():
    gt = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 1.0, 0.0]])
    pred = np.array([[0.5, 0.2, 0.0, 1.0, 0.9], [10.2, 0.1, 0.0, 1.0, 0.8], [30.0, 30.0, 1.0, 0.0, 0.7]])
    pairs, npred, ngt = match(pred, gt, 1.5, True)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)] and npred == 3 and ngt == 2, pairs                  # the second one points 90 degrees off
    pairs, _, _ = match(pred, gt, 1.5, False)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 0), (1, 1)], pairs
    assert match(np.zeros((0, 5)), gt, 3.0, True)[0] == [] and match(pred, np.zeros((0, 4)), 3.0, True)[0] == []
    # E347 by hand: the last two GT ends are 2 m apart (crowded), keypoint 1 is a copy of keypoint 0 (the surplus duplicate), keypoint 3 a false alarm with no neighbour (not a duplicate)
    gt = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0], [12.0, 0.0, 0.0, 1.0]])
    pred = np.array([[0.3, 0.0, 0.0, 1.0, 0.9], [1.5, 0.0, 0.0, 1.0, 0.8], [10.2, 0.0, 0.0, 1.0, 0.7], [30.0, 0.0, 0.0, 1.0, 0.6]])
    cr = crowded_of(gt)
    assert cr.tolist() == [False, True, True], cr
    pairs = match(pred, gt, 3.0, False)[0]
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 0), (2, 1)] and surplus_duplicates(pred, pairs) == 1
    assert surplus_duplicates(pred, pairs, r=1.0) == 0 and surplus_duplicates(pred[:1], []) == 0
    assert sweep_counts(pred, gt, cr, True) == dict(tp=2, pred=4, gt=3, tp15=2, gt_c=2, hit_c=1, dup=1), sweep_counts(pred, gt, cr, True)
    pred_turned = pred.copy()
    pred_turned[2, 2:4] = (1.0, 0.0)                                                                                 # keypoint 2 points 90 degrees off: still a distance match, no heading match
    c = sweep_counts(pred_turned, gt, cr, True)
    assert c["tp"] == 2 and c["hit_c"] == 0 and sweep_counts(pred_turned, gt, cr, False)["hit_c"] == 1, c
    assert sweep_counts(pred[:, :], gt, cr, True)["tp15"] == 2 and sweep_counts(pred[2:], gt, cr, True)["tp15"] == 1
    mk = lambda thr, f1, f15, cw, nd, npd: dict(thr=thr, f1=f1, f1_tol1_5=f15, crowded_recall=cw, n_pred=npd, n_dup=nd)  # noqa: E731
    sw = dict(ent=dict(rows=[mk(0.2, 0.5, 0.3, 0.1, 4, 20), mk(0.3, 0.7, 0.4, 0.6, 1, 10)], best_thr=0.3), ext=dict(rows=[mk(0.2, 0.9, 0.8, 0.5, 2, 10), mk(0.3, 0.6, 0.5, 0.2, 0, 6)], best_thr=0.2))
    g = gate_numbers(sw)                                                                                              # every kind at its OWN best threshold: ent 0.3, ext 0.2
    assert g["thr"] == dict(ent=0.3, ext=0.2) and abs(g["f1_tol3"] - 0.8) < 1e-9 and abs(g["f1_tol1_5"] - 0.6) < 1e-9 and g["crowded_recall"] == dict(ent=0.6, ext=0.5) and abs(g["dup_share"] - 3 / 20) < 1e-9, g
    g3 = gate_numbers(sw, 0.3)
    assert abs(g3["f1_tol3"] - 0.65) < 1e-9 and abs(g3["dup_share"] - 1 / 16) < 1e-9 and g3["thr"] == dict(ent=0.3, ext=0.3), g3
    print("jmaps_eval selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--base", default=None)
    s.add_argument("--max", type=int, default=0)
    s.add_argument("--raster", default=None, choices=("legacy", "exact"), help="score the zone / core / paint IoU against targets rasterised this way (default: as the run was trained)")
    s.add_argument("--workers", type=int, default=3)
    sub.add_parser("selftest")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else score(a)
