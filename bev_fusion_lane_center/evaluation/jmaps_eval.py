"""Score of the junction maps head (E286): maps and entry / exit keypoints on the VAL frames, next to the baseline B's own event heat maps read the same way.

    wsl bash tools/env/_py14.sh jmaps_eval.py score --run e286_j0_s0 [--base runs/e273_p_s0/model.pt] [--max 0]     -> runs/<run>/jmaps_val.json + a printed table
    wsl bash tools/env/_py14.sh jmaps_eval.py selftest

Numbers (VAL 873 frames or the first --max, scene-disjoint from TRAIN, TEST never; every frame counts, frames without a junction carry the false alarms):
  zone               IoU and F1 of the junction zone at p > 0.5 (1 m cells)
  stop / crossing    precision and recall of the painted stop line / pedestrian crossing masks with a tolerance of one 0.5 m cell
  keypoints          entry and exit points: a prediction is a 3 x 3 local maximum of the heat above `thr` (soft-argmax position); it matches a GT end within `tol` metres and, for the head with a heading,
                     within 30 degrees (greedy by distance); recall = matched GT ends / GT ends, precision = matched predictions / predictions; thr 0.3 and 0.5, tol 1.5 and 3.0 m (D79 used 3 m, 30 degrees)
  baseline B         the stop / exit event heat maps of the baseline's pixel stage (centre1 channels 8 / 9), the same extraction, distance only (B has no heading)
  heading            median angle error (degrees) at the matched predictions
  painted split      recall of the entries / exits that are PAINTED (a stop line of another type than the turn stop within 3 m, or a crossing within 4 m: `qjunction.painted_flags`) and of the rest (thr 0.3, 3 m)
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
    """(painted flags of the GT entries, of the GT exits, city) of a frame, aligned with the batch's GT ends; (None, None, city) when the paint sidecar or the junction GT is missing or does not line up"""
    city = None
    try:
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
    except Exception:
        return None, None, city


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
    model = JM.JModel(width=48, d=args.get("d", 64), ctx=args.get("ctx", "dil"))
    model.load_state_dict(ck["state"])
    base_path = a.base or args.get("base") or os.path.join(ROOT, "runs", f"e273_p_s{args.get('seed', 0)}", "model.pt")
    basem = JM.JModel(width=48, d=64, ctx="none")
    basem.load_base(base_path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    basem.to(dev).eval()
    ds, dl = LD.loader("val_clean", bs=8, workers=a.workers, shuffle=False, map_mode="exact", pixels=False)
    acc = Acc()
    n = 0
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(dev).float()
            P, out = model(x, joint=False)
            Pb = basem.pyr(x)
            jz, jp = b["jz"].to(dev).float(), b["jpaint"].to(dev).float()
            pz_all = (out["j1"][:, :2].float().sigmoid() > 0.5).float()
            for c, name in enumerate(("zone", "core")):
                pz, tz = pz_all[:, c:c + 1], jz[:, c:c + 1]
                acc.add(name, tp=float((pz * tz).sum()), fp=float((pz * (1 - tz)).sum()), fn=float(((1 - pz) * tz).sum()))
            pp = (out["j05"].float().sigmoid() > 0.5).float()
            for c, name in enumerate(("stop", "crossing")):
                pr, tg = pp[:, c:c + 1], jp[:, c:c + 1]
                acc.add(name, pred=float(pr.sum()), pred_ok=float((pr * _dil(tg)).sum()), gt=float(tg.sum()), gt_ok=float((tg * _dil(pr)).sum()))
            # baseline B: event heat stop / exit of the pyramid, distance only
            ev = Pb["centre1"][:, 6:12].float()
            fake = dict(j1=torch.cat([torch.zeros_like(ev[:, :2]), ev[:, 2:3], ev[:, 3:4], torch.zeros_like(ev[:, :1]).repeat(1, 4, 1, 1)], 1))
            kps = {"ours": JM.keypoints(out, thr=0.3), "ours5": JM.keypoints(out, thr=0.5), "B": JM.keypoints(fake, thr=0.3), "B5": JM.keypoints(fake, thr=0.5)}
            for i in range(x.shape[0]):
                ent_g = b["ent"][i][b["ent_m"][i]].numpy()
                ext_g = b["ext"][i][b["ext_m"][i]].numpy()
                pe, px, city = painted_of(b["token"][i], len(ent_g), len(ext_g))
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
                                    rc = rc_e if typ == "ent" else rc_x
                                    for k, nm in enumerate(("lt20", "20to35", "gt35")):                                  # range from the ego, E300: the far ends are the hole
                                        acc.add(f"range_{who}_{typ}_{nm}", n=int((rc == k).sum()), hit=int(hit[rc == k].sum()))
                                    if city:
                                        acc.add(f"city_{who}_{typ}_{city}", n=len(gt), hit=int(hit.sum()))
            n += x.shape[0]
            if a.max and n >= a.max:
                break
    res = dict(run=a.run, base=base_path, frames=n, args=args)
    for name in ("zone", "core"):
        z = acc.k[name]
        res[f"{name}_iou"] = z["tp"] / max(z["tp"] + z["fp"] + z["fn"], 1)
        res[f"{name}_f1"] = 2 * z["tp"] / max(2 * z["tp"] + z["fp"] + z["fn"], 1)
    for name in ("stop", "crossing"):
        d = acc.k[name]
        res[f"{name}_prec"], res[f"{name}_rec"] = d["pred_ok"] / max(d["pred"], 1), d["gt_ok"] / max(d["gt"], 1)
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
    out_path = os.path.join(run_dir, "jmaps_val.json")
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    print(f"{a.run}: VAL {n} frames; zone IoU {res['zone_iou']:.3f} F1 {res['zone_f1']:.3f}; core IoU {res['core_iou']:.3f} F1 {res['core_f1']:.3f}; stop line P/R {res['stop_prec']:.3f}/{res['stop_rec']:.3f}; crossing P/R {res['crossing_prec']:.3f}/{res['crossing_rec']:.3f}")
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
    print("jmaps_eval selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--base", default=None)
    s.add_argument("--max", type=int, default=0)
    s.add_argument("--workers", type=int, default=3)
    sub.add_parser("selftest")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else score(a)
