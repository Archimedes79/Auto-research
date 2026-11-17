"""scratch: patch the dev copy of jmaps_eval.py so that a head without the paint branch / zone supervision (E332) is scored (the zone / core / stop / crossing numbers are then n/a)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = '''            jz, jp = b["jz"].to(dev).float(), b["jpaint"].to(dev).float()
            pz_all = (out["j1"][:, :2].float().sigmoid() > 0.5).float()
            for c, name in enumerate(("zone", "core")):
                pz, tz = pz_all[:, c:c + 1], jz[:, c:c + 1]
                acc.add(name, tp=float((pz * tz).sum()), fp=float((pz * (1 - tz)).sum()), fn=float(((1 - pz) * tz).sum()))
            pp = (out["j05"].float().sigmoid() > 0.5).float()
            for c, name in enumerate(("stop", "crossing")):
                pr, tg = pp[:, c:c + 1], jp[:, c:c + 1]
                acc.add(name, pred=float(pr.sum()), pred_ok=float((pr * _dil(tg)).sum()), gt=float(tg.sum()), gt_ok=float((tg * _dil(pr)).sum()))
'''
b = '''            jz, jp = b["jz"].to(dev).float(), b["jpaint"].to(dev).float()
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
'''
assert a in s
s = s.replace(a, b)
a = '''    for name in ("zone", "core"):
        z = acc.k[name]
        res[f"{name}_iou"] = z["tp"] / max(z["tp"] + z["fp"] + z["fn"], 1)
        res[f"{name}_f1"] = 2 * z["tp"] / max(2 * z["tp"] + z["fp"] + z["fn"], 1)
    for name in ("stop", "crossing"):
        d = acc.k[name]
        res[f"{name}_prec"], res[f"{name}_rec"] = d["pred_ok"] / max(d["pred"], 1), d["gt_ok"] / max(d["gt"], 1)
'''
b = '''    for name in ("zone", "core"):
        z = acc.k.get(name)
        res[f"{name}_iou"] = None if z is None else z["tp"] / max(z["tp"] + z["fp"] + z["fn"], 1)
        res[f"{name}_f1"] = None if z is None else 2 * z["tp"] / max(2 * z["tp"] + z["fp"] + z["fn"], 1)
    for name in ("stop", "crossing"):
        d = acc.k.get(name)
        res[f"{name}_prec"], res[f"{name}_rec"] = (None, None) if d is None else (d["pred_ok"] / max(d["pred"], 1), d["gt_ok"] / max(d["gt"], 1))
'''
assert a in s
s = s.replace(a, b)
a = '''    print(f"{a.run}: VAL {n} frames; zone IoU {res['zone_iou']:.3f} F1 {res['zone_f1']:.3f}; core IoU {res['core_iou']:.3f} F1 {res['core_f1']:.3f}; stop line P/R {res['stop_prec']:.3f}/{res['stop_rec']:.3f}; crossing P/R {res['crossing_prec']:.3f}/{res['crossing_rec']:.3f}")
'''
b = '''    na = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731                                          # E332: n/a for the maps a head does not have
    print(f"{a.run}: VAL {n} frames; zone IoU {na(res['zone_iou'])} F1 {na(res['zone_f1'])}; core IoU {na(res['core_iou'])} F1 {na(res['core_f1'])}; stop line P/R {na(res['stop_prec'])}/{na(res['stop_rec'])}; crossing P/R {na(res['crossing_prec'])}/{na(res['crossing_rec'])}")
'''
assert a in s
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("patched", p)
