"""Loop 18 / R2.1: rough localisation (K188) and map trust (K189) on the VAL failure series of loop 17.
For every series (pert_variant, 291 rows, 22 scenes, first half clean, second half failure) and every map condition (scene-constant SD pose error sigma in m, or a wrong map) the dense evidence of the
series input (channels 39 heat, 40-41 direction of the 42-channel cache, i.e. after the hold of the variant) is registered to the SD raster; the single-frame estimates are accumulated causally per scene
(world-frame translation + yaw, confidence-weighted running mean, p18_regis.accumulate).
Prints pose errors per phase (single frame, accumulated, no correction) and the trust of right / wrong maps; writes runs/e18/<out>/<tag>_<cond>.npz with the corrections and the trust that
p18_regis.RegMapWrap (builder evaluation, p17_eval.py --reg) reads.
Usage (gcar, GPU < 2 GB): p18_regis_eval.py --regis regis_a --out regis_eval_a --tags dark_maxdecay_0.95,blur_maxdecay_0.95 --sigmas 1,2,3 [--wrong 1]"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p18_regis as RG  # noqa: E402
import p8_lib as L  # noqa: E402
import sd_lib as SL  # noqa: E402

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"
OUT = BP.RUNS


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    rk = allv.argsort().argsort() + 1.0
    return float((rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def smooth(vals, scene, lam=0.9):
    """causal running mean per scene with forgetting lam (frames in time order)"""
    out, st = np.zeros(len(vals)), {}
    for i, v in enumerate(vals):
        s = int(scene[i])
        a, b = st.get(s, (0.0, 0.0))
        a, b = a * lam + v, b * lam + 1.0
        st[s] = (a, b)
        out[i] = a / b
    return out


def register_series(net, tag, cond, P, sdl, wrong_stub, bs=16, conf_mode="pt", lam=0.9):
    z = np.load(f"{D}/p17/{tag}.meta.npz")
    rows, scene, phase = z["rows"], z["scene"], z["phase"]
    x = np.load(f"{D}/p17/{tag}.npy", mmap_mode="r")
    n = len(rows)
    ras, truth = [], np.zeros((n, 3), np.float32)
    for i, r in enumerate(rows):
        r = int(r)
        if cond == "wrong":
            rng = np.random.RandomState((zlib.crc32(f"{r}:wrong".encode()) + 0) % (2 ** 32))
            ras.append(wrong_stub.raster(r, rng, "wrong", 0.0))
        else:
            sg = float(cond[1:])
            ras.append(wrong_stub.raster(r, np.random.RandomState((zlib.crc32(f"{r}:exact".encode()) + 0) % (2 ** 32)), "exact", sg).astype(np.float32))  # same draw as MapWrap (sigma 0: draw_sd, sigma > 0: sd_raster)
            w, th = SL.scene_pose_error(int(P["scene"][r]), sg)
            d = P["R"][r].T @ w
            truth[i] = [th, d[0], d[1]]
    outs = {k: [] for k in ("th", "t", "pmax", "rho", "trust")}
    with torch.no_grad():
        for s in range(0, n, bs):
            xb = torch.from_numpy(np.asarray(x[s : s + bs], np.float32)).cuda()
            mp = torch.from_numpy(np.stack(ras[s : s + bs])).cuda()
            o = net(xb[:, 39], xb[:, 40:42], mp)
            for k in outs:
                outs[k].append(o[k].cpu().numpy())
    o = {k: np.concatenate(v) for k, v in outs.items()}
    Rego = np.stack([P["R"][int(r)] for r in rows])
    conf = o["trust"] * (o["pmax"] if conf_mode == "pt" else 1.0)
    th_a, t_a, c_a = RG.accumulate(rows, scene, o["th"], o["t"], conf, Rego, lam=lam)
    trust_a = smooth(o["trust"], scene, lam)
    return dict(rows=rows, scene=scene, phase=phase, truth=truth, th=o["th"], t=o["t"], pmax=o["pmax"], rho=o["rho"], trust=o["trust"], conf=conf, th_acc=th_a, t_acc=t_a, c_acc=c_a, trust_acc=trust_a)


def errs(r, key_th, key_t, mask):
    et = np.linalg.norm(r[key_t][mask] - r["truth"][mask][:, 1:3], axis=1)
    ea = np.abs(r[key_th][mask] - r["truth"][mask][:, 0])
    return et, ea


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regis", default="regis_a")
    ap.add_argument("--out", default="regis_eval_a")
    ap.add_argument("--tags", default="dark_maxdecay_0.95,blur_maxdecay_0.95,cam_front3_off_maxdecay_0.95")
    ap.add_argument("--sigmas", default="1,2,3")
    ap.add_argument("--wrong", type=int, default=1)
    ap.add_argument("--conf", default="pt", help="pt = trust x peak probability | t = trust")
    ap.add_argument("--lam", type=float, default=0.9, help="forgetting per frame of the accumulation")
    a = ap.parse_args()
    net = RG.SoftRegister().cuda().eval()
    net.load_state_dict(torch.load(f"{OUT}/e18/{a.regis}/regis.pt", map_location="cuda"))
    P = dict(np.load(f"{D}/poses_val.npz"))
    sdl = L.cache("sd", "val")
    import p17_mapnet as MN

    class Stub:
        gt = rows = ks = None

    wrong_stub = MN.MapWrap(Stub(), "val", P, train=False, mode="wrong")
    wrong_stub.sdl = sdl
    od = f"{OUT}/e18/{a.out}"
    os.makedirs(od, exist_ok=True)
    summ = {}
    conds = [f"s{s}" for s in a.sigmas.split(",")] + (["wrong"] if a.wrong else [])
    for tag in a.tags.split(","):
        for cond in conds:
            r = register_series(net, tag, cond, P, sdl, wrong_stub, conf_mode=a.conf, lam=a.lam)
            np.savez(f"{od}/{tag}_{cond}.npz", **r)
            ph = r["phase"]
            row = {}
            if cond != "wrong":
                sg = float(cond[1:])
                for nm, m in (("cl", ph == 0), ("fail", ph == 1)):
                    e0 = np.linalg.norm(r["truth"][m][:, 1:3], axis=1)
                    et, ea = errs(r, "th", "t", m)
                    et2, ea2 = errs(r, "th_acc", "t_acc", m)
                    seen = r["c_acc"][m] > 1e-6
                    row[nm] = dict(none_t=float(np.mean(e0)), none_a=float(np.mean(np.abs(r["truth"][m][:, 0]))), single_t=float(np.mean(et)), single_a=float(np.mean(ea)), single_med_t=float(np.median(et)),
                                   acc_t=float(np.mean(et2[seen])), acc_a=float(np.mean(ea2[seen])), acc_med_t=float(np.median(et2[seen])), acc_p90_t=float(np.percentile(et2[seen], 90)), seen=float(seen.mean()),
                                   trust=float(np.mean(r["trust"][m])))
                print(f"[{tag} {cond}] " + " | ".join(f"{nm}: err m none {v['none_t']:.2f} single {v['single_t']:.2f} (med {v['single_med_t']:.2f}) acc {v['acc_t']:.2f} (med {v['acc_med_t']:.2f}, p90 {v['acc_p90_t']:.2f}); deg none {v['none_a']:.2f} single {v['single_a']:.2f} acc {v['acc_a']:.2f}; trust {v['trust']:.2f}" for nm, v in row.items()), flush=True)
            else:
                for nm, m in (("cl", ph == 0), ("fail", ph == 1)):
                    row[nm] = dict(trust=float(np.mean(r["trust"][m])), trust_acc=float(np.mean(r["trust_acc"][m])))
                print(f"[{tag} wrong] " + " | ".join(f"{nm}: trust {v['trust']:.2f} acc {v['trust_acc']:.2f}" for nm, v in row.items()), flush=True)
            summ[f"{tag}_{cond}"] = row
        if a.wrong and any(c != "wrong" for c in conds):
            rr = np.load(f"{od}/{tag}_s{a.sigmas.split(',')[0]}.npz")
            rw = np.load(f"{od}/{tag}_wrong.npz")
            for nm, ph in (("cl", 0), ("fail", 1)):
                m = rr["phase"] == ph
                summ[f"{tag}_auroc_{nm}"] = dict(frame=auroc(rr["trust"][m], rw["trust"][m]), acc=auroc(rr["trust_acc"][m], rw["trust_acc"][m]))
            print(f"[{tag}] AUROC right vs wrong map: " + " | ".join(f"{nm}: frame {summ[f'{tag}_auroc_{nm}']['frame']:.3f} acc {summ[f'{tag}_auroc_{nm}']['acc']:.3f}" for nm in ("cl", "fail")), flush=True)
    json.dump(summ, open(f"{od}/summary.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
