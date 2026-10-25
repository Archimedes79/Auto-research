"""Schleife 5 / E60: Routen-bedingter dichter Kopf (Ende-zu-Ende-Loesung 'Karte + Navigationsziel -> Routen-Spurmitte' ohne Handarbeit).
Eingaben: eingefrorene Merkmale (39) + Karten-Raster der Spurmitten (4 Kanaele: Route, alle Karten-Spuren, Spurrichtung dx, dy) mit Pose-Rauschen / veralteter Karte / fehlender Karte / falscher Karte (Training zufaellig, Auswertung fest je Sample).
Ausgaben: Spur-Waermekarte + Richtung (wie bisher) und ROUTEN-Waermekarte (die Spuren, die zum Navigationsziel fuehren). Start aus e37_final (neue Kanaele null-initialisiert).
Aufruf (gcar): route_head.py --name e60_main --steps 4000 [--init_from e37_final] [--maps route+lanes|route|none] ...   Ausgabe wie dense_head: out/loop2/<name>/{model.pt, metrics.json, prob_*.npy (Spur), rprob_*.npy (Route), dir_*.npy, ids_*.npy}"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import copy
import json
import os
import pickle
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_lib as AL  # noqa: E402
import dense_head as DH  # noqa: E402

HOME, OUT = DH.HOME, DH.OUT
D = f"{HOME}/exp_gcar_b123"
N_MAP = 4


def raster(pieces, tans_from=None, n=200, half=50.0, ds=0.25):
    """Stuecke (Liste Polylinien im Fahrzeugsystem) -> Maske [n,n] und Richtung [2,n,n] (Tangente x, y; Zeile = x, Spalte = y)."""
    m = np.zeros((n, n), np.float32)
    dr = np.zeros((2, n, n), np.float32)
    for p in pieces:
        q = AL.resample(np.asarray(p, np.float64), ds)
        if len(q) < 2:
            continue
        t = AL.tangents(q)
        r = np.round((q[:, 0] + half) / (2 * half / n) - 0.5).astype(int)
        c = np.round((q[:, 1] + half) / (2 * half / n) - 0.5).astype(int)
        ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
        m[r[ok], c[ok]] = 1.0
        dr[0][r[ok], c[ok]] = t[ok, 0]
        dr[1][r[ok], c[ok]] = t[ok, 1]
    return m, dr


class RouteData(torch.utils.data.Dataset):
    def __init__(self, part, indices, train, cfg):
        self.part, self.train, self.cfg = part, train, cfg
        self.base = DH.Data("gcar", ["dec32", "logit7"], part, indices, "b123", target=train, with_dir=train)
        self.G = pickle.load(open(f"{D}/graph_gt_{part}.pkl", "rb"))
        self.RT = pickle.load(open(f"{D}/route_gt_{part}.pkl", "rb"))
        self.rh = None
        self.n_all = len(self.G)
        self.ctr = 0
        self.pf = None
        if cfg.get("pert") and not train:
            d_ = f"{D}/pert/{cfg['pert']}"
            self.pf = (np.load(f"{d_}/test_dec32.npy", mmap_mode="r"), np.load(f"{d_}/test_logit7.npy", mmap_mode="r"), np.load(f"{d_}/test_rows.npy"))
        self.pt = None  # E65d: gestoerte TRAIN-Merkmale (Stoerung -> (dec, logit, Zeile -> Index)), traege geoeffnet
        self.pt_names = [p for p in str(cfg.get("pert_train", "")).split(",") if p] if train else []

    def _open_pert(self):
        self.pt = {}
        for p in self.pt_names:
            d_ = f"{D}/pert/{p}"
            rows = np.load(f"{d_}/train_rows.npy")
            self.pt[p] = (np.load(f"{d_}/train_dec32.npy", mmap_mode="r"), np.load(f"{d_}/train_logit7.npy", mmap_mode="r"), {int(r): j for j, r in enumerate(rows)})

    def __len__(self):
        return len(self.base)

    def _rng(self, i):
        if self.train:
            self.ctr += 1
            return np.random.RandomState((torch.initial_seed() % (2 ** 31) + 7919 * int(i) + self.ctr) % (2 ** 32))
        return np.random.RandomState((zlib.crc32(f"{self.part}{i}".encode()) + 17) % (2 ** 32))

    def __getitem__(self, k):
        if self.rh is None and self.train:
            self.rh = np.load(f"{D}/route_heat_{self.part}.npy", mmap_mode="r")
        it = self.base[k]
        i = int(self.base.idx[k])
        c = self.cfg
        if self.pf is not None:
            r_ = int(np.nonzero(self.pf[2] == i)[0][0])
            it = (torch.cat([torch.from_numpy(np.asarray(self.pf[0][r_], np.float32)), torch.from_numpy(np.asarray(self.pf[1][r_], np.float32))], 0), i)
        rng = self._rng(i)
        if self.train and self.pt_names:  # E65d: mit Wahrscheinlichkeit p_use die ECHT gestoerte Version der Zeile (falls vorhanden)
            if self.pt is None:
                self._open_pert()
            cands = [p for p in self.pt_names if i in self.pt[p][2]]
            if cands and rng.rand() < c.get("p_use_pert", 0.85):
                p = cands[int(rng.randint(len(cands)))]
                j = self.pt[p][2][i]
                xp = torch.cat([torch.from_numpy(np.asarray(self.pt[p][0][j], np.float32)), torch.from_numpy(np.asarray(self.pt[p][1][j], np.float32))], 0)
                it = (xp,) + tuple(it[1:])
        # Stoerung der Karten-Eingabe
        if self.train:
            u = rng.rand()
            sxy = 0.0 if u < 0.3 else rng.uniform(0.3, 2.0) * (1.0 if c["noise_train"] else 0.0)
            drop = rng.uniform(0.0, 0.4) if rng.rand() < 0.5 else 0.0
            wrong = rng.rand() < c["p_wrong"]
            none_ = rng.rand() < c["p_none"]
        else:
            sxy, drop, wrong, none_ = c["eval_sxy"], c["eval_drop"], c["eval_wrong"], c["eval_none"]
        src = i if not wrong else (i + self.n_all // 2) % self.n_all
        g, rt = self.G[src], self.RT[src]
        maps = np.zeros((N_MAP, 200, 200), np.float32)
        has = 1.0 if self.RT[i] is not None else 0.0
        if not none_ and rt is not None and c["maps"] != "none":
            d = rng.randn(2) * sxy
            R = AL.rot2(rng.randn() * 2.0 * sxy)
            tf = lambda p: np.asarray(p, np.float64) @ R.T + d
            route_set = set(rt["route"])
            rpieces = [tf(g["pts"][p]) for p in rt["route"] if len(g["pts"][p]) >= 2]
            m_r, _ = raster(rpieces)
            maps[0] = m_r
            if c["maps"] == "route+lanes":
                lp = [tf(p) for q_, p in enumerate(g["pts"]) if len(p) >= 2 and (rng.rand() >= drop)]
                m_l, dr_l = raster(lp)
                maps[1], maps[2:4] = m_l, dr_l
        o = dict(x=it[0], maps=torch.from_numpy(maps), idx=i, has=torch.tensor(has))
        if self.train:
            o["y"], o["dirf"] = it[1], it[2]
            o["yr"] = torch.from_numpy(np.asarray(self.rh[i], np.float32) / 255.0)
        return o


def make_net(init_from, base=48):
    ck = torch.load(f"{OUT}/{init_from}/model.pt", map_location="cuda", weights_only=False) if init_from else None
    cin = 39 + N_MAP
    net = DH.UNet(cin, base, 4, 0.0).cuda()
    mean = torch.zeros(cin)
    std = torch.ones(cin)
    if ck is not None:
        own = net.state_dict()
        for k_, v in ck["state"].items():
            if k_ not in own:
                continue
            if own[k_].shape == v.shape:
                own[k_] = v
            elif v.dim() == 4 and own[k_].shape[0] == v.shape[0] and own[k_].shape[1] > v.shape[1]:  # erste Faltung: neue Eingangskanaele null
                nw = torch.zeros_like(own[k_])
                nw[:, : v.shape[1]] = v
                own[k_] = nw
            elif v.shape[0] == 3 and own[k_].shape[0] == 4:  # letzte 1x1-Faltung: vierte Ausgabe (Route) klein, Bias negativ
                nw = own[k_].clone()
                nw[:3] = v
                if nw.dim() == 4:
                    nw[3] = 0.01 * torch.randn_like(nw[3])
                else:
                    nw[3] = -3.0
                own[k_] = nw
        net.load_state_dict(own)
        mean[:39], std[:39] = ck["mean"][:39], ck["std"][:39]
    return net, mean.cuda(), std.cuda()


def prep(b, mean, std, dev="cuda"):
    x = torch.cat([b["x"].to(dev, non_blocking=True), b["maps"].to(dev, non_blocking=True)], 1)
    return (x - mean[None, :, None, None]) / std[None, :, None, None]


@torch.no_grad()
def predict(net, ds, mean, std, bs=16, workers=4):
    net.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    P_, R_, D_, I_ = [], [], [], []
    for b in dl:
        x = prep(b, mean, std)
        with torch.autocast("cuda", dtype=torch.float16):
            o = net(x).float()
        P_.append(torch.sigmoid(o[:, 0]).half().cpu())
        D_.append(o[:, 1:3].half().cpu())
        R_.append(torch.sigmoid(o[:, 3]).half().cpu())
        I_.append(b["idx"])
    return torch.cat(P_), torch.cat(R_), torch.cat(D_), torch.cat(I_).numpy()


def route_metrics(rprob, ids, part, taus=(0.3, 0.4, 0.5, 0.6), dev="cuda"):
    """Routen-Waermekarte -> Punkte (Ausduennen) -> Abdeckung / Praezision @1,5 m gegen die Soll-Routenpunkte + Erfolgsrate (wie assoc_eval); Punkt-F1 @1 m."""
    from scipy.spatial import cKDTree

    pts = pickle.load(open(f"{D}/route_pts_{part}.pkl", "rb"))
    res = {}
    for tau in taus:
        cov = n_gt = prec = n_pr = 0
        succ = []
        tp = fp = fn = 0
        for s in range(0, len(ids), 32):
            sk = DH.skeleton_points(rprob[s : s + 32].float().to(dev), tau)
            for kk, P in enumerate(sk):
                g = pts[int(ids[s + kk])]
                if g is None:
                    continue
                P = np.asarray(P, np.float64)
                if len(P) == 0:
                    n_gt += len(g)
                    succ.append(0)
                    fn += len(g)
                    continue
                d1 = cKDTree(P).query(g)[0]
                d2 = cKDTree(g).query(P)[0]
                c1, c2 = int((d1 <= 1.5).sum()), int((d2 <= 1.5).sum())
                cov, n_gt, prec, n_pr = cov + c1, n_gt + len(g), prec + c2, n_pr + len(P)
                succ.append(int(c1 / len(g) >= 0.8 and c2 / len(P) >= 0.8))
                tp1, fp1, fn1 = int((d2 <= 1.0).sum()), int((d2 > 1.0).sum()), int((d1 > 1.0).sum())
                tp, fp, fn = tp + tp1, fp + fp1, fn + fn1
        res[str(tau)] = dict(cov=cov / max(n_gt, 1), prec=prec / max(n_pr, 1), success=float(np.mean(succ)) if succ else 0.0, P1=tp / max(tp + fp, 1), R1=tp / max(tp + fn, 1), F1=2 * tp / max(2 * tp + fp + fn, 1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--wpos", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--dir", type=float, default=0.5)
    ap.add_argument("--wroute", type=float, default=1.0)
    ap.add_argument("--maps", default="route+lanes", choices=["route+lanes", "route", "none"])
    ap.add_argument("--noise_train", type=int, default=1)
    ap.add_argument("--p_wrong", type=float, default=0.1)
    ap.add_argument("--p_none", type=float, default=0.15)
    ap.add_argument("--init_from", default="e37_final")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", default="val,test")
    ap.add_argument("--conds", default="clean:0:0,s1:1:0,s2:2:0,d30:0:0.3,s1d30:1:0.3,wrong:0:0:w,none:0:0:n", help="Auswertungsbedingungen name:sigma_m:drop[:w|n]")
    ap.add_argument("--eval_only", type=int, default=0)
    ap.add_argument("--pert", default="", help="E57: Auswertung auf den gestoerten Merkmalen der TEST-Zeilen (pert/<name>)")
    ap.add_argument("--save_pert", type=int, default=0, help="E65d: bei --pert die Spur-Waermekarten der Bedingungen speichern (Abbildung)")
    ap.add_argument("--pert_train", default="", help="E65d: Liste gestoerter TRAIN-Merkmale (gcar_pert.py --part train), die im Training beigemischt werden")
    ap.add_argument("--p_pert_mass", type=float, default=0.55, help="E65d: Anteil der Ziehungen, die auf Zeilen mit gestoerter Version fallen")
    ap.add_argument("--p_use_pert", type=float, default=0.85, help="E65d: Wahrscheinlichkeit, bei einer solchen Zeile die gestoerte Version zu nehmen")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(f"{OUT}/{a.name}", exist_ok=True)
    log = open(f"{OUT}/{a.name}/log.txt", "a")

    def P(*s):
        t = " ".join(str(x) for x in s)
        print(t, flush=True)
        log.write(t + "\n")
        log.flush()

    P("==", time.strftime("%H:%M:%S"), vars(a))
    sp = json.load(open(BP.ROOT + "/data/splits.json"))
    RTtr = pickle.load(open(f"{D}/route_gt_train.pkl", "rb"))
    tr_idx = np.array([i for i in range(sp["n"]["train"]) if RTtr[i] is not None])
    cfg = dict(maps=a.maps, noise_train=a.noise_train, p_wrong=a.p_wrong, p_none=a.p_none, eval_sxy=0.0, eval_drop=0.0, eval_wrong=False, eval_none=False, pert=a.pert, pert_train=a.pert_train, p_use_pert=a.p_use_pert)
    ds = RouteData("train", tr_idx, True, cfg)
    net, mean, std = make_net(a.init_from)
    P("Parameter", sum(p.numel() for p in net.parameters()), "Trainings-Samples mit Route", len(ds))
    step, t0, run, curve = 0, time.time(), [], []
    gema, n_clip, best_m, snap, n_roll, lr_scale, warm_start = None, 0, None, None, 0, 1.0, None
    if a.steps > 0 and not a.eval_only:
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd, eps=a.eps, betas=(0.9, a.beta2))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
        scaler = torch.cuda.amp.GradScaler(growth_interval=200)
        torch.backends.cudnn.benchmark = True
        sampler = None
        if a.pert_train:  # E65d: Zeilen mit gestoerter Version bekommen p_pert_mass der Ziehungen
            prow = set()
            for pn in a.pert_train.split(","):
                prow |= {int(r) for r in np.load(f"{D}/pert/{pn}/train_rows.npy")}
            isp = np.array([int(i) in prow for i in tr_idx])
            w = np.where(isp, a.p_pert_mass / max(int(isp.sum()), 1), (1 - a.p_pert_mass) / max(int((~isp).sum()), 1))
            sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(w), num_samples=a.steps * a.bs + 64, replacement=True)
            P("E65d: gestoerte Zeilen", int(isp.sum()), "von", len(isp), "; Massenanteil", a.p_pert_mass)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=(sampler is None), sampler=sampler, num_workers=a.workers, drop_last=True, persistent_workers=True, prefetch_factor=4)
        net.train()
        while step < a.steps:
            for b in dl:
                x = prep(b, mean, std)
                y, dg, yr = b["y"].cuda(non_blocking=True), b["dirf"].cuda(non_blocking=True), b["yr"].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    o = net(x)
                o = o.float()
                loss = DH.loss_fn(o[:, 0], y, a.lam, a.wpos, a.gamma) + a.dir * DH.dir_loss(o[:, 1:3], dg) + a.wroute * DH.loss_fn(o[:, 3], yr, a.lam, a.wpos, a.gamma)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf")))
                if np.isfinite(gn):
                    gema = gn if gema is None else gema
                    lim = 3.0 * gema
                    if gn > lim:
                        for p_ in net.parameters():
                            if p_.grad is not None:
                                p_.grad.mul_(lim / gn)
                        n_clip += 1
                    gema = 0.98 * gema + 0.02 * min(gn, lim)
                scaler.step(opt)
                scaler.update()
                sched.step()
                mult = lr_scale * (min(1.0, 0.1 + 0.9 * (step - warm_start) / 200.0) if warm_start is not None else 1.0)
                if mult != 1.0:
                    for g_ in opt.param_groups:
                        g_["lr"] *= mult
                run.append(float(loss))
                step += 1
                if step % 200 == 0:
                    curve.append((step, float(np.mean(run[-200:]))))
                    P(f"Schritt {step}/{a.steps} Verlust {curve[-1][1]:.4f} (max {max(run[-200:]):.1f}) ({(time.time() - t0) / step:.2f} s/Schritt) geklippt {n_clip}")
                    if step >= 400:
                        cur = curve[-1][1]
                        best_m = cur if best_m is None else min(best_m, cur)
                        if cur > 1.2 * best_m and snap is not None and n_roll < 6:
                            net.load_state_dict(snap[0])
                            opt.state.clear()
                            scaler.load_state_dict(snap[1])
                            lr_scale *= 0.7
                            warm_start = step
                            n_roll += 1
                            P(f"ROLLBACK bei Schritt {step}")
                        elif cur <= 1.05 * best_m:
                            snap = (copy.deepcopy(net.state_dict()), copy.deepcopy(scaler.state_dict()))
                if step >= a.steps:
                    break
        torch.save(dict(state=net.state_dict(), mean=mean.cpu(), std=std.cpu(), args=vars(a)), f"{OUT}/{a.name}/model.pt")
    elif a.eval_only:
        ck = torch.load(f"{OUT}/{a.name}/model.pt", map_location="cuda", weights_only=False)
        net.load_state_dict(ck["state"])
        mean, std = ck["mean"].cuda(), ck["std"].cuda()
    res = dict(args=vars(a), curve=curve, train_s=time.time() - t0, clipped_steps=n_clip, rollbacks=n_roll)
    parts = {"val": ("val", sp["val"]), "test": ("val", sp["test"]), "train": ("train", sp["train_eval"])}
    RTv = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    for nm in a.eval.split(","):
        part, idx = parts[nm]
        RTe = RTtr if part == "train" else RTv
        idx = np.array([i for i in idx if RTe[i] is not None])
        if a.pert and part == "val":  # gestoerte Merkmale liegen nur fuer (ggf. jede zweite) TEST-Zeile vor
            have = set(np.load(f"{D}/pert/{a.pert}/test_rows.npy").tolist())
            idx = np.array([i for i in idx if int(i) in have])
        for cond in a.conds.split(","):
            f = cond.split(":")
            cname, sxy, dr = f[0], float(f[1]), float(f[2])
            flag = f[3] if len(f) > 3 else ""
            cfg_e = dict(cfg, eval_sxy=sxy, eval_drop=dr, eval_wrong=(flag == "w"), eval_none=(flag == "n"))
            dse = RouteData(part, idx, False, cfg_e)
            prob, rprob, dp, ids = predict(net, dse, mean, std, workers=a.workers)
            gts = DH.load_gt_points("b123", part, ids)
            lane = DH.eval_probs(prob, gts, taus=(0.5, 0.6))
            rm = route_metrics(rprob, ids, part)
            res[f"{nm}_{cname}"] = dict(lane=lane, route=rm)
            b_ = rm["0.5"]
            P(f"[{nm}/{cname}] {len(ids)} Samples | Spur-F1@1 m tau .5/.6: {lane['0.5']['F1@1.0']:.3f}/{lane['0.6']['F1@1.0']:.3f} | Route tau .5: cov {b_['cov']:.3f} prec {b_['prec']:.3f} success {b_['success']:.3f} F1@1m {b_['F1']:.3f}")
            if a.pert and a.save_pert:
                np.save(f"{OUT}/{a.name}/prob_pert_{a.pert}_{cname}.npy", prob.numpy())
                np.save(f"{OUT}/{a.name}/ids_pert_{a.pert}.npy", ids)
            if cname == "clean" and not a.pert:
                np.save(f"{OUT}/{a.name}/prob_{nm}.npy", prob.numpy())
                np.save(f"{OUT}/{a.name}/rprob_{nm}.npy", rprob.numpy())
                np.save(f"{OUT}/{a.name}/dir_{nm}.npy", dp.numpy())
                np.save(f"{OUT}/{a.name}/ids_{nm}.npy", ids)
    mfile = f"{OUT}/{a.name}/metrics{'_pert_' + a.pert if a.pert else ''}.json"
    if a.eval_only and os.path.exists(mfile):  # nur auswerten: vorhandene Kennzahlen (Training, andere Mengen) behalten und ergaenzen
        old = json.load(open(mfile))
        old.update({k: v for k, v in res.items() if k.startswith(("val_", "test_", "train_"))})
        res = old
    json.dump(res, open(mfile, "w"), indent=1)
    P("FERTIG", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
