"""Loop 4 / E44: Schluesselpunkt-Kopf fuer Abzweige (SAM-Road-artig: Knoten als dichte Heatmap + Nicht-Maximum-Unterdrueckung) auf den eingefrorenen Merkmalen.
U-Netz (wie der dichte Kopf) -> 2 Kanaele: Split-Knoten (Ende von Stuecken mit >= 2 Nachfolgern), Merge-Knoten (Anfang von Stuecken mit >= 2 Vorgaengern); Soll aus make_node_gt.py.
Verlust: fokale gewichtete BCE (Falschalarm-Seite lam, Verpasst-Seite wpos; Knoten sind selten -> wpos hoeher). Auswertung: Spitzen (3,5 m Mindestabstand) ueber Schwelle -> Ortsabgleich mit den Soll-Knoten (3 m / 6 m).
Aufruf (gcar): node_head.py --name e44_nodes --steps 5000 [--feats dec32,logit7] [--eval train_eval,val,test]"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_head as DH  # noqa: E402
import graph_eval as GE  # noqa: E402
import graph_from_dense as GD  # noqa: E402

HOME, OUT = DH.HOME, DH.OUT


class NodeData(DH.Data):
    def __init__(self, fg, groups, part, indices, split="b123", target=True):
        super().__init__(fg, groups, part, indices, split, target=False)
        self.want, self.nd = target, None

    def __getitem__(self, k):
        x, i = super().__getitem__(k)
        if not self.want:
            return x, i
        if self.nd is None:
            self.nd = np.load(f"{HOME}/exp_gcar_b123/node_{self.part}.npy", mmap_mode="r")
        return x, torch.from_numpy(np.asarray(self.nd[i], np.float32) / 255.0), i


def peaks(prob, tau, nms=7):
    """prob [B,2,200,200] (GPU, 0..1) -> je Sample und Kanal Array (n, 3): x, y (m), Wert."""
    mx = F.max_pool2d(prob, nms, 1, nms // 2)
    keep = (prob == mx) & (prob >= tau)
    out = [[None, None] for _ in range(prob.shape[0])]
    for b in range(prob.shape[0]):
        for c in range(2):
            rc = keep[b, c].nonzero().float()
            v = prob[b, c][keep[b, c]]
            if len(rc) == 0:
                out[b][c] = np.zeros((0, 3), np.float32)
                continue
            xy = -50.0 + (rc + 0.5) * 0.5
            out[b][c] = torch.cat([xy, v[:, None]], 1).cpu().numpy().astype(np.float32)
    return out


def eval_nodes(prob_all, gts, taus=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7), tols=(3.0, 6.0), dev="cuda", bs=32):
    """prob_all [N,2,200,200] (cpu, half) gegen Soll-Graphen -> {tau: {tol: {split/merge: P,R,F1}}}"""
    G = [GE.junctions(g) for g in gts]
    res = {}
    for tau in taus:
        cnt = {(t, c): [0, 0, 0] for t in tols for c in (0, 1)}
        for i in range(0, len(prob_all), bs):
            pk = peaks(torch.from_numpy(np.asarray(prob_all[i : i + bs], np.float32)).to(dev), tau)
            for k in range(len(pk)):
                for c in (0, 1):
                    P = pk[k][c][:, :2].astype(np.float64)
                    B = np.asarray(G[i + k][c], np.float64).reshape(-1, 2)
                    B = B[(np.abs(B) < 50).all(1)] if len(B) else B
                    for t in tols:
                        m = GE._match_points(P, B, t)
                        cc = cnt[(t, c)]
                        cc[0] += m
                        cc[1] += len(P)
                        cc[2] += len(B)
        r = {}
        for t in tols:
            r[str(t)] = {}
            for c, nm in ((0, "split"), (1, "merge")):
                m, npred, ngt = cnt[(t, c)]
                p, rc = m / max(npred, 1), m / max(ngt, 1)
                r[str(t)][nm] = dict(P=float(p), R=float(rc), F1=float(2 * p * rc / max(p + rc, 1e-9)), n_pred=npred, n_gt=ngt)
        res[str(tau)] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--feats", default="dec32,logit7")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--wpos", type=float, default=8.0)
    ap.add_argument("--train_stride", type=int, default=1)
    ap.add_argument("--eval", default="train_eval,val,test")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", type=float, default=3.0)
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
    groups = a.feats.split(",")
    sp = json.load(open(BP.ROOT + "/data/splits.json"))
    tr_idx = np.arange(sp["n"]["train"])[:: a.train_stride]
    sets = {"train_eval": ("train", sp["train_eval"]), "val": ("val", sp["val"]), "test": ("val", sp["test"])}
    ds = NodeData("gcar", groups, "train", tr_idx)
    mean, std = DH.channel_stats(ds)
    net = DH.UNet(int(mean.shape[0]), a.base, 2).cuda()
    with torch.no_grad():
        net.head.bias.fill_(-4.0)
    P("Eingangskanaele", int(mean.shape[0]), "Parameter", sum(p.numel() for p in net.parameters()), "Trainings-Samples", len(ds))
    mean, std = mean.cuda(), std.cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4, eps=1e-6, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.0, growth_interval=200)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True, prefetch_factor=4)
    net.train()
    step, t0, run, gema, n_clip = 0, time.time(), [], None, 0
    curve = []
    while step < a.steps:
        for bt in dl:
            x, y = bt[0].cuda(non_blocking=True), bt[1].cuda(non_blocking=True)
            x = (x - mean[None, :, None, None]) / std[None, :, None, None]
            with torch.autocast("cuda", dtype=torch.float16):
                o = net(x)
            o = o.float()
            loss = 0.5 * (DH.loss_fn(o[:, 0], y[:, 0], a.lam, a.wpos) + DH.loss_fn(o[:, 1], y[:, 1], a.lam, a.wpos))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = float(nn.utils.clip_grad_norm_(net.parameters(), float("inf")))
            if np.isfinite(gn):
                gema = gn if gema is None else gema
                lim = a.clip * gema
                if gn > lim:
                    for p_ in net.parameters():
                        if p_.grad is not None:
                            p_.grad.mul_(lim / gn)
                    n_clip += 1
                gema = 0.98 * gema + 0.02 * min(gn, lim)
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < 32.0:
                scaler.update(32.0)
            sched.step()
            run.append(float(loss))
            step += 1
            if step % 250 == 0:
                curve.append((step, float(np.mean(run[-250:]))))
                P(f"Schritt {step}/{a.steps} Verlust {curve[-1][1]:.4f} ({(time.time() - t0) / step:.2f} s/Schritt) geklippt {n_clip} Skala {scaler.get_scale():.0f}")
            if step >= a.steps:
                break
    torch.save(dict(state=net.state_dict(), mean=mean.cpu(), std=std.cpu(), args=vars(a)), f"{OUT}/{a.name}/model.pt")
    res = dict(args=vars(a), curve=curve, train_s=time.time() - t0, params=sum(p.numel() for p in net.parameters()))
    net.eval()
    for nm in a.eval.split(","):
        part, idx = sets[nm]
        dse = NodeData("gcar", groups, part, idx, target=False)
        dl2 = torch.utils.data.DataLoader(dse, batch_size=16, shuffle=False, num_workers=a.workers)
        outs, ids = [], []
        with torch.no_grad():
            for x, i in dl2:
                x = (x.cuda() - mean[None, :, None, None]) / std[None, :, None, None]
                with torch.autocast("cuda", dtype=torch.float16):
                    o = net(x).float()
                outs.append(torch.sigmoid(o).half().cpu())
                ids.append(i)
        prob, ids = torch.cat(outs).numpy(), torch.cat(ids).numpy()
        np.save(f"{OUT}/{a.name}/prob_{nm}.npy", prob)
        np.save(f"{OUT}/{a.name}/ids_{nm}.npy", ids)
        r = eval_nodes(prob, GD.load_gt_graphs(part, ids))
        res[nm] = r
        best = max(r.items(), key=lambda kv: kv[1]["6.0"]["split"]["F1"] + kv[1]["6.0"]["merge"]["F1"])
        b6 = best[1]["6.0"]
        P(f"[{nm}] {len(ids)} Samples; bester Schwellwert {best[0]}: Split F1@6m {b6['split']['F1']:.3f} (P {b6['split']['P']:.3f} R {b6['split']['R']:.3f}), Merge F1@6m {b6['merge']['F1']:.3f} (P {b6['merge']['P']:.3f} R {b6['merge']['R']:.3f}); @3m Split {best[1]['3.0']['split']['F1']:.3f} Merge {best[1]['3.0']['merge']['F1']:.3f}")
    json.dump(res, open(f"{OUT}/{a.name}/metrics.json", "w"), indent=1)
    P("FERTIG", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
