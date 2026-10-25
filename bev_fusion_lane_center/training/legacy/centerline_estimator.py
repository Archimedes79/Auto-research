"""Mittellinien-Schaetzer: Spurmitten allein aus der Wahrnehmung (kein Kartenwissen zur Laufzeit).

Pipeline:  eingefrorenes BEVFusion  ->  BEV-Features (512 x 128 x 128) + Vorhersage (6 x 200 x 200)
           -> kleines FCN-Ensemble (4 Netze) -> Abstand zur naechsten Mittellinie (200x200, 0.5 m)
           -> Schwelle -> Ausduennen (Zhang-Suen) -> Verfolgen -> Polylinien im Lidar-System [m]
Das Soll (Karte) wird nur zum Training und zur Auswertung benutzt.

Auswertung: Praezision / Ausbeute / F1 der geschaetzten Mittellinien gegen die Karten-Mittellinien bei Toleranz
0.5 / 1.0 / 1.5 m (Chamfer-artig, wie bei Online-Karten ueblich), Validierung (81 Samples) und Training.
"""
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align_train as A  # noqa: E402
import heat_align as H  # noqa: E402

dev = "cuda"
O = "/mnt/c/ML-Experiment/out/align"
OUT = "/mnt/c/ML-Experiment/out/estimator"
PATHS = [f"{O}/v5dist/heatnet.pt"] + [f"{O}/v7s{s}/heatnet.pt" for s in (1, 2, 3)]


class CenterlineEstimator:
    def __init__(self, paths=PATHS):
        self.nets = []
        for p in paths:
            # EST_CIN=32 fuer PCA-Features; EST_ARCH=unet, EST_BASE fuer das U-Netz
            n = H.make_net(os.environ.get("EST_ARCH", "small"), int(os.environ.get("EST_CIN", "512")), base=int(os.environ.get("EST_BASE", "48")),
                           use_sem=os.environ.get("EST_NO_SEM") != "1",  # EST_NO_SEM=1: nur BEV-Features (Kamera-Radar, kein Lidar-Netz)
                           out_ch=3 if os.environ.get("EST_MULTI") == "1" else 1).to(dev)  # EST_MULTI=1: Netz mit Richtungsausgabe
            n.load_state_dict(torch.load(p))
            n.eval()
            self.nets.append(n)

    @torch.no_grad()
    def distance_field(self, feats, sem):
        """-> Abstand zur naechsten Mittellinie in m (bei 4 m gekappt), [B,200,200]."""
        o = torch.stack([torch.sigmoid(n(feats, sem)) for n in self.nets]).mean(0)
        return o * 8.0 * 0.5

    @staticmethod
    def thin(mask):
        """Zhang-Suen-Ausduennen, mask bool [H,W] (GPU)."""
        img = mask.float()

        def step(img, first):
            p = F.pad(img, (1, 1, 1, 1))
            P2, P3, P4, P5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            P6, P7, P8, P9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            Bn = sum(seq[:8])
            An = sum(((seq[i] == 0) & (seq[i + 1] == 1)).float() for i in range(8))
            c = (Bn >= 2) & (Bn <= 6) & (An == 1)
            c = c & (((P2 * P4 * P6) == 0) & ((P4 * P6 * P8) == 0)) if first else c & (((P2 * P4 * P8) == 0) & ((P2 * P6 * P8) == 0))
            return img * (~(img.bool() & c)).float()

        for _ in range(40):
            new = step(step(img, True), False)
            if torch.equal(new, img):
                break
            img = new
        return img.bool()

    @torch.no_grad()
    def direction_field(self, feats, sem):
        """Nur fuer Netze mit Richtungsausgabe (EST_MULTI=1): gemittelter Doppelwinkel-Vektor (cos 2t, sin 2t) [B,2,200,200]."""
        return torch.stack([torch.tanh(n(feats, sem, full=True)[:, 1:3]) for n in self.nets]).mean(0)

    @staticmethod
    def nms(d, dirs, thr_m, blob, tol=1e-3):
        """Nicht-Maximum-Unterdrueckung quer zur Spurrichtung: behalte Pixel, deren Abstandswert kleiner/gleich dem der
        Nachbarn +-1 px entlang der Normalen ist. Ohne verlaessliche Richtung (Betrag < 0.3) bleibt der Blob unveraendert."""
        B, H_, W_ = d.shape
        th = 0.5 * torch.atan2(dirs[:, 1], dirs[:, 0])  # Richtung (Zeile=x, Spalte=y)
        nrow, ncol = -torch.sin(th), torch.cos(th)  # Normale
        ii, jj = torch.meshgrid(torch.arange(H_, device=d.device).float(), torch.arange(W_, device=d.device).float(), indexing="ij")

        def at(sgn):
            r = ii[None] + sgn * nrow
            c = jj[None] + sgn * ncol
            g = torch.stack([(c + 0.5) / (W_ / 2) - 1, (r + 0.5) / (H_ / 2) - 1], -1)
            return F.grid_sample(d[:, None], g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0]

        ridge = (d <= at(1.0) + tol) & (d <= at(-1.0) + tol) & blob
        reliable = dirs.norm(dim=1) > 0.3
        return torch.where(reliable, ridge, blob)

    def centerline_mask(self, d, thr_m=1.0, blur_px=0.8, low=None, dirs=None):
        """Abstandskarte [B,200,200] -> duenne Mittellinien-Pixel [B,200,200] (bool).
        low (m): Hysterese. Pixel mit Abstand <= thr_m sind sicher; Pixel mit Abstand <= low werden uebernommen,
        wenn sie mit einem sicheren Pixel zusammenhaengen (schliesst Luecken, ohne neue Linien zu erfinden)."""
        db = H.gauss_blur(d, blur_px) if blur_px > 0 else d
        blob = db <= thr_m
        if low is not None and low > thr_m:
            lowm = db <= low
            m = blob
            for _ in range(80):
                m2 = (F.max_pool2d(m.float()[:, None], 3, 1, 1)[:, 0] > 0) & lowm
                if torch.equal(m2, m):
                    break
                m = m2
            blob = m
        if dirs is not None:
            blob = self.nms(db, dirs, thr_m, blob)
        return torch.stack([self.thin(b) for b in blob])

    @staticmethod
    def vectorize(skel, simplify_m=0.4):
        """Duenne Pixel -> Polylinien [m, Lidar-System (x,y)]. Folgt Pfaden zwischen Endpunkten/Abzweigen."""
        pix = set(map(tuple, np.argwhere(skel)))
        nb = {p: [(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (p[0] + a, p[1] + b) in pix] for p in pix}
        seen, lines = set(), []

        def walk(start, first):
            line, prev, cur = [start], start, first
            while True:
                line.append(cur)
                seen.add(cur)
                nxt = [q for q in nb[cur] if q != prev and q not in line[-3:]]
                nxt = [q for q in nxt if q not in seen] or []
                if len(nb[cur]) != 2 or not nxt:
                    break
                prev, cur = cur, nxt[0]
            return line

        nodes = [p for p in pix if len(nb[p]) != 2]
        for s in nodes:
            for f in nb[s]:
                if f not in seen:
                    lines.append(walk(s, f))
        for s in sorted(pix):  # uebrig gebliebene Schleifen
            if s not in seen and nb[s]:
                lines.append(walk(s, nb[s][0]))
        out = []
        for ln in lines:
            if len(ln) < 4:  # zu kurz (< 2 m)
                continue
            a = np.array(ln, np.float32)
            xy = np.stack([-50 + (a[:, 0] + 0.5) * 0.5, -50 + (a[:, 1] + 0.5) * 0.5], 1)  # (x,y) m
            ap = cv2.approxPolyDP(xy.reshape(-1, 1, 2), simplify_m, False).reshape(-1, 2)
            out.append(ap)
        return out


def gt_points(nodes, poly, mask, step=0.5):
    """Mittellinien-Punkte (Soll) dicht (0.5 m), aus den Knoten (3 m) je Linie interpoliert."""
    pts = []
    for pid in np.unique(poly[mask]):
        q = nodes[(poly == pid) & mask]
        for a, b in zip(q[:-1], q[1:]):
            L = np.linalg.norm(b - a)
            if L < 5.0:
                n = max(int(L / step), 1)
                pts.append(a + (b - a) * (np.arange(n)[:, None] / n))
        if len(q):
            pts.append(q[-1:])
    return np.concatenate(pts) if pts else np.zeros((0, 2), np.float32)


def prf(P, G, taus=(0.5, 1.0, 1.5)):
    """Zaehler fuer Praezision/Ausbeute bei Toleranzen. P,G: Punktmengen [n,2] (m)."""
    if len(P) == 0 or len(G) == 0:
        return {t: (0, len(P), 0, len(G)) for t in taus}
    D = torch.cdist(torch.from_numpy(P).float(), torch.from_numpy(G).float())
    dp, dg = D.min(1).values, D.min(0).values
    return {t: (int((dp <= t).sum()), len(P), int((dg <= t).sum()), len(G)) for t in taus}


def evaluate(est, data, thr_m, blur_px, bs=8, gts=None, low=None):
    tot = {t: np.zeros(4) for t in (0.5, 1.0, 1.5)}
    S = data["nodes"].shape[0]
    nodes, poly, mask = data["nodes"].cpu().numpy(), data["poly"], data["mask"].cpu().numpy()
    skels = []
    stats = dict(n=0, bruecken=0, komp_vor=0, komp_nach=0)
    for i in range(0, S, bs):
        sl = slice(i, min(i + bs, S))
        d = est.distance_field(data["feats"][sl], data["sem"][sl])
        dirs = est.direction_field(data["feats"][sl], data["sem"][sl]) if os.environ.get("EST_NMS") == "1" else None
        sk = est.centerline_mask(d, thr_m, blur_px, low, dirs)
        sk = sk.cpu().numpy()
        post = os.environ.get("EST_POST", "")  # Test J: "prune", "prune,bridge"
        if post:
            import postproc as PP

            from scipy import ndimage as ndi

            dn = d.cpu().numpy()
            for b in range(sk.shape[0]):
                s_ = sk[b]
                if "prune" in post:
                    s_ = PP.prune(s_, int(os.environ.get("EST_PRUNE", "5")))
                comp0 = ndi.label(s_, structure=np.ones((3, 3)))[1]
                if "bridge" in post:
                    s_, nbr = PP.bridge(s_, dn[b], int(os.environ.get("EST_BRIDGE", "8")))
                    stats["bruecken"] += nbr
                stats["komp_vor"] += comp0
                stats["komp_nach"] += ndi.label(s_, structure=np.ones((3, 3)))[1]
                stats["n"] += 1
                sk[b] = s_
        skels.append(sk)
    if stats["n"]:
        print(f"Linienbildung: Komponenten je Sample {stats['komp_vor'] / stats['n']:.1f} -> {stats['komp_nach'] / stats['n']:.1f}, "
              f"Bruecken je Sample {stats['bruecken'] / stats['n']:.2f}")
    skels = np.concatenate(skels)
    for i in range(S):
        P = np.argwhere(skels[i]).astype(np.float32)
        P = np.stack([-50 + (P[:, 0] + 0.5) * 0.5, -50 + (P[:, 1] + 0.5) * 0.5], 1)
        P = P[(np.abs(P) < 48).all(1)]
        G = gts[i] if gts is not None else gt_points(nodes[i], poly[i], mask[i])
        for t, v in prf(P, G).items():
            tot[t] += np.array(v, float)
    res = {}
    for t, (tp_p, n_p, tp_g, n_g) in tot.items():
        pr, rc = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
        res[t] = dict(prec=pr, rec=rc, f1=2 * pr * rc / max(pr + rc, 1e-9))
    return res, skels


def main():
    """Aufruf: centerline_estimator.py [netze,mit,komma] [ausgabeordner] [exact]"""
    global OUT
    paths = sys.argv[1].split(",") if len(sys.argv) > 1 else PATHS
    OUT = sys.argv[2] if len(sys.argv) > 2 else OUT
    exact = len(sys.argv) > 3 and sys.argv[3] == "exact"
    os.makedirs(OUT, exist_ok=True)
    # Datenordner per Umgebungsvariable (Standard: Mini); EST_SKIP_TRAIN=1 wertet nur die Validierung aus
    fdir = os.path.expanduser(os.environ.get("EST_FDIR", "~/exp/feats"))
    gdir = os.path.expanduser(os.environ.get("EST_GDIR", "~/exp/graph"))
    skip_train = os.environ.get("EST_SKIP_TRAIN") == "1"
    va = A.load_split("val", fdir, gdir)
    va["poly"] = np.load(f"{gdir}/val_graph.npz")["poly"]
    gts_va = np.load(f"{fdir}/val_gt_pts_exact.npy", allow_pickle=True) if exact else None
    est = CenterlineEstimator(paths)

    print("== Schwellen-Wahl (Validierung): Schwelle der Abstandskarte [m] / Glaettung [px] -> F1 bei 0.5 | 1.0 | 1.5 m")
    best, best_f = None, -1
    fixed = os.environ.get("EST_FIXED")  # z.B. "0.5,0.0": feste Einstellung ohne Suche (gleiche fuer alle Netze)
    grid = [(float(fixed.split(",")[0]), float(fixed.split(",")[1]))] if fixed else [(t, b) for t in (0.5, 0.75, 1.0, 1.5) for b in (0.0, 0.8, 1.5)]
    for thr, blur in grid:
        if True:
            r, _ = evaluate(est, va, thr, blur, gts=gts_va)
            f = [r[t]["f1"] for t in (0.5, 1.0, 1.5)]
            print(f"  thr {thr:4.2f} blur {blur:3.1f}: " + " | ".join(f"{x:.3f}" for x in f), flush=True)
            if np.mean(f) > best_f:
                best_f, best = np.mean(f), (thr, blur)
    thr, blur = best
    print(f"Gewaehlt: Schwelle {thr} m, Glaettung {blur} px (auf Validierung gewaehlt)")

    out = {"thr": thr, "blur": blur}
    for name, data in (("Validierung", va),) + (() if skip_train else (("Training", None),)):
        gts = gts_va
        if data is None:
            tr = A.load_split("train", fdir, gdir)
            tr["poly"] = np.load(f"{gdir}/train_graph.npz")["poly"]
            data = tr
            gts = np.load(f"{fdir}/train_gt_pts_exact.npy", allow_pickle=True) if exact else None
        r, sk = evaluate(est, data, thr, blur, gts=gts)
        out[name] = r
        print(f"== {name}: " + " | ".join(f"Toleranz {t} m: Praezision {r[t]['prec']:.3f}, Ausbeute {r[t]['rec']:.3f}, F1 {r[t]['f1']:.3f}" for t in r))
        if name == "Validierung":
            skel_va = sk
            nl, tl = [], []  # Fragmentierung: Anzahl und Laenge der Polylinien (erste 100 Samples)
            for i in range(min(100, len(sk))):
                ls = est.vectorize(sk[i])
                nl.append(len(ls))
                tl.append(sum(float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) for a in ls))
            print(f"Fragmentierung (erste {len(nl)} Samples): {np.mean(nl):.1f} Polylinien je Sample, Gesamtlaenge {np.mean(tl):.0f} m, mittlere Laenge je Linie {np.sum(tl) / max(np.sum(nl), 1):.1f} m")
    json.dump({k: ({str(t): v for t, v in x.items()} if isinstance(x, dict) else x) for k, x in out.items()}, open(f"{OUT}/metrics.json", "w"), indent=1)

    # Beispielbilder: Soll (blau), geschaetzte Polylinien (rot) auf Abstandskarte
    nodes, poly, mask = va["nodes"].cpu().numpy(), va["poly"], va["mask"].cpu().numpy()
    idxs = [3, 22, 46, 70]
    fig, axs = plt.subplots(1, len(idxs), figsize=(4.2 * len(idxs), 4.4))
    n_lines = []
    for ax, i in zip(axs, idxs):
        d = est.distance_field(va["feats"][i : i + 1], va["sem"][i : i + 1])
        lines = est.vectorize(skel_va[i])
        n_lines.append(len(lines))
        ax.imshow(np.clip(d[0].cpu().numpy(), 0, 4), cmap="Greys_r", vmin=0, vmax=4)
        G = gts_va[i] if gts_va is not None else gt_points(nodes[i], poly[i], mask[i])
        ax.scatter((G[:, 1] + 50) / 0.5 - 0.5, (G[:, 0] + 50) / 0.5 - 0.5, s=3, c="#1a73e8", label="Soll (Karte)")
        for k, ln in enumerate(lines):
            ax.plot((ln[:, 1] + 50) / 0.5 - 0.5, (ln[:, 0] + 50) / 0.5 - 0.5, c="#d93025", lw=1.6, label="geschaetzt" if k == 0 else None)
        ax.set_title(f"Val-Sample {i}: {len(lines)} Polylinien", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 199); ax.set_ylim(199, 0)
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle("Mittellinien-Schaetzer aus der Wahrnehmung (ohne Karte): Soll blau, geschaetzte Linien rot", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/schaetzer_beispiele.png", dpi=130)
    print("Polylinien je Beispiel:", n_lines)


if __name__ == "__main__":
    main()
