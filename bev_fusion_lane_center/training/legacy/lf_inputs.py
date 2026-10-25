"""Zusatz-Eingaben fuer LaneFormer (E34 Karten-Vorwissen, E35 Zeitverfolgung): Datenlieferanten (CPU, im DataLoader) und GPU-Hilfen (Ausrichtung, Vorbild-Merkmale in das aktuelle Bild falten).

PriorProvider: Vorwissen-Kanaele [3,200,200] (befahrbar, Trennlinien, Zebra) aus den globalen Rastern, mit Pose-Stoerung (sxy m, sdeg Grad; Training: zufaellig je Zugriff, Auswertung: fest je Sample) und optional veralteter Karte.
  Modi: 'map' (echtes Vorwissen), 'gt_oracle' (Positivkontrolle: Soll-Mittellinien als Kanal - ein Leck, darf nicht Ergebnis sein), 'map_wrong' (Kontrolle: Vorwissen eines anderen Ortes).
rel_pose: Bewegung zwischen Vorgaenger und aktuellem Bild:  p_t = Q p_(t-1) + s  (Fahrzeug-System, x rechts, y vorn).
fold_prev: Merkmale des Vorgaengers in das Raster des aktuellen Bildes falten (grid_sample)."""
import os

import numpy as np
import torch
import torch.nn.functional as F

import prior_tools as PT

HOME = os.path.expanduser("~")


class PriorProvider:
    def __init__(self, split="b123", mode="map", sxy=0.0, sdeg=0.0, degrade=0.0, train=True):
        self.mode, self.sxy, self.sdeg, self.degrade, self.train, self.split = mode, sxy, sdeg, degrade, train, split
        self.ps = PT.PriorSampler(split) if mode != "gt_oracle" else None
        self.gt = None
        self.ctr = 0

    def _rng(self, i):
        if self.train:
            self.ctr += 1
            return np.random.RandomState((torch.initial_seed() % (2**31) + 7919 * int(i) + self.ctr) % (2**32))
        return np.random.RandomState(1000003 + int(i))

    def get(self, part, i):
        rng = self._rng(i)
        noise = (rng.randn() * self.sxy, rng.randn() * self.sxy, rng.randn() * self.sdeg)
        if self.mode == "gt_oracle":
            if self.gt is None:
                self.gt = {p: np.load(f"{HOME}/exp_crn_{self.split}/feats/{p}_gt_dist_exact.npy", mmap_mode="r") for p in ("train", "val")}
            d = np.asarray(self.gt[part][int(i)], np.float32)
            p = np.zeros((3, 200, 200), np.uint8)
            p[0] = (d <= 1.0).astype(np.uint8)
        else:
            src = int(i)
            if self.mode == "map_wrong":
                n = len(self.ps.poses[part]["loc"])
                src = (int(i) + n // 2) % n
            p = self.ps.get(part, src, noise=noise)
            if self.degrade > 0:
                p = PT.degrade(p, rng, self.degrade)
        return torch.from_numpy(p).float(), torch.tensor(noise, dtype=torch.float32)


class PoseTable:
    """gap = Zahl der Schluessel-Bilder zurueck (1 = unmittelbarer Vorgaenger, 2 = ein Bild uebersprungen). Fuer das Training gibt es nur jede zweite Zeile (gerade Zeilen), daher gap = 2 (1,0 s)."""

    def __init__(self, split="b123", gap=1):
        self.gap = gap
        self.p = {part: dict(np.load(f"{HOME}/exp_gcar_{split}/poses_{part}.npz")) for part in ("train", "val")}

    def prev(self, part, i):
        j = int(self.p[part]["prev"][int(i)])
        for _ in range(self.gap - 1):
            if j < 0:
                break
            j = int(self.p[part]["prev"][j])
        return j

    def rel(self, part, i):
        """(Q [4], s [2], has [1]) fuer den Uebergang Vorgaenger -> aktuelles Bild."""
        P = self.p[part]
        j = self.prev(part, i)
        if j < 0:
            return np.array([1, 0, 0, 1], np.float32), np.zeros(2, np.float32), np.zeros(1, np.float32)
        Rt, Rp = P["R"][int(i)], P["R"][j]
        Q = Rt.T @ Rp
        s = Rt.T @ (P["t"][j] - P["t"][int(i)])
        return Q.reshape(4).astype(np.float32), s.astype(np.float32), np.ones(1, np.float32)


def fold_prev(feat, Q, s, half=50.0):
    """feat [B,C,H,W] im Fahrzeug-System des Vorgaengers; Q [B,4] (zeilenweise 2x2), s [B,2] -> Merkmale im Raster des aktuellen Bildes ([B,C,H,W]) und Gueltigkeit [B,1,H,W]."""
    B, C, H, W = feat.shape
    xs = -half + (torch.arange(H, device=feat.device, dtype=torch.float32) + 0.5) * (2 * half / H)
    ys = -half + (torch.arange(W, device=feat.device, dtype=torch.float32) + 0.5) * (2 * half / W)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    q = torch.stack([X, Y], -1)[None]  # [1,H,W,2]
    v = q - s.view(B, 1, 1, 2)
    Qm = Q.view(B, 2, 2)
    # p = Q^T (q - s)
    px = Qm[:, 0, 0].view(B, 1, 1) * v[..., 0] + Qm[:, 1, 0].view(B, 1, 1) * v[..., 1]
    py = Qm[:, 0, 1].view(B, 1, 1) * v[..., 0] + Qm[:, 1, 1].view(B, 1, 1) * v[..., 1]
    grid = torch.stack([py / half, px / half], -1)
    out = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    ones = torch.ones(B, 1, H, W, device=feat.device)
    valid = F.grid_sample(ones, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return out, valid


def obs_from_logits(raw_logit7):
    """Beobachtung fuer die Ausrichtung aus den 7 Karten-Schichten des Grundlage-Netzes (Rohwerte vor Normierung): [befahrbar, Trenner, Zebra] als Sigmoid."""
    L = raw_logit7.sigmoid()
    return torch.stack([L[:, 1], torch.maximum(L[:, 2], L[:, 3]), L[:, 4]], 1)
