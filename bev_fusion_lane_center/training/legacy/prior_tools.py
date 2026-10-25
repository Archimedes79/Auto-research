"""Karten-Vorwissen im Fahrzeug-System (E34): schneidet aus den globalen Rastern (make_prior.py) ein 200 x 200 Fenster (+-50 m, 0,5 m) mit der Ego-Pose, optional gestoerter Pose.
Stoerung der Pose: Verschiebung (dx, dy) m und Drehung dth Grad im Fahrzeug-System (so wie ein Lokalisierungsfehler wirkt). Kanaele: 0 befahrbar, 1 Trennlinien, 2 Zebrastreifen.
Veraltete Karte: 'age' loescht zufaellig Teile (Linien-/Flaechenbloecke) bzw. versetzt sie (siehe degrade)."""
import json
import os

import cv2
import numpy as np

HOME = os.path.expanduser("~")
LOCS = ["boston-seaport", "singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown"]
RES = 0.5


class PriorSampler:
    def __init__(self, split="b123"):
        d = f"{HOME}/exp_gcar_{split}"
        self.maps = [np.load(f"{d}/prior_{l}.npy", mmap_mode="r") for l in LOCS]
        self.poses = {p: dict(np.load(f"{d}/poses_{p}.npz")) for p in ("train", "val")}

    def window(self, loc, t, R, half=50.0, size=200, noise=(0.0, 0.0, 0.0)):
        dx, dy, dth = noise
        c, s = np.cos(np.radians(dth)), np.sin(np.radians(dth))
        Rn = R @ np.array([[c, -s], [s, c]])
        tn = t + R @ np.array([dx, dy])
        res = 2 * half / size
        h = half - res / 2
        M = np.array([[Rn[0, 1], Rn[0, 0], (tn[0] - (Rn[0, 0] + Rn[0, 1]) * h) / RES],
                      [Rn[1, 1], Rn[1, 0], (tn[1] - (Rn[1, 0] + Rn[1, 1]) * h) / RES]], np.float64)
        M[:, :2] *= res / RES
        img = self.maps[int(loc)]
        out = np.zeros((img.shape[0], size, size), np.uint8)
        for k in range(img.shape[0]):
            out[k] = cv2.warpAffine(np.ascontiguousarray(img[k]), M, (size, size), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
        return out

    def get(self, part, i, **kw):
        P = self.poses[part]
        return self.window(P["loc"][i], P["t"][i], P["R"][i], **kw)


def degrade(prior, rng, frac=0.3, shift_m=3.0, blk=20):
    """Veraltete Karte: Bloecke von blk x blk Pixeln (10 m) mit Wahrscheinlichkeit frac loeschen, die Haelfte der uebrigen Linienbloecke um shift_m verschieben (Neu-Markierung)."""
    out = prior.copy()
    H, W = out.shape[1:]
    for r in range(0, H, blk):
        for c in range(0, W, blk):
            u = rng.rand()
            if u < frac:
                out[:, r : r + blk, c : c + blk] = 0
            elif u < frac + (1 - frac) * 0.15:
                sh = int(round(shift_m / RES))
                dr, dc = rng.randint(-sh, sh + 1), rng.randint(-sh, sh + 1)
                blkv = prior[1, r : r + blk, c : c + blk]
                out[1, r : r + blk, c : c + blk] = 0
                r0, c0 = max(r + dr, 0), max(c + dc, 0)
                r1, c1 = min(r + dr + blk, H), min(c + dc + blk, W)
                out[1, r0:r1, c0:c1] |= blkv[: r1 - r0, : c1 - c0]
    return out
