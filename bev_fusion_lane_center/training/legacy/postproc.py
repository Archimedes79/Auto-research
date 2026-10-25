"""Test J: bessere Linienbildung auf dem duennen Skelett (kein neues Netz).

prune(skel, n):  Seitenaeste und Fragmente unter ca. n Pixeln entfernen (morphologisches Beschneiden: Endpunkte n-mal abtragen,
                 danach von den verbliebenen Endpunkten innerhalb des Original-Skeletts n-mal nachwachsen lassen).
bridge(skel, d, ...): Luecken schliessen. Endpunkte verschiedener Komponenten werden verbunden, wenn der Abstand <= max_px ist,
                 die Verbindung zur Auswaertsrichtung beider Enden passt (cos >= cos_min) und die Abstandskarte d entlang der Strecke
                 im Mittel <= dmax_m zeigt (dort liegt laut Netz Spurmitte).
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

K8 = np.ones((3, 3), int)


def endpoints(s):
    nb = ndi.convolve(s.astype(int), K8, mode="constant") - s.astype(int)
    return s & (nb == 1)


def prune(skel, n=5):
    s = skel.copy()
    for _ in range(n):
        s &= ~endpoints(s)
    grow = endpoints(s)
    for _ in range(n):
        grow = ndi.binary_dilation(grow, K8) & skel
    return s | grow


def _out_dir(skel, p, back=6):
    """Auswaertsrichtung am Endpunkt p: Mittel der Pixel in der Naehe, vom Skelett weg."""
    r0, c0 = p
    sub = skel[max(r0 - back, 0): r0 + back + 1, max(c0 - back, 0): c0 + back + 1]
    rr, cc = np.nonzero(sub)
    rr = rr + max(r0 - back, 0)
    cc = cc + max(c0 - back, 0)
    v = np.array([r0 - rr.mean(), c0 - cc.mean()], float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def bridge(skel, d, max_px=8, cos_min=0.6, dmax_m=1.2):
    """skel: bool [H,W]; d: Abstand in m [H,W] (np). Gibt (Skelett mit Bruecken, Anzahl Bruecken) zurueck."""
    s = skel.copy()
    lab, n = ndi.label(s, structure=K8)
    ep = np.argwhere(endpoints(s))
    if len(ep) < 2:
        return s, 0
    dirs = [_out_dir(s, tuple(p)) for p in ep]
    cand = []
    for i in range(len(ep)):
        for j in range(i + 1, len(ep)):
            a, b = ep[i], ep[j]
            if lab[tuple(a)] == lab[tuple(b)]:
                continue
            v = b - a
            dist = np.linalg.norm(v)
            if dist > max_px or dist < 1.0:
                continue
            u = v / dist
            if np.dot(dirs[i], u) < cos_min or np.dot(dirs[j], -u) < cos_min:
                continue
            t = np.linspace(0, 1, int(dist) + 2)
            pr = (a[0] + t * v[0]).round().astype(int)
            pc = (a[1] + t * v[1]).round().astype(int)
            if d[pr, pc].mean() > dmax_m:
                continue
            cand.append((dist, i, j))
    used, nb = set(), 0
    for dist, i, j in sorted(cand):
        if i in used or j in used:
            continue
        a, b = ep[i], ep[j]
        buf = s.astype(np.uint8)
        cv2.line(buf, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), 1, 1)
        s = buf.astype(bool)
        used.update((i, j))
        nb += 1
    return s, nb
