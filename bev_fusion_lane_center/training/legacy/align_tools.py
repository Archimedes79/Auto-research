"""E34: Ausrichtung des Karten-Vorwissens an die Beobachtung (Karten-Schichten des Grundlage-Netzes) durch Korrelationssuche ueber (dx, dy, dtheta).

Modell der Pose-Stoerung (wie prior_tools.PriorSampler): das gestoerte Vorwissen P' wurde mit t' = t + R d, R' = R Rot(th) gezeichnet, also  P_wahr(q) = P'(Rot(-th) (q - d)),  q in Fahrzeug-Metern (x = Zeile, y = Spalte).
Verfahren ('Beobachtungs-Registrierung'): Vorwissen (befahrbar, Trennlinien, Zebra; leicht unscharf) gegen die vom Netz vorhergesagten Schichten (road_segment, road_divider/lane_divider, ped_crossing; Sigmoid)
ueber ein grobes Gitter (dx, dy in +-R m, th in +-T Grad) bei halber Aufloesung, dann Feinsuche um das Maximum. Bewertung: gewichtete normierte Kreuzkorrelation je Kanal.
Die Suche braucht keine Soll-Daten (nur Beobachtung + Karte + gestoerte Pose)."""
import math

import torch
import torch.nn.functional as F

HALF = 50.0


def gauss_kernel(sigma_px, device):
    r = max(int(math.ceil(3 * sigma_px)), 1)
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    return k / k.sum()


def blur(x, sigma_px):
    if sigma_px <= 0:
        return x
    k = gauss_kernel(sigma_px, x.device)
    C = x.shape[1]
    kh = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    kw = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    r = (k.numel() - 1) // 2
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), kh, groups=C)
    return F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kw, groups=C)


def base_grid(H, W, device):
    """Ausgabe-Raster q (Meter): x je Zeile, y je Spalte (Pixelmitten), [H,W,2] = (x, y)."""
    xs = -HALF + (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * (2 * HALF / H)
    ys = -HALF + (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * (2 * HALF / W)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([X, Y], -1)


def warp(prior, d, th_deg, q=None):
    """prior [B,C,H,W]; d [B,2] = (dx, dy) m; th_deg [B]. Rueckgabe P_wahr(q) = P'(Rot(-th) (q - d)). Alles stapelweise."""
    B, C, H, W = prior.shape
    if q is None:
        q = base_grid(H, W, prior.device)
    th = torch.deg2rad(th_deg).view(B, 1, 1)
    c, s = torch.cos(th), torch.sin(th)
    v = q[None] - d.view(B, 1, 1, 2)  # [B,H,W,2]
    px = c * v[..., 0] + s * v[..., 1]  # Rot(-th) = [[c, s], [-s, c]]
    py = -s * v[..., 0] + c * v[..., 1]
    grid = torch.stack([py / HALF, px / HALF], -1)  # grid_sample: (Spalte = y, Zeile = x)
    return F.grid_sample(prior, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def ncc(a, b, w):
    """a, b [N,C,H,W] -> gewichtete normierte Korrelation [N] (Kanal-Gewichte w [C])."""
    a = a - a.mean((2, 3), keepdim=True)
    b = b - b.mean((2, 3), keepdim=True)
    num = (a * b).sum((2, 3))
    den = a.pow(2).sum((2, 3)).sqrt() * b.pow(2).sum((2, 3)).sqrt() + 1e-6
    return ((num / den) * w.view(1, -1)).sum(1) / w.sum()


@torch.no_grad()
def align(prior, obs, search_m=4.0, search_deg=6.0, coarse_m=1.0, coarse_deg=3.0, w=(1.0, 1.5, 1.0), blur_m=1.0, chunk=64):
    """prior [B,3,H,W] (0/1, gestoerte Pose), obs [B,3,H,W] (Sigmoid der Netz-Schichten road_segment, max(road_divider, lane_divider), ped_crossing).
    Rueckgabe: ausgerichtetes Vorwissen [B,3,H,W], Schaetzung [B,3] = (dx, dy, th_deg)."""
    dev = prior.device
    B, C, H, W = prior.shape
    wt = torch.tensor(w, device=dev, dtype=torch.float32)
    pb = blur(prior.float(), blur_m / 0.5)
    ob = blur(obs.float(), 0.5 / 0.5)
    # grob bei halber Aufloesung
    p2, o2 = F.avg_pool2d(pb, 2), F.avg_pool2d(ob, 2)
    q2 = base_grid(H // 2, W // 2, dev)
    dxs = torch.arange(-search_m, search_m + 1e-6, coarse_m, device=dev)
    ths = torch.arange(-search_deg, search_deg + 1e-6, coarse_deg, device=dev)
    cand = torch.stack(torch.meshgrid(dxs, dxs, ths, indexing="ij"), -1).view(-1, 3)  # [Nc,3]
    best = torch.zeros(B, 3, device=dev)
    for b in range(B):
        sc = []
        for i in range(0, len(cand), chunk):
            cc = cand[i : i + chunk]
            n = len(cc)
            pw = warp(p2[b : b + 1].expand(n, -1, -1, -1), cc[:, :2], cc[:, 2], q2)
            sc.append(ncc(pw, o2[b : b + 1].expand(n, -1, -1, -1), wt))
        sc = torch.cat(sc)
        best[b] = cand[int(sc.argmax())]
    # fein um das Maximum (voller Aufloesung)
    q = base_grid(H, W, dev)
    fd = torch.arange(-coarse_m, coarse_m + 1e-6, 0.5, device=dev)
    ft = torch.arange(-coarse_deg, coarse_deg + 1e-6, 1.5, device=dev)
    fc = torch.stack(torch.meshgrid(fd, fd, ft, indexing="ij"), -1).view(-1, 3)
    est = torch.zeros(B, 3, device=dev)
    out = torch.zeros_like(prior, dtype=torch.float32)
    for b in range(B):
        cc = best[b][None] + fc
        sc = []
        for i in range(0, len(cc), chunk):
            c2 = cc[i : i + chunk]
            n = len(c2)
            pw = warp(pb[b : b + 1].expand(n, -1, -1, -1), c2[:, :2], c2[:, 2], q)
            sc.append(ncc(pw, ob[b : b + 1].expand(n, -1, -1, -1), wt))
        sc = torch.cat(sc)
        est[b] = cc[int(sc.argmax())]
        out[b] = warp(prior[b : b + 1].float(), est[b : b + 1, :2], est[b : b + 1, 2], q)[0]
    return out, est
