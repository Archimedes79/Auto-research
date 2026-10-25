"""Loop 18 / R2.1: differentiable rough localisation of the SD map (K188) and a map-trust output (K189).

K188 SoftRegister: the dense lane evidence of the perception (heat E and travel direction d, 200 x 200, 0.5 m per pixel) is correlated with the SD road raster over a pose grid
(translation +-6 m in 0.5 m steps x yaw +-12 deg in 2 deg steps, 13 x 25 x 25 hypotheses). The correlation is an FFT cross-correlation per yaw, the score of a hypothesis is the share of the lane evidence
that lies on a road (two blur widths, 1.5 px and 5 px) plus the share that runs along the road direction ((d.m)^2 from three product channels). The pose is a SOFT-ARGMAX (softmax over the grid,
window of +-2 px / +-1 yaw step around the peak), so the gradient reaches the evidence (and with it the dense head / builder) and the few parameters (temperature, channel weights, prior widths).
The estimate (theta, t) is the pose error of the map in the sense of the simulator (x_map = R(theta) x + t, as sd_lib.raster_sd(R, d)); the corrected map is rendered with the inverse.
K189 trust: a logistic read-out of (explained evidence share, peak probability, peak margin, log evidence, log road pixels) -> trust in [0, 1] (1 = this map belongs to this place and is usable);
the builder gets the map multiplied by the trust (trust 0 = no map = a situation the map builder has been trained on).
Temporal part (numpy, causal): `accumulate()` carries the world-frame translation (w = R_ego t) and the yaw as a confidence-weighted running mean over the frames of a scene, so that the correction
survives frames in which the perception has lost the lanes.
Usage: imported by p18_regis_train.py / p18_regis_eval.py; `python p18_regis.py` runs the self-test (synthetic known shifts)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N, RES, HALF = 200, 0.5, 50.0
NFFT = 256


def _gauss1d(sig, dev):
    r = int(math.ceil(3 * sig))
    k = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sig) ** 2)
    return k / k.sum()


def blur(x, sig):
    """separable Gaussian blur of [B, C, H, W] (zero padding)"""
    k = _gauss1d(sig, x.device)
    r = (len(k) - 1) // 2
    c = x.shape[1]
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), padding=(r, 0), groups=c)
    return F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), padding=(0, r), groups=c)


def _rot_grid(theta_deg, dev):
    """sampling grid of M_theta(p) = M(R(theta) p): row = x, column = y, normalised coordinates y / 50 (width), x / 50 (height)"""
    a = math.radians(theta_deg)
    c, s = math.cos(a), math.sin(a)
    ax = (torch.arange(N, device=dev, dtype=torch.float32) + 0.5) * RES - HALF
    x, y = torch.meshgrid(ax, ax, indexing="ij")
    sx, sy = c * x - s * y, s * x + c * y
    return torch.stack([sy / HALF, sx / HALF], -1)[None]


class SoftRegister(nn.Module):
    def __init__(self, shift_px=12, yaws=tuple(range(-12, 13, 2)), win_px=2, win_yaw=1):
        super().__init__()
        self.S, self.yaws, self.win_px, self.win_yaw = shift_px, list(yaws), win_px, win_yaw
        self.log_T = nn.Parameter(torch.tensor(math.log(0.02)))
        self.w = nn.Parameter(torch.tensor([1.0, 0.5, 1.0]))  # narrow road, wide road, direction
        self.log_sig_t = nn.Parameter(torch.tensor(math.log(4.0)))  # prior width of the translation (m)
        self.log_sig_a = nn.Parameter(torch.tensor(math.log(8.0)))  # prior width of the yaw (deg)
        self.trust = nn.Linear(5, 1)
        with torch.no_grad():
            self.trust.weight.copy_(torch.tensor([[6.0, 2.0, 4.0, 0.3, 0.0]]))
            self.trust.bias.fill_(-4.0)

    def _maps(self, mp):
        """blurred, peak-normalised map channels before rotation: narrow road, wide road, blurred direction vector"""
        roads = mp[:, 1:2]
        vx, vy = mp[:, 1:2] * mp[:, 2:3], mp[:, 1:2] * mp[:, 3:4]
        f1, f2 = 1.5 * math.sqrt(2 * math.pi), 5.0 * math.sqrt(2 * math.pi)
        return blur(roads, 1.5) * f1, blur(roads, 5.0) * f2, blur(vx, 1.5) * f1, blur(vy, 1.5) * f1

    def forward(self, heat, dirf, mp):
        """heat [B, H, W] (0..1), dirf [B, 2, H, W] (travel direction of the dense head), mp [B, 5, H, W] (sd_lib.raster_sd of the map as posed) -> dict"""
        B, dev = heat.shape[0], heat.device
        E = heat.float().clamp(0, 1)
        d = dirf.float()
        d = d / torch.sqrt((d ** 2).sum(1, keepdim=True) + 1e-4)
        A = torch.stack([E * d[:, 0] ** 2, E * d[:, 0] * d[:, 1] * math.sqrt(2), E * d[:, 1] ** 2], 1)  # [B, 3, H, W]
        ev = E.sum((1, 2))
        pad = (0, NFFT - N, 0, NFFT - N)
        FE = torch.fft.rfft2(F.pad(E[:, None], pad))
        FA = torch.fft.rfft2(F.pad(A, pad))
        rs, rw, vx, vy = self._maps(mp.float())
        w = F.softplus(self.w)
        S = self.S
        idx = torch.arange(-S, S + 1, device=dev) % NFFT
        sc = []
        for th in self.yaws:
            g = _rot_grid(th, dev).expand(B, -1, -1, -1)
            m = F.grid_sample(torch.cat([rs, rw, vx, vy], 1), g, mode="bilinear", padding_mode="zeros", align_corners=False)
            a = math.radians(th)
            c, s = math.cos(a), math.sin(a)
            mx, my = c * m[:, 2] + s * m[:, 3], -s * m[:, 2] + c * m[:, 3]  # direction vectors seen from the perception frame (R^T m)
            Mq = torch.stack([m[:, 0], m[:, 1]], 1)
            Bd = torch.stack([mx * mx, mx * my * math.sqrt(2), my * my], 1)
            FM, FB = torch.fft.rfft2(F.pad(Mq, pad)), torch.fft.rfft2(F.pad(Bd, pad))
            fs = w[0] * torch.conj(FE[:, 0]) * FM[:, 0] + w[1] * torch.conj(FE[:, 0]) * FM[:, 1] + w[2] * (torch.conj(FA) * FB).sum(1)
            cc = torch.fft.irfft2(fs, s=(NFFT, NFFT))
            sc.append(cc[:, idx][:, :, idx])
        sc = torch.stack(sc, 1) / (ev.view(B, 1, 1, 1) + 1.0)  # [B, K, 2S+1, 2S+1]: evidence-normalised score, shift = (row, column) = (x, y)
        K = len(self.yaws)
        th = torch.tensor(self.yaws, device=dev, dtype=torch.float32)
        u = (torch.arange(-S, S + 1, device=dev, dtype=torch.float32)) * RES
        ux, uy = u.view(1, -1, 1), u.view(1, 1, -1)
        ca, sa = torch.cos(torch.deg2rad(th)).view(K, 1, 1), torch.sin(torch.deg2rad(th)).view(K, 1, 1)
        tx, ty = ca * ux - sa * uy, sa * ux + ca * uy  # [K, 2S+1, 2S+1]: translation of the map for the hypothesis (t = R u)
        prior = -0.5 * ((tx ** 2 + ty ** 2) / torch.exp(2 * self.log_sig_t) + th.view(K, 1, 1) ** 2 / torch.exp(2 * self.log_sig_a))
        logit = sc / torch.exp(self.log_T) + prior[None]
        flat = logit.reshape(B, -1)
        p = torch.softmax(flat, 1)
        am = flat.argmax(1)
        ik, rem = am // ((2 * S + 1) ** 2), am % ((2 * S + 1) ** 2)
        ir, ic = rem // (2 * S + 1), rem % (2 * S + 1)
        kk = torch.arange(K, device=dev).view(1, K, 1, 1)
        rr = torch.arange(2 * S + 1, device=dev).view(1, 1, -1, 1)
        cc_ = torch.arange(2 * S + 1, device=dev).view(1, 1, 1, -1)
        win = ((kk - ik.view(B, 1, 1, 1)).abs() <= self.win_yaw) & ((rr - ir.view(B, 1, 1, 1)).abs() <= self.win_px) & ((cc_ - ic.view(B, 1, 1, 1)).abs() <= self.win_px)
        lw = logit.masked_fill(~win, -1e4).reshape(B, -1)
        pw = torch.softmax(lw, 1).reshape(B, K, 2 * S + 1, 2 * S + 1)
        th_hat = (pw * th.view(1, K, 1, 1)).sum((1, 2, 3))
        t_hat = torch.stack([(pw * tx[None]).sum((1, 2, 3)), (pw * ty[None]).sum((1, 2, 3))], 1)
        sbest = sc.reshape(B, -1).gather(1, am[:, None])[:, 0]
        smed = sc.reshape(B, -1).median(1).values
        pmax = p.gather(1, am[:, None])[:, 0]
        feats = torch.stack([sbest, pmax, sbest - smed, torch.log1p(ev) / 8.0, torch.log1p(mp[:, 1].float().sum((1, 2))) / 8.0], 1)
        trust = torch.sigmoid(self.trust(feats))[:, 0]
        return dict(th=th_hat, t=t_hat, pmax=pmax, rho=sbest, ev=ev, trust=trust, feats=feats, argmax=(ik, ir, ic))


def accumulate(rows, scene, th, t, conf, R_ego, lam=0.9, c_min=0.0):
    """causal running mean per scene (rows in time order): world translation w = R_ego t and yaw, weights conf (>= c_min) with forgetting lam per frame
    -> th_acc [n], t_acc [n, 2] (in the ego frame of every frame: R_ego^T w_acc), c_acc [n] (accumulated confidence, 0 = nothing seen yet)"""
    n = len(rows)
    th_a, t_a, c_a = np.zeros(n), np.zeros((n, 2)), np.zeros(n)
    sw, sth, sw2 = {}, {}, {}
    for i in range(n):
        s = int(scene[i])
        sw[s] = sw.get(s, 0.0) * lam
        sth[s] = sth.get(s, 0.0) * lam
        sw2[s] = sw2.get(s, np.zeros(2)) * lam
        c = float(conf[i]) if conf[i] >= c_min else 0.0
        w = R_ego[i] @ t[i]
        sw[s] += c
        sth[s] += c * th[i]
        sw2[s] = sw2[s] + c * w
        if sw[s] > 1e-6:
            th_a[i] = sth[s] / sw[s]
            t_a[i] = R_ego[i].T @ (sw2[s] / sw[s])
        c_a[i] = sw[s]
    return th_a, t_a, c_a


def corrected_pose(R_e, d_e, th_hat, t_hat):
    """pose error of the map as drawn (R_e, d_e) and the estimate (theta, t) of the registration -> transform for sd_lib.raster_sd that draws the corrected map:
    x_map = R_e x + d_e, corrected = R_hat^T (x_map - t_hat) = (R_hat^T R_e) x + R_hat^T (d_e - t_hat)"""
    a = math.radians(th_hat)
    Rh = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return Rh.T @ R_e, Rh.T @ (np.asarray(d_e) - np.asarray(t_hat))


def _selftest():
    """synthetic: lane evidence = the roads of a random layout, the map = the same roads shifted by a known (theta, t) -> estimate within one pixel / 1 deg"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.RandomState(0)

    def roads():
        m = np.zeros((5, N, N), np.float32)
        for _ in range(4):
            a, b = rng.uniform(-40, 40, 2), rng.uniform(-40, 40, 2)
            for s in np.linspace(0, 1, 400):
                p = a + s * (b - a)
                i, j = int((p[0] + 50) / RES), int((p[1] + 50) / RES)
                if 0 <= i < N and 0 <= j < N:
                    t = (b - a) / np.linalg.norm(b - a)
                    m[0, i, j] = m[1, i, j] = 1.0
                    m[2, i, j], m[3, i, j], m[4, i, j] = t[0], t[1], 0.5
        return m

    base = roads()
    errs = []
    reg = SoftRegister().to(dev)
    for th_true, t_true in ((0.0, (0.0, 0.0)), (4.0, (2.0, -1.5)), (-6.0, (-3.0, 3.5)), (2.0, (1.0, 1.0))):
        # perception sees the true layout (base); the map is the layout transformed by the error (x_map = R x + t) -> render it by sampling
        a = math.radians(th_true)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        ax = (np.arange(N) + 0.5) * RES - HALF
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        P = np.stack([X, Y], -1).reshape(-1, 2)
        # map raster value at map-frame point q is the layout at R^T (q - t): inverse-warp the true layout
        Q = (P - np.array(t_true)) @ R  # = R^T (q - t) for row vectors
        ii = np.clip(np.round((Q[:, 0] + HALF) / RES - 0.5).astype(int), 0, N - 1)
        jj = np.clip(np.round((Q[:, 1] + HALF) / RES - 0.5).astype(int), 0, N - 1)
        ok = (np.abs(Q) < HALF).all(1)
        mp = np.zeros((5, N, N), np.float32)
        for ch in range(5):
            mp[ch] = (base[ch][ii, jj] * ok).reshape(N, N)
        # direction vectors of the layout rotate with the map
        v = np.stack([mp[2].reshape(-1), mp[3].reshape(-1)], 1) @ R.T
        mp[2], mp[3] = v[:, 0].reshape(N, N), v[:, 1].reshape(N, N)
        heat = torch.from_numpy(base[1])[None].to(dev)
        heat = blur(heat[:, None], 1.0)[:, 0] * 1.0
        heat = (heat / heat.max()).clamp(0, 1)
        dirf = torch.from_numpy(base[2:4])[None].to(dev)
        o = reg(heat, dirf, torch.from_numpy(mp)[None].to(dev))
        e_t = float(torch.linalg.norm(o["t"][0].cpu() - torch.tensor(t_true, dtype=torch.float32)))
        e_a = float(abs(o["th"][0].cpu() - th_true))
        errs.append((e_t, e_a))
        print(f"true theta {th_true:5.1f} t {t_true} -> estimate theta {float(o['th'][0]):6.2f} t ({float(o['t'][0, 0]):5.2f}, {float(o['t'][0, 1]):5.2f})  err {e_t:.2f} m {e_a:.2f} deg  pmax {float(o['pmax'][0]):.2f} rho {float(o['rho'][0]):.2f} trust {float(o['trust'][0]):.2f}")
    # gradient reaches the evidence
    heat = heat.clone().requires_grad_(True)
    o = reg(heat, dirf, torch.from_numpy(mp)[None].to(dev))
    (o["t"].sum() + o["th"].sum()).backward()
    print("gradient to the evidence: norm", float(heat.grad.norm()))
    assert all(e[0] < 1.0 and e[1] < 2.0 for e in errs), errs
    assert float(heat.grad.norm()) > 0
    print("selftest ok")


if __name__ == "__main__":
    _selftest()
