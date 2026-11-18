"""Loop 18 / R2.1: fit the few parameters of SoftRegister (K188) and the trust read-out (K189) on TRAIN frames.
Data: dense heat / direction of the out-of-fold dense heads (clean TRAIN frames) and of the real failure rows (7 perturbation types, p17_dense_train.py maps); SD map as drawn by the loop-17 simulator
(scene-constant pose error sigma in {0.5, 1, 1.5, 2, 3} m, yaw 2 sigma deg; 25 % maps of another place = wrong, 5 % no map). Loss: Huber of the pose (m, deg) on right maps, BCE of the trust.
Usage (gcar, GPU, < 2 GB): p18_regis_train.py --name regis_a --its 1500 [--fail_perts dark,blur,...]   -> runs/e18/<name>/{regis.pt, log.txt, status.json}"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p18_regis as RG  # noqa: E402
import p8_lib as L  # noqa: E402
import sd_lib as SL  # noqa: E402

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"
OUT = BP.RUNS
ALL_PERTS = "dark,blur,cam_front3_off,fog,radar_off,extr3,cam_front_off"


class RegData(torch.utils.data.Dataset):
    def __init__(self, n, seed, perts, p_fail=0.5, p_wrong=0.25, p_none=0.05, sigmas=(0.5, 1.0, 1.5, 2.0, 3.0)):
        self.n, self.seed, self.p_fail, self.p_wrong, self.p_none, self.sigmas = n, seed, p_fail, p_wrong, p_none, sigmas
        self.perts = [p for p in perts if p]
        self.P = dict(np.load(f"{D}/poses_train.npz"))
        self.clean = []
        for f in range(3):
            d = f"{OUT}/e37_fold{f}"
            self.clean.append((np.load(f"{d}/ids_oof.npy"), np.load(f"{d}/prob_oof.npy", mmap_mode="r"), np.load(f"{d}/dir_oof.npy", mmap_mode="r")))
        self.fail = {pn: (np.load(f"{D}/pert/{pn}/train_rows.npy"), np.load(f"{OUT}/e17/dense_trainoof/{pn}_prob.npy", mmap_mode="r"), np.load(f"{OUT}/e17/dense_trainoof/{pn}_dir.npy", mmap_mode="r")) for pn in self.perts}
        self.sdl = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.sdl is None:
            self.sdl = L.cache("sd", "train")
        rng = np.random.RandomState((self.seed * 1000003 + i) % (2 ** 32))
        if self.perts and rng.rand() < self.p_fail:
            rows, pr, dr = self.fail[self.perts[int(rng.randint(len(self.perts)))]]
        else:
            rows, pr, dr = self.clean[int(rng.randint(3))]
        j = int(rng.randint(len(rows)))
        row = int(rows[j])
        heat, dirf = np.asarray(pr[j], np.float32), np.asarray(dr[j], np.float32)
        sigma = float(self.sigmas[int(rng.randint(len(self.sigmas)))])
        u = rng.rand()
        mode = "none" if u < self.p_none else ("wrong" if u < self.p_none + self.p_wrong else "right")
        sd = self.sdl[row]
        scene = int(self.P["scene"][row])
        truth = np.zeros(3, np.float32)
        if mode == "right" and sd is not None:
            ras = SL.sd_raster(sd, row, scene, sigma, self.P, False).astype(np.float32)
            w, th = SL.scene_pose_error(scene, sigma)
            d = self.P["R"][row].T @ w
            truth = np.array([th, d[0], d[1]], np.float32)
        elif mode == "wrong":
            ras = np.zeros((5, 200, 200), np.float32)
            for _ in range(20):
                r2 = int(rng.randint(len(self.sdl)))
                if r2 != row and self.sdl[r2] is not None and int(self.P["scene"][r2]) != scene:
                    sdm, _, _ = L.draw_sd(self.sdl[r2], r2, 0.0, 0.0, 1)
                    ras = SL.raster_sd(sdm).astype(np.float32)
                    break
        else:
            ras, mode = np.zeros((5, 200, 200), np.float32), "none"
        lab = 1.0 if mode == "right" else 0.0
        return dict(heat=torch.from_numpy(heat), dirf=torch.from_numpy(dirf), mp=torch.from_numpy(ras), lab=torch.tensor(lab), truth=torch.from_numpy(truth), has=torch.tensor(1.0 if mode == "right" else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="regis_a")
    ap.add_argument("--its", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fail_perts", default=ALL_PERTS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    od = f"{OUT}/e18/{a.name}"
    os.makedirs(od, exist_ok=True)
    log = open(f"{od}/log.txt", "a")
    net = RG.SoftRegister().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.its, eta_min=a.lr * 0.05)
    ds = RegData(a.its * a.bs, a.seed, a.fail_perts.split(","))
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, drop_last=True, prefetch_factor=3, persistent_workers=True)
    t0, run = time.time(), {}
    for it, b in enumerate(dl, 1):
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        o = net(b["heat"], b["dirf"], b["mp"])
        h = b["has"]
        et = torch.linalg.norm(o["t"] - b["truth"][:, 1:3], dim=1)
        ea = (o["th"] - b["truth"][:, 0]).abs()
        l_pose = (h * (torch.nn.functional.huber_loss(et, torch.zeros_like(et), reduction="none", delta=1.0) + torch.nn.functional.huber_loss(ea, torch.zeros_like(ea), reduction="none", delta=1.0))).sum() / h.sum().clamp(min=1)
        l_tr = torch.nn.functional.binary_cross_entropy(o["trust"].clamp(1e-4, 1 - 1e-4), b["lab"])
        loss = l_pose + l_tr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            m = {"loss": float(loss), "pose": float(l_pose), "trust_bce": float(l_tr), "t_err_m": float((et * h).sum() / h.sum().clamp(min=1)), "th_err_deg": float((ea * h).sum() / h.sum().clamp(min=1)),
                 "trust_right": float((o["trust"] * b["lab"]).sum() / b["lab"].sum().clamp(min=1)), "trust_wrong": float((o["trust"] * (1 - b["lab"])).sum() / (1 - b["lab"]).sum().clamp(min=1))}
        for k, v in m.items():
            run[k] = 0.95 * run.get(k, v) + 0.05 * v
        if it % 50 == 0 or it == 1:
            msg = f"it {it}/{a.its} {time.time() - t0:.0f}s " + " ".join(f"{k} {v:.3f}" for k, v in run.items()) + f" | T {float(torch.exp(net.log_T)):.4f} w {[round(float(x), 2) for x in torch.nn.functional.softplus(net.w)]} sig_t {float(torch.exp(net.log_sig_t)):.2f} sig_a {float(torch.exp(net.log_sig_a)):.2f}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    torch.save(net.state_dict(), f"{od}/regis.pt")
    json.dump(dict(status="ok", args=vars(a), seconds=time.time() - t0, final={k: float(v) for k, v in run.items()}), open(f"{od}/status.json", "w"))
    print("DONE ok", f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
