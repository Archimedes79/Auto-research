"""The radar encoder MODULE (training/radar_encoder.py) on real nuScenes mini: does it keep the detections, and what does the TASK do to that?

Mirko, 2026-10-26: use the reconstruction as a side loss to help first, but the side loss is not the goal. So three trainings of the
same module (2000 steps, 8 train scenes, 2 held-out scenes, C = 64), each followed by the same honest measurement:
  aux only         side loss alone (what studies/early/radar_codecs.py measured; the information ceiling of the module)
  task only        a proxy task alone: "does the cell hold a moving detection (|vr| > 1 m/s)?" - 1 x 1 head on the features
  task + aux       the proxy task plus the side loss with aux_weight(step): 1.0 decaying to the floor 0.05 (half life 500 steps)
Measurement: kept fraction = share of the detections whose six fields (position, distance, direction, Doppler, RCS, age) all
decode within tolerance, from the oracle (decode the raster itself, no learning) and from the features. Features are probed with a
FRESH 1 x 1 decoder (pass-through start, like the module's own) trained on the frozen features (steps / 2, a lower bound: a longer
probe can only keep more); where the side loss was on, the module's own jointly trained decoder is reported too (meaningless when the
side loss is off). The task score (F1 on occupied cells) shows that the side loss does not buy the
information with the task: it has to stay close to "task only".
Gate (printed, exit code): task + aux keeps >= 0.9 x the oracle on held-out scenes and its task F1 is within 0.03 of task only.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python -u studies/early/radar_encoder_probe.py [steps]
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
import radar_encoder as re  # noqa: E402
import radar_recon_test as rt  # noqa: E402

DEV = rt.DEV
fails = []
MOVING = 1.0                                              # |vr| above this: a moving detection (m/s)


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def moving_target(det, B, grid):
    """(B, NX, NY) 1 where the cell holds a moving detection, and (B, NX, NY) 1 where it holds any detection."""
    idx, ix, iy, cell, _, _, _ = re._slots(det, grid, re.K_SLOTS)
    mov = torch.zeros(B * grid.nx * grid.ny, device=DEV)
    occ = torch.zeros_like(mov)
    occ[cell] = 1.0
    m = det["vr"][idx].abs() > MOVING
    mov[cell[m]] = 1.0
    return mov.view(B, grid.nx, grid.ny), occ.view(B, grid.nx, grid.ny)


def train(mode, steps, frames, seed=0):
    torch.manual_seed(seed)
    enc = re.RadarEncoder(C=64).to(DEV)
    head = nn.Conv2d(64, 1, 1).to(DEV)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = np.random.RandomState(seed)
    for it in range(steps):
        det = rt.batch([frames[i] for i in g.choice(len(frames), 8, replace=False)])
        f, st = enc(det, 8, with_state=True)
        mov, occ = moving_target(det, 8, enc.grid)
        task = (F.binary_cross_entropy_with_logits(head(f)[:, 0], mov, reduction="none") * occ).sum() / occ.sum().clamp(min=1)
        w_aux = {"aux only": 1.0, "task only": 0.0, "task + aux": re.aux_weight(it, 1.0, 0.05, 500)}[mode]
        w_task = 0.0 if mode == "aux only" else 1.0
        loss = w_task * task + w_aux * (enc.aux_loss(f, st) if w_aux > 0 else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return enc.eval(), head.eval()


@torch.no_grad()
def task_f1(enc, head, frames):
    tp = fp = fn = 0
    for i in range(0, len(frames), 8):
        det = rt.batch(frames[i:i + 8])
        mov, occ = moving_target(det, len(frames[i:i + 8]), enc.grid)
        p = (head(enc(det, len(frames[i:i + 8])))[:, 0] > 0).float() * occ
        tp += (p * mov).sum().item()
        fp += (p * (1 - mov)).sum().item()
        fn += ((1 - p) * mov * occ).sum().item()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def probe(enc, steps, train_frames, seed=0):
    """fresh decoder on FROZEN features, trained with the module's tolerance-weighted side loss."""
    dec = enc.new_decoder()
    opt = torch.optim.Adam(dec.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = np.random.RandomState(seed + 1)
    for _ in range(steps):
        det = rt.batch([train_frames[i] for i in g.choice(len(train_frames), 8, replace=False)])
        with torch.no_grad():
            f, st = enc(det, 8, with_state=True)
        loss = enc.aux_loss(f, st, decoder=dec)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return dec.eval()


@torch.no_grad()
def kept(enc, frames, dec=None, use_features=True):
    num = den = 0.0
    per = np.zeros(len(re.TOLERANCES))
    for i in range(0, len(frames), 8):
        fs = frames[i:i + 8]
        det = rt.batch(fs)
        n = len(re._slots(det, enc.grid, enc.K)[0])
        share, pf = enc.kept_fraction(det, len(fs), features=enc(det, len(fs)) if use_features else None, decoder=dec)
        num += share * n
        den += n
        per += np.array(list(pf.values())) * n
    return num / den, per / den


def main(steps):
    tr, te = rt.load(set(range(8))), rt.load({8, 9})
    print(f"RadarEncoder on real nuScenes mini: {len(tr)} train frames, {len(te)} held-out frames, {steps} steps, C = 64")
    ref = re.RadarEncoder(C=64).to(DEV)
    oracle_te, _ = kept(ref, te, use_features=False)
    oracle_tr, _ = kept(ref, tr, use_features=False)
    print(f"oracle (decode the raster itself): held-out {100 * oracle_te:.1f}%, train {100 * oracle_tr:.1f}%")
    print("mode         | task F1 held-out | kept held-out | kept train | gap  | own dec. | violating % per field (" + ", ".join(re.TOLERANCES) + ")")
    res = {}
    for mode in ("aux only", "task only", "task + aux"):
        enc, head = train(mode, steps, tr)
        f1 = task_f1(enc, head, te)
        dec = probe(enc, max(steps // 2, 1), tr)
        k_te, per = kept(enc, te, dec)
        k_tr, _ = kept(enc, tr, dec)
        res[mode] = (f1, k_te, k_tr)
        own = "n/a" if mode == "task only" else f"{100 * kept(enc, te)[0]:.1f}%"
        print(f"{mode:12s} | {f1:16.3f} | {100 * k_te:12.1f}% | {100 * k_tr:9.1f}% | {100 * (k_tr - k_te):+4.1f} | {own:>9s} | "
              + ", ".join(f"{100 * v:.0f}" for v in per), flush=True)
    f1_t, k_t, _ = res["task only"]
    f1_a, k_a, _ = res["task + aux"]
    check(k_a >= 0.9 * oracle_te, f"task + aux keeps >= 90 % of the oracle on held-out scenes ({100 * k_a:.1f}% vs {100 * oracle_te:.1f}%)")
    check(f1_a >= f1_t - 0.03, f"the side loss does not cost the task (F1 {f1_a:.3f} vs task only {f1_t:.3f})")
    print(f"task only keeps {100 * k_t:.1f}% (what the task alone throws away: {100 * (oracle_te - k_t):.1f} points of the oracle)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
    print("RADAR ENCODER GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    sys.exit(1 if fails else 0)
