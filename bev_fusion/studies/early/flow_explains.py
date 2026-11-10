"""Does the velocity field EXPLAIN the motion, or is the next frame only predicted well? (Mirko, 2026-10-26)

Sequence of BEV maps with moving boxes (tests/support/synthetic_world.py: cars, trucks, pedestrians with ground velocities,
lane lines, exact ego motion; frames F0, F1, F2 dt apart, like consecutive nuScenes GT keyframes). A small model of
its own - encoder E (3 x 3 convs) per frame, the temporal block (studies/early/temporal.py: TemporalPropagation velocity
hypotheses + FlowRefine, K iterations, bounded steps) and decoder D - predicts F1 from F0 ONLY through the motion:
  P = D(propagate(E(F0), v)),  v = FlowRefine(E(F0), E(F1), v0 = hypothesis mean)     (E(F1) only for the lookup)
trained on BCE(P, F1) - no velocity label. The decoder could learn to move things itself (a conv net can shift);
the tests separate a good prediction from a velocity field that explains it:
  1 prediction      IoU of D(P) with F1 (vehicles)
  2 velocity        end-point error (m/s) against the GT box velocity on the box cells; on lane cells (aperture)
  3 explanation     the RAW map F0 moved with v alone (no encoder, no decoder) vs F1: IoU on vehicles
  4 extrapolation   v carried to frame 1 (moved with itself, rotated with the ego) moves F1 to t + 2 dt vs F2:
                    a field that only fits one pair does not extrapolate, a physical one does
  5 controls        v = 0 (ego motion only) through the same D and as a raw move; GT velocity as a raw move (the
                    reference of 3 and 4, limited only by the bilinear splat)
Claims (PASS / FAIL): EPE on vehicles < 1 m/s; raw move (3) and extrapolation (4) reach >= 90 % of the GT-velocity
IoU; the learned field beats v = 0 by > 0.1 IoU in 3; the prediction (1) is not much better than the raw move (3)
(< 0.1 IoU: the decoder does not carry the motion).
Image: temp/checks/flow_explains.png. Run from the bev_fusion folder:
  wsl ~/miniconda3/envs/gcar/bin/python studies/early/flow_explains.py [--quick] [--seed N] [--dt 0.25] [--prior 1e-3]
      [--sharp] (softmax-splatting importance + pi entropy 5e-3) [--pyr] (FlowPyramid 2 + 1 + 1 iterations instead of
      4 iterations of FlowRefine at 0.5 m) [--smooth 0.05] (edge-aware smoothness of every flow, weights from the input
      map: one box moves as one body - the first run failed 0 / 5: the field was a patchwork inside each box, the raw
      move scattered the boxes into debris, the decoder repaired it; prediction IoU 0.85, raw move 0.27, v = 0 0.42)
      [--doppler] (simulated radar Doppler, a detection in half the occupied cells, applied to the start field and after
      EVERY refinement iteration - in studies/early/flow_refine.py this kept the stripes static and improved A)
      [--pyr-leak] (the pyramid without detaching between levels: the pre-fix behaviour, compared on seeds)
      [--vmax 20] (soft physical speed bound of the refined field)
      [--rawwarp 1.0] (the explanation as a loss: MSE of the raw F0 moved with v alone against F1, on every refined flow;
      the decoder alone can no longer carry the motion)
"""
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "tests/support")
import scratch  # noqa: E402
from lift_pd import Grid  # noqa: E402
from synthetic_world import gt_bev, random_scene  # noqa: E402
from flow_viz import flow_rgb, wheel  # noqa: E402
from temporal import (FlowPyramid, FlowRefine, TemporalPropagation, doppler_update, landing, propagate,  # noqa: E402
                      sample_at, steps_tensor)

QUICK = "--quick" in sys.argv


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


SEED, DT, PRIOR, K = arg("--seed", 0), arg("--dt", 0.25), arg("--prior", 1e-3), 4
SHARP, PYR, ENT = "--sharp" in sys.argv, "--pyr" in sys.argv, arg("--ent", 5e-3)
SMOOTH = arg("--smooth", 0.0)
DOPPLER = "--doppler" in sys.argv
RAWWARP = arg("--rawwarp", 0.0)                   # > 0: MSE of the RAW map F0 moved with v against F1, on every flow
SS = arg("--ss", 4)                                # GT sub-samples per cell (2 missed the lane lines entirely)
VMAX = arg("--vmax", 0.0)                         # > 0: physical speed bound of the refined field (FlowRefine v_max)
LEAK = "--pyr-leak" in sys.argv                 # FlowPyramid(detach_levels=False): the pre-fix behaviour, for comparison
TAG = ("_sharp" if SHARP else "") + ("_pyr" if PYR else "") + (f"_smooth{SMOOTH}" if SMOOTH > 0 else "") + ("_doppler" if DOPPLER else "") + ("_leak" if LEAK else "") + (f"_vmax{VMAX:g}" if VMAX > 0 else "") + (f"_raw{RAWWARP:g}" if RAWWARP > 0 else "") + (f"_prior{PRIOR:g}" if PRIOR != 1e-3 else "") + (f"_s{SEED}" if SEED else "")
GATE = "--gate" in sys.argv or "--gate-doppler" in sys.argv   # learned soft gate carried vs fresh start, recursive training
GATE_DOPPLER = "--gate-doppler" in sys.argv     # + each candidate's Doppler residual (before the update) as gate input
if GATE:
    TAG = TAG + ("_gatedop" if GATE_DOPPLER else "_gate")
DATA = arg("--data", "synthetic")                 # "nuscenes": real GT boxes (data/cache/gt), see nuscenes_items
if DATA == "nuscenes":
    TAG = "_nusc" + TAG
G = (Grid(x_min=0.0, x_max=60.0, y_min=-26.0, y_max=26.0, res=0.5) if DATA == "nuscenes"          # 120 x 104 (gt_bev)
     else Grid(x_min=0.0, x_max=40.0, y_min=-20.0, y_max=20.0, res=0.5))                           # 80 x 80 cells
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9 else "cpu"
C = 16


def frames_seeded(seed):
    return frames(random.Random(seed))


def dataset(seeds, name):
    """render the sequences of the given seeds (4 processes), cached in temp/checks/ (scratch, not in git)."""
    def build(m):
        import multiprocessing as mp
        with mp.get_context("fork").Pool(4) as pool:
            return pool.map(frames_seeded, seeds[:m], chunksize=8)
    # temp/checks/flow_explains_<name>_dt<DT>_ss<SS>_<first seed>_<n>.pt: a bigger cache serves a smaller n; one built by another frames() is deleted
    return scratch.cached(f"flow_explains_{name}_dt{DT}_ss{SS}_{seeds[0]}", build, len(seeds), sig=scratch.source_sig(frames, frames_seeded))


def frames(rng):
    """-> maps (3, 3, X, Y) of F0, F1, F2 (lanes, vehicles, pedestrians), GT velocity of F0 (2, X, Y), box id of F0,
    ego steps from one frame to the next."""
    s0 = random_scene(rng, n_boxes=(3, 7), x_range=(6, 34), y_range=(-15, 15))
    s1 = s0.step(DT)
    s2 = s1.step(DT)
    # ss = 4: with ss = 2 the sub-samples sit at 0.125 / 0.375 m inside a cell and the 0.15 m lane lines (at y = +-1.75,
    # +-5.25 m) fall exactly between them - F0 had NO lanes (F1 / F2 only dashed pieces after the ego rotation) in
    # every run up to 2026-10-26
    g = [gt_bev(s, G, ss=SS) for s in (s0, s1, s2)]
    maps = torch.stack([torch.stack([x["lanes"], x["vehicles"], x["pedestrians"]]) for x in g])
    st = [(DT, s0.v, s0.omega)]
    return maps, g[0]["velocity"], g[0]["box_id"], st, st          # steps F0 -> F1, F1 -> F2 (the same here)


def nuscenes_item(path, with_v1=False):
    """one cached GT sample (training/gt_bev.py via nuscenes_dataset: keyframes k - 1, k, k + 1 at 0.5 m) -> the item of
    this test: maps (3, 3, X, Y) = (lanes = lane / road divider, vehicles = vehicle + two-wheeler, pedestrians) of F0 =
    k - 1, F1 = k, F2 = k + 1; GT velocity (2, X, Y) of F0 (ground velocity, F0 axes; stationary boxes 0); instance map
    of F0; ego steps F0 -> F1 (the 'prev' chain of k reversed: k -> k - 1 runs with negative dt) and F1 -> F2 (the 'next'
    chain). Returns None without a moving vehicle (> 1 m/s) in F0."""
    import numpy as np
    g = np.load(path, allow_pickle=True)["gt"].item()

    def maps_of(f):
        lanes = np.maximum(f["lane_divider"], f["road_divider"]).astype(np.float32)
        veh = np.maximum(f["vehicle"], f["two_wheeler"]).astype(np.float32)
        return np.stack([lanes, veh, f["pedestrian"].astype(np.float32)])
    vel = g["prev"]["velocity"].astype(np.float32)
    speed = np.hypot(vel[0], vel[1])
    if not ((maps_of(g["prev"])[1] > 0.5) & (speed > 1.0)).any():
        return None
    maps = torch.from_numpy(np.stack([maps_of(g["prev"]), maps_of(g["cur"]), maps_of(g["next"])]))
    s01 = [(-float(dt), float(v), float(w)) for dt, v, w in reversed(g["prev"]["meta"]["steps"])]
    s12 = [(float(dt), float(v), float(w)) for dt, v, w in g["next"]["meta"]["steps"]]
    item = (maps, torch.from_numpy(vel), torch.from_numpy(g["prev"]["instance"].astype(np.int64)), s01, s12)
    if with_v1:                                          # + the GT velocity of F1 (the recursive test scores at F1)
        item = item + (torch.from_numpy(g["cur"]["velocity"].astype(np.float32)),)
    return item


def nuscenes_items(test, n, with_v1=False):
    """n items with a moving vehicle from the GT cache; split by SCENE (one in five scenes is held out: no scene in both
    sets), files in a fixed pseudo-random order (sorted by token). Cached in temp/checks (scratch)."""
    import glob
    import numpy as np
    def build(m):
        out = []
        for f in sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz")):
            scene = np.load(f, allow_pickle=True)["gt"].item()["cur"]["meta"]["scene"]
            if (int(scene[:8], 16) % 5 == 0) != test:
                continue
            it = nuscenes_item(f, with_v1)
            if it is not None:
                out.append(it)
                if len(out) == m:
                    break
        return out
    return scratch.cached("flow_explains_nusc" + ("_v1" if with_v1 else ""), build, n, split="test" if test else "train",
                          deps=("data/cache/gt",), sig=scratch.source_sig(nuscenes_item))


def radar_measure(f0, vgt, sigma_d=0.1, rate=0.5):
    """simulated Doppler of frame 0: radar at the origin, a detection in a random half of the occupied cells (boxes and
    lane lines), d = u . v_ground + noise (lane lines: 0) -> (u, d, valid) for doppler_update."""
    B, N = f0.shape[0], G.nx * G.ny
    xs, ys = G.xy_centres(device=f0.device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    r = torch.sqrt(gx ** 2 + gy ** 2).clamp(min=1e-3)
    u = torch.stack([gx / r, gy / r]).reshape(1, 1, 2, N).expand(B, 1, 2, N)
    d = (u[:, 0] * vgt.reshape(B, 2, N)).sum(1, keepdim=True) + sigma_d * torch.randn(B, 1, N, device=f0.device)
    valid = ((f0.max(1).values > 0.5) & (torch.rand(B, G.nx, G.ny, device=f0.device) < rate)).float().reshape(B, 1, N)
    return u, d, valid


def radar_apply(v, meas, sigma_p=4.0, sigma_d=0.1):
    u, d, valid = meas
    B, N = v.shape[0], G.nx * G.ny
    P = (torch.eye(2, device=v.device) * sigma_p ** 2).view(1, 2, 2, 1).expand(B, 2, 2, N)
    out, _ = doppler_update(v.reshape(B, 2, N), P, u, d, valid, sigma_d=sigma_d)
    return out.view_as(v)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, C, 3, padding=1))
        self.tp = TemporalPropagation(cin=C, grid=G, n_hyp=4, static_hyp=True, init_speed=5.0, importance=SHARP)
        vkw = {"v_max": VMAX} if VMAX > 0 else {}
        self.rf = (FlowPyramid(C, G, factors=(4, 2, 1), iters=(2, 1, 1), detach_levels=not LEAK, **vkw) if PYR
                   else FlowRefine(C, G, radius=3, **vkw))
        self.dec = nn.Sequential(nn.Conv2d(C + 1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 3, 1))
        if GATE:      # [features, fresh v, carried v, their residuals] -> softmax weights (fresh, carried); zero start = 0.5 / 0.5
            self.gate = nn.Sequential(nn.Conv2d(C + 6 + (3 if GATE_DOPPLER else 0), 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, 2, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        self.last_gate = None

    @staticmethod
    def residual(f0, f1, steps, v):
        """(B, 1, X, Y): how badly a velocity explains the next frame at each SOURCE cell - |F0(p) - F1(landing(p))| summed
        over the channels (raw maps, bilinear lookup of F1 at the landing point)."""
        S = steps_tensor(steps, f0.shape[0], f0.device, f0.dtype)
        look = sample_at(f1, landing(S, v, G), G).view_as(f0)
        return (f0 - look).abs().sum(1, keepdim=True)

    def forward(self, f0, f1, steps, v_override=None, vgt=None, v_init=None, v_carried=None):
        e0, e1 = self.enc(f0), self.enc(f1)
        r = self.tp(e0, steps)
        self.ent = r["entropy"]
        imp = self.tp.importance_of(e0.flatten(2)).view(e0.shape[0], G.nx, G.ny) if SHARP else None
        v0, corr = (r["v_mean"] if v_init is None else v_init), None      # v_init: recursive start (carried estimate)
        dres = None
        if DOPPLER and vgt is not None and v_override is None:      # the measurement corrects the start and EVERY iteration
            meas = radar_measure(f0, vgt)
            if GATE_DOPPLER and v_carried is not None:              # disagreement with the radar BEFORE the update (after
                u, d, valid = meas                                   # it both candidates agree radially by construction)
                B_ = v0.shape[0]

                def dop_res(v):
                    pred = (u[:, 0] * v.reshape(B_, 2, -1)).sum(1)
                    return ((pred - d[:, 0]).abs() * valid[:, 0]).view(B_, 1, G.nx, G.ny) / 10.0
                dres = torch.cat([dop_res(v0), dop_res(v_carried), valid.view(B_, 1, G.nx, G.ny)], 1)
            v0 = radar_apply(v0, meas)
            corr = lambda v: radar_apply(v, meas)                     # noqa: E731
            if v_carried is not None:
                v_carried = radar_apply(v_carried, meas)
        if v_carried is not None and GATE and v_override is None:     # learned soft blend of the two starts (no hard choice)
            rf_, rc_ = self.residual(f0, f1, steps, v0), self.residual(f0, f1, steps, v_carried)
            gin = [e0, v0 / 10.0, v_carried / 10.0, rf_, rc_]
            if GATE_DOPPLER:
                gin.append(dres if dres is not None else e0.new_zeros(e0.shape[0], 3, G.nx, G.ny))
            w = torch.softmax(self.gate(torch.cat(gin, 1)), 1)
            self.last_gate = w.detach()
            v0 = w[:, :1] * v0 + w[:, 1:] * v_carried
        if v_override is not None:
            flows = [v_override]
        elif PYR:
            flows = [v0] + self.rf(e0, e1, steps, v0, correct=corr)
        else:
            flows = [v0] + self.rf(e0, e1, steps, v0, iters=K, correct=corr)
        preds = []
        for v in flows:
            p, mass = propagate(e0, steps, v, G, importance=imp)
            preds.append(self.dec(torch.cat([p, mass.clamp(max=1.0)], 1)))
        mix = self.dec(torch.cat([r["feat"], r["mass"].clamp(max=1.0)], 1))
        return flows, preds, mix


def smoothness(v, feat, alpha=10.0):
    """edge-aware first-order smoothness of a flow (B, 2, X, Y) m/s over the 4-neighbours, weights from the map."""
    tot = 0.0
    for dim in (2, 3):
        dv = (v.diff(dim=dim) / 10.0).abs().sum(1)
        w = torch.exp(-alpha * feat.diff(dim=dim).abs().sum(1))
        tot = tot + (w * dv).mean()
    return tot


def pair_loss(model, src, tgt, steps, flows, preds, mix):
    """the training loss of one frame pair (no velocity labels): prediction BCE per refinement output, entropy,
    smoothness, raw warp, zero-motion prior - the same terms as the pair training."""
    loss = F.binary_cross_entropy_with_logits(mix, tgt)
    for k, p in enumerate(preds[1:]):
        loss = loss + 0.8 ** (len(preds) - 2 - k) * F.binary_cross_entropy_with_logits(p, tgt)
    if SHARP:
        loss = loss + ENT * (src.max(1).values * model.ent).mean()
    if SMOOTH > 0:
        loss = loss + SMOOTH * sum(smoothness(v, src) for v in flows)
    if RAWWARP > 0:
        loss = loss + RAWWARP * sum(((propagate(src, steps, v, G)[0] - tgt) ** 2).mean() for v in flows[1:])
    if PRIOR > 0:
        occ = src.max(1).values
        loss = loss + PRIOR * sum((occ * (v / 10.0).pow(2).sum(1)).mean() for v in flows)
    return loss


def train(model, data, iters, rng, bs=8, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for it in range(iters):
        batch = [data[rng.randrange(len(data))] for _ in range(bs)]
        m = torch.stack([b[0] for b in batch]).to(DEV)
        steps = [b[3] for b in batch]
        vg = torch.stack([b[1] for b in batch]).to(DEV)
        flows, preds, mix = model(m[:, 0], m[:, 1], steps, vgt=vg)
        loss = pair_loss(model, m[:, 0], m[:, 1], steps, flows, preds, mix)
        if GATE:          # recursive: the estimate of pair 1 carried to F1 (detached) is blended with pair 2's fresh start
            s12 = [b[4] for b in batch]
            vg1 = torch.stack([b[5] for b in batch]).to(DEV)
            carried = carry(flows[-1].detach(), steps)
            f2, p2, mix2 = model(m[:, 1], m[:, 2], s12, vgt=vg1, v_carried=carried)
            loss = loss + pair_loss(model, m[:, 1], m[:, 2], s12, f2, p2, mix2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            print(f"    iter {it:5d} loss {loss.item():.4f}  ({time.time() - t0:.0f} s)", flush=True)


def rotate_into_next(v, steps):
    """velocity vectors (B, 2, X, Y) from the source axes into the axes after the steps (ego yaw), per sample: steps is a
    list of B step lists (before 2026-10-30 the yaw of sample 0 was used for the whole batch - right only for batch 1,
    which is all the evaluations used)."""
    th = torch.tensor([sum(dt * w for dt, _, w in st) for st in steps], dtype=v.dtype, device=v.device).view(-1, 1, 1)
    th = th.expand(v.shape[0], 1, 1) if th.shape[0] == 1 else th
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], 1)


def carry(v, steps):
    """the velocity field of frame 0 moved to frame 1 (with itself) and rotated into its axes (no mass gate)."""
    moved, _ = propagate(v, steps, v, G, mass_gate=False)
    return rotate_into_next(moved, steps)


def iou(a, b, thr=0.5):
    a, b = a > thr, b > thr
    return (a & b).sum().item(), (a | b).sum().item()


@torch.no_grad()
def evaluate(model, test):
    keys = ("pred", "raw", "extrap", "pred v=0", "raw v=0", "raw GT v", "extrap GT v")
    acc = {k: [0, 0] for k in keys}
    epe = {"vehicles": [0.0, 0], "pedestrians": [0.0, 0], "lanes": [0.0, 0]}
    for maps, vgt, bid, steps, steps12, *_ in test:
        m = maps[None].to(DEV)
        flows, preds, _ = model(m[:, 0], m[:, 1], [steps], vgt=vgt[None].to(DEV))
        v = flows[-1]
        zero = torch.zeros_like(v)
        _, p0, _ = model(m[:, 0], m[:, 1], [steps], v_override=zero)
        vg = vgt[None].to(DEV)
        out = {"pred": torch.sigmoid(preds[-1]), "pred v=0": torch.sigmoid(p0[0]),
               "raw": propagate(m[:, 0], [steps], v, G)[0], "raw v=0": propagate(m[:, 0], [steps], zero, G)[0],
               "raw GT v": propagate(m[:, 0], [steps], vg, G)[0],
               "extrap": propagate(m[:, 1], [steps12], carry(v, [steps]), G)[0],
               "extrap GT v": propagate(m[:, 1], [steps12], carry(vg, [steps]), G)[0]}
        for k, o in out.items():
            ref = m[:, 2] if k.startswith("extrap") else m[:, 1]
            i, u = iou(o[0, 1], ref[0, 1])
            acc[k][0] += i
            acc[k][1] += u
        vc = v[0].cpu()
        e = (vc - vgt).norm(dim=0)
        veh = maps[0, 1] > 0.5
        inner = veh & (-F.max_pool2d(-veh.float()[None, None], 3, 1, 1)[0, 0] > 0.5)   # all 8 neighbours inside too
        masks = {"vehicles": veh, "pedestrians": maps[0, 2] > 0.5,
                 # lane lines are 0.15 m wide in 0.5 m cells (coverage <= 0.3): > 0.3 left the mask empty and printed
                 # "EPE lanes 0.000" in every run up to 2026-10-26 - not a measurement
                 "lanes": (maps[0, 0] > 0.05) & (maps[0, 1:].max(0).values < 0.1),
                 "vehicle interiors": inner, "vehicle edges": veh & ~inner,
                 "vehicles moving (> 1 m/s)": veh & (vgt.norm(dim=0) > 1.0),
                 "vehicles static (< 0.2 m/s)": veh & (vgt.norm(dim=0) < 0.2)}
        for k, mk in masks.items():
            epe.setdefault(k, [0.0, 0])
            epe[k][0] += e[mk].sum().item()
            epe[k][1] += int(mk.sum())
        for k in bid.unique().tolist():                 # object level: the error of each box's MEAN velocity
            cells = (bid == k) & veh
            if k >= 0 and cells.any():
                epe.setdefault("vehicles per box (mean v)", [0.0, 0])
                epe["vehicles per box (mean v)"][0] += (vc[:, cells].mean(1) - vgt[:, cells].mean(1)).norm().item()
                epe["vehicles per box (mean v)"][1] += 1
    res = {"IoU " + k: i / max(1, u) for k, (i, u) in acc.items()}
    res.update({"EPE " + k: s / max(1, n) for k, (s, n) in epe.items()})
    return res




def _seq_meta(path):
    import numpy as np
    g = np.load(path, allow_pickle=True)["gt"].item()
    return path, g["cur"]["meta"]["scene"], g["cur"]["meta"]["timestamp"], g["cur"]["meta"]["sample"], g["prev"]["meta"]["sample"]


def nuscenes_sequences(test, n_seq, length):
    """n_seq sequences of `length` consecutive keyframes of one scene (held-out scenes if test): consecutive cached
    samples chain (the k of sample i is the k - 1 of sample i + 1). -> list of dicts: maps (T, 3, X, Y), velocity
    (T, 2, X, Y) (GT of each keyframe, its axes), steps[t] = ego steps keyframe t -> t + 1. Cached in temp/checks."""
    import glob
    import numpy as np
    path = f"temp/checks/flow_sequences_nusc_{'test' if test else 'train'}_{n_seq}_{length}.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    import multiprocessing as mp
    files = sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz"))
    with mp.get_context("fork").Pool(4) as pool:
        meta = pool.map(_seq_meta, files, chunksize=64)
    by_scene = {}
    for f, scene, ts, tok, prev_tok in meta:
        if (int(scene[:8], 16) % 5 == 0) == test:
            by_scene.setdefault(scene, []).append((ts, f, tok, prev_tok))
    out = []
    for scene in sorted(by_scene):
        rows = sorted(by_scene[scene])
        run = [rows[0]]
        for r in rows[1:]:                                   # keep only chains of directly consecutive keyframes
            run = run + [r] if r[3] == run[-1][2] else [r]
            if len(run) == length - 1:                       # length - 1 samples give length keyframes (prev of the first)
                out.append([x[1] for x in run])
                run = []
                break
        if len(out) == n_seq:
            break
    seqs = []
    for files_ in out:
        gs = [np.load(f, allow_pickle=True)["gt"].item() for f in files_]

        def maps_of(fr):
            return np.stack([np.maximum(fr["lane_divider"], fr["road_divider"]).astype(np.float32),
                             np.maximum(fr["vehicle"], fr["two_wheeler"]).astype(np.float32),
                             fr["pedestrian"].astype(np.float32)])
        frames = [gs[0]["prev"]] + [g["cur"] for g in gs]
        steps = [[(-float(dt), float(v), float(w)) for dt, v, w in reversed(g["prev"]["meta"]["steps"])] for g in gs]
        seqs.append({"maps": torch.from_numpy(np.stack([maps_of(fr) for fr in frames])),
                     "velocity": torch.from_numpy(np.stack([fr["velocity"].astype(np.float32) for fr in frames])),
                     "steps": steps})
    torch.save(seqs, path)
    return seqs


@torch.no_grad()
def sequence_report(model, seqs, image):
    """Per keyframe t of the sequences: the velocity error at F_t (vehicles, all / moving > 1 m/s) of the fresh estimate
    of the pair (F_t, F_t+1) and of the recursive one (the estimate of the previous pair carried to F_t as the START of
    this pair's refinement). Recursion pays if its error falls with t while the fresh one stays flat."""
    T = len(seqs[0]["steps"])
    names = ("fresh", "recursive") + (("gated",) if GATE else ())
    acc = {(n, k, t): [0.0, 0] for n in names for k in ("vehicles", "moving") for t in range(T)}
    for sq in seqs:
        m, vel, steps = sq["maps"][None].to(DEV), sq["velocity"], sq["steps"]
        carried, carried_g = None, None
        for t in range(T):
            vg = vel[t][None].to(DEV)
            fresh = model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg)[0][-1]
            recur = fresh if carried is None else model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg, v_init=carried)[0][-1]
            carried = carry(recur, [steps[t]])
            gated = None
            if GATE:
                gated = fresh if carried_g is None else model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg, v_carried=carried_g)[0][-1]
                carried_g = carry(gated, [steps[t]])
            veh = sq["maps"][t, 1] > 0.5
            for k, mk in (("vehicles", veh), ("moving", veh & (vel[t].norm(dim=0) > 1.0))):
                for n, v in (("fresh", fresh), ("recursive", recur), ("gated", gated)):
                    if v is None:
                        continue
                    e = (v[0].cpu() - vel[t]).norm(dim=0)
                    acc[(n, k, t)][0] += e[mk].sum().item()
                    acc[(n, k, t)][1] += int(mk.sum())
    res = {key: s_ / max(1, n_) for key, (s_, n_) in acc.items()}
    print(f"== recursive speed refinement over {len(seqs)} held-out sequences of {T + 1} keyframes (EPE m/s at F_t)")
    print("  t   vehicles " + " / ".join(names) + "    moving vehicles " + " / ".join(names))
    for t in range(T):
        print(f"  {t}   " + " / ".join(f"{res[(n, 'vehicles', t)]:6.2f}" for n in names) + "            "
              + " / ".join(f"{res[(n, 'moving', t)]:6.2f}" for n in names), flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for i, k in enumerate(("vehicles", "moving")):
        for n, sty in (("fresh", "o--"), ("recursive", "s-"), ("gated", "^-"))[:len(names)]:
            ax[i].plot(range(T), [res[(n, k, t)] for t in range(T)], sty, label=n)
        ax[i].set_xlabel("keyframe t (0.5 s apart)")
        ax[i].set_ylabel("velocity error (m/s)")
        ax[i].set_title({"vehicles": "all vehicles", "moving": "moving vehicles (> 1 m/s)"}[k], fontsize=9)
        ax[i].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(image, dpi=120)
    print(f"image {image}")
    return res


@torch.no_grad()
def evaluate_recursive(model, test):
    """Recursive speed refinement, scored at F1 against the GT velocity of F1 (vehicles, all / moving > 1 m/s):
    fresh      the field estimated from (F1, F2) alone (hypotheses -> refinement), as in the pair test
    carried    the estimate of (F0, F1) carried to F1 (moved with itself, rotated into F1's axes): constant velocity,
               no new data
    recursive  carried as the START of the (F1, F2) refinement (instead of the fresh hypothesis mean): the prediction
               with the carried estimate is compared with the new frame (correlation, Doppler in every iteration)
    Recursion pays if 'recursive' beats both."""
    acc = {}
    for maps, vgt0, bid, s01, s12, vgt1 in test:
        m = maps[None].to(DEV)
        v01 = model(m[:, 0], m[:, 1], [s01], vgt=vgt0[None].to(DEV))[0][-1]
        carried = carry(v01, [s01])
        fresh = model(m[:, 1], m[:, 2], [s12], vgt=vgt1[None].to(DEV))[0][-1]
        recur = model(m[:, 1], m[:, 2], [s12], vgt=vgt1[None].to(DEV), v_init=carried)[0][-1]
        gated = model(m[:, 1], m[:, 2], [s12], vgt=vgt1[None].to(DEV), v_carried=carried)[0][-1] if GATE else None
        veh = maps[1, 1] > 0.5
        masks = {"vehicles": veh, "moving vehicles": veh & (vgt1.norm(dim=0) > 1.0)}
        for name, v in (("fresh", fresh), ("carried", carried), ("recursive", recur), ("gated", gated)):
            if v is None:
                continue
            e = (v[0].cpu() - vgt1).norm(dim=0)
            for k, mk in masks.items():
                a = acc.setdefault((name, k), [0.0, 0])
                a[0] += e[mk].sum().item()
                a[1] += int(mk.sum())
    return {f"EPE F1 {k} {name}": s_ / max(1, n_) for (name, k), (s_, n_) in acc.items()}


def overlay(a, b):
    """a in red, b in green, overlap yellow (vehicles + pedestrians); lanes grey."""
    oa, ob = a[1:].max(0).values.clamp(0, 1), b[1:].max(0).values.clamp(0, 1)
    lane = 0.35 * b[0].clamp(0, 1)
    return torch.stack([torch.maximum(oa, lane), torch.maximum(ob, lane), lane], -1).flip(1).numpy()


@torch.no_grad()
def picture(model, test, path, n=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["F0 (red) / F1 (green)", "GT velocity", "learned velocity", "D(P) (red) vs F1 (green)",
            "raw F0 moved with v (red) vs F1", "extrapolated (red) vs F2 (green)", "v = 0 raw (red) vs F1"]
    fig, ax = plt.subplots(n, len(cols) + 1, figsize=(2.3 * (len(cols) + 1), 2.5 * n), squeeze=False)
    for r in range(n):
        ax[r, -1].axis("off")
    wheel(ax[0, -1], vmax=15.0)
    for r in range(n):
        maps, vgt, bid, steps, steps12, *_ = test[r]
        m = maps[None].to(DEV)
        flows, preds, _ = model(m[:, 0], m[:, 1], [steps], vgt=vgt[None].to(DEV))
        v = flows[-1]
        occ = (maps[0, 1:].max(0).values > 0.3).float()
        ims = [overlay(maps[0], maps[1]), flow_rgb(vgt, occ, 15.0), flow_rgb(v[0].cpu(), occ, 15.0),
               overlay(torch.sigmoid(preds[-1][0]).cpu(), maps[1]),
               overlay(propagate(m[:, 0], [steps], v, G)[0][0].cpu(), maps[1]),
               overlay(propagate(m[:, 1], [steps12], carry(v, [steps]), G)[0][0].cpu(), maps[2]),
               overlay(propagate(m[:, 0], [steps], torch.zeros_like(v), G)[0][0].cpu(), maps[1])]
        for c, im in enumerate(ims):
            ax[r, c].imshow(im, origin="lower")
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=7)
    fig.suptitle(("nuScenes keyframes k-1, k, k+1 (about 0.5 s apart)" if DATA == "nuscenes" else f"dt = {DT} s") + "; velocity: hue = direction, brightness = speed (15 m/s full); x forward = up; "
                 "overlays: red = prediction, green = truth, yellow = both", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    t0 = time.time()
    n_train, n_test, iters = (300, 16, 600) if QUICK else (1500, 48, 3000)
    saved = f"temp/checks/model_explains{TAG}.pt"                    # scratch (not in git): the trained test model
    if "--sequences" in sys.argv:                                     # evaluation only, on the model a run saved
        model = Model().to(DEV)
        model.load_state_dict(torch.load(saved, map_location=DEV))
        model.eval()
        seqs = nuscenes_sequences(True, arg("--n-seq", 24), arg("--length", 5))
        sequence_report(model, seqs, f"temp/checks/flow_sequences{TAG}.png")
        sys.exit(0)
    if DATA == "nuscenes":                            # real GT (fixed data: the seed changes only init and batch order)
        rec = "--recursive" in sys.argv or GATE
        data, test = nuscenes_items(False, n_train, rec), nuscenes_items(True, n_test, rec)
    else:
        data = dataset([100_000 * SEED + i for i in range(n_train)], "train")
        test = dataset([50_000 + i for i in range(n_test)], "test")
    when = "nuScenes keyframes (about 0.5 s)" if DATA == "nuscenes" else f"dt {DT} s"
    print(f"device {DEV}, data {DATA}, seed {SEED}, {when}, prior {PRIOR}, sharp {SHARP}, pyramid {PYR} (leak {LEAK}), smooth {SMOOTH}, raw-warp {RAWWARP}, doppler {DOPPLER}; {n_train} training sequences, {n_test} held-out, "
          f"{iters} steps of 8 ({time.time() - t0:.0f} s to render)", flush=True)
    torch.manual_seed(SEED)
    model = Model().to(DEV)
    train(model, data, iters, random.Random(SEED + 7))
    torch.save(model.state_dict(), saved)
    model.eval()
    r = evaluate(model, test)
    print("== held-out sequences (IoU of vehicles; EPE in m/s)")
    for k, x in r.items():
        print(f"  {k:22s} {x:.3f}")
    if DATA == "nuscenes" and ("--recursive" in sys.argv or GATE):
        print("== recursive speed refinement, scored at F1 (EPE m/s)")
        for k, x in evaluate_recursive(model, test).items():
            print(f"  {k:36s} {x:.3f}")
    claims = [("velocity explains the vehicles: EPE < 1 m/s", r["EPE vehicles"] < 1.0),
              ("raw F0 moved with v alone reaches >= 90 % of the GT-velocity IoU", r["IoU raw"] >= 0.9 * r["IoU raw GT v"]),
              ("extrapolation to t + 2 dt reaches >= 90 % of the GT-velocity IoU", r["IoU extrap"] >= 0.9 * r["IoU extrap GT v"]),
              ("the learned field beats v = 0 by > 0.1 IoU (raw move)", r["IoU raw"] - r["IoU raw v=0"] > 0.1),
              ("the decoder does not carry the motion: prediction - raw move < 0.1 IoU", r["IoU pred"] - r["IoU raw"] < 0.1)]
    print("== claims")
    for name, ok in claims:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    picture(model, test, f"temp/checks/flow_explains{TAG}.png")
    print(f"== {sum(ok for _, ok in claims)} / {len(claims)} claims hold; image temp/checks/flow_explains{TAG}.png "
          f"({time.time() - t0:.0f} s)")
