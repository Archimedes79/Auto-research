"""throw-away (t11 gate): base rate of moving centres in the validation subset of t11_vel_hurdle, and the val objhead.mov@+0.0 a
constant logit would log (the prior level the card's refutation compares with). Reads the GT cache only (no features)."""
import json
import math
import os
import sys

ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path[:0] = [os.path.join(ROOT, "training")]
os.chdir(ROOT)
import torch  # noqa: E402
torch.set_num_threads(2)
import train as tr  # noqa: E402
from losses import NUSC_GROUPS, LossConfig, build_layer_targets, object_head_terms  # noqa: E402
from lift_pd import Grid  # noqa: E402



cfg = tr.TrainConfig(**json.load(open("training/configs/t11_vel_hurdle.json")))
meta = os.path.join(os.path.expanduser(cfg.data_root), cfg.version)    # the devkit's tables, read light (no NuScenes object: RAM)
scene_tok = {s["token"] for s in json.load(open(os.path.join(meta, "scene.json"))) if s["name"] in set(cfg.val_scenes)}
sample_rows = json.load(open(os.path.join(meta, "sample.json")))
nxt = {s["token"]: s["next"] for s in sample_rows}
samples = [s["token"] for s in sample_rows if s["scene_token"] in scene_tok and s["prev"] and s["next"]]   # = FrontRadarDataset
pos = {t: i for i, t in enumerate(samples)}
all_starts = []                                                          # = ClipDataset
for t0 in samples:
    chain, cur = [pos[t0]], t0
    while len(chain) < cfg.seq_len:
        cur = nxt[cur]
        if cur not in pos:
            break
        chain.append(pos[cur])
    if len(chain) == cfg.seq_len:
        all_starts.append(chain)
sub = tr.val_subset(all_starts, cfg.eval_batches * cfg.batch_size)
starts = list(sub.dataset[i] for i in sub.indices) if hasattr(sub, "indices") else sub
sig = json.load(open("runs/t10_dual_s0/gt.json"))["gt_signature"]


class _Val:
    samples = samples

    @staticmethod
    def _gt(token):
        with __import__("numpy").load(os.path.join(cfg.gt_cache, f"{token}_{sig}.npz"), allow_pickle=True) as z:
            return z["gt"].item()


val = _Val
print(f"{len(scene_tok)} val scenes, {len(samples)} samples, {len(all_starts)} clips, subset {len(starts)}", flush=True)
grid, lc = Grid(), LossConfig(**cfg.loss)
X, Y = grid.nx, grid.ny
zero = {k: torch.zeros(1, c, X, Y) for k, c in (("hm", 3), ("off", 2), ("size", 2), ("rot", 2), ("rad", 1), ("vel", 2))}
per, n_all, n_mov = [], 0, 0                       # per (batch, keyframe): (moving centres, centres)
for b in range(0, len(starts), cfg.batch_size):
    for k in range(cfg.seq_len):
        gts = [val._gt(val.samples[starts[c][k]])["cur"] for c in range(b, min(b + cfg.batch_size, len(starts)))]
        T = build_layer_targets(gts, lc, None, 0.0, "cpu")
        _, t = object_head_terms({kk: v.expand(len(gts), -1, -1, -1) for kk, v in zero.items()}, T["boxes"], T["w_obj"], grid,
                                 NUSC_GROUPS, max_obj=lc.max_obj)
        m = t["mask"] > 0
        mv = (t["vel"].norm(dim=-1) > 0.5) & m
        per.append((int(mv.sum()), int(m.sum())))
        n_all, n_mov = n_all + int(m.sum()), n_mov + int(mv.sum())


def logged(p):                                     # val term of a constant probability p: mean over (batch, keyframe) of the mean BCE
    vals = [-(a * math.log(p) + (n - a) * math.log(1 - p)) / max(n, 1) for a, n in per]
    return sum(vals) / len(vals)


p = n_mov / n_all
H = -(p * math.log(p) + (1 - p) * math.log(1 - p))
print(f"val subset: {len(starts)} clips x {cfg.seq_len} keyframes, {n_all} supervised centres, {n_mov} moving (> 0.5 m/s): base rate {p:.4f}; "
      f"entropy H(p) {H:.4f} nats; logged val term of the constant p {logged(p):.4f}, of the init prior 0.2 {logged(0.2):.4f}; "
      f"20 % below the prior level: {0.8 * logged(p):.4f}")
