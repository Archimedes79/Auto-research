"""The datasets of a run config and what turns their items into the batches the network reads: clips of consecutive keyframes, the move to the device
and the ego motion of the gaps. The config is read by attribute only (no import of the trainer)."""
import math
import os

import numpy as np
import torch

from bev_fusion.net.prep.feature_cache import CachedFrontRadarDataset, FeatureStore
from bev_fusion.net.prep.nuscenes_dataset import FrontRadarDataset, available_scenes


def make_datasets(cfg):
    from nuscenes.nuscenes import NuScenes
    root = os.path.expanduser(cfg.data_root)
    nusc = NuScenes(cfg.version, dataroot=root, verbose=False)

    def scenes(split, names):
        if cfg.version.endswith("mini") and split not in ("mini_train", "mini_val"):
            split = "mini_" + split
        toks = available_scenes(nusc, split)
        if names:
            toks = [t for t in toks if nusc.get("scene", t)["name"] in set(names)]
        return toks
    kw = dict(n_cam=cfg.n_cam, cam_stride=cfg.cam_stride, with_gt=True, nusc=nusc,
              cache_dir=cfg.gt_cache or None, radar_window=cfg.radar_window,
              radar_per_frame=cfg.model.get("radar_per_frame", False))        # the model's switch: one radar set per camera frame (batch["radar_seq"])
    out = []
    for split, names in ((cfg.train_split, cfg.train_scenes), (cfg.val_split, cfg.val_scenes)):
        if cfg.feature_cache:
            out.append(CachedFrontRadarDataset(root, cfg.version, scenes(split, names),
                                               store=FeatureStore(cfg.feature_cache), radar_cache=cfg.radar_cache or None,
                                               camera_features=cfg.model.get("camera_features", "dualbev"),
                                               lane_images=cfg.lane_images or None,
                                               lane_images_older=bool(cfg.model.get("camera_student")) and cfg.model.get("student_frames", "current") == "all",
                                               store2=FeatureStore(cfg.feature_cache2) if cfg.feature_cache2 else None, **kw))
        else:
            out.append(FrontRadarDataset(root, cfg.version, scenes(split, names), **kw))
    return out


class ClipDataset(torch.utils.data.Dataset):
    """Clips of `length` consecutive keyframes of ONE scene (sequence training): item = [base[i], base[i + 1], ...], a clip
    starts at every sample whose next `length - 1` keyframes are also in `base.samples` (the dataset already leaves out the
    first and last keyframe of a scene, so the chain stays inside the scene)."""

    def __init__(self, base, length):
        self.base, self.length = base, length
        pos = {t: i for i, t in enumerate(base.samples)}
        self.starts = []
        for t in base.samples:
            chain, cur = [pos[t]], t
            while len(chain) < length:
                cur = base.nusc.get("sample", cur)["next"]
                if cur not in pos:
                    break
                chain.append(pos[cur])
            if len(chain) == length:
                self.starts.append(chain)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        return [self.base[j] for j in self.starts[i]]


def collate_clips(clips, fn):
    """list of clips (lists of items) -> list over time of batches: [fn([clip[t] for clip in clips]) for t]."""
    return [fn([c[t] for c in clips]) for t in range(len(clips[0]))]


def to_device(batch, device, non_blocking=False):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        elif k == "radar":
            out[k] = {a: b.to(device, non_blocking=non_blocking) for a, b in v.items()}
        elif k == "radar_seq":
            out[k] = [{a: b.to(device, non_blocking=non_blocking) for a, b in r.items()} for r in v]
        elif k == "steps":
            out[k] = [[tuple(t.to(device, non_blocking=non_blocking) for t in s) for s in fr] for fr in v]
        else:
            out[k] = v
    return out


def gap_ego_motion(steps):
    """The ego steps [(dt_i, v_i, omega_i)] (batch["gap_steps"]: time 0 -> the target keyframe, negative dt backwards) as ONE
    splat step (dt, v, omega) that lands a static point where the chain does: the chain is composed with the splat's own
    motion model p' = R(-omega dt)(p - d), d = (v dt, v omega dt^2 / 2); omega = heading change / dt, v = forward chord /
    dt (what splat.ego_speed_yawrate gives from the two poses). dt is the TRUE gap (0.40-0.60 s on nuScenes), not the
    nominal 0.5 s."""
    x = y = th = T = 0.0                                   # the start ego origin in the current ego frame, heading, time
    for dt, v, w in steps:
        px, py = x - v * dt, y - 0.5 * v * w * dt * dt
        c, s = math.cos(-w * dt), math.sin(-w * dt)
        x, y = c * px - s * py, s * px + c * py
        th += w * dt
        T += dt
    if abs(T) < 1e-9:
        return 0.0, 0.0, 0.0
    chord_x = -(math.cos(th) * x - math.sin(th) * y)       # the end ego in the start frame (x component)
    return T, chord_x / T, th / T


def _one_copy(rows, device):
    """float32 tensor of the nested Python lists `rows` on `device` with ONE host-to-device copy, made from pinned memory without waiting (D54). The values are
    those of torch.tensor(rows, dtype=torch.float32, device=device), which converts on the host as well; the old code made one tensor, i.e. one copy from
    pageable memory (a copy that first waits until the GPU has finished everything queued) per element of the result: three per dt and three per clip step."""
    flat = torch.tensor(rows, dtype=torch.float32)
    if torch.device(device).type == "cuda":
        return flat.pin_memory().to(device, non_blocking=True)
    return flat.to(device)


def batch_ego(gap_steps, dts, device, strict=False):
    """{nominal dt != 0: (dt, v, omega)} as (B,) tensors: the true gap and the gap-mean ego motion of every sample, from batch["gap_steps"]
    (the dataset makes them from the poses: the network input needs no ground truth). gap_steps holds the previous and the next keyframe only, i.e. dt = -0.5
    and +0.5 s: any other dt
    would silently get the 0.5 s gap; strict (LossConfig.hygiene_2026_11_02) raises instead. All tensors are views of one (n_dt, 3, B) tensor (one copy to the
    GPU, D54)."""
    keys, rows = [], []
    for dt in dts:
        if abs(dt) < 1e-6:
            continue
        if strict and abs(abs(dt) - 0.5) > 1e-6:
            raise ValueError(f"batch_ego: dt {dt:+.2f} s has no ego gap (gap_steps holds the keyframe before and after, dt = -0.5 and +0.5 s); the sample builder must provide the gap of that dt")
        key = "prev" if dt < 0 else "next"
        vals = [gap_ego_motion(g[key]) for g in gap_steps]
        keys.append(dt)
        rows.append([[v[i] for v in vals] for i in range(3)])
    if not keys:
        return {}
    flat = _one_copy(rows, device)
    return {dt: (flat[j, 0], flat[j, 1], flat[j, 2]) for j, dt in enumerate(keys)}


def clip_step(prev_batch, device):
    """(dt, v, omega) as (B,) tensors from the previous keyframe of each clip (or of a scene, in the evaluation) to the current one: the ego steps of the
    previous batch toward its next keyframe (batch["gap_steps"], from the poses), composed into one splat step; the model's memory pose advances by it."""
    vals = [gap_ego_motion(g["next"]) for g in prev_batch["gap_steps"]]
    flat = _one_copy([[v[i] for v in vals] for i in range(3)], device)           # one copy to the GPU, three views
    return flat[0], flat[1], flat[2]


def val_subset(ds, n):
    """n samples spread evenly over the whole validation set (all scenes), not its first batches."""
    if n >= len(ds):
        return ds
    return torch.utils.data.Subset(ds, sorted({int(i) for i in np.linspace(0, len(ds) - 1, n).round()}))
