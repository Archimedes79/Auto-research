"""Offline distillation of the student encoder (experiment t22, stage A): the student (`student_encoder.py`) reads a key-frame JPEG and is taught to reproduce the CACHED outputs of
the frozen teachers, H16 = DualBEV (`data/cache/cam_feats`) and H8 = BEVFusion (`data/cache/cam_feats_bevfusion`). Nothing of the chain is loaded; the teachers are read only.

    python training/distill_student.py --name A1_s0 --split small --freeze-backbone               # stage A1: ImageNet backbone frozen, FPN + heads trained (about 5 epochs)
    python training/distill_student.py --name A2_s0 --split large --epochs 6 --batch 16 --lr 1e-3 --backbone-mult 0.1      # stage A2: all weights, backbone at 0.1 x lr
    options: --backbone efficientvit_b1|resnet18  --workers 4  --max-steps N (a short run)  --resume (continue an existing run folder)  --force (skip the resource gate)
From the bev_fusion folder (the paths are relative to it). Output `runs/<name>/`: config.json, log.jsonl (a row per 20 steps: the loss terms, lr, s_per_step, samples_per_s, data_wait_s,
gpu_gb), ckpt_*.pt every 200 steps (resumes; the trainkit engine), metrics.json at the end (validation under the averaged weights, per head and per range band, samples/s, peak GPU memory).

Samples: the key frames of the split's scenes (`splits/<name>.json`): the tokens of the cache shards (a shard is named by its scene) that have a JPEG in `data/cache/img_front` (the
cache holds exactly the samples of the dataset: small 2283 / 378, large 428 / 82 scenes). No devkit is read. Loss, equal weights, per cell: KL(teacher || student) of the depth
distribution on H16 and H8, context (1 - cosine) + MSE on H16 and H8; every term is logged. No augmentation (a flip would not flip the cached targets). fp32.
Validation bands: the ground range of an image row on a flat road, near < 15 m, mid 15 - 30 m, far 30 - 60 m (a row proxy; the lower half of the image holds only the first 16 m, the
horizon is at row 76 of the 256), in the DualBEV window. Stop: the CONTROL file of the trainkit protocol (`python -m trainsuite stop <run>`) -> checkpoint, STOPPED, exit 75.
Exit codes: 3 = a resource is short (free host RAM < 3 GB or more than 6 GB of the GPU in use by others; --force overrides), 2 = the run folder has checkpoints and --resume is missing.
Test: tests/test_student_encoder.py
"""
import argparse
import contextlib
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_REPO = HERE                                           # the engine: a pinned copy next to this file (a run snapshot) wins, else the live autoresearch/ found by walking up (as train.py)
while not os.path.isdir(os.path.join(_REPO, "autoresearch", "trainkit")) and os.path.dirname(_REPO) != _REPO:
    _REPO = os.path.dirname(_REPO)
if os.path.join(_REPO, "autoresearch") not in sys.path:
    sys.path.append(os.path.join(_REPO, "autoresearch"))
from trainkit.config import Config  # noqa: E402
from trainkit.loop import Hooks, fit  # noqa: E402
from trainkit.util import prepare_run_dir, resolve_device, seed_all  # noqa: E402

import splits  # noqa: E402
from feature_cache import FeatureStore  # noqa: E402
from lane_branch import image_path, load_image  # noqa: E402
from student_encoder import StudentEncoder, h8_warp_grid  # noqa: E402

MIN_RAM_MB, MAX_GPU_MB = 3000, 6144                    # the resource gate: free host RAM, GPU memory already in use by others
# flat-ground range of an image row (nuScenes CAM_FRONT: fy 1266.4 px, cy 491.5 px at 1600 x 900, 1.5 m above the road) in the JPEG's window (resize 0.44, top 140 rows cut)
Y_HORIZON, F_PX, CAM_Z = 0.44 * 491.5 - 140, 0.44 * 1266.4, 1.5
BANDS = (("near", 0.0, 15.0), ("mid", 15.0, 30.0), ("far", 30.0, 60.0))


@dataclass
class DistillConfig(Config):
    split: str = "small"
    backbone: str = "efficientvit_b1"
    freeze_backbone: bool = False
    backbone_mult: float = 0.1                          # the backbone's lr relative to the heads' when it trains
    epochs: int = 5
    max_steps: int = 0                                  # > 0: this many steps instead of the epochs (a short run)
    batch: int = 16
    workers: int = 4
    img_root: str = "data/cache/img_front"
    feats16: str = "data/cache/cam_feats"               # DualBEV (118 + 80 at 16 x 44)
    feats8: str = "data/cache/cam_feats_bevfusion"      # BEVFusion (118 + 80 at 32 x 88)
    splits_dir: str = "splits"
    accum: int = 1
    lr: float = 1e-3
    warmup: int = 100
    keep: int = 2
    save_every: int = 200
    eval_every: int = 1000
    RESUME_FREE = Config.RESUME_FREE + ("workers",)


# ------------------------------------------------------------------------------------------------ data
def split_tokens(store, scenes, img_root):
    """sorted sample_data tokens of the key frames of `scenes`: the tokens in the shards of these scenes that have a JPEG; a scene without a shard is an error"""
    shard_scene = [os.path.basename(p)[:-len(".npy")] for p in store.shards]
    have = set(shard_scene)
    missing = [s for s in scenes if s not in have]
    if missing:
        raise ValueError(f"{len(missing)} scene(s) have no shard in {store.root}, e.g. {missing[:3]}")
    want = set(scenes)
    return sorted(t for t, (sid, _) in store.index.items() if shard_scene[sid] in want and os.path.exists(image_path(img_root, t)))


class KeyFrames(torch.utils.data.Dataset):
    """one key frame: the JPEG (uint8, 3 x 256 x 704) and the teachers' outputs (float16): probabilities p16 / p8, context c16 / c8"""

    def __init__(self, tokens, img_root, store16, store8):
        miss = [t for t in tokens if not store8.has(t)]
        if miss:
            raise ValueError(f"{len(miss)} token(s) are in the DualBEV cache but not in the BEVFusion cache, e.g. {miss[:2]}")
        self.tokens, self.img_root, self.store16, self.store8 = tokens, img_root, store16, store8

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        tok = self.tokens[i]
        (p16, c16), (p8, c8) = self.store16.get(tok), self.store8.get(tok)
        return {"img": load_image(self.img_root, tok), "p16": p16, "c16": c16, "p8": p8, "c8": c8}


# ------------------------------------------------------------------------------------------------ loss and validation
def cell_maps(out, tgt):
    """the per-cell terms, each (B, h, w): kl<k> = KL(teacher || student) over the 118 depth bins, cos<k> = cosine similarity of the 80 context channels, mse<k> = their mean squared
    error; k = 16 (DualBEV) and 8 (BEVFusion). The teacher probabilities are renormalised (they are stored as float16)."""
    m = {}
    for k in ("16", "8"):
        t = tgt["p" + k].float()
        t = t / t.sum(1, keepdim=True)
        m["kl" + k] = (t * (t.clamp_min(1e-8).log() - out["logit" + k].float().log_softmax(1))).sum(1)
        c, tc = out["ctx" + k].float(), tgt["c" + k].float()
        m["cos" + k] = F.cosine_similarity(c, tc, dim=1)
        m["mse" + k] = (c - tc).pow(2).mean(1)
    return m


def total_of(mean):
    """the loss from the cell means: kl + (1 - cos) + mse on both heads, equal weights"""
    return sum(mean["kl" + k] + 1 - mean["cos" + k] + mean["mse" + k] for k in ("16", "8"))


def distill_terms(out, tgt):
    """the training terms: kl<k>, cosl<k> (= 1 - cosine), mse<k>, total"""
    mean = {n: v.mean() for n, v in cell_maps(out, tgt).items()}
    terms = {n: v for n, v in mean.items() if not n.startswith("cos")}
    terms.update({"cosl" + k: 1 - mean["cos" + k] for k in ("16", "8")})
    terms["total"] = total_of(mean)
    return terms


def band_rows(ys):
    """{band: row indices} of the cell rows whose centres lie at the pixel rows `ys` of the JPEG's window: the flat-ground range f z / (y - y_horizon) falls in the band (rows above the
    horizon or beyond 60 m are in none)"""
    ys = np.asarray(ys, dtype=np.float64)
    d = np.where(ys > Y_HORIZON, F_PX * CAM_Z / np.maximum(ys - Y_HORIZON, 1e-9), np.inf)
    return {n: np.flatnonzero((d >= a) & (d < b)) for n, a, b in BANDS}


def head_rows(model):
    """pixel rows (JPEG window) of the centres of the output rows of H16 and H8"""
    rows8 = h8_warp_grid()[1][:, 0, 1].numpy() if model.grid8 is not None else 8 * np.arange(32) + 3.5
    return {"16": 16 * np.arange(16) + 7.5, "8": rows8}


@torch.no_grad()
def validate(model, loader, device):
    """flat dict: kl<k>, cos<k>, mse<k> per head, the same per range band (<name>.near / .mid / .far), total (the training loss on the validation frames) and n"""
    model.eval()
    rows, width, n = {}, {}, 0
    for batch in loader:
        img, tgt = to_device(batch, device)
        for name, m in cell_maps(model(img), tgt).items():
            rows[name] = rows.get(name, 0) + m.sum((0, 2)).double().cpu()            # per output row, summed over samples and columns
            width[name] = m.shape[2]
        n += len(img)
    ys, res, mean = head_rows(model), {"n": n}, {}
    for name, r in rows.items():
        per_row = (r / (n * width[name])).numpy()
        mean[name] = res[name] = float(per_row.mean())                              # every row has the same number of cells
        for band, idx in band_rows(ys["16" if name.endswith("16") else "8"]).items():
            res[f"{name}.{band}"] = float(per_row[idx].mean()) if len(idx) else float("nan")
    res["total"] = float(total_of(mean))
    return res


def to_device(batch, device):
    return batch["img"].to(device), {k: v.to(device) for k, v in batch.items() if k != "img"}


# ------------------------------------------------------------------------------------------------ run
class Throughput(Hooks):
    """samples per second of the training steps (data wait and compute, not the validation or the checkpoint between steps) and the data wait per step, over the last 20 steps"""

    def __init__(self, batch, window=20, skip=5):
        self.size, self.skip, self.wait, self.durs, self.waits = batch, skip, 0.0, deque(maxlen=window), deque(maxlen=window)
        self.t0, self.steps, self.total = None, 0, 0.0

    def begin_step(self, n):
        self.t0, self.wait = time.time(), 0.0

    @contextlib.contextmanager
    def data_wait(self):
        t = time.time()
        yield
        self.wait += time.time() - t

    def end_step(self):
        dur = time.time() - self.t0
        self.durs.append(dur)
        self.waits.append(self.wait)
        self.steps += 1
        if self.steps > self.skip:                     # the first steps hold the start-up (cudnn, worker spin-up)
            self.total += dur
        return {"samples_per_s": self.size * len(self.durs) / sum(self.durs), "data_wait_s": sum(self.waits) / len(self.waits)}

    def mean_rate(self):
        n = self.steps - self.skip
        return self.size * n / self.total if n > 0 and self.total > 0 else None


def mem_available_mb(path="/proc/meminfo"):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def gpu_used_mb():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return max(int(x) for x in out.split() if x.isdigit())
    except Exception:                                  # noqa: BLE001 - no nvidia-smi, a timeout, an unreadable answer
        return None


def check_resources(device, ram_mb, gpu_mb):
    """the reason a start is refused, else None. ram_mb = free host RAM (None: unreadable, no gate), gpu_mb = GPU memory in use (device cuda only; None: unreadable = refused). This process
    has no CUDA context yet, so what is in use belongs to others."""
    if ram_mb is not None and ram_mb < MIN_RAM_MB:
        return f"free host RAM {ram_mb} MB < {MIN_RAM_MB} MB"
    if device.startswith("cuda") and (gpu_mb is None or gpu_mb > MAX_GPU_MB):
        return "GPU memory unreadable (nvidia-smi)" if gpu_mb is None else f"{gpu_mb} MB of the GPU are in use by others (limit {MAX_GPU_MB} MB)"
    return None


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="small")
    ap.add_argument("--backbone", default="efficientvit_b1", choices=["efficientvit_b1", "resnet18"])
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-mult", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="steps instead of epochs (a short run)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    for k, v in (("out-root", "runs"), ("img-root", DistillConfig.img_root), ("feats16", DistillConfig.feats16), ("feats8", DistillConfig.feats8), ("splits-dir", DistillConfig.splits_dir)):
        ap.add_argument("--" + k, default=v)
    return ap.parse_args(argv)



def build_data(cfg):
    """({"train", "val"} -> KeyFrames of the split's scenes, the two loaders)"""
    lists = splits.load(cfg.split, cfg.splits_dir)
    store16, store8 = FeatureStore(cfg.feats16), FeatureStore(cfg.feats8)
    sets = {part: KeyFrames(split_tokens(store16, lists[part], cfg.img_root), cfg.img_root, store16, store8) for part in ("train", "val")}
    if len(sets["train"]) < cfg.batch:
        raise SystemExit(f"split {cfg.split!r}: {len(sets['train'])} training samples for batch {cfg.batch}")

    def loader(ds, train):
        return torch.utils.data.DataLoader(ds, cfg.batch, shuffle=train, drop_last=train, num_workers=cfg.workers, pin_memory=False, persistent_workers=cfg.workers > 0)
    return sets, loader(sets["train"], True), loader(sets["val"], False)


def run(cfg, out_dir, resumed, stop_check=None):
    """build data and student, train with the trainkit loop, validate under the averaged weights, write metrics.json"""
    device = cfg.device
    sets, train_loader, val_loader = build_data(cfg)
    cfg.steps = cfg.max_steps or cfg.epochs * (len(sets["train"]) // cfg.batch)
    if cfg.steps <= cfg.eval_every:
        cfg.eval_every = 0                             # a short run: the validation at its end only
    model = StudentEncoder(cfg.backbone)
    pass
    model.to(device)
    cfg.lr_groups = model.lr_groups(cfg.lr, cfg.backbone_mult)
    n_par, n_train = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{cfg.name}: {cfg.backbone}{' (backbone frozen)' if cfg.freeze_backbone else ''}, {n_par / 1e6:.2f} M parameters, {n_train / 1e6:.2f} M trained; split {cfg.split}: "
          f"{len(sets['train'])} train / {len(sets['val'])} val key frames, {cfg.steps} steps of batch {cfg.batch} on {device}", flush=True)

    def step_fn(batch, info):
        img, tgt = to_device(batch, device)
        terms = distill_terms(model(img), tgt)
        ok = bool(torch.isfinite(terms["total"]))
        if ok:
            info.backward(terms["total"])
        return {k: float(v) for k, v in terms.items()}, ok, None

    def evaluate(ema):
        with ema.applied(model):
            return validate(model, val_loader, device)

    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    meter, t0 = Throughput(cfg.batch), time.time()
    model, _, ema, _ = fit(cfg, model, torch.nn.Module(), train_loader, step_fn, out_dir, device, evaluate=evaluate, hooks=meter, stop_check=stop_check)
    with ema.applied(model):
        val = validate(model, val_loader, device)
    metrics = {"name": cfg.name, "split": cfg.split, "backbone": cfg.backbone, "freeze_backbone": cfg.freeze_backbone, "n_train": len(sets["train"]), "n_val": len(sets["val"]),
               "steps": cfg.steps, "batch": cfg.batch, "device": device, "resumed": resumed, "weights": "ema", "params_m": n_par / 1e6, "trained_params_m": n_train / 1e6,
               "train_samples_per_s": meter.mean_rate(), "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None,
               "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if cuda else None, "seconds": time.time() - t0, "val": val}
    with open(os.path.join(out_dir, "metrics.json.tmp"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    os.replace(os.path.join(out_dir, "metrics.json.tmp"), os.path.join(out_dir, "metrics.json"))
    print(f"done: val total {val['total']:.4f} (kl16 {val['kl16']:.3f} cos16 {val['cos16']:.3f} kl8 {val['kl8']:.3f} cos8 {val['cos8']:.3f}), "
          f"{metrics['train_samples_per_s'] or float('nan'):.1f} samples/s", flush=True)


def main(argv=None, stop_check=None):
    a = parse(argv)
    device = resolve_device(a.device)
    if not a.force:
        why = check_resources(device, mem_available_mb(), gpu_used_mb() if device.startswith("cuda") else None)
        if why:
            print(f"refused to start: {why} (--force overrides)", flush=True)
            return 3
    cfg = DistillConfig(name=a.name, out_root=a.out_root, split=a.split, backbone=a.backbone, freeze_backbone=a.freeze_backbone, backbone_mult=a.backbone_mult, epochs=a.epochs,
                        max_steps=a.max_steps, batch=a.batch, lr=a.lr, workers=a.workers, log_every=a.log_every, seed=a.seed, device=device, img_root=a.img_root, feats16=a.feats16,
                        feats8=a.feats8, splits_dir=a.splits_dir)
    out_dir = os.path.join(cfg.out_root, cfg.name)
    ckpts = [f for f in os.listdir(out_dir) if f.startswith("ckpt_")] if os.path.isdir(out_dir) else []
    if ckpts and not a.resume:
        print(f"{out_dir} has checkpoints: --resume continues it, or choose another --name", flush=True)
        return 2
    out_dir = prepare_run_dir(cfg)                     # the CONTROL / STOPPED files of an earlier stop are moved aside
    seed_all(cfg.seed)
    run(cfg, out_dir, bool(ckpts), stop_check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
