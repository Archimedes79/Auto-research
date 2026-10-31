"""Distil the frozen DualBEV image heads into the light stride-8 encoder (plan.md 2.7, step 2 of the recipe).

Teacher: the cached outputs of the frozen DualBEV heads (training/feature_cache.py: depth probability (118) + context (80) at
16 x 44, float16, one shard per scene). Student: training/light_encoder.LightImageEncoder (ImageNet-initialised backbone, FPN,
camera-aware heads, stride 8 = 32 x 88 cells). Loss: KL(teacher || student pooled 2 x 2) + relative MSE of the pooled context
(`light_encoder.distill_loss`). Input: the images of the cached frames exactly as the teacher saw them (the dataset's `_image`
and camera vector). Train frames: the cached frames of the official train scenes; validation: every 8th cached frame of the
official val scenes (at most --val-frames), under the eval-mode student.
Reported (eval.json): KL, relative context error, and the mean absolute error of the expected depth to the teacher's by the
teacher's range (< 20 / 20-40 / > 40 m): how close the student is to the teacher; the teacher's own error against reality is
not measured here (no lidar), a student cannot beat it by distillation alone - the BEV task loss does that (recipe step 3).
Resumable (checkpoint every 500 steps), 2 loader threads (shared machine), bf16 autocast on CUDA.
Run from the bev_fusion folder (GPU, after the t1 / t4 rounds: tests/run_when_gpu_free.sh):
  python training/distill_light_encoder.py --backbone resnet18 [--steps 6000 --batch 16]
Smoke test (nuScenes mini, CPU, 20 steps): python training/distill_light_encoder.py --smoke
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_encoder import camera_vector, nuscenes_front_ida  # noqa: E402
from feature_cache import FeatureStore  # noqa: E402
from gt_bev import pose_matrix  # noqa: E402
from light_encoder import LightImageEncoder, distill_loss, expected_depth, n_params  # noqa: E402
from nuscenes_dataset import DST_HW, FrontRadarDataset  # noqa: E402


class Frames:
    """token -> (image (3, 256, 704), camera vector (27), teacher prob, teacher ctx): the teacher's own input, via the dataset code."""

    def __init__(self, nusc, root, store):
        self.nusc, self.root, self.store = nusc, root, store
        self.ida = nuscenes_front_ida(dst_hw=DST_HW)

    def _image_uint8(self, sd):
        return FrontRadarDataset._image_uint8(self, sd)

    def _image(self, sd):
        return FrontRadarDataset._image(self, sd)

    def __call__(self, token):
        sd = self.nusc.get("sample_data", token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
        T = torch.tensor(pose_matrix(cs), dtype=torch.float32)
        cv = camera_vector(K[None], self.ida[None, :2, :2], self.ida[None, :2, 2], T[None])[0]
        t_prob, t_ctx = self.store.get(token)
        return self._image(sd), cv, t_prob, t_ctx


def split_tokens(store, split_names):
    """tokens of the cached frames per official split ('train' / 'val' / 'mini_train' / 'mini_val'), by the shard's scene name."""
    from nuscenes.utils.splits import create_splits_scenes
    names = create_splits_scenes()
    scene_of = {i: os.path.basename(p)[:-4] for i, p in enumerate(store.shards)}
    out = {}
    for split in split_names:
        want = set(names[split])
        out[split] = sorted(t for t, (sid, _) in store.index.items() if scene_of[sid] in want)
    return out


def collate(items):
    return tuple(torch.stack(x) for x in zip(*items))


class Prefetch:
    """a batch stream from a thread pool (PIL decoding releases the GIL), two batches ahead."""

    def __init__(self, frames, tokens, batch, threads, seed):
        self.frames, self.tokens, self.batch, self.rng = frames, tokens, batch, np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(threads)
        self.queue = [self._submit() for _ in range(2)]

    def _submit(self):
        idx = self.rng.integers(0, len(self.tokens), self.batch)
        return [self.pool.submit(self.frames, self.tokens[i]) for i in idx]

    def next(self):
        futs = self.queue.pop(0)
        self.queue.append(self._submit())
        return collate([f.result() for f in futs])


@torch.no_grad()
def evaluate(model, frames, tokens, device, batch=16):
    model.eval()
    kl, rel, n = 0.0, 0.0, 0
    err = {"< 20 m": [0.0, 0], "20-40 m": [0.0, 0], "> 40 m": [0.0, 0]}
    for i in range(0, len(tokens), batch):
        img, cv, tp, tc = collate([frames(t) for t in tokens[i:i + batch]])
        img, cv, tp, tc = img.to(device), cv.to(device), tp.to(device), tc.to(device)
        prob, ctx = model.image_features(img, cv)
        _, k, r = distill_loss(prob, ctx, tp, tc)
        kl, rel, n = kl + float(k) * len(img), rel + float(r) * len(img), n + len(img)
        ed_s = expected_depth(torch.nn.functional.avg_pool2d(prob.float(), 2))
        ed_t = expected_depth(tp.float())
        e = (ed_s - ed_t).abs()
        for name, m in (("< 20 m", ed_t < 20), ("20-40 m", (ed_t >= 20) & (ed_t < 40)), ("> 40 m", ed_t >= 40)):
            err[name][0] += float(e[m].sum())
            err[name][1] += int(m.sum())
    model.train()
    return {"kl": kl / n, "ctx_rel_mse": rel / n, "depth_mae_to_teacher_m": {k: v[0] / max(v[1], 1) for k, v in err.items()}, "frames": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-frames", type=int, default=400)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="data/cache/cam_feats")
    ap.add_argument("--dataroot", default="~/data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-stride", type=int, default=8)
    a = ap.parse_args()
    splits = ("train", "val")
    if a.smoke:
        a.cache, a.dataroot, a.version, a.steps, a.batch, a.val_frames, a.val_every, a.warmup = \
            "data/cache/cam_feats_mini", "~/data/nuscenes", "v1.0-mini", 20, 2, 8, 10, 5
        splits = ("mini_train", "mini_val")
    device = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    torch.set_num_threads(a.threads)
    out = a.out or f"runs/distill_{a.backbone}{'_smoke' if a.smoke else ''}"
    os.makedirs(out, exist_ok=True)
    from nuscenes.nuscenes import NuScenes
    root = os.path.expanduser(a.dataroot)
    nusc = NuScenes(a.version, root, verbose=False)
    store = FeatureStore(a.cache)
    toks = split_tokens(store, splits)
    train_t, val_t = toks[splits[0]], toks[splits[1]][::8][:a.val_frames]
    if not val_t:
        if not a.smoke:
            raise SystemExit("no cached frame of the validation scenes: build the feature cache of the val split first")
        val_t = train_t[::4][:a.val_frames]                   # smoke only: the mini cache has no mini_val scene
        print("smoke: no cached mini_val frame, validating on train frames", flush=True)
    print(f"teacher cache {len(store)} frames; train {len(train_t)}, val {len(val_t)}; device {device}", flush=True)
    frames = Frames(nusc, root, store)
    model = LightImageEncoder(a.backbone, out_stride=a.out_stride).to(device).train()
    decay = [p for p in model.parameters() if p.ndim >= 2]
    plain = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4}, {"params": plain, "weight_decay": 0.0}], lr=a.lr)
    step, ck = 0, os.path.join(out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, weights_only=True)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step = s["step"]
        print(f"resumed at step {step}", flush=True)
    data = Prefetch(frames, train_t, a.batch, a.threads, seed=step)
    print(f"{a.backbone}: {n_params(model) / 1e6:.1f} M parameters, {a.steps} steps of {a.batch}", flush=True)
    t0, run = time.time(), []
    log = open(os.path.join(out, "log.jsonl"), "a")
    while step < a.steps:
        f = (step + 1) / a.warmup if step < a.warmup else 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))
        for g in opt.param_groups:
            g["lr"] = a.lr * max(f, 0.01)
        img, cv, tp, tc = (x.to(device) for x in data.next())
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            prob, ctx = model.image_features(img, cv)
        loss, kl, rel = distill_loss(prob, ctx, tp, tc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        run.append((loss.item(), kl.item(), rel.item()))
        if not math.isfinite(run[-1][0]):
            raise RuntimeError(f"non-finite loss at step {step}")
        if step % 50 == 0 or step == 1:
            m = np.mean(run[-50:], 0)
            rec = {"step": step, "loss": m[0], "kl": m[1], "ctx_rel_mse": m[2], "seconds": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"  step {step:5d} loss {m[0]:.4f} (kl {m[1]:.4f}, ctx {m[2]:.4f}) {time.time() - t0:.0f} s", flush=True)
        if step % 500 == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
        if step % a.val_every == 0 or step == a.steps:
            ev = evaluate(model, frames, val_t, device)
            ev["step"] = step
            print(f"  eval step {step}: kl {ev['kl']:.4f}, ctx {ev['ctx_rel_mse']:.4f}, depth MAE to teacher {ev['depth_mae_to_teacher_m']}", flush=True)
            json.dump(ev, open(os.path.join(out, "eval.json"), "w"), indent=1)
    torch.save({"model": model.state_dict(), "backbone": a.backbone, "out_stride": a.out_stride, "step": step},
               os.path.join(out, "student.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
