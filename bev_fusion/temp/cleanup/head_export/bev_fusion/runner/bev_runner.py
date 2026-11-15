"""The runner: the five-layer system (training/architecture.md) run on frames, without ground truth, from a run's own code snapshot.

    python runner/bev_runner.py check      <config.json> [--split val] [--n 8 | --scene scene-0003 [--first N]]      build the input, check it, write nothing
    python runner/bev_runner.py precompute <config.json> [--split val] [--n 8 | --scene scene-0003 [--first N]] [--out DIR]
    python runner/bev_runner.py network    <run> <frames DIR> [--out DIR] [--dts 0 0.5] [--cold] [--device cpu|cuda]
    python runner/bev_runner.py full       <run> [--config C] [--split val] [--n 8 | --scene scene-0003 [--first N]] [--out DIR] [--dts 0 0.5] [--device]

precompute  layer 1 alone: the network input of chosen frames (the dataset's item through the precompute step, no ground truth, checked
            against interface.INPUTS), one .pt per frame and index.json (`check` does the same and writes nothing; the radar fan is also
            compared with the ground truth's own radar layer there).
network     layers 2 - 5 alone: the frames of a precompute folder through the run's network (EMA weights of its last checkpoint), in time
            order; the memory is carried from a frame to the next keyframe of the same scene, a gap restarts it. Writes per frame the
            boxes at every dt (the post-processing: model.boxes = extract.select_boxes) and the probability maps of dt 0.
full        both, in one process, frame by frame (nothing is written between the layers); the ground truth is not read (a frame
            without annotations runs too).
The code is the run's own snapshot (RUN/code, the code that trained it); `--code training` runs a checkpoint under the live code instead
(the state-dict keys of a checkpoint do not change with the five-layer build). A snapshot from before the build lacks precompute.py and
the batch's gap_steps and is refused with that hint. The ego step between frames and the true gap to dt = +0.5 s come from the poses
(batch["gap_steps"]), the way a deployed system would supply its own odometry. Outputs (--out, default RUN/runner/<mode>):
    objects.json   per frame: token, index, and for every dt the boxes in the ego frame of that frame (class, score, centre, size or radius,
                   yaw, velocity)
    maps.npz       sem (frames, layers, X, Y) float16: sigmoid for areas, lines and objects, the distance in metres for distance layers
                   (dt 0), names, kinds, tokens and the grid
Test: tests/test_runner.py (a real run against its stored prototype outputs). Without arguments this prints the guide.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MAP_KINDS = ("area", "line", "object", "dist", "centreline", "stationary")     # layers with a map to write (a logit: sigmoid; dist: metres, >= 0)


def use_code(run, code):
    """put the code folder first on sys.path: `code` if given, else RUN/code (it must hold precompute.py); returns its path"""
    folder = os.path.abspath(code) if code else os.path.join(os.path.abspath(run), "code")
    if not os.path.exists(os.path.join(folder, "precompute.py")):
        raise SystemExit(f"{folder} has no precompute.py: a snapshot from before the five-layer build cannot be run here - "
                         f"use `--code training` to run its checkpoint under the live code")
    for p in (os.path.join(ROOT, ".."), folder):                   # the repository (autogt/) behind the code folder
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return folder


def load_model(run, device):
    """(cfg, model, loss, step, checkpoint): the last checkpoint of RUN under its EMA weights, eval mode (as the evaluation loads a run)"""
    import train as T
    with open(os.path.join(run, "config.json")) as f:
        cfg = T.TrainConfig(**json.load(f))
    model, loss, _ = T.build(cfg)
    model, loss = model.to(device), loss.to(device)
    ckpts = sorted(glob.glob(os.path.join(run, "ckpt_*.pt")))
    if not ckpts:
        raise SystemExit(f"no checkpoint in {run}")
    ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    ck.pop("opt", None), ck.pop("rng", None)                       # running needs neither: no optimiser is built, the random state is left alone
    ema = T.EMA(model, cfg.ema)
    step, _ = T.load_checkpoint(ckpts[-1], model, loss, None, ema, ck=ck)
    ema.apply(model)
    model.eval()
    return cfg, model, loss, step, ckpts[-1]


def record(d):
    """one box of model.boxes -> the planner record: boxes have size [length, width] and yaw, circles a radius and no yaw"""
    rec = {"class": d["group"], "score": float(d["score"]), "centre": [float(d["cx"]), float(d["cy"])], "velocity": [float(d["vx"]), float(d["vy"])]}
    if "radius" in d:
        return dict(rec, radius=float(d["radius"]), size=None, yaw=None)
    return dict(rec, radius=None, size=[float(d["length"]), float(d["width"])], yaw=float(d["yaw"]))


class Network:
    """the run's network on a stream of frames: the memory carried from a frame to the next keyframe of the scene"""

    def __init__(self, run, device, dts, cold=False):
        import train as T
        from interface import check_outputs
        self.T, self.check_outputs, self.dts, self.device, self.always_cold = T, check_outputs, tuple(float(d) for d in dts), device, cold
        self.cfg, self.model, self.loss, self.step, self.ckpt = load_model(run, device)
        self.spec = self.model.spec
        self.kept = [i for i, (_, kind, _) in enumerate(self.spec) if kind in MAP_KINDS]       # the layers that have a map to write
        self.state, self.prev, self.next_token = None, None, None

    @torch.no_grad()
    def __call__(self, batch, row):
        T, token = self.T, row["token"]
        cold = self.always_cold or self.state is None or self.prev is None or token != self.next_token   # a gap in the scene restarts the memory
        kw = {} if cold else {"state": self.state, "step": T.clip_step(self.prev, self.device)}
        if getattr(self.cfg, "gap_ego", False) and T.supports_ego(self.model):
            if "gap_steps" not in batch:
                raise SystemExit("this run decodes dt = +-0.5 s with the true gap (gap_ego) but the frames carry no gap_steps: make them again with `precompute`")
            kw["ego"] = T.batch_ego(batch["gap_steps"], self.dts, self.device)             # the true gap to dt = +-0.5 s, from the poses
        with T.amp_context(self.cfg.amp, self.device):                                      # as the evaluation runs it (train.forward_loss): bf16 and fp16 on CUDA
            res = self.model(T.to_device(batch, self.device), dts=self.dts, **kw)
        self.check_outputs(res, self.dts, raise_on=True)
        self.state, self.prev, self.next_token = T.detach_tree(res["state"]), batch, row["next"]
        boxes = {f"{dt:+.1f}": [record(d) for d in self.model.boxes(res, dt, k=100, threshold=0.05, rescore=True)[0]] for dt in self.dts}
        sem = res["out"][0.0 if 0.0 in self.dts else self.dts[0]]["sem"][0].float().cpu()
        maps = np.stack([sem[i].clamp(min=0).numpy() if self.spec[i][1] == "dist" else torch.sigmoid(sem[i]).numpy()
                         for i in self.kept]).astype(np.float16)
        return boxes, maps


def write(out, net, rows, boxes, maps, extra):
    os.makedirs(out, exist_ok=True)
    grid = net.model.grid
    info = {"run": extra["run"], "checkpoint": os.path.basename(net.ckpt), "step": int(net.step), "code": extra["code"], "dts": list(net.dts),
            "frame": "ego (vehicle) frame of each frame's own time, metres: x forward, y left; yaw in rad counter-clockwise from x; "
                     "velocity = world velocity of the object in the ego axes, m/s",
            "frames": [{"index": r["index"], "token": r["token"], "objects": b} for r, b in zip(rows, boxes)]}
    json.dump(info, open(os.path.join(out, "objects.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, "maps.npz"), sem=np.stack(maps), names=[net.spec[i][0] for i in net.kept], kinds=[net.spec[i][1] for i in net.kept],
                        tokens=[r["token"] for r in rows],
                        grid=json.dumps({"res": grid.res, "x_min": grid.x_min, "x_max": grid.x_max, "y_min": grid.y_min, "y_max": grid.y_max}))
    print(f"runner: {len(rows)} frames -> {out} (objects.json, maps.npz)")


def frame_indices(ds, scene, n, first):
    """the dataset indices to run: one scene in time order (its first `first` frames if given), or n frames spread over the split"""
    if scene:
        names = {s["token"]: s["name"] for s in ds.nusc.scene}
        idx = [i for i, t in enumerate(ds.samples) if names[ds.nusc.get("sample", t)["scene_token"]] == scene]
        if not idx:
            raise SystemExit(f"no frame of {scene} in the split")
        idx = sorted(idx, key=lambda i: ds.nusc.get("sample", ds.samples[i])["timestamp"])      # time order: the runner carries a memory
        return idx[:first] if first else idx
    n = min(n, len(ds))
    return [int(i * (len(ds) - 1) / max(n - 1, 1)) for i in range(n)]


def network_inputs(config, split="val", n=8, scene="", first=0, with_gt=True):
    """the network input of chosen frames, one batch of one frame at a time: -> (cfg, generator of (batch, row)). The batch is the dataset's
    item through the precompute step, without the ground truth, and passes interface.check_inputs; row = {index, token, next (the token of the
    next keyframe of the scene, "" at its end), cover_<map>, fan_equals_gt (when the ground truth was read)}. `config` = a training config
    (training/configs/*.json or runs/<name>/config.json): data, caches and the model's grid. The data loaders and the trainer come from the
    code folder in use (use_code); they are not part of the network, which is why this lives here and not in training/precompute.py."""
    import train as T
    from feature_cache import collate_cached
    from interface import check_inputs
    from model import ModelConfig
    from nuscenes_dataset import collate
    from precompute import MAPS, Precompute
    cfg = T.read_config(config)
    train_ds, val_ds = T.make_datasets(cfg)
    ds = val_ds if split == "val" else train_ds
    if not with_gt:                                               # the network input needs none: the dataset does not read (or build) it
        getattr(ds, "dataset", ds).with_gt = False
    fn = collate_cached if cfg.feature_cache else collate
    pre = Precompute.from_config(ModelConfig(**cfg.model))

    def run():
        for i in frame_indices(ds, scene, n, first):
            batch = pre(fn([ds[i]]))
            row = {}
            if "gt" in batch:                                     # the one definition: the fan equals the ground truth's own radar layer
                g = torch.as_tensor(np.asarray(batch["gt"][0]["cur"]["radar"])) > 0
                row["fan_equals_gt"] = float(((batch["vis_radar"][0, 0] > 0) == g).float().mean()) if g.shape == batch["vis_radar"].shape[-2:] else float("nan")
            batch.pop("gt", None)                                 # the network input: no ground truth
            check_inputs(batch, raise_on=True)
            token = batch["tokens"][0]
            row.update(index=i, token=token, next=ds.nusc.get("sample", token)["next"],
                       **{f"cover_{k}": round(float(batch[k].mean()), 4) for k in MAPS if k in batch})
            yield batch, row
    return cfg, run()


def precompute_frames(config, split, n, scene, first, out, write_frames):
    """`check` / `precompute`: build and check the network input of chosen frames; with write_frames also one .pt per frame (the batch: the
    BEVInputs + maps, no ground truth) and index.json (code commit, the rows with the cover of every map)"""
    import subprocess
    import time
    from precompute import MAPS
    t0 = time.time()
    cfg, frames = network_inputs(config, split, n, scene, first)
    out = out or os.path.join("temp", "precompute", f"{os.path.basename(config).replace('.json', '')}_{split}")
    if write_frames:
        os.makedirs(out, exist_ok=True)
    rows = []
    for batch, row in frames:
        rows.append(row)
        if write_frames:
            torch.save(batch, os.path.join(out, f"{row['token']}.pt"))
    fan_gt = [r["fan_equals_gt"] for r in rows if "fan_equals_gt" in r]
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "training"], capture_output=True, text=True).stdout.strip())
    summary = {"config": config, "split": split, "frames": len(rows), "code": commit + ("+dirty" if dirty else ""), "maps": list(MAPS),
               "radar_static_vr": cfg.model.get("rad_static_vr", 1.0), "radar_fan": "autogt.coverage.radar_cells of the calibrated radar poses", "rows": rows}
    if write_frames:
        json.dump(summary, open(os.path.join(out, "index.json"), "w"), indent=1)
    if fan_gt:
        assert min(fan_gt) >= 0.999, f"vis_radar differs from the ground truth's radar layer (share of equal cells per frame: {fan_gt})"
    mean = {k: round(sum(r[f"cover_{k}"] for r in rows) / len(rows), 3) for k in MAPS if f"cover_{k}" in rows[0]}
    print(f"runner {'precompute' if write_frames else 'check'}: {len(rows)} frames of {split} pass interface.check_inputs ({time.time() - t0:.0f} s), "
          f"mean cover {mean}" + (f", fan equals the GT's radar layer on {min(fan_gt):.4f} or more of the cells" if fan_gt else "")
          + (f"; written to {out}" if write_frames else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "precompute", "network", "full"])
    ap.add_argument("target", help="precompute: a training config; network / full: a run folder (runs/<name>)")
    ap.add_argument("frames", nargs="?", default="", help="network: the folder written by `precompute`")
    ap.add_argument("--config", default="", help="full: the data / grid config (default RUN/config.json)")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--scene", default="")
    ap.add_argument("--first", type=int, default=0, help="with --scene: only its first N frames")
    ap.add_argument("--out", default="")
    ap.add_argument("--dts", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--code", default="", help="a code folder to run instead of RUN/code (e.g. training)")
    ap.add_argument("--cold", action="store_true", help="network / full: restart the memory at every frame (no carried state)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    for k in ("target", "frames", "config", "out", "code"):          # every path against the caller's folder, before we work from the bev_fusion folder
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    os.chdir(ROOT)
    run = a.target
    if a.mode in ("check", "precompute"):
        use_code(ROOT, a.code or os.path.join(ROOT, "training"))
        precompute_frames(a.target, a.split, a.n, a.scene, a.first, a.out, a.mode == "precompute")
        return
    code = use_code(run, a.code)
    net = Network(run, a.device, a.dts, a.cold)
    rows, boxes, maps = [], [], []
    if a.mode == "network":
        index = json.load(open(os.path.join(a.frames, "index.json")))
        if not all("next" in r for r in index["rows"]):
            raise SystemExit(f"{a.frames} was written before the runner existed (its rows have no 'next'): make the frames again with `precompute`")
        if net.cfg.model.get("rad_static_gate") and index.get("radar_static_vr") != net.cfg.model.get("rad_static_vr", 1.0):
            raise SystemExit(f"the frames' radar_static was made with rad_static_vr {index.get('radar_static_vr')}, the run uses "
                             f"{net.cfg.model.get('rad_static_vr', 1.0)}: make the frames again with that config")
        stream = ((torch.load(os.path.join(a.frames, f"{r['token']}.pt"), weights_only=False), r) for r in index["rows"])
    else:
        _, stream = network_inputs(a.config or os.path.join(run, "config.json"), a.split, a.n, a.scene, a.first, with_gt=False)
    for batch, row in stream:
        b, m = net(batch, row)
        rows.append(row), boxes.append(b), maps.append(m)
    write(a.out or os.path.join(run, "runner", a.mode), net, rows, boxes, maps, {"run": os.path.relpath(run, ROOT), "code": os.path.relpath(code, ROOT)})


if __name__ == "__main__":
    main() if len(sys.argv) > 1 else print(__doc__)
