"""Training of the toy network on the problem-independent engine trainkit (autoresearch/trainkit/README.md).

    python training/train.py <config.json>             train into <out_root>/<name>; the same command again resumes from the newest checkpoint
    python training/train.py --check <config.json>     the static check: every key known, the values sane

In a run of the research system the worker composes the config (the base config, the arm's settings, the seed, the run's folders) and starts
this script from the run's snapshot.
"""
import argparse
import sys
from dataclasses import dataclass

import numpy as np
import torch

from gt import generate as GT
from net.loss.loss import DiscLoss
from net.model.net import DiscNet
from net.prep import shards as S
from trainkit.config import Config, read_config
from trainkit.data import EpochLoader
from trainkit.loop import fit
from trainkit.util import prepare_run_dir, seed_all


@dataclass
class TrainConfig(Config):
    name: str = "toy"
    steps: int = 400
    lr: float = 3e-3
    warmup: int = 20
    accum: int = 1
    batch_size: int = 16
    width: int = 8                   # channels of the network
    pos_weight: float = 1.0          # weight of the disc pixels in the loss
    ema: float = 0.98
    log_every: int = 20
    eval_every: int = 200
    save_every: int = 100
    keep: int = 1
    device: str = "cpu"
    threads: int = 2                 # CPU threads: small, so a toy run does not disturb other work
    data_root: str = ""              # the toy's data folder (in a run: <data root of the machine>/toy)
    gt_supersample: int = 4

    RESUME_FREE = Config.RESUME_FREE + ("threads", "data_root")


def check(cfg):
    """problems of a config that make a run pointless (an unknown key is already an error of read_config)"""
    bad = [f"{k} must be at least 1" for k in ("steps", "batch_size", "width", "accum", "threads", "gt_supersample") if getattr(cfg, k) < 1]
    if cfg.device != "cpu":
        bad.append("the toy runs on the CPU (device cpu)")
    if not cfg.data_root or "{" in cfg.data_root:
        bad.append(f"data_root {cfg.data_root!r}: an absolute folder (the worker fills in the machine's data root)")
    return bad


def train(cfg):
    torch.set_num_threads(cfg.threads)
    sig = GT.signature(cfg.gt_supersample)
    tr, va = S.load_split("train", cfg.data_root, sig), S.load_split("val", cfg.data_root, sig)
    tensors = lambda d: (torch.from_numpy(d["frames"].copy())[:, None], torch.from_numpy(d["masks"].astype(np.float32))[:, None])      # noqa: E731
    x_val, y_val = tensors(va)
    loader = EpochLoader(torch.utils.data.TensorDataset(*tensors(tr)), cfg.batch_size, cfg.seed, drop_last=True)
    seed_all(cfg.seed)                                              # before the network is made: its start weights are a function of the seed
    model, loss = DiscNet(cfg.width), DiscLoss(cfg.pos_weight)

    def step_fn(batch, info):
        x, y = batch
        value = loss(model(x), y)
        info.backward(value)
        return {"total": float(value.detach())}, bool(torch.isfinite(value)), None

    def evaluate(ema):
        with ema.applied(model), torch.no_grad():
            logits = model(x_val)
            pred, truth = logits > 0, y_val > 0.5
            return {"total": float(loss(logits, y_val)), "iou": float((pred & truth).sum()) / max(1.0, float((pred | truth).sum()))}

    out_dir = prepare_run_dir(cfg)
    fit(cfg, model, loss, loader, step_fn, out_dir, cfg.device, evaluate=evaluate)
    print(f"trained {cfg.name}: {cfg.steps} steps, {sum(p.numel() for p in model.parameters())} parameters, ground truth {sig}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = read_config(TrainConfig, a.config)
    except TypeError as e:
        print(f"config {a.config}: {e}", file=sys.stderr)
        return 2
    bad = check(cfg)
    if bad:
        print("config: " + "; ".join(bad), file=sys.stderr)
        return 2
    if a.check:
        print(f"config {a.config}: ok")
        return 0
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
