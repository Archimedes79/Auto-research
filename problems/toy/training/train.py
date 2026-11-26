"""Training of the toy network on the project-independent engine trainkit (autoresearch/trainkit/README.md).

    bash tools/py.sh training/train.py <config.json>            train; the same command again resumes from the newest checkpoint
    bash tools/py.sh training/train.py --check <config.json>    the static check: every key known, the values sane (the data are the 'prepare' item's job)
"""
import argparse
import json
import os
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
from trainkit.util import check_engine, prepare_run_dir, seed_all


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
    data_root: str = ""              # "" = data/store of the problem (or TOY_DATA)
    gt_supersample: int = 4
    compare_to: str = ""             # run folder name of the control this run's card is compared with (evaluation/score.py)

    RESUME_FREE = Config.RESUME_FREE + ("threads", "data_root", "compare_to")


def check(cfg):
    """problems of a config that make a run pointless (an unknown key is already an error of read_config)"""
    bad = [f"{k} must be at least 1" for k in ("steps", "batch_size", "width", "accum", "threads", "gt_supersample") if getattr(cfg, k) < 1]
    if cfg.device != "cpu":
        bad.append("the toy runs on the CPU (device cpu)")
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
    with open(os.path.join(out_dir, "gt.json"), "w", encoding="utf-8") as f:
        json.dump({"signature": sig, "version": sig}, f)            # what Training shows as the truth a run trained on
    fit(cfg, model, loss, loader, step_fn, out_dir, cfg.device, evaluate=evaluate)
    print(f"trained {cfg.name}: {cfg.steps} steps, {sum(p.numel() for p in model.parameters())} parameters", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    print(check_engine(os.path.dirname(os.path.abspath(__file__))), flush=True)
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
