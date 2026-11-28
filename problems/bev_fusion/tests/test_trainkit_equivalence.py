"""Golden test of the BEV training: train() of training/train.py on the engine autoresearch/trainkit trains the real network and loss, tiny and on the CPU, on
synthetic clips, and must give the pinned losses, run record and weights bit for bit. The run skips a step with a non-finite loss (an incident dump), is stopped
(CONTROL file: checkpoint, STOPPED, exit 75) and resumed. A changed value means that the trainer, the engine, the network or the loss changed what this config
does; when that is meant, pin the new values (--pin prints them) and say why in the commit. After the golden run it runs the engine's own rules
(trainkit.selftest).

    python tests/test_trainkit_equivalence.py [--pin]      (from the problem folder, with its module path; CPU, 2 threads, about half a minute)
"""
import functools
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from training import train as T
from net.model.model import ModelConfig
from tests.support import synthetic_batch
from trainkit import control

# steps 1, 2, 4 (3 is skipped); 2, 4
GOLDEN = {"train totals": [29.28260326385498, 23.38815450668335, 27.408717155456543], "val totals": [33.12045764923096, 25.308552742004395],
          # files, log, incidents, config (the config without the removed key uwso_T: the same run)
          "run record": "fc39674bad178264", "weights": "4bed3499a2ffe731"}
MODEL =dict(pretrained="", trunk_pretrained="", oversample=1, c_cam=8, c_rad=64, c_fuse=16, trunk_width=0.25, radar_kind="delta_slots", motion_block=True)
CONFIG = dict(name="golden", device="cpu", workers=0, batch_size=1, accum=2, steps=4, warmup=2, lr=2e-3, ema=0.2, grad_clip=1.0, grad_clip_split=("motion_block",),
              lr_groups={"radar": {"lr": 1e-3, "warmup": 3}}, freeze=("aux_rad",), log_every=1, eval_every=2, save_every=2, keep=1, eval_batches=2, seq_len=2,
              seq_bptt=True, flip_aug=0.5, consistency=0.5, consistency_warmup=(0.0, 0.5), radar_aux=1.0, grad_diag_every=2, probe=True,
              feature_cache="synthetic", gt_cache="", model=MODEL)
WALL = ("t", "s_per_step", "t_data", "t_step", "time")          # wall-clock fields of a log row and of an incident dump


@functools.lru_cache(maxsize=None)
def keyframe(i):
    return synthetic_batch.keyframe(ModelConfig(**MODEL), i)


class Keyframes(torch.utils.data.Dataset):
    """the first n keyframes; on_fetch(k, item) sees the k-th item that is read (from 1) and returns the item to give"""

    def __init__(self, n, on_fetch=lambda k, item: item):
        self.samples, self.on_fetch, self.fetched = [f"k{i}" for i in range(n)], on_fetch, 0
        self.nusc = self                                         # ClipDataset follows nusc.get("sample", token)["next"]

    def get(self, table, token):
        return {"next": f"k{int(token[1:]) + 1}"}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        self.fetched += 1
        return self.on_fetch(self.fetched, keyframe(i))


def golden_run():
    """-> what the run leaves behind: the logged totals, and digests of the run folder (files, log rows, incident dumps) and of the final weights and EMA"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = T.TrainConfig(**CONFIG, out_root=tmp)
        out = os.path.join(tmp, cfg.name)

        def first_start(k, item):                                # a step reads 4 key frames (2 micro-batches, clips of 2): read 9 opens step 3
            if k == 9:                                           # a target that is not a number: step 3 is skipped
                cur = item["gt"]["cur"]
                return {**item, "gt": {**item["gt"], "cur": {**cur, "drivable_area": cur["drivable_area"] * np.nan}}}
            if k == 10:
                control.request_stop(out, "golden")              # seen at the next step boundary
            return item
        try:
            T.train(cfg, datasets=(Keyframes(5, first_start), Keyframes(3)))
            raise AssertionError("the stop request must end the run")
        except SystemExit as e:
            assert e.code == 75, e.code                          # the exit code of a stop: the research worker reads exactly this one
        T.train(cfg, datasets=(Keyframes(5), Keyframes(3)))      # the same start again resumes from the checkpoint of step 3
        files = sorted(os.listdir(out))
        read = lambda text: {k: v for k, v in json.loads(text).items() if k not in WALL}          # noqa: E731
        rows = [read(ln) for ln in open(os.path.join(out, "log.jsonl"))]
        dumps = [read(open(os.path.join(out, f)).read()) for f in files if f.startswith("incident_")]
        ck = torch.load(os.path.join(out, f"ckpt_{cfg.steps:08d}.pt"), map_location="cpu", weights_only=False)
        weights = hashlib.sha256()
        for part in ("state", "ema"):
            for k, v in ck[part].items():
                weights.update(f"{part}.{k} {v.dtype} {tuple(v.shape)}".encode() + v.numpy().tobytes())
        config = {k: v for k, v in json.load(open(os.path.join(out, "config.json"))).items() if k != "out_root"}      # the settings with their defaults
        return {"train totals": [r["total"] for r in rows if "val" not in r], "val totals": [r["val"]["total"] for r in rows if "val" in r],
                "run record": hashlib.sha256(json.dumps([files, rows, dumps, config], sort_keys=True).encode()).hexdigest()[:16], "weights": weights.hexdigest()[:16]}


def main(argv):
    torch.set_num_threads(2)                                     # the pinned bits are those of two threads
    got = golden_run()
    if "--pin" in argv:
        print("GOLDEN = " + json.dumps(got))
        return 0
    from trainkit import selftest as engine                      # the engine's rules a golden run does not reach (atomic saves, superseded rows, warm
    engine.main()                                                # start, fp16, probes): the problem's tests run them, no other list does
    bad = [k for k in got if got[k] != GOLDEN.get(k)]
    for k, v in got.items():
        print(f"  {'FAIL' if k in bad else 'ok  '} {k}: {v}" + (f"   pinned: {GOLDEN.get(k)}" if k in bad else ""))
    print("TRAINKIT EQUIVALENCE (golden BEV training): " + (f"FAIL ({', '.join(bad)})" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
