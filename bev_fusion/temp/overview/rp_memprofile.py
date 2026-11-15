"""scratch: where do the 8 GB of the evaluation adapter's build go? (RssAnon after each step; copy of bev_adapter.build's steps, read-only use of the repository)"""
import gc
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]


def mem(tag):
    d = {}
    for ln in open("/proc/self/status"):
        k = ln.split(":")[0]
        if k in ("RssAnon", "RssFile"):
            d[k] = round(int(ln.split()[1]) / 1048576.0, 2)
    print(f"{tag:40s} {d}", flush=True)


mem("start")
import torch  # noqa: E402
mem("torch imported")
from evaluation import bev_adapter as A  # noqa: E402
os.chdir(ROOT)
code = A.setup_paths("runs/t20_base_s0")
ns = A._chain()
mem("chain imported")
run = "runs/t20_base_s0"
cfg = ns.T.TrainConfig(**json.load(open(os.path.join(run, "config.json"))))
A.freeze_scenes(cfg)
cfg.workers, cfg.seq_len, cfg.batch_size = 0, 1, 1
from nuscenes.nuscenes import NuScenes  # noqa: E402
nusc = NuScenes(cfg.version, dataroot=os.path.expanduser(cfg.data_root), verbose=False)
gc.collect()
mem("NuScenes trainval loaded")
print({t: len(getattr(nusc, t)) for t in ("sample", "sample_data", "sample_annotation", "ego_pose", "instance", "scene")})
del nusc
gc.collect()
mem("NuScenes deleted")
tr, va = ns.T.make_datasets(cfg)
gc.collect()
mem("datasets made")
print(len(tr), len(va))
model, loss, _ = ns.T.build(cfg)
gc.collect()
mem("model built")
