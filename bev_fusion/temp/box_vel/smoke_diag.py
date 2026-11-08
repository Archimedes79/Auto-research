"""Smoke test of train_diag.py (CPU, nuScenes mini, the 4 cached samples of the trainer's self-test, trunk width 0.25, 3 steps per arm):
every arm trains, the log rows carry the diag.* columns, the l1 arm's trained term equals today's L1 term, the zero-predictor term is
finite. Run (from bev_fusion/): OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python temp/box_vel/smoke_diag.py
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "training"), HERE]

import torch  # noqa: E402

torch.set_num_threads(2)


def main():
    import train_diag
    from nuscenes.nuscenes import NuScenes
    nusc = None
    for kind in ("l1", "huber", "moving"):
        T = train_diag.install(kind)
        from feature_cache import CachedFrontRadarDataset, FeatureStore, used_frames
        cache = "data/cache/cam_feats_mini"
        store = FeatureStore(cache)
        tmp = tempfile.mkdtemp()
        try:
            cfg = T.TrainConfig(name="smoke", out_root=tmp, data_root="~/data/nuscenes", version="v1.0-mini", train_split="train",
                                val_split="val", feature_cache=cache, gt_cache=os.path.join(tmp, "gt"), batch_size=2, accum=1, workers=0,
                                steps=3, warmup=1, log_every=1, eval_every=0, save_every=0, keep=1, device="cpu", seq_len=1,
                                model={"trunk_width": 0.25, "object_direction": True})
            nusc = nusc or NuScenes(cfg.version, dataroot=os.path.expanduser(cfg.data_root), verbose=False)
            base = CachedFrontRadarDataset(os.path.expanduser(cfg.data_root), cfg.version, store=store, n_cam=3, cam_stride=2,
                                           with_gt=True, nusc=nusc, cache_dir=cfg.gt_cache)
            ok = [t for t in base.samples if all(store.has(x) for v in used_frames(base, [t]).values() for x in v)]
            base.samples = ok[:4]
            T.train(cfg, datasets=(base, base))
            rows = [json.loads(ln) for ln in open(os.path.join(tmp, "smoke", "log.jsonl"))]
            r = rows[-1]
            d = {k: v for k, v in r.items() if k.startswith("diag.")}
            assert {"diag.vel", "diag.vel_l1", "diag.vel_zero", "diag.act_vel", "diag.act_off", "diag.gn_vel0"} <= set(d), sorted(d)
            if kind == "l1":
                assert abs(d["diag.vel"] - d["diag.vel_l1"]) < 1e-6, d
            assert all(v == v for v in d.values()), d
            print(f"{kind}: ok, last row {json.dumps({k: round(v, 5) for k, v in d.items()})}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        for name in ("train_diag", "train", "losses", "object_head"):          # fresh modules for the next arm's wrappers
            sys.modules.pop(name, None)
    print("smoke_diag ok")


if __name__ == "__main__":
    main()
