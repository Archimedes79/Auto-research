"""crn_export - run CRN (R50, camera + radar, 4 key frames) on the val infos of the lane project and dump its raw detections (card t26, G0, the object row of the SOA table).

Runs in the env `crn` (python 3.8, torch 1.9.1), from the bev_fusion folder, ONE GPU (CUDA-only voxel pooling), batch 1:
    ~/miniconda3/envs/crn/bin/python evaluation/crn/crn_export.py [--out DIR] [--limit N] [--tf32] [--workers W]
    ... --dry-run        resolve the exp, the data links, the info file and the input files and print the command; no model, no CUDA
    ... --check-load     on the CPU (no CUDA): build the model, load the checkpoint strictly, read sample 0 through the dataset (no forward)
    ... --oracle         on the CPU (no CUDA, no model): CRN's OWN ground-truth boxes (dataset.get_gt) as the "detections" through the same writers, score 1; a perfect CRN

What it is: the exp class CRNLightningModel of ~/work/CRN (unchanged, nothing in that repo is edited) with return_depth = False (the depth labels are not on disk and `eval_step` ignores
them), the checkpoint ckpt/CRN_r50_256x704_128x128_4key.pth loaded strictly, `eval_step` per sample under torch.cuda.amp.autocast (what Lightning's precision=16 does around test_step),
TF32 OFF for matmul and cuDNN (torch 1.9 has both ON by default; --tf32 turns them on, for the comparison of gate (b)). The samples are the info file's `nuscenes_infos_val.pkl` =
the link data/nuScenes/nuscenes_infos_val.pkl -> ~/data/nuscenes_full/nuscenes_infos_val_crn.pkl (1706 samples of 43 scenes); --limit N takes N samples spread evenly over them (the
key-frame history of a sample is read from the whole list, as in the full run).
Output in --out (default ~/work/CRN/outputs/t26_crn/tf32_<off|on>[_n<N>]):
    results_nusc.json   CRN's own writer (DetNuscEvaluator._format_bbox): global frame, top 500 per sample: the input of crn_convert.py and of crn_devkit_gate.py
    crn_dump.pkl        {"meta": {...}, "samples": {token: {boxes (n, 9) = x y z l w h yaw vx vy in CRN's ego (the mean of its six camera poses), scores, labels,
                        ego2global_translation, ego2global_rotation}}}: the raw predictions, the input of crn_gates.py tf32; meta holds the settings, peak GPU memory, ms per sample
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CRN = os.path.expanduser("~/work/CRN")
EXP_MODULE, EXP_FILE = "exps.det.CRN_r50_256x704_128x128_4key", "exps/det/CRN_r50_256x704_128x128_4key.py"
CKPT = "ckpt/CRN_r50_256x704_128x128_4key.pth"
DATA_ROOT = "data/nuScenes"
KEY_IDXES = (0, -2, -4, -6)                         # the exp: key_idxes [-2, -4, -6] after the key frame itself (checked against the model in the real run)
CAMS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
LINKS = ("samples", "maps", "v1.0-trainval", "radar_pv_filter", "radar_bev_filter", "nuscenes_infos_val.pkl")


# ------------------------------------------------------------------------------------------------ what is read, without torch
def pick(n, limit):
    """indices of the samples to run: all, or `limit` spread evenly"""
    if not limit or limit >= n:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (limit - 1))) for i in range(limit)}) if limit > 1 else [0]


def input_files(infos, indices, data_root=DATA_ROOT):
    """(image files, radar PV files) the dataset reads for the samples: the six cameras of the key frame and of the three older key frames (the dataset's own index rule:
    idx + key, forward to the first sample of the same scene)"""
    imgs, radar = set(), set()
    for idx in indices:
        for k in KEY_IDXES:
            cur = idx + k
            while infos[cur]["scene_token"] != infos[idx]["scene_token"]:
                cur += 1
            for cam in CAMS:
                f = infos[cur]["cam_infos"][cam]["filename"]
                imgs.add(os.path.join(data_root, f))
                radar.add(os.path.join(data_root, "radar_pv_filter", os.path.basename(f) + ".bin"))
    return sorted(imgs), sorted(radar)


def missing_files(files):
    with ThreadPoolExecutor(4) as pool:
        ok = list(pool.map(os.path.exists, files))
    return [f for f, o in zip(files, ok) if not o]


def default_out(a):
    return os.path.join(CRN, "outputs", "t26_crn", ("tf32_on" if a.tf32 else "tf32_off") + (f"_n{a.limit}" if a.limit else "") + ("_oracle" if a.oracle else ""))


def command_of(a, out):
    return " ".join(["~/miniconda3/envs/crn/bin/python", "evaluation/crn/crn_export.py", "--out", out] + (["--tf32"] if a.tf32 else []) + (["--limit", str(a.limit)] if a.limit else [])
                    + ["--workers", str(a.workers)] + (["--oracle"] if a.oracle else []))


def dry_run(a, out, infos, idx):
    print(f"exp        {EXP_FILE} (class CRNLightningModel, return_depth overridden to False)")
    print(f"checkpoint {CKPT}: {os.path.getsize(CKPT) / 1e6:.0f} MB")
    for n in LINKS:
        p = os.path.join(DATA_ROOT, n)
        print(f"link       {p} -> {os.path.realpath(p)}" + ("" if os.path.exists(p) else "   MISSING"))
    print(f"info       {len(infos)} samples in {len({i['scene_name'] for i in infos})} scenes; selected {len(idx)}" + (f" (--limit {a.limit}: spread evenly)" if a.limit else " (all)"))
    imgs, radar = input_files(infos, idx)
    miss_i, miss_r = missing_files(imgs), missing_files(radar)
    print(f"files      {len(imgs)} images + {len(radar)} radar PV files for the 4 key frames x 6 cameras of the selected samples: {len(miss_i)} + {len(miss_r)} missing" + (f" (first {(miss_i + miss_r)[0]})" if miss_i + miss_r else ""))
    print(f"settings   batch 1, 1 GPU, {a.workers} workers, TF32 matmul / cuDNN {'ON' if a.tf32 else 'OFF'}, autocast fp16 (Lightning precision=16), no depth labels, weights fp32 as stored")
    print(f"output     {out}/results_nusc.json  and  {out}/crn_dump.pkl")
    print(f"command    {command_of(a, out)}")
    try:
        import importlib
        mod = importlib.import_module(EXP_MODULE)
        print(f"import     {EXP_MODULE}.CRNLightningModel: {'found' if hasattr(mod, 'CRNLightningModel') else 'MISSING'} (the class only, nothing built)")
    except Exception as e:                                  # noqa: BLE001  a dry run reports what does not resolve
        print(f"import     {EXP_MODULE}: FAILED {type(e).__name__}: {e}")
    return 1 if miss_i or miss_r else 0


# ------------------------------------------------------------------------------------------------ the model and its inputs
def export_class():
    from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel

    class CRNExport(CRNLightningModel):
        """CRN's exp unchanged except that the depth labels are not read"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.return_depth = False
    return CRNExport


def build(out):
    import torch
    model = export_class()(gpus=1, batch_size_per_device=1, default_root_dir=out)
    assert model.key_idxes == [-2, -4, -6] and model.val_info_paths.endswith("nuscenes_infos_val.pkl"), (model.key_idxes, model.val_info_paths)
    ck = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ck["state_dict"])                 # strict: a missing or extra key is an error
    print(f"checkpoint loaded strictly: epoch {ck['epoch']}, step {ck['global_step']}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    return model


def settings_dict(a, idx, infos):
    import torch
    from exps.base_exp import CLASSES
    commit = subprocess.run(["git", "-C", CRN, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"tf32": a.tf32, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32, "autocast": "fp16", "batch": 1, "workers": a.workers,
            "n": len(idx), "limit": a.limit, "indices": idx, "tokens": [infos[i]["sample_token"] for i in idx], "info": os.path.realpath(os.path.join(DATA_ROOT, "nuscenes_infos_val.pkl")),
            "checkpoint": os.path.realpath(CKPT), "crn_commit": commit, "torch": torch.__version__, "class_names": list(CLASSES), "date": time.strftime("%Y-%m-%dT%H:%M:%S")}


def write_outputs(results, metas, out, meta):
    """CRN's own json writer (the global boxes) and the raw dump"""
    from evaluators.det_evaluators import DetNuscEvaluator
    from exps.base_exp import CLASSES
    DetNuscEvaluator(class_names=CLASSES, output_dir=out)._format_bbox(results, metas, out)
    samples = {m["token"]: {"boxes": r[0].astype("float32"), "scores": r[1].astype("float32"), "labels": r[2].astype("int64"),
                            "ego2global_translation": m["ego2global_translation"], "ego2global_rotation": m["ego2global_rotation"]} for r, m in zip(results, metas)}
    path = os.path.join(out, "crn_dump.pkl")
    with open(path + ".tmp", "wb") as f:
        pickle.dump({"meta": meta, "samples": samples}, f)
    os.replace(path + ".tmp", path)
    print(f"wrote {os.path.join(out, 'results_nusc.json')} and {path}: {len(samples)} samples", flush=True)


def run_model(a, out, infos, idx):
    import torch
    from torch.utils.data import DataLoader, Subset
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    print(f"TF32 matmul {torch.backends.cuda.matmul.allow_tf32}, cuDNN {torch.backends.cudnn.allow_tf32} (read back); {len(idx)} samples, batch 1, {a.workers} workers; output {out}", flush=True)
    model = build(out)
    dl = model.val_dataloader()
    assert len(dl.dataset) == len(infos), (len(dl.dataset), len(infos))
    loader = DataLoader(Subset(dl.dataset, idx), batch_size=1, num_workers=a.workers, shuffle=False, collate_fn=dl.collate_fn)
    model.cuda().eval()
    results, metas, ms = [], [], []
    t0 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, batch in enumerate(loader):
            torch.cuda.synchronize()
            t = time.time()
            res = model.eval_step(batch, i, "test")[0]
            torch.cuda.synchronize()
            ms.append((time.time() - t) * 1e3)
            results.append(res[:3])
            metas.append(res[3])
            if (i + 1) % 50 == 0 or i + 1 == len(loader):
                print(f"{i + 1}/{len(loader)} samples, {(time.time() - t0) / (i + 1):.2f} s per sample (model {sum(ms[-50:]) / len(ms[-50:]):.0f} ms), "
                      f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB allocated, {torch.cuda.max_memory_reserved() / 2**30:.2f} GB reserved", flush=True)
    meta = settings_dict(a, idx, infos)
    meta.update(peak_mem_allocated_gb=torch.cuda.max_memory_allocated() / 2**30, peak_mem_reserved_gb=torch.cuda.max_memory_reserved() / 2**30, ms_per_sample=sum(ms) / len(ms),
                device=torch.cuda.get_device_name(0), seconds=time.time() - t0, oracle=False)
    write_outputs(results, metas, out, meta)
    return 0


def run_oracle(a, out, infos, idx):
    """no model: CRN's own training targets of the samples (dataset.get_gt: boxes in its ego, yaw, velocity) as detections of score 1 -> the same writers.
    A perfect CRN: through crn_convert.py they must come out as our GT boxes (up to the differences of the labels)."""
    import numpy as np
    import exps.base_exp as B
    from datasets.nusc_det_dataset import NuscDatasetRadarDet
    ds = NuscDatasetRadarDet(ida_aug_conf=B.ida_aug_conf, bda_aug_conf=B.bda_aug_conf, rda_aug_conf=B.rda_aug_conf, img_backbone_conf=B.backbone_img_conf, classes=B.CLASSES,
                             data_root=DATA_ROOT, info_paths=os.path.join(DATA_ROOT, "nuscenes_infos_val.pkl"), is_train=False, key_idxes=list(KEY_IDXES[1:]),
                             return_image=False, return_depth=False, return_radar_pv=False, remove_z_axis=True)
    results, metas = [], []
    for i in idx:
        ret = ds[i]
        boxes = np.nan_to_num(ret[8].numpy().astype("float64"))        # a box without a neighbour has a NaN velocity in the infos
        results.append([boxes, np.ones(len(boxes)), ret[9].numpy()])
        metas.append(ret[7])
    meta = {"oracle": True, "n": len(idx), "tokens": [infos[i]["sample_token"] for i in idx], "class_names": list(B.CLASSES), "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_outputs(results, metas, out, meta)
    return 0


def check_load(a, out, infos, idx):
    """CPU only: strict checkpoint load, the dataset's sample 0 (images, radar PV, matrices): shapes and no NaN. No forward pass (the voxel pooling is CUDA-only)."""
    import torch
    model = build(out)
    sample = model.val_dataloader().dataset[idx[0]]
    imgs, mats, metas, radar = sample[0], sample[1], sample[7], sample[11]
    print(f"sample {idx[0]} {metas['token']}: images {tuple(imgs.shape)} (key frames x cameras x 3 x H x W), radar PV {tuple(radar.shape)}, "
          f"sensor2ego {tuple(mats.shape)}, finite {bool(torch.isfinite(imgs).all() and torch.isfinite(radar).all())}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None, help="output folder (default ~/work/CRN/outputs/t26_crn/tf32_<off|on>[_nN])")
    p.add_argument("--limit", type=int, default=0, help="N samples spread evenly over the infos (0 = all 1706)")
    p.add_argument("--tf32", action="store_true", help="TF32 on for matmul and cuDNN (default off, our rule)")
    p.add_argument("--workers", type=int, default=2, help="data loader workers (JPEG decoding of 24 images per sample; 2 on the shared box)")
    p.add_argument("--dry-run", action="store_true", help="resolve everything and print the command; no model, no CUDA")
    p.add_argument("--check-load", action="store_true", help="CPU: build the model, load the checkpoint, read sample 0; no CUDA, no forward")
    p.add_argument("--oracle", action="store_true", help="CPU: CRN's own GT boxes as detections through the same writers; no model, no CUDA")
    a = p.parse_args(argv)
    cpu_only = a.dry_run or a.check_load or a.oracle
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""             # before torch is imported: nothing of this run touches the GPU
        os.environ.setdefault("OMP_NUM_THREADS", "2")
    out = os.path.abspath(a.out or default_out(a))           # before the chdir: a relative --out is relative to where the command was typed
    sys.path.insert(0, CRN)
    os.chdir(CRN)
    with open(os.path.join(DATA_ROOT, "nuscenes_infos_val.pkl"), "rb") as f:
        infos = pickle.load(f)
    idx = pick(len(infos), a.limit)
    if a.dry_run:
        return dry_run(a, out, infos, idx)
    if cpu_only:
        import torch
        torch.set_num_threads(2)
        assert not torch.cuda.is_available()
    os.makedirs(out, exist_ok=True)
    return check_load(a, out, infos, idx) if a.check_load else run_oracle(a, out, infos, idx) if a.oracle else run_model(a, out, infos, idx)


if __name__ == "__main__":
    raise SystemExit(main())
