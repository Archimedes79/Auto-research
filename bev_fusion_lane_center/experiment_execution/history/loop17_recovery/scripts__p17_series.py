"""Loop 17: failure series of the VAL scenes as an input cache for the loop-16 builders.
Protocol of E79 / E105 / E114 (p13_temporal_val.py): rows of a scene in time order (every third frame, 1.5 s apart), the first half clean, the second half the features of one perturbation, ego poses known.
For every perturbation and input variant (single = no fusion, maxdecay:a = spatial hold, ema_hold_mon:a = EMA + heat-monitor hold) the 42-channel builder input of every frame is written to
~/exp_gcar_b123/p17/<pert>_<variant>.npy (fp16 [N, 42, 200, 200]) with <pert>_<variant>.meta.npz (rows, scene, phase 0 = clean / 1 = failure, pos, prev = index of the previous frame of the scene or -1, flag).
Usage (gcar, GPU): p17_series.py --perts dark,blur,cam_front3_off --variants single,maxdecay:0.95,ema_hold_mon:0.5"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p13_temporal_val as TV  # noqa: E402
import p10_infer as PI  # noqa: E402
import p8_lib as L  # noqa: E402

D = L.D
OUTD = f"{os.path.expanduser('~')}/exp_gcar_b123/p17"


E17 = "/mnt/c/ML-Experiment/out/loop2/e17"


class PertReaderD(TV.PertReader):
    """perturbed frames with the dense maps (heat, direction) of another dense head (p17_dense.py --tag)"""

    def __init__(self, pn, mean, std, tag):
        self.dec = np.load(f"{D}/pert/{pn}/val_dec32.npy", mmap_mode="r")
        self.lg = np.load(f"{D}/pert/{pn}/val_logit7.npy", mmap_mode="r")
        rows = np.load(f"{D}/pert/{pn}/val_rows.npy")
        self.j = {int(r): i for i, r in enumerate(rows)}
        self.mean, self.std = mean, std
        d = f"{E17}/dense_{tag}"
        assert (np.load(f"{d}/{pn}_ids.npy") == rows).all()
        self.prob, self.dirf = np.load(f"{d}/{pn}_prob.npy", mmap_mode="r"), np.load(f"{d}/{pn}_dir.npy", mmap_mode="r")


def sequences(pn, es, P, min_frames=8, dense=None):
    rd = TV.PertReader(pn, es.mean, es.std) if dense is None else PertReaderD(pn, es.mean, es.std, dense)
    rows_p = [r for r in es.rows if r in rd.j]
    by = {}
    for r in rows_p:
        by.setdefault(int(P["scene"][r]), []).append(r)
    seqs = [sorted(v, key=lambda i: P["ts"][i]) for v in by.values() if len(v) >= min_frames]
    return rd, seqs


def build(pn, variants, thr=0.6):
    es = PI.EvalSet("val_clean")
    P = dict(np.load(f"{D}/poses_val.npz"))
    kof = {int(r): k for k, r in enumerate(es.rows)}
    rd, seqs = sequences(pn, es, P)
    read = lambda r, fail: rd(r) if fail else es.batch([kof[r]], "cuda")[0]
    order = [r for s in seqs for r in s]
    n = len(order)
    os.makedirs(OUTD, exist_ok=True)
    for vs in variants:
        var, a = (vs.split(":") + ["0"])[:2]
        a = float(a)
        tag = f"{pn}_{vs.replace(':', '_')}"
        mm = np.lib.format.open_memmap(f"{OUTD}/{tag}.npy", mode="w+", dtype=np.float16, shape=(n, 42, 200, 200))
        phase, pos, prev, flag, scene = [], [], [], [], []
        i0 = 0
        t0 = time.time()
        for rows in seqs:
            f0 = len(rows) // 2
            fused, flags, _ = TV.fuse_scene(rows, read, f0, P, var, a, thr)
            for j, x in enumerate(fused):
                mm[i0 + j] = x.cpu().numpy()
                phase.append(int(j >= f0))
                pos.append(j - f0)
                prev.append(i0 + j - 1 if j > 0 else -1)
                flag.append(bool(flags[j]))
                scene.append(int(P["scene"][rows[j]]))
            i0 += len(rows)
            del fused
            torch.cuda.empty_cache()
        mm.flush()
        np.savez(f"{OUTD}/{tag}.meta.npz", rows=np.asarray(order), scene=np.asarray(scene), phase=np.asarray(phase), pos=np.asarray(pos), prev=np.asarray(prev), flag=np.asarray(flag))
        print(f"[{pn}] {vs:16s} frames {n} flagged fail {np.mean([f for f, p in zip(flag, phase) if p]):.2f}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perts", default="dark,blur,cam_front3_off")
    ap.add_argument("--variants", default="single,maxdecay:0.95,ema_hold_mon:0.5")
    ap.add_argument("--par_thr", type=float, default=0.6)
    a = ap.parse_args()
    for pn in a.perts.split(","):
        build(pn, a.variants.split(","), a.par_thr)
    print("DONE", time.strftime("%H:%M:%S"))
