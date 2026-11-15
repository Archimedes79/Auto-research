"""scratch (analysis, no verdict) stage 2 of the recall-by-paint test: the t20_base_s0 network (last checkpoint, EMA weights, its own code snapshot) on the CPU, through the REAL
evaluation adapter (evaluation/bev_adapter.build: the dataset, the checkpoint loader, Subject.predict, the seg metric), on the frames of rp_frames.json, memory EMPTY (state
reset, as the stored card: protocol.state = reset), all three dt's as the evaluation does (dt 0 is used here).
Per frame: temp/overview/rp_out/<i>.npz with the dt-0 probability maps (lane_divider, road_divider, centreline) and rp_out/<i>.counts.pkl with the seg metric's per-frame counts
(additive: reduce() over any subset gives the dataset-level numbers). Resumable (finished frames are skipped).
python temp/overview/rp_stage2_infer.py [--limit N] [--max-seconds S]      CPU only: run it through rp_py.sh (OMP_NUM_THREADS=2, nice, CUDA_VISIBLE_DEVICES empty)"""
import argparse
import json
import os
import pickle
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
RUN = "runs/t20_base_s0"
SIG = "0.5_4_6a7aa7864a"
KEEP = ("lane_divider", "road_divider", "centreline")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def mem_now():
    """current resident memory split into anonymous and file-backed pages (GB), from /proc/self/status"""
    out = {}
    for ln in open("/proc/self/status"):
        k = ln.split(":")[0]
        if k in ("VmRSS", "RssAnon", "RssFile", "VmHWM"):
            out[k] = round(int(ln.split()[1]) / 1048576.0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N frames of the list only (the list is shuffled over scenes)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this many seconds (finished frames are kept)")
    ap.add_argument("--full-tables", action="store_true", help="load the whole trainval devkit tables (7-8 GB RAM)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_available(), "CUDA visible: refusing to touch the GPU"
    from evaluation import bev_adapter as A
    from evalsuite import FrameRef  # noqa: F401
    t0 = time.time()
    os.chdir(ROOT)
    frames = json.load(open(os.path.join(HERE, "rp_frames.json")))
    if not a.full_tables:                                     # the devkit tables of the 83 scenes only: ~1 GB instead of 7-8 GB (rp_nusc_subset.py)
        sys.path.insert(0, HERE)
        import rp_nusc_subset
        rp_nusc_subset.restrict({f["scene"] for f in frames})
    source, subject, task = A.build(RUN, device="cpu", split="val", workers=0)
    d = source.describe()
    import gc
    gc.collect()
    print(f"built in {time.time() - t0:.0f} s; max RSS {rss_gb():.1f} GB; now {mem_now()}; subject {subject.describe()}", flush=True)
    print("source:", d, flush=True)
    assert d["gt_signature"] == SIG, f"GT signature {d['gt_signature']} is not {SIG}: refusing (the dataset could try to build GT)"
    assert subject.device == "cpu"
    torch.set_num_threads(2)
    seg = [m for m in task.metrics() if m.name == "seg"][0]
    if a.limit:
        frames = frames[:a.limit]
    by_tok = {}
    for sc in source.scenes():
        for r in source.frames(sc):
            by_tok[r.key] = r
    miss = [f for f in frames if f["token"] not in by_tok]
    print(f"{len(frames)} frames asked, {len(miss)} not in the val source (first/last keyframes of a scene are left out by the dataset)", flush=True)
    os.makedirs(os.path.join(HERE, "rp_out"), exist_ok=True)
    json.dump([f["i"] for f in miss], open(os.path.join(HERE, "rp_missing.json"), "w"))
    done, times = 0, []
    t_start = time.time()
    for f in frames:
        if f["token"] not in by_tok:
            continue
        out = os.path.join(HERE, "rp_out", f"{f['i']}.npz")
        if os.path.exists(out) and os.path.exists(out.replace(".npz", ".counts.pkl")):
            done += 1
            continue
        if a.max_seconds and time.time() - t_start > a.max_seconds:
            print("time budget used: stopping", flush=True)
            break
        t1 = time.time()
        sample = source.load(by_tok[f["token"]])
        subject.reset()                                       # empty memory: as the stored card (state reset)
        pred = subject.predict(sample)
        cnt = seg.frame(sample, pred)
        o0 = pred.outputs["dt"][0.0]
        np_ = {k: o0["prob"][k].astype("float16") for k in KEEP if k in o0["prob"]}
        import numpy as np
        np.savez(out + ".tmp.npz", **np_)
        os.replace(out + ".tmp.npz", out)
        pickle.dump(cnt, open(out.replace(".npz", ".counts.pkl"), "wb"))
        dt_s = time.time() - t1
        times.append(dt_s)
        done += 1
        print(f"frame {f['i']} ({f['scene']}): {dt_s:.1f} s, done {done}, max RSS {rss_gb():.1f} GB, now {mem_now()}", flush=True)
    print(f"finished: {done} frames in rp_out; mean {sum(times) / max(len(times), 1):.1f} s per new frame", flush=True)


if __name__ == "__main__":
    main()
