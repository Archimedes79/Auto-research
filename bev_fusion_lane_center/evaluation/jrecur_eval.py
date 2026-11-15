"""Read-out of the pair stage with the recursive feature state (E354): the dump of `jends_eval.py dump` in STREAM order (scene by scene, time order, batch 1, the state carried), so that the stored scorers read it unchanged.

    wsl bash tools/env/_py14.sh jrecur_eval.py dump --run e354_r_s0 --seed 0 --state stream                 # the delivered mode of arm r -> runs/<run>/ends_val.pkl, ends_head.json
    wsl bash tools/env/_py14.sh jrecur_eval.py dump --run e354_z_s0 --seed 0 --state off                    # the delivered mode of arm z (trained without a state)
    wsl bash tools/env/_py14.sh jrecur_eval.py dump --run e354_r_s0 --seed 0 --state off|shuffle|trunc3 --tag off|shuffle|trunc3     # another read-out of the same checkpoint -> the sibling folder runs/<run>__<tag>/
    wsl bash tools/env/_py14.sh jends_eval.py score --run e354_r_s0 --seed 0 --thr 0.1 [--snap]             # the stored scorer, unchanged; a tagged read-out: --run e354_r_s0__shuffle

States: `stream` the state of the keyframe before, from the first to the last keyframe of a scene, empty at a scene start (`lane_seq_data.stream_order`: the previous keyframe of the row is not the row before it); `off` never a state;
`trunc3` empty at every third keyframe of a scene (the training distribution of `--seq 3`); `shuffle` the STREAM state of the keyframe with the same index of the NEXT scene of the list (the last scene reads the first; the donor's last
state when the donor is shorter), warped with the frame's own pose: a real state of the wrong place. The donor states of `shuffle` come from a first stream pass (kept as float16 on the CPU, about 0.3 GB).
The untagged dump is the delivered mode of the run (stream for a run trained with `--state on`, off for `--state off`); every other mode needs `--tag` and goes to runs/<run>__<tag>/ with a copy of model.pt and config.json, which is all
`jends_eval.py score`, `jtime.py`, `jmaneuver.py` and `jpair_pr.py` read of a run next to ends_val.pkl (they take the dump by the run name only). Tests: training/grid/test_jrecur.py
"""
import argparse
import json
import os
import pickle
import shutil
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jrecur as JR  # noqa: E402
import lane_gt_data as LD  # noqa: E402
import lane_seq_data as LS  # noqa: E402
import p16_data as PD  # noqa: E402
import qendnet as QEN  # noqa: E402

STATES = ("stream", "off", "shuffle", "trunc3")


def state_plan(reset, mode):
    """reset (n,) bool of the stream (`stream_order`) -> per frame the stream index of the frame whose STATE it reads, -1 = none. stream / trunc3 read the frame before (index - 1), shuffle a frame of the next scene"""
    assert mode in STATES, mode
    starts = [i for i, r in enumerate(reset) if r] + [len(reset)]
    segs = list(zip(starts[:-1], starts[1:]))
    src = []
    for si, (s0, s1) in enumerate(segs):
        d0, d1 = segs[(si + 1) % len(segs)]
        for n in range(s1 - s0):
            if n == 0 or mode == "off" or (mode == "trunc3" and n % 3 == 0):
                src.append(-1)
            elif mode == "shuffle":
                src.append(min(d0 + n - 1, d1 - 1))
            else:
                src.append(s0 + n - 1)
    return src


def run_stream(m, ds, order, src, pose, dev, donor=None, keep=False, thr=0.5, workers=2):
    """the frames of `order` one by one. Frame i reads the state of frame src[i] (the state carried from frame i - 1, or `donor[src[i]]`), warped by the pose of ITS previous keyframe.
    -> (frames: the dicts of `jends_eval.dump`, counts of `qendnet.end_counts`, states (float16, CPU) when keep). keep = a donor pass: the maps head only, no frames"""
    i_of_row = {int(ds.bd.rows[k]): i for i, k in enumerate(ds.ks)}
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [i_of_row[r] for r, _ in order]), batch_size=1, shuffle=False, num_workers=workers, collate_fn=LD.collate)
    frames, tot, states, last = [], {}, [], None
    with torch.no_grad():
        for i, b in enumerate(dl):
            row = int(b["row"][0])
            assert row == order[i][0], (row, order[i])
            state = rel = None
            if src[i] >= 0:
                assert donor is not None or src[i] == i - 1, "without donor states a frame reads the frame before it"
                state = last if donor is None else donor[src[i]].to(dev).float()
                prow = int(pose["prev"][row])
                assert prow >= 0, f"row {row} reads a state but has no previous keyframe"
                R, t = LS.rel_pose(pose["R"][row], pose["t"][row], pose["R"][prow], pose["t"][prow])
                rel = (torch.from_numpy(R)[None].to(dev), torch.from_numpy(t)[None].to(dev))
            x = b["x"].to(dev).float()
            if keep:
                _, jout = m.jmodel(x, state=state, rel=rel)
                states.append(jout["state"].half().cpu())
            else:
                _, jout, out = m(x, gt=None, state=state, rel=rel)
                for k, v in QEN.end_counts(out, b, thr).items():
                    tot[k] = tot.get(k, 0) + v
                frames.append(dict(token=b["token"][0], row=row, kp_pos=out["kp_pos"][0].cpu().numpy(), kp_dir=out["kp_dir"][0].cpu().numpy(), kp_p=out["kp_p"][0].cpu().numpy(), valid=out["valid"][0].cpu().numpy(),
                                   logit=out["logit"][0].float().cpu().numpy(), allowed=out["allowed"][0].cpu().numpy(), scale=out["scale"][0].float().cpu().numpy() if "scale" in out else None))
            last = jout["state"]
    return frames, tot, states


def dump(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":                                                                                                                          # D100: a dump next to two trainings crashed one of them
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jrecur_eval dump {a.run}")
    m, args = JR.load_model(a.run, dev)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    own = "stream" if args["state"] == "on" else "off"
    assert a.tag or a.state == own, f"the untagged dump of {a.run} is its delivered mode ({own}): --state {a.state} needs --tag"
    assert a.tag is None or a.tag.isalnum(), "--tag: letters and digits (it is part of a folder name)"
    LD.QData.DEFAULT_JUNCTION = True
    ds = LD.QData("val_clean", map_mode="error", pixels=False)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    pose = dict(np.load(f"{PD.D}/poses_val.npz"))
    order = LS.stream_order(pose, [int(ds.bd.rows[k]) for k in ds.ks])
    order = order[:a.max] if a.max else order
    reset = [r for _, r in order]
    donor = None
    if a.state == "shuffle":
        donor = run_stream(m, ds, order, state_plan(reset, "stream"), pose, dev, keep=True)[2]
    frames, tot, _ = run_stream(m, ds, order, state_plan(reset, a.state), pose, dev, donor=donor, thr=a.thr)
    r = lambda x, y: x / max(y, 1)  # noqa: E731
    head = dict(ent_recall=r(tot["ent_hit"], tot["ent_n"]), ent_precision=r(tot["ent_ok"], tot["ent_kp"]), ext_recall=r(tot["ext_hit"], tot["ext_n"]), ext_precision=r(tot["ext_ok"], tot["ext_kp"]),
                pair_recall=r(tot["pairs_hit"], tot["pairs_n"]), pair_precision=r(tot["pairs_ok"], tot["pairs_kp"]), counts=tot, thr=a.thr, frames=len(frames), order=not args["no_order"],
                evidence=not args["no_evidence"], maps=args["maps"], state=a.state, scene_starts=int(sum(reset)), gate_mean=float(m.gate().mean()))
    src_dir = os.path.join(ROOT, "runs", a.run)
    out_dir = src_dir + ("__" + a.tag if a.tag else "")
    if a.tag:                                                                                             # the sibling folder of a tagged read-out: what the scorers read of a run
        os.makedirs(out_dir, exist_ok=True)
        for f in ("model.pt", "config.json"):
            shutil.copyfile(os.path.join(src_dir, f), os.path.join(out_dir, f))
    pickle.dump(frames, open(os.path.join(out_dir, "ends_val.pkl"), "wb"))
    json.dump(head, open(os.path.join(out_dir, "ends_head.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in head.items() if k != "counts"}, indent=1))
    print("written", os.path.relpath(out_dir, ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dump")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--state", required=True, choices=STATES)
    s.add_argument("--tag", default=None, help="another read-out than the delivered mode of the run: written to runs/<run>__<tag>/")
    s.add_argument("--thr", type=float, default=0.5)
    s.add_argument("--max", type=int, default=0, help="only the first N frames of the stream (a test of the pipeline; the score needs all 873)")
    dump(ap.parse_args())
