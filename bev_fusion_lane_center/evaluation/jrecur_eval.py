"""Read-out of the pair stage with the recursive feature state (E354): the dump of `jends_eval.py dump` in STREAM order (scene by scene, time order, batch 1, the state carried), so that the stored scorers read it unchanged.

    wsl bash tools/env/_py14.sh jrecur_eval.py dump --run e354_r_s0 --seed 0 --state stream                 # the delivered mode of arm r -> runs/e354_r_s0/ends_val.pkl, ends_head.json
    wsl bash tools/env/_py14.sh jrecur_eval.py dump --run e354_r_s0 --seed 0 --state off|shuffle|trunc3     # another read-out of the same checkpoint -> its OWN run folder runs/e354_r_s0__off | __shuffle | __trunc3
    wsl bash tools/env/_py14.sh jrecur_eval.py dump --run e354_z_s0 --seed 0 --state off                    # arm z (trained without a state): off is its only mode -> runs/e354_z_s0/
    wsl bash tools/env/_py14.sh jrecur_eval.py dump --run e341_p_s0 --seed 0 --state off                    # a pair run from before E354 through THIS driver (zero gate) -> runs/e341_p_s0__e354drv/: must reproduce the stored cards of the run
    wsl bash tools/env/_py14.sh jends_eval.py score --run e354_r_s0__shuffle --seed 0 --thr 0.1 [--snap]    # the stored scorers, unchanged: a read-out is a run folder

Frames: ONLY the rows of data/splits.json['val'] (poses_val.npz and the val feature files also hold the TEST rows: never iterated), 873 frames in 22 scenes (asserted). States: `stream` the state of the keyframe before, from the first
to the last keyframe of a scene, empty at a scene start (`lane_seq_data.stream_order`); `off` never a state; `trunc3` empty at every third keyframe of a scene (the training distribution of `--seq 3`); `shuffle` a real state of the
wrong place: frame n of a scene reads the STREAM state that frame n of the NEXT scene of the list reads (cyclic; the state after that scene's frame (n - 1) modulo its length), warped with the frame's OWN relative pose; the donor
states come from a first stream pass (float16 on the CPU, about 0.3 GB).
A read-out other than the delivered mode of the run gets a sibling run folder with a COPY of model.pt and config.json: `jends_eval.py score`, `jtime.py`, `jmaneuver.py`, `jpair_pr.py` take runs/<run>/ends_val.pkl, model.pt (args.maps,
args.seed, the pyramid tensors) and config.json by the run name only. A run trained with `--state off` is read with `off` only (its state weights are untrained). ends_head.json also carries the mode, the mean |gate| and, per frame
index in the scene, the RMS of the state and the share of the fused features that the state causes (`StateFuse.probe`): drift shows there. Tests: training/grid/test_jrecur.py
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

STATES = ("stream", "off", "shuffle", "trunc3", "trunc")                                                  # trunc = empty at every K-th keyframe with K = the run's --seq (the training distribution; trunc3 = the fixed period 3)


def target(run, trained, mode):
    """the run folder name of a read-out; trained = args['state'] of the run (on / off; None = a pair run from before E354)"""
    assert mode in STATES, mode
    assert trained == "on" or mode == "off", f"{run} was trained without a state ({trained}): its state weights are untrained, only --state off reads it"
    if trained is None:
        return run + "__e354drv"
    return run if mode == ("stream" if trained == "on" else "off") else run + "__" + mode


def state_plan(reset, mode, period=3):
    """reset (n,) bool of the stream (`stream_order`) -> per frame the stream index of the frame whose STATE it reads, -1 = none. stream / trunc3 / trunc read the frame before (index - 1), shuffle a frame of the next scene;
    period = the reset period of `trunc` (the --seq of the run); trunc3 always 3"""
    period = 3 if mode == "trunc3" else int(period)
    assert period >= 2, period
    assert mode in STATES, mode
    starts = [i for i, r in enumerate(reset) if r] + [len(reset)]
    segs = list(zip(starts[:-1], starts[1:]))
    src = []
    for si, (s0, s1) in enumerate(segs):
        d0, d1 = segs[(si + 1) % len(segs)]
        for n in range(s1 - s0):
            if n == 0 or mode == "off" or (mode in ("trunc3", "trunc") and n % period == 0):
                src.append(-1)
            elif mode == "shuffle":
                src.append(d0 + (n - 1) % (d1 - d0))
            else:
                src.append(s0 + n - 1)
    return src


def val_stream():
    """-> (the VAL data set, the VAL poses, [(row, reset)] the rows of data/splits.json['val'] in stream order); 873 frames in 22 scenes, every one with stored lane GT"""
    rows = [int(r) for r in json.load(open(os.path.join(ROOT, "data", "splits.json")))["val"]]
    ds = LD.QData("val_clean", map_mode="error", pixels=False, junction=True)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    assert sorted(int(ds.bd.rows[k]) for k in ds.ks) == sorted(rows), "the rows of QData('val_clean') are not the rows of data/splits.json['val']"
    pose = dict(np.load(f"{PD.D}/poses_val.npz"))
    order = LS.stream_order(pose, rows)
    assert len(order) == 873 and sum(r for _, r in order) == 22 and len({int(pose["scene"][r]) for r, _ in order}) == 22, (len(order), sum(r for _, r in order))
    return ds, pose, order


def run_stream(m, ds, order, src, pose, dev, donor=None, keep=False, thr=0.5, workers=2):
    """the frames of `order` one by one. Frame i reads the state of frame src[i] (the state carried from frame i - 1, or `donor[src[i]]`), warped by the pose of ITS previous keyframe.
    -> (frames: the dicts of `jends_eval.dump`, the counts of `qendnet.end_counts`, per frame the RMS of its state, per frame the state's share of the fused features (None without a state), the states (float16, CPU) when keep).
    keep = a donor pass: the maps head only, no frames"""
    i_of_row = {int(ds.bd.rows[k]): i for i, k in enumerate(ds.ks)}
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [i_of_row[r] for r, _ in order]), batch_size=1, shuffle=False, num_workers=workers, collate_fn=LD.collate)
    fuse = m.jmodel.jm.fuse
    frames, tot, rms, share, states, last = [], {}, [], [], [], None
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
            fuse.probe, fuse.contrib = True, []
            if keep:
                _, jout = m.jmodel(x, state=state, rel=rel)
                states.append(jout["state"].half().cpu())
            else:
                _, jout, out = m(x, gt=None, state=state, rel=rel)
                for k, v in QEN.end_counts(out, b, thr).items():
                    tot[k] = tot.get(k, 0) + v
                frames.append(dict(token=b["token"][0], row=row, kp_pos=out["kp_pos"][0].cpu().numpy(), kp_dir=out["kp_dir"][0].cpu().numpy(), kp_p=out["kp_p"][0].cpu().numpy(), valid=out["valid"][0].cpu().numpy(),
                                   logit=out["logit"][0].float().cpu().numpy(), allowed=out["allowed"][0].cpu().numpy(), scale=out["scale"][0].float().cpu().numpy() if "scale" in out else None))
            fuse.probe = False
            last = jout["state"]
            rms.append(float(last.pow(2).mean().sqrt()))
            share.append(fuse.contrib[0] if fuse.contrib else None)
    return frames, tot, rms, share, states


def by_index(reset, values):
    """the mean of the per-frame values per frame index in its scene (None values left out) -> list over the index"""
    out, n = {}, 0
    for r, v in zip(reset, values):
        n = 0 if r else n + 1
        if v is not None:
            out.setdefault(n, []).append(v)
    return [float(np.mean(out[k])) if k in out else None for k in range(max(out) + 1)] if out else []


def dump(a):
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU (the stored E341 cards were); --cpu only for tests, the device is written into ends_head.json"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":                                                                                                                        # D100: a dump next to two trainings crashed one of them
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jrecur_eval dump {a.run}")
    m, args = JR.load_model(a.run, dev)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    name = target(a.run, args.get("state"), a.state)                                                      # before anything long runs
    ds, pose, order = val_stream()
    reset = [r for _, r in order]
    donor = run_stream(m, ds, order, state_plan(reset, "stream"), pose, dev, keep=True)[4] if a.state == "shuffle" else None
    period = int(args.get("seq") or 3)
    frames, tot, rms, share, _ = run_stream(m, ds, order, state_plan(reset, a.state, period), pose, dev, donor=donor, thr=a.thr)
    assert len(frames) == 873
    r = lambda x, y: x / max(y, 1)  # noqa: E731
    head = dict(ent_recall=r(tot["ent_hit"], tot["ent_n"]), ent_precision=r(tot["ent_ok"], tot["ent_kp"]), ext_recall=r(tot["ext_hit"], tot["ext_n"]), ext_precision=r(tot["ext_ok"], tot["ext_kp"]),
                pair_recall=r(tot["pairs_hit"], tot["pairs_n"]), pair_precision=r(tot["pairs_ok"], tot["pairs_kp"]), counts=tot, thr=a.thr, frames=len(frames), order=not args["no_order"],
                evidence=not args["no_evidence"], maps=args["maps"], state=a.state, device=dev, trunc_period=(3 if a.state == "trunc3" else period if a.state == "trunc" else None), trained_state=args.get("state"), source_run=a.run, scenes=int(sum(reset)), gate_mean=float(m.gate().mean()),
                state_rms_by_index=by_index(reset, rms), state_contrib_by_index=by_index(reset, share))
    src_dir, out_dir = os.path.join(ROOT, "runs", a.run), os.path.join(ROOT, "runs", name)
    if name != a.run:                                                                                     # the read-out's own run folder: what the scorers read of a run
        os.makedirs(out_dir, exist_ok=True)
        for f in ("model.pt", "config.json"):
            shutil.copyfile(os.path.join(src_dir, f), os.path.join(out_dir, f))
    pickle.dump(frames, open(os.path.join(out_dir, "ends_val.pkl"), "wb"))
    json.dump(head, open(os.path.join(out_dir, "ends_head.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in head.items() if k != "counts"}, indent=1))
    print("written runs/" + name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dump")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--state", required=True, choices=STATES)
    s.add_argument("--thr", type=float, default=0.5)
    s.add_argument("--cpu", action="store_true", help="tests only: dump on the CPU (a card's read-outs are dumped on the GPU)")
    dump(ap.parse_args())
