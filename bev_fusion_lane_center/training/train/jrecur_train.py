"""Trainer of the pair stage with the recursive feature state (E354, D103); start it through `autoresearch/tools/run.py`. The loop of `jends_train.py` (frozen, E343) over SEQUENCES of keyframes.

    wsl bash tools/env/_py14.sh jrecur_train.py --name e354_r_s0 --seed 0 --maps runs/e332_ma_s0/model.pt --e2e --scales --maps-w 1.0 --steps 3000 --seq 3 --bs 4 --state on --workers 4 [--smoke]
    ... --state off                                       # the equal-budget twin z: the same module and sampler, the state always empty

A batch is `--bs` sequences of `--seq` consecutive keyframes of one TRAIN scene (`data/lane_seq_data.py`). Frame 1 runs cold; with `--state on` frame k reads the state of frame k - 1 of its sequence (the 2 m features after the dilated
blocks, `grid/jrecur.py`), warped by the relative ego pose. The loss of E341 (pair loss + `--maps-w` x maps loss) is taken on every frame and SUMMED; one backward pass through all frames, the state is never detached.
Trainable: the pair head and the maps head with the fusion (`--e2e` is required: the fusion lives in the maps head); the pyramid stays frozen. The model starts from the maps run and a zero gate, i.e. as the start of `jends_train.py`.
Output `runs/<name>/` as `trainer.fit` writes it (log.txt / train.log, log.jsonl, status.json, config.json, ckpt.pt every 500 steps with resume, model.pt, .done); model.pt = the state + the args with the names `jends_eval.py score` and
`grid/jrecur.load_model` read (seed, maps, maps_args, no_order, no_evidence, scales, state, seq). Every step line carries the mean |gate| (`gate`), the peak GPU memory (`gpu_gb`) and the loss per frame index (`l_f<k>`); at step 50 and every 100 steps
a line with the seconds per step of the last steps (warm-up left out). The plain maps run `<name>_maps` of `jends_train.py` is not written (the maps head has the fusion: `jmaps.JModel` does not load it).
"""
import argparse
import json
import os
import re
import sys
import time

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jends as JE  # noqa: E402
import jgt as JG  # noqa: E402
import jmaps as JM  # noqa: E402
import jrecur as JR  # noqa: E402
import lane_gt_data as LD  # noqa: E402
import lane_seq_data as LS  # noqa: E402
import trainer as T  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps", required=True, help="the junction maps run, runs/<run>/model.pt (E341: runs/e332_ma_s<seed>/model.pt)")
    ap.add_argument("--e2e", action="store_true", help="required: the maps head (with the fusion) trains with the pair head")
    ap.add_argument("--scales", action="store_true", help="E303: two handle scales per pair")
    ap.add_argument("--maps-w", type=float, default=1.0)
    ap.add_argument("--cross-w", type=float, default=1.0)
    ap.add_argument("--seq", type=int, default=3, help="keyframes per sequence")
    ap.add_argument("--bs", type=int, default=4, help="sequences per batch")
    ap.add_argument("--state", default="on", choices=("on", "off"), help="off: the state is always empty (the control)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="20 steps")
    return ap.parse_args(argv)


def check_start(a, maps_args, folder):
    """before anything is written: the flags, the seed of the maps run, the width of the end Gaussians, and a folder that already holds a run is only RESUMED (the same recipe), never another run's"""
    assert a.e2e, "--e2e is required: the fusion lives in the maps head"
    m = re.search(r"_s(\d+)[/\\]", str(a.maps).replace("\\", "/") + "/")
    assert m is None or int(m.group(1)) == a.seed, f"--seed {a.seed} does not match the maps run {a.maps}: `jends_eval.py score` would refuse the result"
    want, sig = maps_args.get("sigma"), float(os.environ.get("BFLC_SIG", JG.SIG))
    assert want is None or abs(sig - float(want)) < 1e-9, f"BFLC_SIG={sig} but the maps run was trained with sigma {want}: export BFLC_SIG={want}"
    p = os.path.join(folder, "config.json")
    if os.path.exists(p):
        c = json.load(open(p))
        old = {k: (c.get("args") or {}).get(k) for k in ("seed", "state", "seq", "bs")}
        assert c.get("design") == "jrecur" and old == {k: getattr(a, k) for k in old}, f"{folder} holds another run ({c.get('design')}, {old}): use another --name"


def set_trainable(model):
    """the pair head and the maps head (fusion included) train, the pyramid is frozen -> the parameters that train"""
    ps = list(model.je.parameters()) + list(model.jmodel.jm.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    return ps


def seq_loss(m, b, a):
    """the E341 loss of every frame of the sequences, summed; frame k reads the state of frame k - 1 (never detached) when a.state is on -> (loss, info: the means over the frames, the loss per frame index, the gate)"""
    dev = next(m.parameters()).device
    state, total, info, K = None, 0.0, {}, len(b["frames"])
    for k, fb in enumerate(b["frames"]):
        x = fb["x"].to(dev, non_blocking=True).float()
        gt = {n: fb[n].to(dev) for n in ("ent", "ext", "ent_m", "ext_m")}
        rel = None if state is None else (b["rel_R"][:, k - 1].to(dev), b["rel_t"][:, k - 1].to(dev))
        _, jout, out = m(x, gt=gt, state=state, rel=rel)
        loss, inf = JE.jends_loss(out, fb, cross_w=a.cross_w)
        lm, im = JM.jmaps_loss(jout, fb)
        loss = loss + a.maps_w * lm
        inf.update({"j_" + n: v for n, v in im.items()})
        for n, v in inf.items():
            info[n] = info.get(n, 0.0) + float(v) / K
        info[f"l_f{k + 1}"] = float(loss)
        total = total + loss
        state = jout["state"] if a.state == "on" else None
    info["gate"] = float(m.gate().mean())
    return total, info


def make_loss(a, run):
    """`seq_loss` + the step clock: at call 50 and every 100 calls a line with the seconds per step since the last line (the first 10 calls of the process are warm-up) and the peak GPU memory"""
    n, mark = [0], [None, 0]

    def loss_fn(m, b):
        loss, info = seq_loss(m, b, a)
        info["gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        n[0] += 1
        if n[0] == 10:
            mark[:] = [time.time(), 10]
        if n[0] == 50 or n[0] % 100 == 0:
            g = m.gate()
            run.log(f"clock: {(time.time() - mark[0]) / (n[0] - mark[1]):.3f} s/step over the last {n[0] - mark[1]} steps of this process, peak GPU {info['gpu_gb']:.2f} GB, gate mean |g| {float(g.mean()):.4f} max {float(g.max()):.4f}")
            mark[:] = [time.time(), n[0]]
        return loss, info
    return loss_fn


def main(argv=None):
    a = parse_args(argv)
    T.seed_all(a.seed)
    mp = a.maps if os.path.isabs(a.maps) else os.path.join(ROOT, a.maps)
    ck = torch.load(mp, map_location="cpu")
    run = T.Run(a.name)
    check_start(a, ck["args"], run.dir)
    LD.QData.DEFAULT_JUNCTION = True
    LD.QData.DEFAULT_PSCALE = a.scales                                                                   # the fitted handle scales cost 0.15 s per frame: only for the scale head
    model = JR.JRecurModel(ck["args"], scales=a.scales)
    miss, extra = model.jmodel.load_state_dict(ck["state"], strict=False)
    assert not extra and all(k.startswith("jm.fuse.") for k in miss), (miss, extra)
    steps = 20 if a.smoke else a.steps
    ps = set_trainable(model)
    ds, dl = LS.loader("train", K=a.seq, bs=a.bs, workers=a.workers, shuffle=True, map_mode="train", pixels=False)
    args = dict(vars(a), maps=mp, maps_args=ck["args"], no_order=False, no_evidence=False, steps=steps)
    run.log(f"jrecur state={a.state} seq={a.seq} bs={a.bs} ({a.seq * a.bs} frames per step): {steps} steps, lr {a.lr}, trainable {sum(p.numel() for p in ps) / 1e6:.3f} M (fusion {sum(p.numel() for p in model.jmodel.jm.fuse.parameters()) / 1e3:.1f} k), "
            f"maps {mp}, train sequences {len(ds)} of {len(ds.ds)} frames (missing lane GT: {ds.ds.missing})")
    T.fit(model, make_loss(a, run), run, steps=steps, loader=dl, lr=a.lr, ckpt_every=500, seed=a.seed, args=args, params=ps, device="cuda" if torch.cuda.is_available() else "cpu",
          config=dict(design="jrecur", order=True, evidence=True, e2e=True, scales=a.scales, state=a.state, seq=a.seq, maps=mp))
    run.log(f"done {a.name}: gate mean |g| {float(model.gate().mean()):.4f}")


if __name__ == "__main__":
    main()
