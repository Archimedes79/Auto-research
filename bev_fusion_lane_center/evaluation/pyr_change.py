"""E393 read-out: how much a stage-2 run moved B''s pyramid - its OUTPUT features on fixed VAL frames and its weights - against the B' it started from.

    wsl bash tools/env/_py14.sh pyr_change.py --run e393_e_s0 --start e372_s4_s0 --seed 0 [--frames 100] [--cpu]
    wsl bash tools/env/_py14.sh pyr_change.py selftest

Both B' come from a jlive run's model.pt (`b_state`, `jlive_train.load_b`): `--run` is the trained arm, `--start` the run whose b_state the arm started from
(`--b-state-from`). The frames: `--frames N` evenly spaced indices of the VAL split in the order of the E365 cache of B' seed `--seed`
(`numpy.linspace(0, n - 1, N).round()`; VAL 873 -> the same 100 frames for every arm of a seed). Both pyramids run in float32, eval mode, no gradient.
Written: runs/<run>/pyr_change_vs_<start>_n<N>.json with, per output of the pyramid (feat2, feat1, feat05 - what the node head reads - and centre1,
road2, lines05), rel = ||F_end - F_start|| / ||F_start|| over all frames (Frobenius norms summed over the frames); `feat_rel` = the same over feat2,
feat1 and feat05 together (E393 section 1.4: the 'in structure only' measure, below 0.01 = the output barely moved); and the weights:
`weight_rel` = ||W_end - W_start|| / ||W_start|| over every pyramid tensor, per tensor in `weights`. A json that exists is never overwritten.
Tests: evaluation/tests/test_pyr_change.py (the math on toy pyramids, a mutant that compares the start with itself, the frame choice).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/train"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))

FEATS = ("feat2", "feat1", "feat05")                     # what the node head reads (E393 section 1.2)
OUTS = FEATS + ("centre1", "road2", "lines05")


def frame_indices(n, k):
    """k evenly spaced indices of n frames (sorted, unique; all n when k >= n)"""
    return sorted(set(int(i) for i in np.linspace(0, n - 1, min(k, n)).round()))


def weight_change(start, end):
    """(rel over all tensors, {name: rel}) of two pyramid state dicts with the same keys; rel = ||end - start|| / ||start||"""
    assert list(start) == list(end), "the two pyramids differ in their tensors"
    per, num, den = {}, 0.0, 0.0
    for k in start:
        a, b = start[k].double(), end[k].double()
        d, s = float((b - a).norm() ** 2), float(a.norm() ** 2)
        per[k] = math.sqrt(d / s) if s > 0 else (0.0 if d == 0 else math.inf)
        num, den = num + d, den + s
    return math.sqrt(num / den) if den > 0 else 0.0, per


def output_change(pyr_start, pyr_end, batches):
    """{output: rel} and feat_rel: the pyramids run on the same input batches (an iterable of tensors); sums of squares over all frames"""
    num, den = {k: 0.0 for k in OUTS}, {k: 0.0 for k in OUTS}
    with torch.no_grad():
        for x in batches:
            a, b = pyr_start(x.float()), pyr_end(x.float())
            for k in OUTS:
                num[k] += float((b[k].double() - a[k].double()).pow(2).sum())
                den[k] += float(a[k].double().pow(2).sum())
    rel = {k: math.sqrt(num[k] / den[k]) if den[k] > 0 else 0.0 for k in OUTS}
    fr = math.sqrt(sum(num[k] for k in FEATS) / sum(den[k] for k in FEATS)) if sum(den[k] for k in FEATS) > 0 else 0.0
    return rel, fr


def measure(run, start, seed, frames=100, device="cuda", bs=4, runs=None):
    import jlive_train as JL
    import lane_live_tok as LLT
    out = os.path.join(runs or JL.T.RUNS, run, f"pyr_change_vs_{start}_n{frames}.json")
    assert not os.path.exists(out), f"{out} exists: a read-out is never overwritten"
    ms, me = JL.load_b(start, device, runs), JL.load_b(run, device, runs)
    wr, per = weight_change({k: v.cpu() for k, v in ms.pyr.state_dict().items()}, {k: v.cpu() for k, v in me.pyr.state_dict().items()})
    ds = LLT.LiveData("val", seed)
    idx = frame_indices(len(ds), frames)
    batches = (torch.stack([ds[i]["x"] for i in idx[j:j + bs]]).to(device) for j in range(0, len(idx), bs))
    rel, fr = output_change(ms.pyr.eval(), me.pyr.eval(), batches)
    res = dict(run=run, start=start, seed=seed, split="val", frames=len(idx), frame_rule=f"numpy.linspace(0, {len(ds) - 1}, {frames}).round() of LiveData('val', {seed})",
               feat_rel=fr, rel=rel, weight_rel=wr, weights=per, device=device, in_structure_only_floor=0.01)
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out + ".tmp", out)
    print(f"{run} vs {start} (VAL, {len(idx)} frames): feat_rel {fr:.5f} ({', '.join(f'{k} {rel[k]:.5f}' for k in FEATS)}), weight_rel {wr:.5f} -> {out}")
    return res


def selftest():
    import pyramid
    torch.manual_seed(0)
    a, b = pyramid.PyramidHead(width=8), pyramid.PyramidHead(width=8)
    b.load_state_dict(a.state_dict())
    xs = [torch.randn(2, a.c_in, 32, 32)]
    rel, fr = output_change(a.eval(), b.eval(), xs)
    assert fr == 0.0 and all(v == 0.0 for v in rel.values()) and weight_change(a.state_dict(), b.state_dict())[0] == 0.0, "the same pyramid moved"
    with torch.no_grad():
        next(iter(b.parameters())).add_(0.05)
    rel, fr = output_change(a.eval(), b.eval(), xs)
    wr, per = weight_change(a.state_dict(), b.state_dict())
    assert fr > 0 and wr > 0 and sum(v > 0 for v in per.values()) == 1, (fr, wr)
    assert frame_indices(873, 100)[:3] == [0, 9, 18] and len(frame_indices(873, 100)) == 100 and frame_indices(5, 100) == [0, 1, 2, 3, 4]
    print("pyr_change selftest ok: no change -> 0, one tensor moved -> output and weight change > 0 in that tensor only, the frame choice")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(prog="pyr_change", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="the trained jlive run (its model.pt b_state)")
    ap.add_argument("--start", required=True, help="the jlive run whose b_state the run started from (--b-state-from)")
    ap.add_argument("--seed", type=int, required=True, help="B''s seed: the VAL frame order of its cache")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    measure(a.run, a.start, a.seed, a.frames, "cpu" if a.cpu or not torch.cuda.is_available() else "cuda", a.bs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
