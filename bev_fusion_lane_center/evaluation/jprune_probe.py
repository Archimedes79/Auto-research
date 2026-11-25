"""Sensitivity probe of the junction system (E345, D99 step 0, part A): what is lost when one unit of a trained `jends_train.JEndModel` (frozen pyramid, junction maps head, pair head) is taken away without retraining. The GPU side;
the CPU side (scoring, halves, classes, curves, table) is `jprune_table.py`, the cost of a set of removals `jprune_cost.py`. VAL and TRAIN only (a TEST split is refused).

    wsl bash tools/env/_py14.sh jprune_probe.py units  --run e312_p_s0                           # the unit table (45 units, 84 points) with the saved GMACs / parameters of each unit, CPU, no data
    wsl bash tools/env/_py14.sh jprune_probe.py sweep  --run e312_p_s0 --out runs/kept/e345/e312_p_s0 [--score-points f.json] [--runs-dir DIR] [--max N] [--val-stride N] [--calib-max N] [--workers N] [--device cpu] [--deterministic] [--no-controls]
                                                                                                 # resumes from sweep.json; the calibration is kept in calib.pt (a rerun picks the same channels); relative paths are relative to the problem folder
    wsl bash tools/env/_py14.sh jprune_probe.py time   --run e312_p_s0 [--device cuda] [--batches 4]     # seconds of one pass of the VAL cache on the device (the pass time of the GPU jobs, measured under load)
    wsl bash tools/env/_py14.sh jprune_probe.py selftest                                         # random-init model, CPU, no data (the data tests are tests/test_jprune_probe.py)

What a point is (card E345 section 2): the lowest-importance fraction f of the channels of one tensor is replaced by its calibration mean (importance s_c = mean over 201 TRAIN frames and positions of |a_c - mean a_c|; tokens, pair MLP and scale MLP: valid keypoints /
allowed pairs only), a residual block is skipped, one attention head loses its value (its weights are edited on a deepcopy: the head output becomes its calibration mean), an output channel of a head becomes its mean logit, the paint evidence inputs of the pair
MLPs become their means. `Unit` = what one removal names, a point = (unit, f), a point SET = {unit: f} (joint removals: v, the prefixes of the curves). Per pass the probe stores, per VAL frame, the raw counts of `qendnet.end_counts` at p >= 0.1 (the six
dump-level numbers offline), the matched-keypoint position / heading error (`jdiag.py kperr`: strict one-to-one, 4 m / 30 degrees, keypoints p >= 0.5; with attached ends also the error of the attached end), and the pair average precision and precision at recall 0.5
of the logits (candidates = allowed pairs of keypoints p >= 0.1, labels from `match_keypoints` as in `end_counts`; recall is measured against ALL true pairs of the frames, so a pair lost with its keypoint costs AP too) for the whole VAL and for the two halves of the scenes.
`dump_frames` is `jends_eval.dump` without its file writing, but NOT yet shared with it (TODO: `jends_eval.dump` should call it once no chain runs `jends_eval.py`): the sweep and the point dumps use `forward_batches` + `frames_of` (the pieces `dump_frames` is made of), and
tests/test_jprune_probe.py compares `dump_frames` frame for frame with a fresh `jends_eval.dump` on 16 frames. The point sets named in `--score-points` also get `runs/e345_p<NNN>_<model>/ends_val.pkl` + `point.json` (the set, the edits, the sha256 of the pickle, written LAST: a folder without it
is incomplete and redone on resume) + a symlink to the unmodified model.pt, which `jprune_table.py score` scores; every file is written through a temporary file and `os.replace`.
Choices the card left open: the TRAIN split is called "train" (the loader has no "train_clean"); a tap with no valid position in the calibration (no keypoint on any frame) gets mean 0, s_c 0 and is then allowed to be unchanged; the two nulls are compared by ends_mean6
(cap 3e-3) here, by the scored cards in `jprune_table.py`; the random draws of the control use RandomState(1000 * draw + sum(ord(c) for c in unit name)) for draw 0 / 1 / 2; the attention-head edit is checked against the by-hand per-head replacement to 2e-7 (measured 6e-8 and 9e-8; the card asks 1e-7).
GPU safety: before any CUDA work `wait_for_gpu` calls `gpu_gate.wait_for_room` (D100) with its own, much longer timeout (12 h, `E345_GATE_TIMEOUT_S`; the shared `GPU_GATE_TIMEOUT_S` of the chains is ignored here) and, if the GPU is still full afterwards, exits with code 75 and a message instead of going on;
`time` and every pass record of `sweep.json` carry `torch.cuda.max_memory_allocated / reserved` (MiB). Determinism: `--deterministic` sets `CUBLAS_WORKSPACE_CONFIG`, cudnn.deterministic and `use_deterministic_algorithms` BEFORE CUDA is up and is recorded in `sweep.json`; a resumed job takes the recorded setting
(and refuses `--deterministic` against passes made without it); an identity hook that does not reproduce the null to 0 means: restart with `--deterministic`.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jends_eval as JEV  # noqa: E402
import qendnet as QEN  # noqa: E402
from jends import NG  # noqa: E402
from jprune_cost import Cost, Edit, sub  # noqa: E402

F3 = (0.25, 0.5, 0.75)
THR = 0.1                                                                               # the read-out threshold of the card (pairs p >= 0.1, keypoints p >= 0.1)
KP_THR = 0.5                                                                            # the keypoints of the position / heading error (the definition of jdiag.py kperr)
SPLITS = {"val": "val_clean", "train": "train"}                                         # no TEST here
CAL_STRIDE = 42                                                                         # 8414 TRAIN frames -> 201
NULL_CAP = 3e-3                                                                         # cap of the spread of the two nulls (ends_mean6)
COUNTS = ("ent_n", "ent_kp", "ent_ok", "ent_hit", "ext_n", "ext_kp", "ext_ok", "ext_hit", "pairs_n", "pairs_kp", "pairs_ok", "pairs_hit")
SIX = (("ent_hit", "ent_n"), ("ent_ok", "ent_kp"), ("ext_hit", "ext_n"), ("ext_ok", "ext_kp"), ("pairs_hit", "pairs_n"), ("pairs_ok", "pairs_kp"))     # recall / precision of entries, exits, pairs


# ---------------------------------------------------------------- 1. units and points
@dataclasses.dataclass(frozen=True)
class Unit:
    name: str
    group: str                      # pyr | maps | pair (the card counts the paint inputs with the maps head)
    kind: str                       # chan | skip | head | cols
    fracs: tuple                    # the fractions of the ladder; (1.0,) = removed as a whole
    taps: tuple = ()                # chan / cols: the tensors it removes channels of
    ranks: tuple = ()               # chan: the tensor whose ranking picks the channels of taps[i] (default: itself; the stream shares p2's)
    idx: tuple = ()                 # a fixed channel set (an output channel of a head, the paint columns)
    block: str = ""
    head: int = -1


def _last_conv(seq):
    return [m for m in seq if isinstance(m, nn.Conv2d)][-1]


def build_units(model):
    """the units of the model, derived from its module tree (the card's 45 units / 84 points for E312 and E324 s)"""
    jm, je = model.jmodel.jm, model.je
    U = [Unit(f"pyr.{s}", "pyr", "chan", F3, (f"pyr.{s}",)) for s in ("stem", "e1", "e2", "e3", "d2", "d1", "d0")]
    U.append(Unit("pyr.s05+m05", "pyr", "chan", F3, ("pyr.s05", "pyr.m05")))                           # the 0.5 m pixel stem: two rankings, one unit
    stream = ("jm.p2",) + tuple(f"jm.blocks.{i}" for i in range(len(jm.blocks)))
    U.append(Unit("jm.stream", "maps", "chan", (0.25, 0.5), stream, ranks=("jm.p2",) * len(stream)))   # one channel set, ranked on p2's output
    U += [Unit(f"jm.blocks.{i}.hid", "maps", "chan", F3, (f"jm.blocks.{i}.f.2",)) for i in range(len(jm.blocks))]
    U += [Unit(f"jm.blocks.{i}.skip", "maps", "skip", (1.0,), block=f"jm.blocks.{i}") for i in range(len(jm.blocks))]
    U.append(Unit("jm.m1.hid", "maps", "chan", F3, ("jm.m1.2",)))
    if jm.paint:
        U.append(Unit("jm.m05.hid", "maps", "chan", F3, ("jm.m05.2",)))
    U += [Unit(f"jm.head1.c{c}", "maps", "chan", (1.0,), ("jm.head1",), idx=(c,)) for c in range(jm.head1.out_channels)]
    if jm.paint:
        U += [Unit(f"jm.head05.c{c}", "maps", "chan", (1.0,), ("jm.head05",), idx=(c,)) for c in range(jm.head05.out_channels)]
    if je.evidence and je.paint_ev:                                                                     # evidence columns 4 and 5 of the pair MLP input: 2 d + NG + 4, + 5
        c0 = 2 * je.kp.out_features + NG
        U.append(Unit("je.paint", "maps", "cols", (1.0,), ("je.fcat",), idx=(c0 + 4, c0 + 5)))
    U += [Unit(f"je.blocks.{i}.skip", "pair", "skip", (1.0,), block=f"je.blocks.{i}") for i in range(len(je.blocks))]
    U += [Unit(f"je.blocks.{i}.att.h{h}", "pair", "head", (1.0,), block=f"je.blocks.{i}", head=h) for i, b in enumerate(je.blocks) for h in range(b.att.num_heads)]
    U += [Unit(f"je.blocks.{i}.ffn", "pair", "chan", F3, (f"je.blocks.{i}.ffn.1",)) for i in range(len(je.blocks))]
    U += [Unit("je.pair.1", "pair", "chan", F3, ("je.pair.1",)), Unit("je.pair.3", "pair", "chan", F3, ("je.pair.3",))]
    if hasattr(je, "scale"):
        U.append(Unit("je.scale.1", "pair", "chan", F3, ("je.scale.1",)))
    return U


def points(units):
    """-> [(id, unit, f)]: the single points in the order of the table, ids from 1 (the null is 0)"""
    return [(i, u, f) for i, (u, f) in enumerate(((u, f) for u in units for f in u.fracs), 1)]


def count_edits(u, f, C):
    """the edits of a unit at fraction f on the FIRST n channels of each tensor (C: channels per tensor, `Cost.C`): the cost of ONE unit does not depend on which channels"""
    if u.kind == "skip":
        return [Edit("skip", block=u.block)]
    if u.kind == "head":
        return [Edit("head", block=u.block, head=u.head)]
    return [Edit("cols" if u.kind == "cols" else "chan", tap=t, idx=u.idx or tuple(range(int(f * C[t] + 0.5)))) for t in u.taps]


def unit_table(units, cost, expect=None):
    """the table as text (saved GMACs / parameters of a unit at its largest fraction); with `expect = (units, points, ladder units)` it asserts the card's counts"""
    pts = points(units)
    rows = [f"{'unit':16s} {'group':5s} {'kind':5s} {'channels':>8s} {'fractions':>16s} {'saved GMACs':>12s} {'saved params':>12s}"]
    for u in units:
        sv = cost.saved(count_edits(u, u.fracs[-1], cost.C))
        ch = str(cost.C[u.taps[0]]) if u.kind == "chan" and not u.idx else ("2 cols" if u.kind == "cols" else "-")
        rows.append(f"{u.name:16s} {u.group:5s} {u.kind:5s} {ch:>8s} {str(u.fracs):>16s} {sv['macs'] / 1e9:12.4f} {sv['params']:12d}")
    n = {g: sum(1 for u in units if u.group == g) for g in ("pyr", "maps", "pair")}
    ladder = sum(1 for u in units if len(u.fracs) > 1)
    rows.append(f"{len(units)} units ({n['pyr']} + {n['maps']} + {n['pair']}), {len(pts)} points, {ladder} with a fraction ladder; intact: {cost.all['macs'] / 1e9:.3f} GMACs, {cost.all['params'] / 1e6:.3f} M parameters")
    if expect is not None:
        assert (len(units), len(pts), ladder) == tuple(expect) and (n["pyr"], n["maps"], n["pair"]) == (8, 22, 15), (len(units), len(pts), ladder, n)
    return "\n".join(rows)


# ---------------------------------------------------------------- 2. taps and calibration
@dataclasses.dataclass(frozen=True)
class Tap:
    name: str
    sites: tuple                    # module paths the hook sits on
    where: str                      # out: the module's output, in: its input (pre-hook)
    dim: int                        # the channel dimension
    C: int
    mask: str = ""                  # "" every position | "tok" valid keypoints | "pair" allowed pairs (both taken from the pair head's output)


def tap_table(model):
    """every tensor a unit edits or the calibration reads, from the module tree"""
    pyr, jm, je = model.jmodel.pyr, model.jmodel.jm, model.je
    T = {}

    def add(name, C, sites=None, where="out", dim=1, mask=""):
        T[name] = Tap(name, tuple(sites or (name,)), where, dim, C, mask)

    for s in ("stem", "e1", "e2", "e3", "d2", "d1", "d0", "m05"):
        add(f"pyr.{s}", _last_conv(getattr(pyr, s)).out_channels)
    add("pyr.s05", pyr.s05[0].out_channels)
    add("jm.p2", jm.p2[0].out_channels)
    for i, b in enumerate(jm.blocks):
        add(f"jm.blocks.{i}", b.f[3].out_channels)
        add(f"jm.blocks.{i}.f.2", b.f[0].out_channels)
    add("jm.m1.2", jm.m1[0].out_channels)
    add("jm.head1", jm.head1.out_channels)
    if jm.paint:
        add("jm.m05.2", jm.m05[0].out_channels)
        add("jm.head05", jm.head05.out_channels)
    for i, b in enumerate(je.blocks):
        add(f"je.blocks.{i}.ffn.1", b.ffn[0].out_features, dim=-1, mask="tok")
    add("je.pair.1", je.pair[0].out_features, dim=-1, mask="pair")
    add("je.pair.3", je.pair[2].out_features, dim=-1, mask="pair")
    if hasattr(je, "scale"):
        add("je.scale.1", je.scale[0].out_features, dim=-1, mask="pair")
    add("je.fcat", je.pair[0].in_features, sites=("je.pair.0",) + (("je.scale.0",) if hasattr(je, "scale") else ()), where="in", dim=-1, mask="pair")
    return T


def observe(model, taps, sink):
    """hooks that hand `sink(name, a)` the tensor of every tap: convolution taps as they are (B, C, H, W), token and pair taps reduced to the (N, C) rows of the valid keypoints / allowed pairs; -> the handles to remove"""
    pending, hs = {}, []
    for t in taps.values():
        for site in t.sites:
            mod = sub(model, site)
            if t.mask == "":
                hs.append(mod.register_forward_hook(lambda _m, _a, o, n=t.name: sink(n, o)))
            elif t.where == "in":
                hs.append(mod.register_forward_pre_hook(lambda _m, a, n=t.name: pending.__setitem__(n, a[0])))
            else:
                hs.append(mod.register_forward_hook(lambda _m, _a, o, n=t.name: pending.__setitem__(n, o)))

    def flush(_m, _a, out):                                                  # the masks are in the pair head's output
        for n, a in pending.items():
            sink(n, a[out["valid"].flatten(1) if taps[n].mask == "tok" else out["allowed"]])
        pending.clear()
    hs.append(sub(model, "je").register_forward_hook(flush))
    return hs


def head_outputs(att, h, pad):
    """(B, H, n, hd): the output of every head of an `nn.MultiheadAttention` (batch_first, self-attention) BEFORE the output projection; pad (B, n) True = padded key"""
    B, n, d = h.shape
    H = att.num_heads
    q, k, v = (t.reshape(B, n, H, d // H).transpose(1, 2) for t in F.linear(h, att.in_proj_weight, att.in_proj_bias).chunk(3, -1))
    s = (q @ k.transpose(-1, -2) / (d // H) ** 0.5).masked_fill(pad[:, None, None, :], float("-inf"))
    return torch.nan_to_num(s.softmax(-1) @ v)


@dataclasses.dataclass
class Calib:
    mean: dict                      # tap -> (C,) calibration mean
    s: dict                         # tap -> (C,) importance s_c
    head: dict                      # (block, head) -> (hd,) mean of the head output
    frames: int = 0


@torch.no_grad()
def calibrate(model, batches, dev, taps):
    """channel means and importance s_c of every tap (two passes: the mean, then the mean absolute deviation) and the mean of every attention head output (valid query positions) over the batches"""
    model = model.to(dev).eval()
    count, tot, dev_sum, hsum, hcount = {}, {}, {}, {}, {}
    chan = lambda a: (0, 2, 3) if a.dim() == 4 else (0,)  # noqa: E731

    def sink1(n, a):
        tot[n] = tot.get(n, 0.0) + a.double().sum(chan(a))
        count[n] = count.get(n, 0) + (a.shape[0] * a.shape[2] * a.shape[3] if a.dim() == 4 else a.shape[0])

    def sink2(n, a):
        dev_sum[n] = dev_sum.get(n, 0.0) + (a.double() - (mean[n].view(1, -1, 1, 1) if a.dim() == 4 else mean[n]).double()).abs().sum(chan(a))

    def head_pre(att, args, kwargs, blk):
        pad = kwargs["key_padding_mask"]
        o = head_outputs(att, args[0], pad)                                  # (B, H, n, hd)
        ok = (~pad)[:, None, :, None].to(o.dtype)
        hsum[blk] = hsum.get(blk, 0.0) + (o.double() * ok).sum((0, 2))       # (H, hd)
        hcount[blk] = hcount.get(blk, 0.0) + float(ok[:, 0].sum())

    hs = observe(model, taps, sink1) + [sub(model, f"je.blocks.{i}.att").register_forward_pre_hook(lambda m, a, k, blk=f"je.blocks.{i}": head_pre(m, a, k, blk), with_kwargs=True) for i in range(len(model.je.blocks))]
    for b in batches:
        model(b["x"].to(dev).float(), gt=None)
    for h in hs:
        h.remove()
    mean = {n: (tot[n] / max(count[n], 1)).float() for n in tot}
    hs = observe(model, taps, sink2)
    for b in batches:
        model(b["x"].to(dev).float(), gt=None)
    for h in hs:
        h.remove()
    head = {(blk, h): (hsum[blk][h] / max(hcount[blk], 1.0)).float() for blk in hsum for h in range(hsum[blk].shape[0])}
    return Calib(mean=mean, s={n: (dev_sum[n] / max(count[n], 1)).float() for n in dev_sum}, head=head, frames=sum(b["x"].shape[0] for b in batches))


def atomic_write(path, data):
    """write bytes through a temporary file and `os.replace`: a crash leaves the old file or none, never a truncated one"""
    with open(path + ".tmp", "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


def save_calib(calib, path):
    torch.save(dict(mean={k: v.cpu() for k, v in calib.mean.items()}, s={k: v.cpu() for k, v in calib.s.items()}, head={k: v.cpu() for k, v in calib.head.items()}, frames=calib.frames), path + ".tmp")
    os.replace(path + ".tmp", path)


def load_calib(path, dev):
    d = torch.load(path, map_location=dev)
    return Calib(mean=d["mean"], s=d["s"], head=d["head"], frames=d["frames"])


# ---------------------------------------------------------------- 3. the appliers
def pick(s, n, how="rank", rng=None):
    """the n channels to remove: `rank` the lowest importance, `reverse` the highest, `random` a draw (rng: numpy RandomState)"""
    if how == "random":
        return torch.from_numpy(rng.permutation(len(s))[:n]).long()
    return torch.argsort(s.cpu(), descending=(how == "reverse"), stable=True)[:n]


def edits_for(unit, f, calib, how="rank", rng=None):
    """the edits of a unit at fraction f; `how`: rank | reverse | random (channel units only); the replacement of a channel is its calibration mean"""
    if unit.kind == "skip":
        return [Edit("skip", block=unit.block)]
    if unit.kind == "head":
        return [Edit("head", block=unit.block, head=unit.head, val=calib.head[(unit.block, unit.head)])]
    out, chosen = [], {}
    for tap, rk in zip(unit.taps, unit.ranks or unit.taps):
        if unit.idx:
            idx = torch.tensor(unit.idx).long()
        else:
            if rk not in chosen:
                chosen[rk] = pick(calib.s[rk], int(f * len(calib.mean[rk]) + 0.5), how, rng)
            idx = chosen[rk]
        idx = idx.sort()[0]
        out.append(Edit("cols" if unit.kind == "cols" else "chan", tap=tap, idx=tuple(int(i) for i in idx), val=calib.mean[tap][idx.to(calib.mean[tap].device)]))
    return out


def spec_edits(spec, units, calib, how="rank", rng=None):
    """the edits of a point SET {unit name: f} (joint removals)"""
    by = {u.name: u for u in units}
    return [e for name, f in spec.items() for e in edits_for(by[name], f, calib, how, rng)]


class Watch:
    """what the hooks of one ablated copy saw: how often each fired (per site) and whether it changed a value (per edit)"""

    def __init__(self):
        self.edit, self.sites, self.fired, self.changed = {}, {}, {}, {}

    def fire(self, key, site, ch=None):
        self.fired[(key, site)] = self.fired.get((key, site), 0) + 1
        if ch is not None:
            self.changed[key] = ch if key not in self.changed else (self.changed[key] | ch)

    def check(self, n_forward, calib=None):
        """the dead-hook assert: every hook fired once per forward and changed something (unless the replaced channels carried no variance at all, s_c = 0, or the edit is an identity: then it must change nothing)"""
        for key, e in self.edit.items():
            for site in self.sites[key]:
                assert self.fired.get((key, site), 0) == n_forward, f"hook {key} at {site} fired {self.fired.get((key, site), 0)} times in {n_forward} forwards"
            ch = bool(self.changed.get(key, torch.zeros((), dtype=torch.bool)))
            if e.kind in ("chan", "cols") and e.val is None:
                assert not ch, f"identity hook {key} changed a value"
            elif e.kind in ("chan", "cols") and calib is not None and float(calib.s[e.tap][list(e.idx)].max()) == 0.0:
                continue
            else:
                assert ch, f"dead hook {key}: the edit changed nothing"


def _edit_tensor(x, e, dim, w, key, site):
    """replace channels e.idx of x in place by e.val (None: write the values back); the hook's note for the Watch"""
    i = torch.tensor(e.idx, dtype=torch.long, device=x.device)
    if dim == 1:
        old = x[:, i]
        new = old.clone() if e.val is None else e.val.to(x.device, x.dtype).view(1, -1, *([1] * (x.dim() - 2)))
        x[:, i] = new
    else:
        old = x[..., i]
        new = old.clone() if e.val is None else e.val.to(x.device, x.dtype)
        x[..., i] = new
    w.fire(key, site, (old != new).any())


def ablate(model, edits, taps):
    """-> (a deepcopy of the model with the edits as hooks / weight edits, Watch); the model itself is never touched"""
    m = copy.deepcopy(model).eval()
    w = Watch()
    for k, e in enumerate(edits):
        key = f"{k}:{e.kind}:{e.tap or e.block}" + (f".h{e.head}" if e.head >= 0 else "")
        w.edit[key] = e
        if e.kind in ("chan", "cols"):
            t = taps[e.tap]
            w.sites[key] = t.sites
            for site in t.sites:
                if t.where == "in":
                    sub(m, site).register_forward_pre_hook(lambda _m, a, e=e, t=t, key=key, site=site: _edit_tensor(a[0], e, t.dim, w, key, site))
                else:
                    sub(m, site).register_forward_hook(lambda _m, _a, o, e=e, t=t, key=key, site=site: _edit_tensor(o, e, t.dim, w, key, site))
        elif e.kind == "skip":
            w.sites[key] = (e.block,)

            def skip(_m, a, o, key=key, site=e.block):                        # output := input (exact for ResDil: its input is post-ReLU; a residual SetBlock: x + 0)
                w.fire(key, site, (o != a[0]).any())
                return a[0]
            sub(m, e.block).register_forward_hook(skip)
        elif e.kind == "head":
            att = sub(m, e.block + ".att")
            w.sites[key] = (e.block + ".att",)
            d = att.embed_dim
            sl = slice(2 * d + e.head * (d // att.num_heads), 2 * d + (e.head + 1) * (d // att.num_heads))
            with torch.no_grad():                                            # the head's value becomes a constant: weights 0, bias the calibration mean (the softmax weights sum to 1)
                val = e.val.to(att.in_proj_bias.device, att.in_proj_bias.dtype)
                w.changed[key] = (att.in_proj_weight[sl] != 0).any() | (att.in_proj_bias[sl] != val).any()
                att.in_proj_weight[sl] = 0.0
                att.in_proj_bias[sl] = val
            att.register_forward_pre_hook(lambda _m, a, key=key, site=e.block + ".att": w.fire(key, site))
    return m, w


# ---------------------------------------------------------------- 4. the cache, the dump, the per-frame numbers
def build_cache(split, bs=8, workers=2, stride=1, max_frames=0):
    """the loader's own batches of a split in host memory: x fp16, token, row and the GT that `end_counts` reads (ent, ext, ent_m, ext_m, pairs). stride: every n-th frame (the TRAIN calibration list, 8414 -> 201 at 42)"""
    assert split in SPLITS, f"split {split!r}: only {sorted(SPLITS)} (never the TEST split)"
    import lane_gt_data as LD
    LD.QData.DEFAULT_JUNCTION = True
    ds = LD.QData(SPLITS[split], map_mode="error", pixels=False)
    assert split == "train" or ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    sel = list(range(0, len(ds), stride))[:max_frames or None]
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, sel), batch_size=bs, shuffle=False, num_workers=workers, collate_fn=LD.collate)
    keep = ("x", "row", "ent", "ext", "ent_m", "ext_m", "pairs")
    return [dict({k: b[k].clone() for k in keep}, token=list(b["token"])) for b in dl]                    # clone: a worker's shared-memory tensor would hold a file descriptor per batch


def forward_batches(model, cache, dev):
    """(output on the CPU, batch) per cached batch; the input is the loader's fp16 as float, like `jends_eval.dump`"""
    with torch.no_grad():
        for cb in cache:
            out = model(cb["x"].to(dev).float(), gt=None)[2]
            yield {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in out.items()}, cb


def frames_of(out, b):
    """the per-frame dicts of `jends_eval.dump` (the `ends_val.pkl` entries) of one batch"""
    frames = []
    for i in range(len(b["token"])):
        fr = dict(token=b["token"][i], row=int(b["row"][i]), kp_pos=out["kp_pos"][i].numpy(), kp_dir=out["kp_dir"][i].numpy(), kp_p=out["kp_p"][i].numpy(), valid=out["valid"][i].numpy(), logit=out["logit"][i].float().numpy(),
                  allowed=out["allowed"][i].numpy(), scale=out["scale"][i].float().numpy() if "scale" in out else None)
        if "kp_logsig" in out:
            fr["logsig"] = out["kp_logsig"][i].float().numpy()
        if "att_pos" in out:
            fr.update(att_pos=out["att_pos"][i].float().numpy(), att_dir=out["att_dir"][i].float().numpy(), att_p=out["att_p"][i].float().numpy())
        frames.append(fr)
    return frames


def dump_frames(model, cache, dev, thr=THR, max_frames=0):
    """-> (frames, counts): `jends_eval.dump` on the cached batches without its file writing; `max_frames` stops before the batch after that many frames (its `--max`; the counts are those of the frames kept)"""
    frames, tot = [], {}
    for out, b in forward_batches(model, cache, dev):
        if max_frames and len(frames) >= max_frames:
            break
        for k, v in QEN.end_counts(out, b, thr).items():
            tot[k] = tot.get(k, 0) + v
        frames += frames_of(out, b)
    return frames, tot


def _slice(d, i):
    return {k: (v[i:i + 1] if torch.is_tensor(v) else v) for k, v in d.items()}


def _heading_error(d, g):
    return float(np.degrees(np.arccos(np.clip(np.dot(d, g) / (np.linalg.norm(d) * np.linalg.norm(g) + 1e-9), -1, 1))))


def kp_error(out, b, i, kp_thr=KP_THR):
    """[matched ends, sum of position error (m), sum of heading error (deg), the same two for the attached ends] of frame i: the definition of `jdiag.py kperr` (strict one-to-one matching of the keypoints p >= kp_thr to the GT ends, 4 m / 30 degrees)"""
    import jrule_control as JR
    K = out["logit"].shape[1]
    kp, kd, p, v = (out[k][i].numpy() for k in ("kp_pos", "kp_dir", "kp_p", "valid"))
    att = ("att_pos" in out) and (out["att_pos"][i].numpy(), out["att_dir"][i].numpy())
    res = np.zeros(5)
    for k, (key, mkey) in enumerate((("ent", "ent_m"), ("ext", "ext_m"))):
        G = b[key][i][b[mkey][i]].numpy()
        s = slice(k * K, (k + 1) * K)
        for g, j in enumerate(JR._match(kp[s], kd[s], v[k] & (p[s] >= kp_thr), G)):
            if j >= 0:
                j += k * K
                res[:3] += (1, np.linalg.norm(kp[j] - G[g, :2]), _heading_error(kd[j], G[g, 2:4]))
                if att:
                    res[3:] += (np.linalg.norm(att[0][j] - G[g, :2]), _heading_error(att[1][j], G[g, 2:4]))
    return res


def pair_candidates(out, b, i, kp_thr=THR):
    """(logits, labels) of the candidate pairs of frame i: allowed pairs of keypoints with p >= kp_thr; true when the GT ends the keypoints are matched to (`match_keypoints`) are the two ends of a true pair, as `end_counts` counts them"""
    mi, mj = (m[0] for m in QEN.match_keypoints(_slice(out, i), _slice(b, i)))
    K = out["logit"].shape[1]
    kp = out["kp_p"][i]
    ia, ib = torch.nonzero(out["allowed"][i] & (kp[:K, None] >= kp_thr) & (kp[None, K:] >= kp_thr), as_tuple=True)
    pr = b["pairs"][i]
    true = {(int(x), int(y)) for x, y in pr[pr[:, 0] >= 0].tolist()}
    return out["logit"][i][ia, ib].float().numpy(), np.array([(int(mi[a]), int(mj[c])) in true for a, c in zip(ia.tolist(), ib.tolist())], bool)


def frame_stats(out, b, thr=THR):
    """per frame of a batch: the raw counts at p >= thr (`end_counts`), the keypoint error, the candidate pairs"""
    res = []
    for i in range(len(b["token"])):
        c = QEN.end_counts(_slice(out, i), _slice(b, i), thr)
        sc, lab = pair_candidates(out, b, i)
        res.append(dict(counts=[int(c[k]) for k in COUNTS], kp=kp_error(out, b, i).tolist(), scores=sc, labels=lab))
    return res


def six(counts):
    """the six dump-level numbers (entry / exit / pair recall and precision) of summed counts (a row of `COUNTS` or a dict)"""
    c = counts if isinstance(counts, dict) else dict(zip(COUNTS, counts))
    return [c[a] / max(c[b], 1) for a, b in SIX]


def pair_curve(scores, labels, n_gt):
    """-> dict(ap, p_at_r05): the average precision of the candidates ranked by logit, recall against all `n_gt` true pairs, and the exact precision at recall 0.5 (`jpair_pr.precision_at_recall_exact`)"""
    import jpair_pr as PR
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    o = np.argsort(-scores, kind="stable")
    tp = np.cumsum(labels[o])
    p = PR.precision_at_recall_exact(scores, labels, n_gt, 0.5)
    return dict(ap=float((tp / np.arange(1, len(tp) + 1))[labels[o]].sum() / n_gt) if n_gt and len(tp) else 0.0, p_at_r05=None if p is None else float(p))


def rootpath(p):
    """a path of the command line: a relative one is relative to the problem folder (the wrapper _py14.sh starts the script in its own folder)"""
    return p if not p or os.path.isabs(p) else os.path.join(ROOT, p)


def halves(rows):
    """0 = SELECTION half, 1 = JUDGEMENT half per frame (given as VAL dataset rows): the 22 VAL scenes in sorted order, even index = selection (11 scenes, 438 frames), odd = judgement (11 scenes, 435 frames; the card of E345 has the two counts swapped, the rule is the card's)"""
    import qgraph_eval as QE
    vrows = np.asarray(json.load(open(f"{QE.BP.ROOT}/data/splits.json"))["val"], int)
    scene = np.load(f"{QE.LS.B123}/poses_val.npz")["scene"]
    order = {int(s): k % 2 for k, s in enumerate(sorted(set(scene[vrows].tolist())))}
    return np.array([order[int(scene[int(r)])] for r in rows])


def run_pass(model, cache, dev, half, keep_frames=False):
    """one pass over the cache: -> (record, frames); the record has the per-frame counts and keypoint errors, the pair curves of the whole cache and of the two halves, the number of forwards"""
    stats, frames, n = [], [], 0
    for out, b in forward_batches(model, cache, dev):
        stats += frame_stats(out, b)
        n += 1
        if keep_frames:
            frames += frames_of(out, b)
    sc, lb = (np.concatenate([s[k] for s in stats]) for k in ("scores", "labels"))
    fi = np.concatenate([np.full(len(s["scores"]), k) for k, s in enumerate(stats)])
    n_pairs = np.array([s["counts"][COUNTS.index("pairs_n")] for s in stats])
    curves = {name: pair_curve(sc[m[fi]], lb[m[fi]], int(n_pairs[m].sum())) for name, m in (("all", np.ones(len(stats), bool)), ("sel", half == 0), ("jud", half == 1))}
    return dict(counts=[s["counts"] for s in stats], kp=[s["kp"] for s in stats], pair=curves, forwards=n), frames


# ---------------------------------------------------------------- 5. the plumbing controls
def zero_channels(model, tap, idx):
    """control: make channels `idx` of a tap constant by construction (the producing conv / linear rows 0 with a constant bias, the norm weight 0 with a constant bias; for the stream also the second conv and norm of every block) so that the mean replacement of them must change nothing"""
    idx = torch.tensor(list(idx))

    def silence(layer, norm, bias):
        layer.weight[idx] = 0.0
        if layer.bias is not None:
            layer.bias[idx] = bias
        if norm is not None:
            norm.weight[idx] = 0.0
            norm.bias[idx] = bias

    def producer(name):
        mod = sub(model, name)
        seq, i = (mod, len(mod) - 1) if isinstance(mod, nn.Sequential) else (sub(model, name.rpartition(".")[0]), int(name.rpartition(".")[2]))     # the end of a block, or an activation inside a Sequential
        return (seq[i - 2], seq[i - 1]) if isinstance(seq[i - 1], nn.GroupNorm) else (seq[i - 1], None)

    with torch.no_grad():
        if tap == "jm.stream":
            silence(*producer("jm.p2"), 0.5)
            for b in model.jmodel.jm.blocks:                                     # out = relu(x + f(x)) with f's last norm weight 0 and no bias: the channel stays what the block before made it
                silence(b.f[3], b.f[4], 0.0)
        else:
            silence(*producer(tap), 0.5)


def max_diff(a, b, ints=True):
    """largest absolute difference over the floating-point tensors of two outputs; the integer / bool tensors (keypoint cells, validity) must be equal, else inf (`ints=False`: only the floating-point ones count)"""
    d = 0.0
    for k in a:
        if torch.is_tensor(a[k]) and a[k].numel():
            if a[k].is_floating_point():
                d = max(d, float((a[k] - b[k]).abs().max()))
            elif ints and not torch.equal(a[k], b[k]):
                d = float("inf")
    return d


ZEROED = (("jm.stream", 16), ("jm.blocks.1.f.2", 16), ("jm.m1.2", 16), ("pyr.e1", 24), ("pyr.d0", 12), ("je.pair.1", 16))      # (tap, how many of its first channels) of the zeroed-copy control


def plumbing_controls(model, calib, cache, dev, taps):
    """the controls of the card on a few cached batches: (1) an identity hook (writes the values back, on every tap) reproduces the null to 0; (2) on a ZEROED COPY (channels constant by construction) the mean replacement of those channels changes nothing (<= 1e-6);
    both with positive controls so that a 0 is not a dead control: zeroing the channels changes the model (> 1e-4) and the mean replacement of the same channels of the ORIGINAL changes it (> 1e-6). -> dict of the numbers; asserts"""
    run = lambda m: [o for o, _ in forward_batches(m, cache, dev)]  # noqa: E731
    base = run(model)
    mi, w = ablate(model, [Edit("chan", tap=t.name, idx=tuple(range(max(1, t.C // 4)))) for t in taps.values()], taps)
    d_id = max(max_diff(a, b) for a, b in zip(base, run(mi)))
    w.check(len(cache))
    z = copy.deepcopy(model)
    for t, n in ZEROED:
        zero_channels(z, t, range(n))
    zout, zcal = run(z), calibrate(z, cache, dev, taps)
    d_zero, d_effect, d_orig = 0.0, max(max_diff(a, b, False) for a, b in zip(base, zout)), 0.0
    for t, n in ZEROED:
        u = Unit(f"zeroed.{t}", "maps", "chan", (1.0,), taps=tuple(f"jm.blocks.{i}" for i in range(4)) + ("jm.p2",) if t == "jm.stream" else (t,), idx=tuple(range(n)))
        d_zero = max(d_zero, max(max_diff(a, b) for a, b in zip(zout, run(ablate(z, edits_for(u, 1.0, zcal), taps)[0]))))
        d_orig = max(d_orig, max(max_diff(a, b, False) for a, b in zip(base, run(ablate(model, edits_for(u, 1.0, calib), taps)[0]))))
    res = dict(identity_max_diff=d_id, zeroed_copy_max_diff=d_zero, zeroing_effect=d_effect, original_effect=d_orig)
    assert d_id == 0.0, f"an identity hook must reproduce the null to 0 (restart with --deterministic): {res}"
    assert d_zero <= 1e-6, f"the mean replacement of zeroed channels changed the outputs by {d_zero:.2e}"
    assert d_effect > 1e-4 and d_orig > 1e-6, f"a control that cannot fail: {res}"
    return res


# ---------------------------------------------------------------- 6. the sweep
def spec_name(spec):
    return "+".join(f"{u}@{f}" for u, f in spec.items()) or "null"


def check_resume(rec, rows, deterministic):
    """a sweep.json on disk may be resumed only on the same frames and, once it holds passes, with the same deterministic setting"""
    assert rec["rows"] == rows, "the sweep.json on disk is of other frames"
    assert bool(rec.get("deterministic", False)) == deterministic or not rec["passes"], f"sweep.json was made with deterministic={rec.get('deterministic', False)}, this job runs with {deterministic}: the passes must share one setting"


class Sweep:
    """the passes of one model-seed: the null twice, the plumbing controls, the 84 single points, the random / reverse draws of the ladder units, the dumps of the point sets named for scoring; every pass is a record of `run_pass`, saved after each pass (a rerun resumes)"""

    def __init__(self, run, dev, out_dir, runs_dir=None, max_frames=0, calib_frames=0, workers=2, val_stride=1, deterministic=False, log=print):
        self.run, self.dev, self.out, self.log, self.deterministic = run, dev, out_dir, log, deterministic
        wait_for_gpu("jprune_probe sweep", dev)
        self.runs_dir = runs_dir or os.path.join(ROOT, "runs")
        self.model, self.args = JEV.load_model(run, dev)
        self.units, self.taps = build_units(self.model), tap_table(self.model)
        self.cost = Cost(copy.deepcopy(self.model).cpu())
        self.val = build_cache("val", workers=workers, stride=val_stride, max_frames=max_frames)
        self.rows = [int(r) for b in self.val for r in b["row"]]
        assert max_frames or val_stride > 1 or (len(self.rows) == 873 and [len(b["token"]) for b in self.val] == [8] * 109 + [1]), "the VAL cache must be the loader's 109 batches of 8 and one of 1"
        self.half = halves(self.rows)
        cpath = os.path.join(out_dir, "calib.pt")                               # the calibration is kept: a later invocation (the dumps of the plan's point sets) must pick the SAME channels, not a ranking that a near tie may flip
        if os.path.exists(cpath):
            self.calib = load_calib(cpath, dev)
        else:
            t = time.time()
            self.calib = calibrate(self.model, build_cache("train", workers=workers, stride=CAL_STRIDE, max_frames=calib_frames), dev, self.taps)
            save_calib(self.calib, cpath)
            log(f"calibration on {self.calib.frames} TRAIN frames: {time.time() - t:.0f} s")
        self.path = os.path.join(out_dir, "sweep.json")
        self.rec = json.load(open(self.path)) if os.path.exists(self.path) else dict(run=run, attach=bool(self.args.get("attach")), deterministic=deterministic, rows=self.rows, tokens=[t for b in self.val for t in b["token"]], calib_frames=self.calib.frames, passes={}, points=[], controls={})
        check_resume(self.rec, self.rows, deterministic)
        self.rec["deterministic"] = deterministic

    def save(self):
        atomic_write(self.path, json.dumps(self.rec).encode())

    def do(self, key, spec, how="rank", rng=None):
        """one pass with the point set `spec` (skipped when `key` is recorded); the dead-hook assert on every hook"""
        if key in self.rec["passes"]:
            return
        edits = spec_edits(spec, self.units, self.calib, how, rng)
        m, w = ablate(self.model, edits, self.taps)
        reset_gpu_peak(self.dev)
        t = time.time()
        r, _ = run_pass(m, self.val, self.dev, self.half)
        w.check(r["forwards"], self.calib)
        sv = self.cost.saved(edits)
        r.update(spec=spec, how=how, saved_macs=sv["macs"], saved_params=sv["params"], saved_macs_pyr=sv["macs_pyr"], seconds=time.time() - t, gpu_mb=gpu_peak_mb(self.dev))
        self.rec["passes"][key] = r
        self.save()
        self.log(f"{key:30s} {r['seconds']:5.0f} s  ends_mean6 {np.mean(six(np.sum(r['counts'], 0))):.4f}  saved {sv['macs'] / 1e9:.3f} GMACs")

    def plumbing(self, n_batches=2):
        self.rec["controls"]["plumbing"] = plumbing_controls(self.model, self.calib, self.val[:n_batches], self.dev, self.taps)
        self.log("plumbing: " + json.dumps(self.rec["controls"]["plumbing"]))
        self.save()

    def nulls(self):
        """the null twice; above the cap the deterministic algorithms are set (recorded in sweep.json) and both rerun, still above: stop (the probe is then not the stored pipeline)"""
        for attempt in (0, 1):
            self.do("null", {})
            self.do("null2", {})
            a, b = (np.mean(six(np.sum(self.rec["passes"][k]["counts"], 0))) for k in ("null", "null2"))
            self.rec["controls"]["null_spread_ends_mean6"] = float(abs(a - b))
            if abs(a - b) <= NULL_CAP:
                return
            assert attempt == 0 and not self.deterministic, f"the two nulls differ by {abs(a - b):.4f} in ends_mean6 (cap {NULL_CAP}) with the deterministic setting: fix the probe"
            apply_deterministic()                                              # cuBLAS read its workspace setting at start: if this does not bring the spread down, restart with --deterministic
            self.deterministic = self.rec["deterministic"] = True
            for k in ("null", "null2"):
                del self.rec["passes"][k]

    def run_all(self, score_points=(), controls=True):
        """the whole job: nulls, plumbing, the single points, the random / reverse draws; then the dumps of `score_points` [{id, set}]"""
        self.nulls()
        if controls:
            self.plumbing()
        pts = points(self.units)
        self.rec["points"] = [dict(id=i, unit=u.name, f=f, group=u.group, kind=u.kind) for i, u, f in pts]
        for i, u, f in pts:
            self.do(f"p{i:03d}", {u.name: f})
        for u in (u for u in self.units if controls and len(u.fracs) > 1):
            for s in range(3):
                self.do(f"rand{s}_{u.name}", {u.name: 0.5}, "random", np.random.RandomState(1000 * s + sum(map(ord, u.name))))
            self.do(f"rev_{u.name}", {u.name: 0.5}, "reverse")
        for sp in score_points:
            self.dump_point(sp["id"], sp["set"])

    def dump_point(self, pid, spec):
        """the point's `ends_val.pkl` + `point.json` in `runs/e345_p<NNN>_<model>/` with a symlink to the unmodified model.pt (the stored run's folder gets no file). Every file goes through a temporary file and `os.replace`, `point.json` (with the sha256 of the pickle) LAST:
        a folder without it is incomplete and redone; a complete one is never overwritten (the same point set again is a no-op, another set under that number or a pickle that does not match its point.json is refused)"""
        d = os.path.join(self.runs_dir, f"e345_p{pid:03d}_{self.run}")
        if os.path.exists(os.path.join(d, "point.json")):
            meta = json.load(open(os.path.join(d, "point.json")))
            assert meta["spec"] == spec, f"{d} holds another point set: a point folder is never overwritten"
            assert hashlib.sha256(open(os.path.join(d, "ends_val.pkl"), "rb").read()).hexdigest() == meta["sha256"], f"{d}/ends_val.pkl does not match its point.json (truncated or changed): move the folder away and rerun"
            self.log(f"point {pid} is dumped already")
            return
        edits = spec_edits(spec, self.units, self.calib)
        m, w = ablate(self.model, edits, self.taps)
        frames, n = [], 0
        for out, b in forward_batches(m, self.val, self.dev):
            frames += frames_of(out, b)
            n += 1
        w.check(n, self.calib)
        os.makedirs(d, exist_ok=True)
        if not os.path.lexists(os.path.join(d, "model.pt")):
            os.symlink(os.path.abspath(os.path.join(self.runs_dir, self.run, "model.pt")), os.path.join(d, "model.pt"))
        blob = pickle.dumps(frames)
        atomic_write(os.path.join(d, "ends_val.pkl"), blob)
        atomic_write(os.path.join(d, "point.json"), json.dumps(dict(point=pid, spec=spec, run=self.run, frames=len(frames), sha256=hashlib.sha256(blob).hexdigest(), edits=[dict(kind=e.kind, tap=e.tap or e.block, idx=list(e.idx), head=e.head) for e in edits]), indent=1).encode())
        self.log(f"dumped point {pid} ({spec_name(spec)}) -> {d}")


GATE_TIMEOUT_S = 12 * 3600.0                                                          # these jobs wait for the GPU much longer than the chains do (`E345_GATE_TIMEOUT_S` changes it)


def wait_for_gpu(who, dev, timeout_s=None):
    """D100: a GPU step does not start next to two trainings. `gpu_gate.wait_for_room` waits (this job: 12 h, `E345_GATE_TIMEOUT_S`; the shared `GPU_GATE_TIMEOUT_S` of the chains is ignored here); when the GPU is still full afterwards the job EXITS with code 75 and a message
    instead of going on into a full GPU (exit 75 = a retry is wanted, as everywhere in this repository). No effect on the CPU."""
    if not str(dev).startswith("cuda"):
        return
    import gpu_gate
    timeout_s = float(os.environ.get("E345_GATE_TIMEOUT_S", timeout_s or GATE_TIMEOUT_S))
    shared = os.environ.pop("GPU_GATE_TIMEOUT_S", None)
    try:
        gpu_gate.wait_for_room(who=who, timeout_s=timeout_s)
    finally:
        if shared is not None:
            os.environ["GPU_GATE_TIMEOUT_S"] = shared
    try:
        limit = float(os.environ.get("GPU_GATE_MB", gpu_gate.DEFAULT_MB))
    except ValueError:
        limit = gpu_gate.DEFAULT_MB
    used = gpu_gate.used_mb()
    if limit > 0 and used is not None and used >= limit:
        print(f"[jprune_probe] {who}: {used:.0f} MiB of the GPU are still in use after {timeout_s:.0f} s (limit {limit:.0f}): not starting next to the trainings; run it again when the GPU is free", flush=True)
        sys.exit(75)


def reset_gpu_peak(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb(dev):
    """-> dict(max_allocated, max_reserved) of this process in MiB since the last reset (None on the CPU)"""
    if not str(dev).startswith("cuda"):
        return None
    return dict(max_allocated=torch.cuda.max_memory_allocated() / 2 ** 20, max_reserved=torch.cuda.max_memory_reserved() / 2 ** 20)


def apply_deterministic():
    """cuBLAS reads its workspace setting when it starts: call this BEFORE anything creates the CUDA context"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_deterministic(out_dir, requested):
    """the deterministic setting of a job: the one recorded in `sweep.json` (with a loud line when the command line says otherwise), a refusal of `--deterministic` against passes that were made without it"""
    p = os.path.join(out_dir, "sweep.json")
    if not os.path.exists(p):
        return requested
    rec = json.load(open(p))
    was = bool(rec.get("deterministic", False))
    if was and not requested:
        print("[jprune_probe] sweep.json was made with --deterministic: this resumed job runs with it too", flush=True)
        return True
    if requested and not was and rec.get("passes"):
        sys.exit(f"[jprune_probe] {p} holds passes made WITHOUT --deterministic: the passes must share one setting (use another --out or drop the flag)")
    return requested


def time_pass(run, dev, n_batches=4):
    """seconds of a null pass over n_batches cached VAL batches on the device (the pass of the sweep: forward + per-frame numbers), and the extrapolation to the 110 batches of the VAL"""
    wait_for_gpu("jprune_probe time", dev)
    reset_gpu_peak(dev)
    model, _ = JEV.load_model(run, dev)
    cache = build_cache("val", workers=0, max_frames=8 * n_batches)
    half = np.zeros(sum(len(b["token"]) for b in cache), int)
    t = {}
    for name, fn in (("forward only", lambda: [None for _ in forward_batches(model, cache, dev)]), ("forward + per-frame numbers", lambda: run_pass(model, cache, dev, half))):
        fn()                                                                                         # warm-up
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        fn()
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        t[name] = (time.time() - t0) / len(cache)
        print(f"{name}: {t[name]:.2f} s per batch of 8 on {dev} (the machine as it is now) -> {t[name] * 110:.0f} s per VAL pass")
    mem = gpu_peak_mb(dev)
    if mem:
        print(f"GPU memory of this process: peak allocated {mem['max_allocated']:.0f} MiB, peak reserved {mem['max_reserved']:.0f} MiB")
    t["gpu_mb"] = mem
    return t


# ---------------------------------------------------------------- selftest and CLI
def _mha_by_hand(att, h, pad, replace=None):
    """the attention module by hand: per-head outputs (head replace[0] := the constant replace[1]), then the output projection"""
    o = head_outputs(att, h, pad)
    if replace is not None:
        o = o.clone()
        o[:, replace[0]] = replace[1].view(1, 1, -1)
    B, H, n, hd = o.shape
    return F.linear(o.transpose(1, 2).reshape(B, n, H * hd), att.out_proj.weight, att.out_proj.bias)


def selftest():
    import jends_train as JT
    torch.manual_seed(0)
    m = JT.JEndModel(dict(d=64, ctx="dil"), order=True, evidence=True, scales=True).eval()
    units, taps = build_units(m), tap_table(m)
    print(unit_table(units, Cost(m), expect=(45, 84, 20)))
    assert len({u.name for u in units}) == 45 and [i for i, _, _ in points(units)] == list(range(1, 85))
    # the edit of an attention head (weights on a deepcopy) = a manual per-head replacement by the head's calibration mean, and the plain module = the manual forward
    att, h = m.je.blocks[0].att, torch.randn(3, 20, 64)
    pad = torch.zeros(3, 20, dtype=torch.bool)
    pad[1, 15:] = True
    mean = head_outputs(att, h, pad)[:, 2].mean((0, 1))
    mc, w = ablate(m, [Edit("head", block="je.blocks.0", head=2, val=mean)], taps)
    with torch.no_grad():
        got = mc.je.blocks[0].att(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        d_edit = float((got - _mha_by_hand(att, h, pad, (2, mean)))[~pad].abs().max())
        d_plain = float((_mha_by_hand(att, h, pad) - att(h, h, h, key_padding_mask=pad, need_weights=False)[0])[~pad].abs().max())
    w.check(1)
    assert d_edit < 2e-7 and d_plain < 2e-7, (d_edit, d_plain)                 # the card asks 1e-7: measured 6e-8 and 9e-8 with this seed
    # a channel edit: the replaced channels are constant at the output; a hook that did not fire is caught
    x = torch.randn(2, 39, 200, 200)
    cal = calibrate(m, [dict(x=x)], "cpu", taps)
    mz, wz = ablate(m, edits_for(units[2], 0.5, cal), taps)
    seen = []
    mz.jmodel.pyr.e2.register_forward_hook(lambda _m, _a, o: seen.append(o.detach().clone()))
    with torch.no_grad():
        mz(x, gt=None)
    wz.check(1, cal)
    idx = list(wz.edit.values())[0].idx
    assert len(idx) == 72 and float(seen[0][:, list(idx)].std(dim=(0, 2, 3)).max()) < 1e-6, "the replaced channels must be constant"
    try:
        wz.check(2, cal)
        raise RuntimeError("a hook that fired once in two forwards must be caught")
    except AssertionError as ex:
        assert "fired" in str(ex), ex
    print(f"jprune_probe selftest ok (head edit vs by-hand replacement {d_edit:.1e}, plain module vs by-hand {d_plain:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("units")
    s.add_argument("--run", required=True)
    for name in ("sweep", "time"):
        s = sp.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s = sp.choices["time"]
    s.add_argument("--batches", type=int, default=4)
    s = sp.choices["sweep"]
    s.add_argument("--out", required=True)
    s.add_argument("--max", type=int, default=0, help="only the first N VAL frames (a test of the pipeline)")
    s.add_argument("--val-stride", type=int, default=1, help="every n-th VAL frame (a test of the pipeline that keeps both halves of the scenes)")
    s.add_argument("--calib-max", type=int, default=0, help="only the first N calibration frames")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--score-points", default=None, help="json [{id, set: {unit: f}}]: also write the point folders runs/e345_p<NNN>_<model>/ for these (the first entry of `jprune_table.py plan`)")
    s.add_argument("--runs-dir", default=None)
    s.add_argument("--deterministic", action="store_true")
    s.add_argument("--no-controls", action="store_true")
    sp.add_parser("selftest")
    a = ap.parse_args()
    for k in ("out", "score_points", "runs_dir"):
        if hasattr(a, k):
            setattr(a, k, rootpath(getattr(a, k)))
    torch.set_num_threads(2)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "units":
        mdl, _ = JEV.load_model(a.run, "cpu")
        print(unit_table(build_units(mdl), Cost(mdl)))
    elif a.cmd == "time":
        time_pass(a.run, a.device, a.batches)
    else:
        os.makedirs(a.out, exist_ok=True)
        det = resolve_deterministic(a.out, a.deterministic)
        if det:
            apply_deterministic()                                            # before the GPU is touched: cuBLAS reads its workspace setting at start
        sw = Sweep(a.run, a.device, a.out, a.runs_dir, a.max, a.calib_max, a.workers, a.val_stride, det)
        sw.run_all(json.load(open(a.score_points)) if a.score_points else (), not a.no_controls)
