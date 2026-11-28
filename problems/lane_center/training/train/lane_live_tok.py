"""E372 (stage S4 of the token champion, D113): the LIVE token producer. B' (a `QGraph` run, `runs/e292_a_s<S>`: pyramid + node head + the Matrix-NMS layer) runs inside the training step with its node head TRAINABLE, and the tokens and the
labels that `lane_tokcache.py` stores for the frozen B' are made again from its live output with the graph kept, so that the loss of the token head reaches B''s node head. Card: `experiments/E372_s4-joint-finetune-b-in-the-loop/experiment.md`.
Derived data: none (nothing is cached here); the code, its selftest and the fixture are the record.

    ds, dl = loader("train" | "val", seed=0, bs=8, workers=4)                          # the frames of the E365 cache of B' seed `seed` (TokData(split, res=2.0)), asserted in the cache's order
    b = next(iter(dl))                                                                 # x (B, 39, 200, 200) f16, table (list of the GT tables), token (list), row (B,), targets (the dense node-loss maps of B)
    src = LiveSource("runs/e292_a_s0", K=512, train_nodes=True).to(dev)                # B' loaded, everything frozen but `model.nodes.*`
    out = src(b["x"], b["table"], b["token"], b["row"])                                # out["batch"]: the batch of `lane_ptok_data` (head input + labels); out["no"]: B''s node-stage outputs WITH the graph
    loss_h, info_h = jtoken.jtoken_loss(head(out["batch"]), out["batch"])              # the head's loss (the head must not cut its inputs: PLAN)
    loss_b, info_b = b_loss(out["no"], b["targets"])                                   # B''s own node loss (its E273 / E292 form, `qgraph.node_loss`)
    python training/train/lane_live_tok.py check   [--split val] [--seed 0] [--n 20]                           # the guards, the frame order and the checkpoint of the cache, node_targets against the maps of `lane_gt_data.QData`, bit for bit
    python training/train/lane_live_tok.py fixture [--split val] [--n 40] [--seeds 0 1] [--gpu] [--bs 4] [--out FILE]   # the live batch against the cache (`PTokData`), field by field; --gpu: must be bit for bit (exit 1 otherwise): the rule below
    python training/train/lane_live_tok.py grad    [--split val] [--seed 0] [--n 2] [--head e373_cb_s0]          # the card's gradient check on real frames with the real B' (CPU): the head's loss reaches every node-head tensor and no frozen stage; the gradient norms
    python training/train/lane_live_tok.py selftest                                                            # no cache, no GPU (a toy `QGraph`); tests: net/tests/test_lane_live.py (the hand mutants)

Frames: TRAIN / VAL as the cache (`lane_tok_data.TokData(split, res=2.0)`); TEST guard: VAL is exactly `data/splits.json['val']` (`check_rows`; the feature files also hold the TEST rows, never read).
LiveData item: x (39, 200, 200) f16 (normalised as QData / TokData), token, row, table (the E357 GT table of the frame on the 2 m lattice); NO `QData` item (it fits the long-link scales per frame: too slow). `collate` adds `targets`: the dense maps of B''s own node
loss, `node_targets(table)` = `qtargets.grid_targets` on the table: exists (50, 50), flags (8, 50, 50), offset / dirv (2, 50, 50), sstart, uout, uin (50, 50); the tier / vis / ztier maps of `grid_targets` are not made (the node loss does not read them).
LiveSource.forward(x, tables, token, row, token_drop=0.0, seed=0, gate=True) -> dict(no, batch) = `tokens(x)` -> (no, tk) (pyramid + node stage + live tokens: device work, asynchronous on a GPU) followed by `labelled(tk, tables, token, row, token_drop, seed, gate)` -> batch
(the label step: it reads the tokens back to the host and runs on the CPU). B' runs in float32 also inside the trainer's bf16 autocast (the cache was made in float32). `no` = `QGraph.node_stage` outputs (e = the damped logit, e_raw = the head's own).
`batch` = the dict of `lane_ptok_data.collate` (tok (B, K, 64), pos, dirv, e, e_raw, s, uout, uin, flags, valid, ytype, y, ylong, yscale, gt_out, token, row, tokens_np, label) + `lab` (the per-frame dicts of `lane_tokcache.label_frame`: counts, pair gate) and
`label_s` (seconds of the CPU label loop): the TOKEN INPUTS (tok, pos, dirv, e, e_raw, s, uout, uin, flags) carry the graph to B''s node head; valid and every label are constants. The tokens are those of `lane_tokcache.select_tokens` (the K cells with the largest e, a stable
argsort = the named index budget: no gradient through the choice; probabilities from the whole map; K = 512 as the cache), `tok` rounded to float16 as the cache stores it with the gradient passed straight through (a plain .half() would round the gradient too). The labels
are `label_frame` + `densify` on the DETACHED live tokens (numpy: a graph there would raise), the gate `allowed` from `jtok.pair_geometry` on them (statistics only: `ypair_gate`, `n_reach`; gate=False skips it), ylong on the TOKEN chord as `PTokData` defines it.
Token drop: this module's `loader()` has none; `forward(token_drop=, seed=)` is the STATIC draw of `PTokData` (seeded by (seed, frame token), the same removal every epoch), meant for a degraded read-out. The champion run used none: the joint trainer passes token_drop=0
(or uses ONE mechanism only: the trainer's own per-step token drop, which works on this batch as it is). A frame with more target pairs than `lane_tokcache.PMAX` raises `LabelOverflow(RuntimeError)` (`.token`, `.pairs`).

FIXTURE on the CPU (2026-11-17, first 40 VAL frames, B' seeds 0 and 1, 2 threads): the label step on the CACHED tokens is exactly the cache (valid, ytype, y, ylong, yscale, gt_out, gt_id / gd / gt_flags / gt_aux, the `tokens_np` fields pos, dirv, e_p, flags_p, s_p, valid and the label counts: no difference on the 80 frames). The whole live
path: at least 511 of the 512 cells in both lists on every frame, but no frame with the same ORDERED list (neighbouring ranks swap); over the common cells the largest |diff| tok 1.6e-2, pos 2.1e-3 m, dirv 2.6e-2, e_raw 4.6e-3, s 2.4e-2, uout / uin 2.0e-2, flags 2.1e-2, e 1.15 (one
Matrix-NMS neighbour order flipped, so e_p of that token 0.28 in tokens_np; the other tokens_np fields: pos 2.1e-3 m, dirv 2.6e-2, flags_p 3.8e-3, s_p 3.3e-3); differing entries (80 frames): valid 5, ytype 7, y 24, ylong 0, label 8, gt_out 1, tokens_np valid 5; gd
inf = a token assigned to a GT node in one path only (a near-threshold token). The same B' on the CPU in float64 differs from float32 by 3.4e-5 at most (the same 512 cells). The gap to the cache is 60 - 700 times that (pos 2.1e-3 m: 60, e_raw 4.6e-3:
135, uout / uin 2.0e-2: 590, s 2.4e-2: 700): it is not CPU float noise, and it is the size of float32 against TF32-rounded convolutions (emulated on the CPU: pos 1.5e-3 m, e_raw 2.8e-3, s 1.8e-2), so the reading 'the cache was made with the GPU's default convolution rounding' is LIKELY, NOT PROVEN.
The tok gap (1.56e-2 = 2^-6, two float16 ulps for |tok| in [8, 16): the cached tok is float16, max |tok| 14.5 on these frames) is a rounding quantum and does not discriminate that reading from any other noise. `fixture --gpu` is the decisive check, by the rule below.

GPU FIXTURE ACCEPTANCE (the rule of the parity check, written before its first GPU run; the check is a DIAGNOSTIC: it feeds no verdict):
  PASS        on the first --n frames in batches of --bs 4 (as the cache build; n a multiple of 4), for EVERY seed given, in BOTH parts (`labels` = the label step on the cached tokens, `live` = the whole path): the K cells are the same ORDERED list on every frame and every field of
              `compare_frame` is exactly zero (floats 0.0, integer / boolean entries 0, the label counts equal). Exit code 0, json `ok` true.
  FAIL = BUG  any difference in the `labels` part (no model is involved: it must be exact on every machine and every bs); exactly one field differs and the others are exact (a field mix-up); different cells with identical e (a tie rule); `live` differs while the CPU `labels` part passes
              on the same frames and the environment below equals the cache build's.
  FAIL = ENV  to look at before any judgement of the code: another torch / CUDA / cuDNN / driver than at the cache build (the json `env` holds torch.__version__, torch.version.cuda, the cudnn version, the GPU name and the checkpoint sha of every seed; the cache's meta.json does not
              record the build's versions: ask who built it), another GPU model, a cuDNN algorithm change under memory pressure (repeat on an idle GPU), --bs other than 4, n not a multiple of 4 (refused on --gpu).
  RESULT (2026-11-17 / 18, GPU, 200 VAL frames, B' seeds 0 and 1, runs `E372.fixture_gpu_bs4` / `E372.fixture_gpu_bs8`, json `runs/e372_fixture/fixture_gpu_bs{4,8}.json`, code 0b7b6e4369 / 99d5f570de): **bs 4 PASS: the whole live path is exactly the cache on every frame of both seeds, every field zero, the same ordered K cells, the label counts equal (label loop 33 - 34 ms per batch of 4)**; bs 8 FAIL = ENV (the batch-size dependence of cuDNN, not a bug: the `labels` part is exact at bs 8): tok 1.56e-2 (two float16 ulps), pos 1.8e-3 / 2.0e-3 m, dirv 2.2e-2 / 6.5e-2, e up to 2.7 on one Matrix-NMS neighbour-order flip, valid differs in 5 slots, ytype 5 - 6, y 16 - 27 pair entries, label 6 - 13 per 200 frames (label loop 71 - 80 ms per batch of 8): the live read-out of the S4 card therefore runs B' at bs 4 (the head at bs 8 as the cache dump).
  A PASS proves parity at bs 4 only: cuDNN picks its algorithms by shape, and the trainer reads at the batch size of its own step. The live read-out of the reference arm must repeat the check at ITS batch size (`--bs`: the result line and the json say which bs ran; at bs != 4 a
  difference is the batch-size dependence of the GPU, not a bug, and its size is what that read-out inherits against the cache).

PLAN for the joint trainer (the head's trainer is `train/jtoken_train.py`; the joint loop is the lane chat's):
  per step    b = next(dl); out = src(b["x"], b["table"], b["token"], b["row"]); loss = jtoken_loss(head(out["batch"]), out["batch"]) + w * b_loss(out["no"], b["targets"])   (w = 1, the card); one backward, one clip, one optimiser step
  overlap     the head's forward reads only `head_inputs(tk)` (bit for bit the same output as with the whole batch, tested; token_drop 0): a trainer can run `no, tk = src.tokens(x)`, queue `head(head_inputs(tk))`, and only then call `src.labelled(tk, ...)` (the CPU label
              loop, with its device sync) so that the head's forward overlaps with the labels; the loss needs the labels (`labelled`'s batch), the forward does not
  parameters  head.parameters() + src.trainable() (the node head: 38 tensors, 99 304 parameters), both at lr 1e-4: `trainer.fit` sets every group of its optimiser from ONE schedule, so the two rates must be equal (they are, in the card); its clip (3.0) is `clip_grad_norm_(model.parameters())` (trainer.py: `params=` only changes what the optimiser steps): parameters that are not in `model` are NOT clipped, so the joint trainer must pass a `model` whose parameters() hold
              the head AND `src.trainable()` (one wrapper module) or clip the node head itself; a silent False is the risk of the whole stage: `load_model` / `build_head` and the trainer's init loading build a head with live_inputs False, so build it with live_inputs True (put it in the RECIPE), assert
              `head.live_inputs` in the loss function at every step and refuse a zero head-loss gradient norm on the node head at step 1 (a joint run would otherwise train B' on its own node loss only, with plausible numbers);
              the node head's gradient from B's own loss alone has a norm of 8.7 on 2 VAL frames and 17.3 - 17.6 on 8 (`grad`, B' seeds 0 / 1, step 0), so a clip of 3.0 over them is active from the first step. By design NO gradient reaches B' through: `valid`, the allowed-pair gate, the short / long split at 8 m (`cand`), the choice of the K tokens (the argsort budget) and, under prior=rule, the c-rule matrix: name these in the D55 sentence of the S4 card. `src.model.state_dict()` is a complete `QGraph` state (9.3 MB, the pyramid
              8.0 MB): `qpredict.load_model` reads a model.pt made of it (args as the run's). B' trains on its node loss alone here: E292's phase 9 also had the link, section and assignment terms, whose heads are not in this path
  the head    must keep its inputs: `JTokenHead(live_inputs=True)` (`jtoken.py`; False = the champion's cut `_const`, bit for bit). `gradient_check` toggles it. Live inputs add ~80 MB of saved activations at bs 8 (the pair-geometry graph);
              under `torch.no_grad()` (the read-out) the same code gives the live tokens without a graph. At the champion start (2 - 8 VAL frames, step 0) the head's loss gives the node head a GRADIENT of norm 4.0 (2 frames) / 8.1 and 6.7 (8 frames, seeds 0 / 1): 0.28 - 0.32
              of the gradient-norm sum with B's own loss. That is not the card's secondary number (the share of the parameter UPDATE norm: Adam rescales each parameter's step); measure that one from the optimiser's steps
  autocast    `src` switches autocast off for B' (float32 = the cache's numbers); the head may stay in bf16 as in S1. The cache was made with the GPU's default convolution settings (nothing in the cache build or in `trainer.py` sets cudnn / TF32 flags): do not set any in the trainer
  a resume    the node head and the head are in the checkpoint; the pyramid is not trained, its weights come from the run (`src.state_dict()` holds them too)
  labels      `label_batch` runs on the CPU in the main process, serial between the tokens and the loss (see overlap): 8 - 14 ms per frame, 0.08 - 0.11 s per batch of 8 on this box (median 31 - 43 ms per batch of 4, load ~6, 2 threads). `jtok.pair_geometry` for the gate runs on
              the device (0.42 s on the CPU for a batch of 8: not there). gate: the pair gate only feeds the statistics `ypair_gate` and `n_reach` of the label dicts (not y, ylong, yscale, valid, ytype, gt_out: tested equal); the trainer may pass gate=False (saves ~109 MB, a device
              sync and a host copy per step at bs 8; `lab[b]['n_reach']` is then 0 and `ypair_gate` False); the default True is the cache's parity (the fixture uses it). `label_frame` asserts at most 4096 target pairs per frame (PMAX; the cache's maximum is 646 on TRAIN, 536 on VAL,
              but tokens that move under the head's loss may grow near-duplicates): `LabelOverflow(RuntimeError)` names the frame (`.token`) and its count (`.pairs`); the trainer decides (skip the step and count it, abort after N); log `lab[b]['n_pairs']`
  memory      bs 8, K = 512: head 2.12 GB (S1 peak, bf16) + B' node head 1.23 GB of saved activations (float32; 307 MB at bs 2, counted with saved-tensor hooks on the CPU, x 4) + live head inputs 0.08 GB + backward temporaries (the 7 x 7 unfold buffers, 0.25 GB each) ~0.6 GB
              + pyramid transient ~0.3 GB (no_grad) + its retained maps 0.08 GB = about 4.5 GB of the 12 GB card; float32 for B' fits, bf16 is not needed (and would not be the cache's tokens)
  speed       S1 head 0.108 s per step at bs 8 (0.2 s on a shared GPU; measured in the S1 / S3 logs) + B' forward ~13 ms per frame at batch 1 (~0.1 s at bs 8) + node-head backward ~0.05 s + labels ~0.09 s (measured on the CPU) = ~0.35 s per step (~0.5 s shared): 3000 steps = 18 - 25 min per run.
              The B' forward time, the node-head backward time and the sum are ESTIMATES until the first GPU smoke; so is the memory sum above except its measured parts (the head's 2.12 GB, the node head's saved activations by hooks, the label step on the CPU)
"""
import argparse
import contextlib
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "net/prep", "net/model", "net/loss", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import bflc_paths as BP  # noqa: E402  (where code, data and runs are)
import jtok as JT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import qgraph as QG  # noqa: E402
import loss_qgraph as LQG  # noqa: E402
import qpredict as QP  # noqa: E402
import qtargets as QT  # noqa: E402

NODE_KEYS = ("exists", "flags", "offset", "dirv", "sstart", "uout", "uin")                              # the maps `qgraph.node_loss` reads (B''s dense target)
GRAPH_KEYS = tuple(k for k in LP.TOKEN_FIELDS if k != "valid")                                          # the token inputs of the head that carry the graph (valid is a constant)
NP_KEYS = ("pos", "dirv", "e_p", "flags_p", "s_p")                                                     # the float fields of `tokens_np` (what the read-out writer reads; plus cell and valid)
STATS = ("n_gt", "n_gt_assigned", "n_links", "n_links_c", "gt_out", "n_nosrc", "n_reach", "n_pairs", "n_conflict", "n_valid", "n_deliv", "n_deliv_near", "n_gt_deliv_cov", "n_e50_in")      # per-frame counts of a label dict, as the cache stores them


# ------------------------------------------------------------------------------------------------------------------------------ B's own loss
def node_targets(table):
    """the dense maps of B''s node loss (NODE_KEYS, float32 numpy) from the GT table of a frame on the 2 m lattice: `qtargets.grid_targets` with no points (the tier maps are not needed, the node loss reads exists, flags, offset, dirv, sstart, uout, uin)"""
    M = len(table["pos"])
    g = QT.grid_targets(dict(table, pts=np.zeros((0, 2)), tier=np.zeros(M, np.uint8), vis=np.ones(M, bool)))
    return {k: g[k] for k in NODE_KEYS}


def b_loss(no, targets):
    """B''s own node loss, the plain dense one of its training (`loss_qgraph.node_loss`, the weight is the caller's): no = node-stage outputs, targets = the `targets` of `collate` -> (loss, info)"""
    return LQG.node_loss(no, targets)


# ------------------------------------------------------------------------------------------------------------------------------ frames
def check_rows(split, rows):
    """the TEST guard: the rows of VAL are exactly `data/splits.json['val']`, those of TRAIN are distinct (the feature files also hold the TEST rows)"""
    assert len(set(rows)) == len(rows), "a row twice in the cache"
    if split == "val":
        with open(os.path.join(ROOT, "data", "splits.json")) as fh:
            sv = [int(r) for r in json.load(fh)["val"]]
        assert [int(r) for r in rows] == sv, "the rows of the VAL cache are not data/splits.json['val'] (the feature files also hold TEST rows)"


def check_order(cache_dir, rows, tokens):
    """the frames (rows, tokens) are those of the cache directory's frames.json, in its order"""
    with open(os.path.join(cache_dir, "frames.json")) as fh:
        fr = json.load(fh)
    assert [int(r) for r in fr["rows"]] == [int(r) for r in rows] and list(fr["tokens"]) == list(tokens), f"{cache_dir}: its frames are not the frames of this data (rows, tokens), in this order"


def check_run_sha(cache_dir, run):
    """the cache directory was made from this very checkpoint: its meta.json `run_sha` (`lane_tokcache`: the first 16 hex digits of the sha256 of the run's model.pt) is that of `run`/model.pt (a path relative to the project folder) -> the sha"""
    with open(os.path.join(cache_dir, "meta.json")) as fh:
        want = json.load(fh)["run_sha"]
    got = LC._sha(os.path.join(BP.resolve(run), "model.pt"))
    assert got == want, f"{run}/model.pt (sha {got}) is not the checkpoint that the cache {cache_dir} was made from (run_sha {want})"
    return got


class LiveData(torch.utils.data.Dataset):
    """the frames of `lane_tok_data.TokData(split, res=2.0, feats=True)` in the order of the E365 cache of B' seed `seed` (asserted against its frames.json); the item is x, token, row, table. `td` (and `cache_dir`, the directory
    whose frames.json is checked instead of the cache of `seed`) are injected by the selftest only"""

    def __init__(self, split="train", seed=0, ident="e292_a", td=None, cache_dir=None):
        injected = td is not None
        if not injected:
            import lane_tok_data as LT
            td = LT.TokData(split, res=2.0, feats=True)
        self.td, self.split, self.seed = td, split, int(seed)
        self.rows, self.tokens = td.rows, td.tokens
        check_rows(split, self.rows)                                        # the TEST guard: VAL = data/splits.json['val'] exactly
        if cache_dir or not injected:
            check_order(cache_dir or LP.cache_dir(split, seed, ident), self.rows, self.tokens)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        o = self.td[i]
        return dict(x=o["x"], token=o["token"], row=o["row"], table={k: v for k, v in o.items() if k not in ("x", "token", "row")})


def collate(b):
    tg = [node_targets(o["table"]) for o in b]
    return dict(x=torch.stack([o["x"] for o in b]), table=[o["table"] for o in b], token=[o["token"] for o in b], row=torch.tensor([o["row"] for o in b]),
                targets={k: torch.from_numpy(np.stack([t[k] for t in tg])) for k in NODE_KEYS})


def loader(split="train", seed=0, bs=8, workers=4, shuffle=None, ident="e292_a"):
    shuffle = split == "train" if shuffle is None else shuffle
    ds = LiveData(split, seed, ident)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)


# ------------------------------------------------------------------------------------------------------------------------------ live tokens and labels
class _Round16(torch.autograd.Function):
    """the float16 rounding of `tok` as the cache stores it, the gradient passed straight through (a plain .half().float() would round the gradient to float16)"""

    @staticmethod
    def forward(ctx, x):
        return x.half().float()

    @staticmethod
    def backward(ctx, g):
        return g


def live_tokens(no, K):
    """`lane_tokcache.select_tokens` of the node-stage outputs `no` (the K best cells by e, a stable argsort: an index choice, no gradient through it; every gather keeps the graph) with `tok` through `_Round16` -> dict of tensors (B, K, ...)"""
    tk = LC.select_tokens(no, None, K)
    tok = no["tok"].flatten(2).gather(2, tk["cell"].long()[:, None, :].expand(-1, no["tok"].shape[1], -1)).transpose(1, 2)
    return dict(tk, tok=_Round16.apply(tok))


class LabelOverflow(RuntimeError):
    """a frame whose live tokens have more target pairs than `lane_tokcache.PMAX` (4096: the cache's own maximum is 646 on TRAIN, 536 on VAL, but tokens that move under the head's loss may grow near-duplicates): `label_frame` asserts, this names it. `.token` = the
    frame, `.pairs` = its pair count. The trainer decides: skip the step and count it, abort after N"""

    def __init__(self, token, pairs):
        super().__init__(f"frame {token}: {pairs} target pairs exceed PMAX {LC.PMAX}")
        self.token, self.pairs = token, pairs


def label_batch(tk, tables, token, row, K, token_drop=0.0, seed=0, gate=True):
    """the label side of a live batch from the DETACHED tokens `tk` (numpy: `label_frame`, `densify`) and the GT tables of the frames -> (the dict `lane_ptok_data.collate` makes of the items: valid, ytype, y, ylong, yscale, gt_out, token, row, tokens_np, label;
    the label dicts of `lane_tokcache.label_frame`; the seconds of the per-frame CPU loop). The gate `allowed` is that of `jtok.pair_geometry` on the live tokens; it only feeds statistics (`ypair_gate`, `n_reach` of the label dicts): gate=False passes None
    (the pair geometry, its device sync and its host copy are skipped; y, ylong, yscale, valid, ytype and the rest are the same). A frame over PMAX target pairs raises `LabelOverflow`"""
    allowed = None
    if gate:
        with torch.no_grad():
            allowed = JT.pair_geometry(tk["pos"][:, :K].detach(), tk["dirv"][:, :K].detach(), tk["valid"][:, :K])[1].cpu().numpy()
    tn = {k: tk[k].detach().cpu().numpy() for k in ("cell", "valid", "pos", "dirv", "e_p", "flags_p", "s_p")}
    items, labs, t0 = [], [], time.perf_counter()
    for b in range(len(tables)):
        try:
            lab = LC.label_frame({k: tn[k][b] for k in ("pos", "dirv", "valid", "e_p")}, tables[b], K=K, allowed=None if allowed is None else allowed[b])
        except AssertionError as e:
            m = re.match(r"(\d+) target pairs exceed PMAX", str(e))
            if m is None:
                raise
            raise LabelOverflow(token[b], int(m.group(1))) from e
        keep = np.ones(K, bool)
        if token_drop > 0:
            from jtok_eval import keep_nodes
            keep = keep_nodes(token[b], K, token_drop, seed)                  # as PTokData: seeded by (seed, the frame's token)
        y, _, ys = LP.densify(lab["ypair"], lab["ypair_long"], lab["ypair_scale"], K, keep if token_drop > 0 else None)
        pos = tn["pos"][b].astype(np.float64)
        valid = tn["valid"][b] & keep
        items.append(dict(valid=valid, ytype=np.where(keep, lab["gt_type"], 0).astype(np.int64), y=y, ylong=LP.chord_long(y, pos), yscale=ys,
                          gt_out=int(lab["gt_out"]), token=token[b], row=int(row[b]),
                          tokens_np=dict(pos=pos, dirv=tn["dirv"][b].astype(np.float64), e_p=tn["e_p"][b].astype(np.float64), flags_p=tn["flags_p"][b].astype(np.float64), s_p=tn["s_p"][b].astype(np.float64),
                                         cell=tn["cell"][b].astype(np.int64), valid=valid.copy()),
                          label=dict(gt_id=lab["gt_id"], gd=lab["gd"], gt_flags=lab["gt_flags"], gt_aux=lab["gt_aux"])))
        labs.append(lab)
    return LP.collate(items), labs, time.perf_counter() - t0


def head_inputs(tk):
    """the keys the token head reads (`jtoken.INPUTS`: GRAPH_KEYS + valid) of the live tokens `tk`, graph kept: the head's forward reads nothing else, so a trainer can queue `head(head_inputs(tk))` before the label step (token_drop 0: the valid of a drop is the labels')"""
    return {k: tk[k] for k in GRAPH_KEYS + ("valid",)}


def live_batch(tk, tables, token, row, K, token_drop=0.0, seed=0, gate=True):
    """the batch of `lane_ptok_data` from the live tokens `tk` (`live_tokens`) of a live B': the token inputs with the graph, the labels as constants (module docstring)"""
    lb, labs, secs = label_batch(tk, tables, token, row, K, token_drop, seed, gate)
    batch = head_inputs(tk)
    batch.update({k: v.to(tk["pos"].device) if torch.is_tensor(v) else v for k, v in lb.items()})
    return dict(batch, lab=labs, label_s=secs)


class LiveSource(nn.Module):
    """B' as a live token source. `run`: a run folder (relative to the project folder; `qpredict.load_model`, on the CPU: move the module with .to) or a `QGraph`. Everything is frozen except `model.nodes.*` (nothing when `train_nodes` is False) and,
    with `train_pyr` (E393, needs `train_nodes`), `model.pyr.*`; the pyramid is always in eval mode (GroupNorm only: the mode changes nothing) and under no_grad unless `train_pyr` (`train()` of this module does not change that), the node head follows
    the module's mode. K = the tokens per frame (512 as the cache). `node_params` / `pyr_params` = the trained tensors of each part (`trainable` = both)"""

    def __init__(self, run, K=512, train_nodes=True, train_pyr=False):
        super().__init__()
        assert train_nodes or not train_pyr, "train_pyr needs train_nodes (the pyramid's gradient reaches it through the node head)"
        self.model = run if isinstance(run, nn.Module) else QP.load_model(BP.resolve(run), "cpu")
        self.K, self.train_pyr = K, bool(train_pyr)
        for p in self.model.parameters():
            p.requires_grad_(False)
        if train_nodes:
            for p in self.model.nodes.parameters():
                p.requires_grad_(True)
        if self.train_pyr:
            for p in self.model.pyr.parameters():
                p.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        self.model.nodes.train(mode)
        return self

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def node_params(self):
        return [p for p in self.model.nodes.parameters() if p.requires_grad]

    def pyr_params(self):
        return [p for p in self.model.pyr.parameters() if p.requires_grad]

    def tokens(self, x):
        """pyramid + node stage + `live_tokens` -> (no, tk): the node-stage outputs and the live tokens, all with the graph of the node head (and of the pyramid with `train_pyr`). Device work only (asynchronous on a GPU: nothing is read back to the host)"""
        dev = next(self.model.parameters()).device
        with torch.autocast(device_type=dev.type, enabled=False):         # float32, as the cache was made, also inside the trainer's autocast
            with contextlib.nullcontext() if self.train_pyr else torch.no_grad():
                P = self.model.pyr(x.to(dev).float())
            no = self.model.node_stage(P)
            return no, live_tokens(no, self.K)

    def labelled(self, tk, tables, token, row, token_drop=0.0, seed=0, gate=True):
        """the batch of `lane_ptok_data` from the tokens of `tokens` and the GT tables: the label step, which reads the tokens back to the host (a sync) and runs on the CPU (`label_batch`, `LabelOverflow`); gate=False skips the pair gate"""
        with torch.autocast(device_type=tk["pos"].device.type, enabled=False):
            return live_batch(tk, tables, token, row, self.K, token_drop, seed, gate)

    def forward(self, x, tables, token, row, token_drop=0.0, seed=0, gate=True):
        no, tk = self.tokens(x)
        return dict(no=no, batch=self.labelled(tk, tables, token, row, token_drop, seed, gate))


# ------------------------------------------------------------------------------------------------------------------------------ check: the frames and the node targets
def check(split="val", seed=0, ident="e292_a", n=20, log=print):
    """the guards and the order of the cache (LiveData asserts them), the cache's checkpoint (`check_run_sha`), then `node_targets(table)` against the maps of the `lane_gt_data.QData` item of the same frame (the loader B' was trained with), bit for bit, on n evenly
    spaced frames -> dict"""
    ds = LiveData(split, seed, ident)
    check_run_sha(LP.cache_dir(split, seed, ident), f"runs/{ident}_s{seed}")
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    bad = {k: 0 for k in NODE_KEYS}
    for i in idx:
        want, got = ds.td.qd[int(i)], node_targets(ds.td.table(int(i)))
        assert want["token"] == ds.tokens[i] and int(want["row"]) == ds.rows[i], f"frame {i}: QData is on another frame"
        for k in NODE_KEYS:
            bad[k] += int(not (want[k].numpy().dtype == got[k].dtype and np.array_equal(want[k].numpy(), got[k])))
    res = dict(split=split, seed=seed, frames=len(ds), checked=len(idx), differing_frames=bad, ok=not any(bad.values()))
    log(f"{split}: {len(ds)} frames = the rows / tokens of the cache of B' seed {seed} in its order, the cache was made from runs/{ident}_s{seed}/model.pt" + (", VAL = data/splits.json['val']" if split == "val" else ""))
    log(f"node targets vs QData on {len(idx)} frames: " + ", ".join(f"{k} {'equal' if not v else str(v) + ' DIFFER'}" for k, v in bad.items()))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ fixture: the live batch against the cache
def compare_frame(a, w):
    """two versions of one frame (numpy dicts: the GRAPH_KEYS (K, ...), valid, ytype (K,), y, ylong, yscale (K, K, ...), gt_out, cell (K,) = the flat cells of the tokens, label = dict gt_id, gd, gt_flags, gt_aux (K, ...), tnp = the `tokens_np` fields NP_KEYS and
    valid (K, ...) that the read-out writer reads) -> dict. The tokens are matched by their CELL (a float-noise swap of two neighbours in the ranking or at the budget's edge must not misalign the fields); same_order = the K cells are the same list; common /
    only_a / only_w = the cells in both lists / in one only; floats = the largest absolute difference per field over the common cells (NaN = NaN counts as equal), ints = the number of differing entries (the pair fields over pairs of common cells)"""
    ia, iw = {int(c): k for k, c in enumerate(a["cell"])}, {int(c): k for k, c in enumerate(w["cell"])}
    com = sorted(set(ia) & set(iw))
    ja, jw = np.array([ia[c] for c in com], int), np.array([iw[c] for c in com], int)
    tok = lambda d, k, j: np.asarray(d[k])[j]  # noqa: E731
    pair = lambda d, k, j: np.asarray(d[k])[np.ix_(j, j)]  # noqa: E731
    gap = lambda x, y: float(np.where(np.isnan(x) & np.isnan(y), 0.0, np.where(np.isnan(x) | np.isnan(y), np.inf, np.abs(x.astype(np.float64) - y.astype(np.float64)))).max(initial=0.0))  # noqa: E731   (one NaN: inf)
    floats = {k: gap(tok(a, k, ja), tok(w, k, jw)) for k in GRAPH_KEYS}
    floats.update(yscale=gap(pair(a, "yscale", ja), pair(w, "yscale", jw)), gd=gap(tok(a["label"], "gd", ja), tok(w["label"], "gd", jw)))
    floats.update({"np_" + k: gap(tok(a["tnp"], k, ja), tok(w["tnp"], k, jw)) for k in NP_KEYS})
    ints = {k: int((tok(a, k, ja) != tok(w, k, jw)).sum()) for k in ("valid", "ytype")}
    ints.update({k: int((pair(a, k, ja) != pair(w, k, jw)).sum()) for k in ("y", "ylong")})
    ints.update(label=sum(int((tok(a["label"], k, ja) != tok(w["label"], k, jw)).sum()) for k in ("gt_id", "gt_flags", "gt_aux")), gt_out=int(a["gt_out"] != w["gt_out"]), np_valid=int((tok(a["tnp"], "valid", ja) != tok(w["tnp"], "valid", jw)).sum()))
    return dict(same_order=bool(np.array_equal(a["cell"], w["cell"])), common=len(com), only_a=len(ia) - len(com), only_w=len(iw) - len(com), floats=floats, ints=ints)


def _vs_cache(got, pt, idx):
    """compare_frame of every frame j of the batch `got` (token fields, labels, `tokens_np`, `label`, `lab`) with the cache item pt[idx[j]] -> (the results, the number of frames with the same cell list whose label counts (`lab`) differ from the cache's)"""
    cpu = lambda t: t.detach().cpu().numpy()  # noqa: E731
    out, stats = [], 0
    for j, i in enumerate(idx):
        live = {k: cpu(got[k])[j] for k in GRAPH_KEYS + ("valid", "ytype", "y", "ylong", "yscale")}
        live.update(gt_out=int(got["gt_out"][j]), cell=got["tokens_np"][j]["cell"], label={k: cpu(v)[j] for k, v in got["label"].items()}, tnp={k: got["tokens_np"][j][k] for k in NP_KEYS + ("valid",)})
        o = pt[i]
        out.append(compare_frame(live, dict({k: o[k] for k in live if k not in ("gt_out", "cell", "label", "tnp")}, gt_out=o["gt_out"], cell=o["tokens_np"]["cell"], label=o["label"], tnp=o["tokens_np"])))
        stats += int(out[-1]["same_order"] and any(got["lab"][j][k] != pt.c.f[k][i] for k in STATS))
    return out, stats


def _summary(cmp, stats):
    agg = dict(frames=len(cmp), same_order=sum(c["same_order"] for c in cmp), common_min=min(c["common"] for c in cmp), only_live_max=max(c["only_a"] for c in cmp), only_cache_max=max(c["only_w"] for c in cmp),
               floats={k: float(np.max([c["floats"][k] for c in cmp])) for k in cmp[0]["floats"]}, ints={k: sum(c["ints"][k] for c in cmp) for k in cmp[0]["ints"]}, frames_with_other_label_counts=stats)
    agg["exact"] = agg["same_order"] == len(cmp) and not any(agg["floats"].values()) and not any(agg["ints"].values()) and not stats
    return agg


def fixture(split="val", n=40, seeds=(0, 1), gpu=False, ident="e292_a", bs=4, log=print):
    """the first n frames of the split, B' of every seed in `seeds`, against the cache (`PTokData`) by `compare_frame`, in batches of `bs` (4 = the batches of the cache build), frozen B' in eval mode under no_grad. Two parts per seed: `labels` = the label step on
    the CACHED tokens (no model: must be exactly the cache's labels on every machine and every bs) and `live` = the whole live path (on the GPU at bs 4 it must be exactly the cache; on the CPU the forward of B' differs from the GPU-made cache by more than float
    noise: the numbers say how much). exact = the K cells are the same list on every frame and every difference is zero -> dict (`ok` = both parts exact for every seed; `env` = the versions, the GPU, the cache's checkpoint sha, bs, n; `parity_at_cache_bs` = bs 4)"""
    assert n > 0 and len(seeds) > 0 and bs > 0, f"nothing to compare: n {n}, seeds {list(seeds)}, bs {bs}"
    assert n % bs == 0 or not gpu, f"--gpu: n {n} is not a multiple of bs {bs}: a short last batch is not a batch of the cache build"
    dev = "cuda" if gpu else "cpu"
    assert not gpu or torch.cuda.is_available(), "--gpu: no GPU visible"
    res, shas = {}, {}
    for s in seeds:
        shas[s] = check_run_sha(LP.cache_dir(split, s, ident), f"runs/{ident}_s{s}")
        src = LiveSource(f"runs/{ident}_s{s}", 512, train_nodes=False).to(dev).eval()
        ds, pt = LiveData(split, s, ident), LP.PTokData(split, s, ident)
        assert n <= len(ds) and pt.rows[:n] == ds.rows[:n] and pt.tokens[:n] == ds.tokens[:n] and pt.K == src.K, "the cache is not the frames / K of the live source"
        cmp = dict(labels=[], live=[])
        stats, secs = dict(labels=0, live=0), []
        for i0 in range(0, n, bs):
            idx = list(range(i0, min(i0 + bs, n)))
            b = collate([ds[i] for i in idx])
            cached = {k: torch.from_numpy(np.stack([np.array(pt.c.f[k][i]) for i in idx])) for k in ("cell", "valid", "pos", "dirv", "e_p", "flags_p", "s_p")}
            lb, labs, _ = label_batch(cached, b["table"], b["token"], b["row"], pt.K)                     # the label step on the cache's own tokens
            want = LP.collate([pt[i] for i in idx])
            r, st = _vs_cache(dict(want, **lb, lab=labs), pt, idx)
            cmp["labels"] += r
            stats["labels"] += st
            with torch.no_grad():
                got = src(b["x"], b["table"], b["token"], b["row"])["batch"]
            secs.append(got["label_s"])
            r, st = _vs_cache(got, pt, idx)
            cmp["live"] += r
            stats["live"] += st
        res[s] = {k: _summary(cmp[k], stats[k]) for k in cmp}
        res[s]["live"]["label_s_per_batch_median"] = float(np.median(secs))
        for k, a in res[s].items():
            log(f"seed {s} ({dev}, bs {bs}), {k}: {n} frames, the K cells are the same list on {a['same_order']}; cells in both lists >= {a['common_min']}, in one only <= {a['only_live_max']} (live) / {a['only_cache_max']} (cache); exact: {a['exact']}")
            log("   largest |diff| over the common cells: " + ", ".join(f"{c} {v:.2e}" for c, v in a["floats"].items()))
            log("   differing entries: " + ", ".join(f"{c} {v}" for c, v in a["ints"].items()) + f"; frames with other label counts {a['frames_with_other_label_counts']}"
                + (f"; label loop {a['label_s_per_batch_median'] * 1000:.0f} ms per batch of {bs}" if k == "live" else ""))
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0) if gpu else None, run_sha={str(s): v for s, v in shas.items()}, bs=bs, n=n, split=split, device=dev)
    log("environment: " + json.dumps(env) + ("" if bs == 4 else f"; bs {bs} is not the batch size of the cache build (4): a difference here may be the batch-size dependence of cuDNN, parity at bs 4 is the claim of the acceptance rule"))
    return dict(res, ok=all(a["exact"] for r in res.values() for a in r.values()), labels_ok=all(r["labels"]["exact"] for r in res.values()), device=dev, env=env, parity_at_cache_bs=bs == 4)


def gradient_check(src, head, b):
    """the gradient check of the card on one batch `b` of `collate` (x, table, token, row, targets); `src` a trainable `LiveSource`, `head` a `jtoken.JTokenHead` (it needs its `live_inputs` option). (1) With `live_inputs` False (the champion's default: `jtoken._const` cuts the
    inputs at the batch) the head's loss gives the node head no gradient. (2) With `live_inputs` True it reaches EVERY node-head parameter and no frozen one. (3) B''s own loss reaches the node head on its own. (4) An AdamW step on both groups leaves the pyramid (weights
    and buffers) as it was. -> dict(head_loss, b_loss, params, grad_norm_from_head, grad_norm_from_b: the norms of the node head's GRADIENT from each loss; the card's secondary number is the share of the parameter UPDATE norm, which Adam rescales per
    parameter: not the same). The step trains `src` and `head` and leaves `head.live_inputs` True: pass throw-away copies"""
    import jtoken as JK
    import loss_jtoken as LJK  # noqa: E402
    assert hasattr(head, "live_inputs"), "the head has no `live_inputs` option (jtoken.JTokenHead)"
    args = (b["x"], b["table"], b["token"], b["row"])
    opt = torch.optim.AdamW([dict(params=list(head.parameters()), lr=1e-3), dict(params=src.trainable(), lr=1e-3)])
    states = {n: v.clone() for n, v in src.model.pyr.state_dict().items()}
    norm = lambda: float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in src.trainable())))  # noqa: E731

    def head_step(live):
        opt.zero_grad(set_to_none=True)
        out = src(*args)
        assert int(out["batch"]["y"].sum()) > 0, "the batch gives the head positive pairs"
        head.live_inputs = live
        loss, info = LJK.jtoken_loss(head(out["batch"]), out["batch"])
        loss.backward()
        return loss, info

    head_step(live=False)
    assert all(p.grad is None for p in src.trainable()), "a head that cuts its inputs (`live_inputs` False) gives B' no gradient"
    loss, info = head_step(live=True)
    dead = [n for n, p in src.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    assert not dead and info["pos_pairs"] > 0, f"the head's loss does not reach the node head's {dead}"
    assert not [n for n, p in src.named_parameters() if not n.startswith("model.nodes.") and (p.requires_grad or p.grad is not None)], "the frozen stages got a gradient"
    g_head = norm()
    opt.zero_grad(set_to_none=True)
    lb, ib = b_loss(src(*args)["no"], b["targets"])                                                              # B''s own loss reaches the node head on its own
    lb.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in src.trainable()) and sorted(ib) == ["dir", "e", "flags", "off", "s", "u"]
    g_b = norm()
    opt.step()
    assert all(torch.equal(states[n], v) for n, v in src.model.pyr.state_dict().items()), "the pyramid (weights and statistics) changed in a training step"
    return dict(head_loss=float(loss), b_loss=float(lb), params=len(src.trainable()), grad_norm_from_head=g_head, grad_norm_from_b=g_b)


def grad(split="val", seed=0, n=2, head=None, ident="e292_a", log=print):
    """`gradient_check` on the first n real frames: the real B' of `seed` on the CPU, the head = a fresh `JTokenHead` or the run `runs/<head>` (the champion's stage-1 head `e373_cb_s0`). It reports the head's share of the GRADIENT norm on the node head at step 0 on a few frames: a
    guide to the size of the head's pull, not the card's secondary number (the share of the parameter UPDATE norm: Adam rescales every parameter's step, so the two differ)"""
    import jtoken as JK
    import loss_jtoken as LJK  # noqa: E402
    ds = LiveData(split, seed, ident)
    if head:
        sys.path.insert(0, os.path.join(ROOT, "training", "train"))
        import jtoken_train as JTR
        hd = JTR.load_model(head, "cpu")[0]
    else:
        hd = JK.JTokenHead()
    r = gradient_check(LiveSource(f"runs/{ident}_s{seed}", 512, True), hd, collate([ds[i] for i in range(n)]))
    share = r["grad_norm_from_head"] / (r["grad_norm_from_head"] + r["grad_norm_from_b"])
    log(f"{split} {n} frames, B' seed {seed}, head {head or 'fresh'}: head loss {r['head_loss']:.3f}, B's node loss {r['b_loss']:.3f}; all {r['params']} node-head tensors get a gradient from the head's loss, no frozen stage does, the pyramid is unchanged by a step; "
        f"gradient norm on the node head: from the head's loss {r['grad_norm_from_head']:.3f}, from B's loss {r['grad_norm_from_b']:.3f} (the head's share {share:.3f})")
    return dict(r, share=share)


# ------------------------------------------------------------------------------------------------------------------------------ selftest
class _Toy:
    """the stand-in of `TokData` for the selftest: frame i has one GT node at (-9, 2 i + 0.5) and x = i everywhere"""

    def __init__(self, rows, tokens):
        self.rows, self.tokens = list(rows), list(tokens)

    def __getitem__(self, i):
        return dict(toy_table([[-9.0, 2.0 * i + 0.5]], [], []), x=torch.full((39, 4, 4), float(i)).half(), token=self.tokens[i], row=self.rows[i])


def refuses(fn, text):
    """fn must fail with an AssertionError whose message holds `text`; another exception is reported as such, a call that is accepted raises a RuntimeError (which no test catches)"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    except Exception as e:
        raise AssertionError(f"raised {type(e).__name__} ({e}) instead of refusing ({text})") from e
    raise RuntimeError(f"accepted although it must be refused ({text})")


def toy_table(pos, links, longs, flags=None):
    """a GT table (the E357 keys) of nodes at `pos`, all heading (1, 0); links = [(src, dst)], longs = their class"""
    pos, M = np.asarray(pos, float).reshape(-1, 2), len(pos)
    return dict(pos=pos, dirv=np.tile([1.0, 0.0], (M, 1)), cell=np.clip(np.floor((pos + QT.HALF) / QT.RES), 0, QT.N - 1).astype(np.int16), flags=np.zeros((M, 8), bool) if flags is None else np.asarray(flags, bool),
                piece=np.zeros(M, np.int32), sstart=np.eye(1, M, 0, dtype=bool)[0] if M else np.zeros(0, bool), src=np.array([a for a, _ in links], np.int32), dst=np.array([b for _, b in links], np.int32),
                long=np.array(longs, bool), scale=np.full((len(links), 2), np.nan), conn=np.zeros(len(links), bool))


def toy_no(B=2, seed=0, rank_cells=((20, 27), (21, 27), (22, 27)), grad=True):
    """node-stage outputs of B frames in which the cells `rank_cells` have the largest existence (logits 5, 4, 3 in that order, all others -8), no offset, heading (1, 0): the tokens of a lane along x, the first at (-9, 5) m. KNOWN values (the
    selftest checks them by hand): the section-start logit s is 2, -1, 0 at those cells and 0 elsewhere, the split flag (channel 2) is 1, 0, -1 there and every flag logit is 0 elsewhere; e_raw, uout, uin and tok are random. The leaves carry a gradient when `grad`"""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    e, s, flags = torch.full((B, 50, 50), -8.0), torch.zeros(B, 50, 50), torch.zeros(B, 8, 50, 50)
    for k, (i, j) in enumerate(rank_cells):
        e[:, i, j], s[:, i, j], flags[:, 2, i, j] = 5.0 - k, (2.0, -1.0, 0.0)[k % 3], (1.0, 0.0, -1.0)[k % 3]
    c = -QT.HALF + (torch.arange(50.0) + 0.5) * QT.RES
    dr = torch.zeros(B, 2, 50, 50)
    dr[:, 0] = 1.0 + 0.1 * r(B, 50, 50).abs()
    no = dict(e=e, e_raw=e + 0.5 * r(B, 50, 50), pos=torch.stack([c.view(50, 1).expand(50, 50), c.view(1, 50).expand(50, 50)])[None].repeat(B, 1, 1, 1), off=torch.zeros(B, 2, 50, 50), dir_raw=dr,
              flags=flags, s=s, uout=r(B, 50, 50), uin=r(B, 50, 50), tok=r(B, 6, 50, 50))
    return {k: v.clone().requires_grad_(grad) for k, v in no.items()}


def _t_targets():
    fl = np.zeros((3, 8), bool)
    fl[1, 2] = fl[2, 0] = True                                                                                  # split, stop
    t = toy_table([[-9.3, 4.2], [-7.0, 5.0], [10.0, 5.0]], [(0, 1), (1, 2)], [False, True], fl)                  # node 0 in cell (20, 27), the second link is long (chord 17 m)
    g = node_targets(t)
    assert sorted(g) == sorted(NODE_KEYS) and all(g[k].dtype == np.float32 for k in NODE_KEYS)
    assert g["exists"].shape == (50, 50) and g["flags"].shape == (8, 50, 50) and g["offset"].shape == (2, 50, 50) and g["dirv"].shape == (2, 50, 50)
    assert np.argwhere(g["exists"] > 0).tolist() == [[20, 27], [21, 27], [30, 27]], "the cell of a node is (floor((x + 50) / 2), floor((y + 50) / 2))"
    assert np.allclose(g["offset"][:, 20, 27], [-0.3, -0.8], atol=1e-6) and np.allclose(g["offset"][:, 21, 27], [0.0, 0.0]), "the offset is pos - the cell centre (x first)"
    assert g["flags"][2, 21, 27] == 1 and g["flags"][0, 30, 27] == 1 and g["flags"].sum() == 2, "split is channel 2, stop channel 0 (qtargets.FLAGS)"
    assert g["sstart"][20, 27] == 1 and g["sstart"].sum() == 1 and np.allclose(g["dirv"][:, 20, 27], [1.0, 0.0]) and g["dirv"][0].sum() == 3
    assert g["uout"][21, 27] == 1 and g["uin"][30, 27] == 1 and g["uout"].sum() == 1 and g["uin"].sum() == 1, "a long link marks its source (uout) and its destination (uin)"
    assert all(float(np.abs(v).sum()) == 0 for v in node_targets(toy_table(np.zeros((0, 2)), [], [])).values()), "a frame without GT nodes has empty maps"
    # B's loss: a perfect prediction costs almost nothing, a wrong one costs more
    tg = {k: torch.from_numpy(v)[None] for k, v in g.items()}
    ex = tg["exists"]
    no = dict(e=ex * 24 - 12, flags=tg["flags"] * 24 - 12, off=tg["offset"], dir_raw=tg["dirv"] + (1 - ex)[:, None] * torch.tensor([0.0, 1.0]).view(1, 2, 1, 1), s=tg["sstart"] * 24 - 12, uout=tg["uout"] * 24 - 12, uin=tg["uin"] * 24 - 12)
    good, info = b_loss(no, tg)
    assert float(good) < 0.01 and sorted(info) == ["dir", "e", "flags", "off", "s", "u"], (float(good), info)
    assert float(b_loss(dict(no, e=-no["e"]), tg)[0]) > 1.0 and float(b_loss(no, {k: v.transpose(-1, -2) for k, v in tg.items()})[0]) > 0.1


def _t_tokens():
    K = 16
    no = toy_no()
    tk = live_tokens(no, K)
    ref = LC.select_tokens({k: v.detach() for k, v in no.items()}, None, K)
    assert sorted(tk) == sorted(ref) and all(torch.equal(tk[k], ref[k]) for k in ref if k != "tok"), "the live selection is `select_tokens`, bit for bit"
    assert torch.equal(tk["tok"], ref["tok"].float()) and tk["tok"].dtype == torch.float32, "tok = the float16 rounding of the cache, as float32"
    assert tk["cell"][0, :3].tolist() == [20 * 50 + 27, 21 * 50 + 27, 22 * 50 + 27] and tk["valid"][0].tolist() == [True] * 3 + [False] * 13
    assert sorted(GRAPH_KEYS) == ["dirv", "e", "e_raw", "flags", "pos", "s", "tok", "uin", "uout"] and sorted(head_inputs(tk)) == sorted(GRAPH_KEYS + ("valid",)), "the head's inputs: tok, pos, dirv, e, e_raw, s, uout, uin, flags + valid"
    batch = live_batch(tk, [toy_table([[-9.0, 5.0]], [], [])] * 2, ["a", "b"], [1, 2], K)
    assert sorted(k for k in batch if k not in ("lab", "label_s")) == sorted(GRAPH_KEYS + ("valid", "ytype", "y", "ylong", "yscale", "gt_out", "token", "row", "tokens_np", "label"))
    for k in GRAPH_KEYS:
        assert batch[k].requires_grad and torch.equal(batch[k], ref[k].float() if k == "tok" else ref[k]), k
    assert not any(torch.is_tensor(batch[k]) and batch[k].requires_grad for k in ("valid", "ytype", "y", "ylong", "yscale", "gt_out", "row")), "the labels are constants"
    # the graph reaches every map of B, at the K chosen cells only (the choice itself has no gradient); the gradient of tok is not rounded to float16 (1e-9 would underflow there)
    w = {k: 1.0 + torch.rand(batch[k].shape, generator=torch.Generator().manual_seed(1)) for k in GRAPH_KEYS}
    (sum((batch[k] * w[k]).sum() for k in GRAPH_KEYS if k != "tok") + (batch["tok"] * 1e-9).sum()).backward()
    for k_n in ("pos", "dir_raw", "e", "e_raw", "s", "uout", "uin", "flags", "tok"):
        gr = no[k_n].grad
        assert gr is not None and float(gr.abs().sum()) > 0, f"the graph does not reach no['{k_n}']"
        per_cell = gr.abs().flatten(2).amax(1) if gr.dim() == 4 else gr.abs().flatten(1)
        assert (per_cell > 0).sum(1).tolist() == [K, K], f"no['{k_n}']: the gradient is not at exactly the K = {K} chosen cells"
    assert abs(float(no["tok"].grad.abs().max()) / 1e-9 - 1) < 1e-6 and no["tok"].grad.dtype == torch.float32, "the gradient of tok must not be rounded to float16"


def _t_labels():
    B, K = 2, 16
    no = toy_no(B)
    tk = live_tokens(no, K)
    fl = np.zeros((3, 8), bool)
    fl[1, 0], fl[2, 2] = True, True                                                                             # node 1 is a stop, node 2 a split
    tabs = [toy_table([[-9.0, 5.3], [-7.0, 5.0], [-5.0, 4.6]], [(0, 1), (1, 2)], [False, False], fl), toy_table(np.zeros((0, 2)), [], [])]      # the GT nodes lie 0.3 / 0 / 0.4 m from the tokens
    out = live_batch(tk, tabs, ["f0", "f1"], [7, 8], K)
    assert out["y"].shape == (B, K, K) and out["y"].dtype == torch.uint8 and out["ylong"].shape == (B, K, K) and out["yscale"].shape == (B, K, K, 2) and out["ytype"].dtype == torch.int64 and out["row"].tolist() == [7, 8]
    lb = out["label"]
    assert lb["gt_id"][0].tolist() == [0, 1, 2] + [-1] * 13 and (lb["gt_id"][1] == -1).all(), "the three tokens take the three GT nodes in the order of their existence"
    assert np.allclose(lb["gd"][0][:3].numpy(), [0.3, 0.0, 0.4], atol=1e-6) and torch.isnan(lb["gd"][0][3:]).all() and torch.isnan(lb["gd"][1]).all(), "gd = the distance of a token to its GT node, NaN without one"
    assert lb["gt_flags"][0][1].tolist() == [True] + [False] * 7 and lb["gt_flags"][0][2].tolist() == [False, False, True] + [False] * 5 and not lb["gt_flags"][0][0].any() and not lb["gt_flags"][0][3:].any() and not lb["gt_flags"][1].any(), "the raw flags of the assigned node"
    assert lb["gt_aux"][0][0].tolist() == [True, False, False] and not lb["gt_aux"][0][1:].any() and not lb["gt_aux"][1].any(), "the section start of node 0; no long link"
    assert out["ytype"][0].tolist() == [1, 4, 2] + [0] * 13 and out["ytype"][1].sum() == 0, "lane point, stop = entry (4), split = fork (2); the padding and the empty frame are type 0"
    assert out["y"][0].nonzero().tolist() == [[0, 1], [1, 2]] and int(out["y"][1].sum()) == 0 and not out["ylong"].any() and torch.isnan(out["yscale"]).all()
    assert out["valid"][0].tolist() == [True] * 3 + [False] * 13 and out["gt_out"].tolist() == [0, 0] and out["token"] == ["f0", "f1"]
    assert out["lab"][0]["n_reach"] == 2 and out["lab"][0]["n_pairs"] == 2 and out["lab"][1]["n_gt"] == 0, "both pairs lie inside the gate of pair_geometry on the live tokens (forward along x)"
    tn = out["tokens_np"][0]
    assert tn["pos"].dtype == np.float64 and tn["cell"].dtype == np.int64 and tn["cell"][:3].tolist() == [1027, 1077, 1127] and np.allclose(tn["pos"][:3], [[-9, 5], [-7, 5], [-5, 5]])
    assert sorted(tn) == ["cell", "dirv", "e_p", "flags_p", "pos", "s_p", "valid"], "no GT field in tokens_np"
    # the values of tokens_np by hand (sigmoid of the toy logits: e 5, 4, 3 then -8; s 2, -1, 0 then 0; split flag 1, 0, -1 then 0; every other flag 0), the headings (1, 0), valid for the first three
    fp = np.full((K, 8), 0.5)
    fp[:3, 2] = [0.73105858, 0.5, 0.26894142]
    assert np.allclose(tn["e_p"][:3], [0.99330715, 0.98201379, 0.95257413], atol=1e-6) and np.allclose(tn["e_p"][3:], 3.3535013e-4, atol=1e-7), "e_p = sigmoid(e) of the whole map, gathered"
    assert np.allclose(tn["s_p"][:3], [0.88079708, 0.26894142, 0.5], atol=1e-6) and np.allclose(tn["s_p"][3:], 0.5, atol=1e-6) and np.allclose(tn["flags_p"], fp, atol=1e-6), "s_p and flags_p = their own sigmoids"
    assert np.allclose(tn["dirv"], [[1.0, 0.0]] * K, atol=1e-6) and tn["valid"].tolist() == [True] * 3 + [False] * 13 and tn["valid"].dtype == bool
    # the long chord class follows the TOKEN chord (12 m), not the table's class of the link; a chord of exactly 8 m is short (long = above 8 m)
    o2 = live_batch(live_tokens(toy_no(1, rank_cells=((20, 27), (26, 27))), K), [toy_table([[-9.0, 5.0], [3.0, 5.0]], [(0, 1)], [False])], ["g"], [0], K)
    assert o2["y"][0].nonzero().tolist() == [[0, 1]] and bool(o2["ylong"][0, 0, 1]) and int(o2["ylong"].sum()) == 1
    o4 = live_batch(live_tokens(toy_no(1, rank_cells=((20, 27), (24, 27))), K), [toy_table([[-9.0, 5.0], [-1.0, 5.0]], [(0, 1)], [False])], ["i"], [0], K)
    assert o4["y"][0].nonzero().tolist() == [[0, 1]] and not o4["ylong"].any(), "a chord of exactly 8 m is short"
    # a GT node that no token covers: the link into it is lost (gt_out counts it before the contraction)
    o3 = live_batch(live_tokens(toy_no(1), K), [toy_table([[-9.0, 5.0], [-7.0, 5.0], [-7.0, 40.0]], [(0, 1), (1, 2)], [False, True])], ["h"], [0], K)
    assert o3["gt_out"].tolist() == [1] and o3["lab"][0]["gt_out"] == 1 and o3["lab"][0]["n_links"] == 2 and o3["lab"][0]["n_links_c"] == 1
    # the pair gate only feeds statistics: without it y, ylong, yscale, valid, ytype and gt_out are the same
    og = live_batch(tk, tabs, ["f0", "f1"], [7, 8], K, gate=False)
    for k in ("valid", "ytype", "y", "ylong", "yscale", "gt_out"):
        assert np.array_equal(og[k].numpy(), out[k].numpy(), equal_nan=True), f"{k} differs without the gate"
    assert og["lab"][0]["n_reach"] == 0 and og["lab"][0]["n_pairs"] == 2 and not og["lab"][0]["ypair_gate"].any() and out["lab"][0]["ypair_gate"][:2].all(), "gate=False: no pair is counted as reachable, nothing else changes"
    # a frame over PMAX target pairs is named, with its token and its count (the trainer decides)
    pmax, LC.PMAX = LC.PMAX, 1
    try:
        live_batch(tk, [tabs[1], tabs[0]], ["fa", "fb"], [8, 7], K)                                              # the second frame has two pairs, the first none
        raise RuntimeError("accepted although it must be refused (more target pairs than PMAX)")
    except LabelOverflow as e:
        assert e.token == "fb" and e.pairs == 2 and isinstance(e, RuntimeError) and "exceed PMAX" in str(e), str(e)
    finally:
        LC.PMAX = pmax
    # the loader's token drop: the draw of `jtok_eval.keep_nodes` seeded by (seed, token); a removed token is not valid, has type none and no pair
    from jtok_eval import keep_nodes
    seed = next(s for s in range(1, 50) if not keep_nodes("f0", K, 0.5, s)[:3].all() and keep_nodes("f0", K, 0.5, s)[:3].any() and not np.array_equal(keep_nodes("f0", K, 0.5, s)[:3], keep_nodes("f0", K, 0.5, 0)[:3]))
    keep = torch.from_numpy(keep_nodes("f0", K, 0.5, seed))
    od = live_batch(tk, tabs[:1] * 2, ["f0", "f0"], [7, 7], K, token_drop=0.5, seed=seed)
    assert od["valid"][0].tolist() == (out["valid"][0] & keep).tolist() and od["valid"][0].tolist() != out["valid"][0].tolist() and od["tokens_np"][0]["valid"].tolist() == od["valid"][0].tolist()
    assert (od["ytype"][0][~keep] == 0).all() and not od["y"][0][~keep].any() and not od["y"][0][:, ~keep].any() and int(od["y"][0].sum()) < int(out["y"][0].sum())


def _t_data():
    import tempfile
    with open(os.path.join(ROOT, "data", "splits.json")) as fh:
        rows = json.load(fh)["val"]
    toks = [f"t{i}" for i in range(len(rows))]
    ds = LiveData("val", 0, td=_Toy(rows, toks))
    assert len(ds) == len(rows) and ds[5]["token"] == "t5" and ds[5]["row"] == rows[5] and float(ds[5]["x"].float().mean()) == 5.0 and float(ds[5]["table"]["pos"][0, 1]) == 10.5, "item i is frame i"
    b = collate([ds[3], ds[4]])
    assert b["x"].shape == (2, 39, 4, 4) and b["row"].tolist() == [rows[3], rows[4]] and b["token"] == ["t3", "t4"] and len(b["table"]) == 2 and sorted(b["targets"]) == sorted(NODE_KEYS)
    assert [float(t["pos"][0, 1]) for t in b["table"]] == [6.5, 8.5], "table[i] is the table of frame i"
    assert b["targets"]["exists"].shape == (2, 50, 50) and b["targets"]["exists"][0].nonzero().tolist() == [[20, 28]] and b["targets"]["exists"][1].nonzero().tolist() == [[20, 29]], "targets[i] belong to frame i"
    refuses(lambda: LiveData("val", 0, td=_Toy(rows[::-1], toks)), "VAL cache are not data/splits.json['val']")                   # the TEST guard
    refuses(lambda: LiveData("val", 0, td=_Toy(rows[:-1], toks[:-1])), "VAL cache are not data/splits.json['val']")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "frames.json"), "w") as fh:
            json.dump(dict(rows=[1, 2, 3], tokens=["a", "b", "c"]), fh)
        check_order(d, [1, 2, 3], ["a", "b", "c"])
        refuses(lambda: check_order(d, [1, 3, 2], ["a", "b", "c"]), "not the frames of this data")                  # the rows alone, the tokens alone
        refuses(lambda: check_order(d, [1, 2, 3], ["a", "c", "b"]), "not the frames of this data")
        refuses(lambda: check_order(d, [1, 2], ["a", "b"]), "not the frames of this data")
        with open(os.path.join(d, "frames.json"), "w") as fh:                                                      # LiveData itself runs the order check (here against an injected cache directory)
            json.dump(dict(rows=rows, tokens=toks), fh)
        LiveData("val", 0, td=_Toy(rows, toks), cache_dir=d)
        refuses(lambda: LiveData("val", 0, td=_Toy(rows, toks[::-1]), cache_dir=d), "not the frames of this data")
        os.makedirs(os.path.join(d, "run"))                                                                        # the checkpoint of the cache: sha256("abc") starts ba7816bf8f01cfea
        with open(os.path.join(d, "run", "model.pt"), "wb") as fh:
            fh.write(b"abc")
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(dict(run_sha="ba7816bf8f01cfea"), fh)
        assert check_run_sha(d, os.path.join(d, "run")) == "ba7816bf8f01cfea"
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(dict(run_sha="0123456789abcdef"), fh)
        refuses(lambda: check_run_sha(d, os.path.join(d, "run")), "is not the checkpoint that the cache")


def _t_compare():
    K = 6
    rng = np.random.default_rng(0)
    pair = lambda *s: rng.random(s).astype(np.float32)  # noqa: E731
    a = dict(tok=pair(K, 64), pos=pair(K, 2), dirv=pair(K, 2), e=pair(K), e_raw=pair(K), s=pair(K), uout=pair(K), uin=pair(K), flags=pair(K, 8), valid=np.arange(K) < 4, ytype=np.arange(K) % 3, y=(pair(K, K) > 0.7).astype(np.uint8),
             ylong=pair(K, K) > 0.8, yscale=np.where(pair(K, K, 2) > 0.5, pair(K, K, 2), np.nan).astype(np.float32), gt_out=1, cell=np.arange(K) * 7 + 3,
             label=dict(gt_id=np.arange(K, dtype=np.int32) - 2, gd=np.where(np.arange(K) > 1, pair(K), np.nan).astype(np.float32), gt_flags=pair(K, 8) > 0.5, gt_aux=pair(K, 3) > 0.5),
             tnp=dict(pos=pair(K, 2).astype(np.float64), dirv=pair(K, 2).astype(np.float64), e_p=pair(K).astype(np.float64), flags_p=pair(K, 8).astype(np.float64), s_p=pair(K).astype(np.float64), valid=np.arange(K) < 4))
    cp = lambda d: {k: (cp(v) if isinstance(v, dict) else np.copy(v)) for k, v in d.items()}  # noqa: E731
    c = compare_frame(a, cp(a))
    assert c["same_order"] and c["common"] == K and c["only_a"] == c["only_w"] == 0 and not any(c["floats"].values()) and not any(c["ints"].values()), "a frame equals its copy (NaN = NaN)"
    perm = np.array([3, 0, 5, 1, 4, 2])                                                                          # the same tokens in another order: aligned by cell, nothing differs
    w = dict(cp(a), cell=a["cell"][perm], label={k: v[perm] for k, v in a["label"].items()}, tnp={k: v[perm] for k, v in a["tnp"].items()})
    for k in ("tok", "pos", "dirv", "e", "e_raw", "s", "uout", "uin", "flags", "valid", "ytype"):
        w[k] = a[k][perm]
    for k in ("y", "ylong", "yscale"):
        w[k] = a[k][np.ix_(perm, perm)]
    c = compare_frame(a, w)
    assert not c["same_order"] and c["common"] == K and not any(c["floats"].values()) and not any(c["ints"].values()), "a reordering is not a difference"
    for k, edit, field, kind in (("tok", lambda d: d["tok"].__setitem__((2, 5), d["tok"][2, 5] + 0.5), "tok", "floats"), ("e", lambda d: d["e"].__setitem__(3, d["e"][3] + 1.0), "e", "floats"),
                                 ("y", lambda d: d["y"].__setitem__((1, 3), 1 - d["y"][1, 3]), "y", "ints"), ("ytype", lambda d: d["ytype"].__setitem__(4, 9), "ytype", "ints"),
                                 ("valid", lambda d: d["valid"].__setitem__(0, False), "valid", "ints"), ("ylong", lambda d: d["ylong"].__setitem__((2, 2), ~d["ylong"][2, 2]), "ylong", "ints"),
                                 ("gt_id", lambda d: d["label"]["gt_id"].__setitem__(1, 77), "label", "ints"), ("gt_out", lambda d: d.__setitem__("gt_out", 2), "gt_out", "ints"),
                                 ("yscale", lambda d: d["yscale"].__setitem__((0, 0, 0), np.nan if not np.isnan(d["yscale"][0, 0, 0]) else 0.5), "yscale", "floats"),
                                 ("np_e_p", lambda d: d["tnp"]["e_p"].__setitem__(2, d["tnp"]["e_p"][2] + 0.5), "np_e_p", "floats"), ("np_s_p", lambda d: d["tnp"]["s_p"].__setitem__(3, 0.123), "np_s_p", "floats"),
                                 ("np_flags_p", lambda d: d["tnp"]["flags_p"].__setitem__((1, 4), 0.99), "np_flags_p", "floats"), ("np_dirv", lambda d: d["tnp"]["dirv"].__setitem__((2, 0), 9.0), "np_dirv", "floats"),
                                 ("np_pos", lambda d: d["tnp"]["pos"].__setitem__((4, 1), 9.0), "np_pos", "floats"), ("np_valid", lambda d: d["tnp"]["valid"].__setitem__(1, False), "np_valid", "ints")):
        w = cp(a)
        edit(w)
        c = compare_frame(a, w)
        got = c[kind].get(field)
        assert got is not None and got != 0 and (kind == "floats" or got == 1) and sum(1 for f_ in c["floats"].values() if f_ != 0) + sum(1 for i_ in c["ints"].values() if i_ != 0) == 1, (k, c)
    w = cp(a)
    w["tok"][2, 5] += 0.25
    assert abs(compare_frame(a, w)["floats"]["tok"] - 0.25) < 1e-6
    w = cp(a)                                                                                                    # another token at the edge of the budget: counted, and its own fields are not compared
    w["cell"][5], w["tok"][5], w["ytype"][5] = 999, 9.0, 5
    c = compare_frame(a, w)
    assert c["common"] == K - 1 and c["only_a"] == 1 and c["only_w"] == 1 and not any(c["floats"].values()) and c["ints"]["ytype"] == 0 and not c["same_order"]


def _toy_model(seed=0):
    torch.manual_seed(seed)
    m = QG.QGraph(width=16, f05=16, d=64, soft_nms=["mnms", 1.75, 0.1])                                         # d = 64: the node token of the real B' (the head reads 64 channels)
    with torch.no_grad():
        m.nodes.head1.bias[0] = 3.0                                                                              # existence p = 0.95 at the start: many valid tokens
    return m


def _t_source():
    import jtoken as JK
    import loss_jtoken as LJK  # noqa: E402
    K, B = 64, 2
    x = torch.randn(B, 39, 200, 200).half()
    m = _toy_model()
    src = LiveSource(m, K=K)
    names = [n for n, p in src.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("model.nodes.") for n in names) and len(names) == len(list(m.nodes.parameters())) and len(src.trainable()) == len(names), "only the node head trains"
    assert not any(p.requires_grad for n, p in src.named_parameters() if not n.startswith("model.nodes.")), "the pyramid, links, map encoder and assigner are frozen"
    src.train()
    assert not any(mm.training for mm in m.pyr.modules()) and m.nodes.training, "the pyramid stays in eval mode, the node head follows the module"
    src.eval()
    assert not m.nodes.training
    src.train()
    assert not any(p.requires_grad for p in LiveSource(_toy_model(), K=K, train_nodes=False).parameters()), "train_nodes=False: nothing trains"
    # a live forward without a graph, and inside the trainer's autocast: B' stays in float32
    tab0 = [toy_table([[-9.0, 5.0]], [], [])] * B
    with torch.no_grad():
        o0 = src(x, tab0, ["a", "b"], [0, 1])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            o1 = src(x, tab0, ["a", "b"], [0, 1])
        o2 = src(x, tab0, ["a", "b"], [0, 1], token_drop=0.5, seed=1)                                            # the loader's token drop reaches the labels through the source
        no_, tk_ = src.tokens(x)                                                                                 # forward = tokens + labelled, bit for bit
        b_ = src.labelled(tk_, tab0, ["a", "b"], [0, 1])
    assert int(o0["batch"]["valid"].sum()) >= 8 and not o0["batch"]["tok"].requires_grad, "enough valid tokens for the tests (the toy bias)"
    assert all(torch.equal(o0["batch"][k], o1["batch"][k]) for k in GRAPH_KEYS), "B' runs in float32 inside an autocast"
    assert 0 < int(o2["batch"]["valid"].sum()) < int(o0["batch"]["valid"].sum()), "token_drop and seed are passed on"
    assert sorted(o0["no"]) == sorted(no_) and all(torch.equal(o0["no"][k], no_[k]) for k in no_), "forward = tokens + labelled: the node-stage outputs"
    assert sorted(o0["batch"]) == sorted(b_) and all(np.array_equal(o0["batch"][k].numpy(), b_[k].numpy(), equal_nan=True) for k in GRAPH_KEYS + ("valid", "ytype", "y", "ylong", "yscale", "gt_out", "row")), "forward = tokens + labelled: the batch"
    head = JK.JTokenHead(d=32, heads=4, layers=2, ffn=64).eval()                                                 # the head reads the INPUT keys only: the tokens alone give the same output as the whole batch (token_drop 0)
    with torch.no_grad():
        h1, h2 = head(head_inputs(tk_)), head(b_)
    assert all(torch.equal(h1[k], h2[k]) for k in h1), "the head's forward reads only the head inputs"
    # tables made of the tokens themselves: two GT nodes at the first allowed pair of valid tokens, linked
    bt = o0["batch"]
    allowed = JT.pair_geometry(bt["pos"], bt["dirv"], bt["valid"])[1]
    tabs = []
    for b in range(B):
        i, j = [int(v) for v in allowed[b].nonzero()[0]]
        t = toy_table(bt["pos"][b][[i, j]].double().numpy(), [(0, 1)], [False])
        t["dirv"] = bt["dirv"][b][[i, j]].double().numpy()
        tabs.append(t)
    with torch.no_grad():
        og = src(x, tabs, ["a", "b"], [0, 1])["batch"]["lab"]
        on = src(x, tabs, ["a", "b"], [0, 1], gate=False)["batch"]["lab"]
    assert sum(lab["n_reach"] for lab in og) > 0 and all(lab["n_reach"] == 0 for lab in on), "gate=False reaches the labels through the source"
    tg = {k: torch.from_numpy(np.stack([node_targets(t)[k] for t in tabs])) for k in NODE_KEYS}
    r = gradient_check(src, JK.JTokenHead(d=32, heads=4, layers=2, ffn=64), dict(x=x, table=tabs, token=["a", "b"], row=[0, 1], targets=tg))        # the card's gradient check
    assert r["params"] == len(names) and r["grad_norm_from_head"] > 0 and r["grad_norm_from_b"] > 0
    o = LiveSource(_toy_model(), K=K, train_nodes=False)(x, tabs, ["a", "b"], [0, 1])
    assert not o["batch"]["tok"].requires_grad and not o["batch"]["pos"].requires_grad, "train_nodes=False: no graph in the batch"


def _t_vs_cache():
    K = 16
    got = live_batch(live_tokens(toy_no(2), K), [toy_table([[-9.0, 5.0], [-7.0, 5.0]], [(0, 1)], [False]), toy_table(np.zeros((0, 2)), [], [])], ["a", "b"], [1, 2], K)
    cpu = lambda t: t.detach().cpu().numpy()  # noqa: E731
    items = []
    for j in range(2):                                                                                           # the cache items (`PTokData`) that equal this batch: float16 tok, the same arrays
        o = {k: cpu(got[k])[j] for k in ("pos", "dirv", "e", "e_raw", "s", "uout", "uin", "flags", "valid", "ytype", "y", "ylong", "yscale")}
        items.append(dict(o, tok=cpu(got["tok"])[j].astype(np.float16), gt_out=int(got["gt_out"][j]), tokens_np=dict(got["tokens_np"][j]), label={k: cpu(v)[j] for k, v in got["label"].items()}))
    counts = {k: np.array([lab[k] for lab in got["lab"]]) for k in STATS}

    class Fake:                                                                                                 # the stand-in of PTokData: the items and the stored label counts
        c = type("Cache", (), {"f": counts})()

        def __getitem__(self, i):
            return items[i]

    out, stats = _vs_cache(got, Fake(), [0, 1])
    assert all(c["same_order"] and not any(c["floats"].values()) and not any(c["ints"].values()) for c in out) and stats == 0, "a batch equals the cache items made of it"
    items[1]["tokens_np"]["s_p"] = items[1]["tokens_np"]["s_p"] + 0.1                                           # one field of tokens_np of frame 1 differs: found, in that frame only
    out, stats = _vs_cache(got, Fake(), [0, 1])
    assert out[1]["floats"]["np_s_p"] > 0.09 and out[0]["floats"]["np_s_p"] == 0 and sum(1 for f_ in out[1]["floats"].values() if f_ != 0) == 1 and stats == 0
    counts["n_pairs"][0] += 1                                                                                    # a label count of frame 0 differs
    assert _vs_cache(got, Fake(), [0, 1])[1] == 1, "the label counts are compared with the cache's"


def _t_pyr():
    """E393 `train_pyr`: the pyramid trains with the node head (nothing else of B'), the tokens are the frozen source's bit for bit, a loss on the live tokens reaches every pyramid tensor the node head reads (`lines05` and the rows 1 - 5 of
    `centre1` get nothing), no graph under an outer no_grad; without `train_pyr` the pyramid holds no gradient; `train_pyr` without `train_nodes` is refused"""
    K = 64
    x = torch.randn(2, 39, 200, 200).half()
    sp, s0 = LiveSource(_toy_model(), K=K, train_pyr=True), LiveSource(_toy_model(), K=K)
    names = {n for n, p in sp.named_parameters() if p.requires_grad}
    assert names == {"model.pyr." + n for n, _ in sp.model.pyr.named_parameters()} | {"model.nodes." + n for n, _ in sp.model.nodes.named_parameters()}, "train_pyr: the pyramid and the node head train, nothing else"
    assert len(sp.pyr_params()) == len(list(sp.model.pyr.parameters())) and len(sp.node_params()) == len(list(sp.model.nodes.parameters())) and len(sp.trainable()) == len(sp.pyr_params()) + len(sp.node_params())
    assert not s0.pyr_params() and len(s0.node_params()) == len(s0.trainable())
    refuses(lambda: LiveSource(_toy_model(), K=K, train_nodes=False, train_pyr=True), "train_pyr needs train_nodes")
    sp.train()
    assert not any(mm.training for mm in sp.model.pyr.modules()), "the pyramid stays in eval mode"
    got = {}
    for src in (sp, s0):
        no, tk = src.tokens(x)
        sum(tk[k].float().sum() for k in GRAPH_KEYS).backward()
        got[src is sp] = tk
    assert all(torch.equal(got[True][k].detach(), got[False][k].detach()) for k in GRAPH_KEYS + ("valid", "cell")), "train_pyr changes the tokens"
    held = {n for n, p in sp.model.pyr.named_parameters() if p.grad is not None and bool(p.grad.ne(0).any())}
    assert held == {n for n, _ in sp.model.pyr.named_parameters() if not n.startswith("lines05.")}, f"the loss on the tokens does not reach every pyramid tensor the node head reads: {sorted(set(n for n, _ in sp.model.pyr.named_parameters()) - held)}"
    cw, cb = sp.model.pyr.centre1.weight.grad.flatten(1), sp.model.pyr.centre1.bias.grad
    read = [0] + list(range(6, 12))
    assert bool(cw[1:6].eq(0).all()) and bool(cb[1:6].eq(0).all()) and bool(cw[read].ne(0).any(1).all()), "centre1: only the channels 0 and 6 - 11 are read"
    assert all(p.grad is None for p in s0.model.pyr.parameters()), "without train_pyr the pyramid holds no gradient"
    with torch.no_grad():
        assert not sp.tokens(x)[1]["tok"].requires_grad, "train_pyr under an outer no_grad: no graph"


def _t_guards():
    refuses(lambda: fixture(n=0), "nothing to compare")                                                          # no vacuous PASS
    refuses(lambda: fixture(seeds=()), "nothing to compare")
    refuses(lambda: fixture(n=6, gpu=True, bs=4), "not a multiple of bs")                                        # a short last batch is not a batch of the cache build


def selftest():
    torch.manual_seed(0)
    _t_targets()
    _t_tokens()
    _t_labels()
    _t_data()
    _t_compare()
    _t_vs_cache()
    _t_guards()
    _t_source()
    _t_pyr()
    print("lane_live_tok selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check")
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n", type=int, default=20)
    s = sub.add_parser("fixture")
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    s.add_argument("--gpu", action="store_true", help="on the GPU: every difference must be exactly zero (exit code 1 otherwise; the acceptance rule is in the module docstring)")
    s.add_argument("--bs", type=int, default=4, help="the batch size of the live batches: 4 = the cache build's (the parity of the acceptance rule); another size repeats the check at the trainer's batch size")
    s.add_argument("--out", default=None)
    s = sub.add_parser("grad")
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n", type=int, default=2)
    s.add_argument("--head", default=None, help="a head run under runs/ (e.g. e373_cb_s0, the champion's stage-1 head of seed 0); default a fresh head")
    sub.add_parser("selftest")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))                                      # a shared box: 2 threads unless told otherwise
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "check":
        sys.exit(0 if check(a.split, a.seed, n=a.n)["ok"] else 1)
    elif a.cmd == "grad":
        grad(a.split, a.seed, a.n, a.head)
    else:
        r = fixture(a.split, a.n, tuple(a.seeds), a.gpu, bs=a.bs)
        if a.out:
            with open(a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out), "w") as fh:
                json.dump(r, fh, indent=1)
        ok = r["ok"] if a.gpu else r["labels_ok"]                                                           # the CPU has no say on the model's numerics, only on the label step
        print(f"FIXTURE bs {a.bs} " + ("PASS" if ok else "FAIL") + (" (the whole path is exactly the cache" + ("" if r["parity_at_cache_bs"] else f", at bs {a.bs}, NOT the cache build's bs 4") + ")" if a.gpu else " (CPU: the label step on the cached tokens; the model's numbers: see above, exact only on the GPU)"))
        sys.exit(0 if ok else 1)
