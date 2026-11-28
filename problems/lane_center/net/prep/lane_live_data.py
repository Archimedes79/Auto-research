"""The data of the live token path (stage 2, the predict step): the frames of the token cache with their GT tables, and the label step that makes the batch of
`lane_ptok_data` from LIVE tokens (the tokens of a lane head B' that runs in the step, `lane_net.LaneNet.tokens`) exactly as `lane_tokcache` stores it for the frozen B'.

    ds, dl = loader("train" | "val", seed=0, bs=8, workers=4)        # the frames of the cache of B' seed `seed`, in its order (asserted); TRAIN shuffled
    b = next(iter(dl))                                               # x (B, 39, 200, 200) f16, table (the GT tables), token, row (B,), targets (the dense maps of B''s node loss)
    batch = live_batch(tk, b["table"], b["token"], b["row"], K, allowed)        # tk = the live tokens; allowed = the pair gate (B, K, K) of the caller or None

Frames: `lane_tok_data.TokData(split, res=2.0, feats=True)`; TEST guard: VAL is exactly `data/splits.json['val']` (`check_rows`; the feature files also hold the TEST
rows, never read). `collate` adds `targets` = `node_targets(table)`: exists (50, 50), flags (8, 50, 50), offset / dirv (2, 50, 50), sstart, uout, uin (50, 50).
The batch: the token inputs of the head (`head_inputs`: tok, pos, dirv, e, e_raw, s, uout, uin, flags + valid) keep the graph of the tokens; valid and every label
(ytype, y, ylong, yscale, gt_out, tokens_np, label) are constants made from the DETACHED tokens on the host (`label_frame` + `densify`; ylong on the TOKEN chord), plus
`lab` (the per-frame label dicts: counts) and `label_s` (seconds of the CPU loop). The pair gate only feeds the statistics `ypair_gate` / `n_reach`; it is computed by
the caller (the preprocessing does not import the model). A frame with more target pairs than `lane_tokcache.PMAX` raises `LabelOverflow`.
Tests: net/tests/test_lane_net.py"""
import json
import os
import re
import time

import numpy as np
import torch

import bflc_paths as BP
import lane_ptok_data as LP
import lane_tokcache as LC
import qtargets as QT

NODE_KEYS = ("exists", "flags", "offset", "dirv", "sstart", "uout", "uin")                              # the maps `qgraph.node_loss` reads (B''s dense target)
GRAPH_KEYS = tuple(k for k in LP.TOKEN_FIELDS if k != "valid")                                          # the token inputs of the head that carry the graph (valid is a constant)


def node_targets(table):
    """the dense maps of B''s node loss (NODE_KEYS, float32 numpy) from the GT table of a frame on the 2 m lattice: `qtargets.grid_targets` (the node loss reads exists, flags, offset, dirv, sstart, uout, uin)"""
    g = QT.grid_targets(table)
    return {k: g[k] for k in NODE_KEYS}


def check_rows(split, rows):
    """the TEST guard: the rows of VAL are exactly `data/splits.json['val']`, those of TRAIN are distinct (the feature files also hold the TEST rows)"""
    assert len(set(rows)) == len(rows), "a row twice in the cache"
    if split == "val":
        with open(os.path.join(BP.ROOT, "data", "splits.json")) as fh:
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


class LabelOverflow(RuntimeError):
    """a frame whose live tokens have more target pairs than `lane_tokcache.PMAX` (4096: the cache's own maximum is 646 on TRAIN, 536 on VAL, but tokens that move under the head's loss may grow near-duplicates): `label_frame` asserts, this names it. `.token` = the
    frame, `.pairs` = its pair count. The trainer decides: skip the step and count it, abort after N"""

    def __init__(self, token, pairs):
        super().__init__(f"frame {token}: {pairs} target pairs exceed PMAX {LC.PMAX}")
        self.token, self.pairs = token, pairs


def host_tokens(tk):
    """the token fields that the label step and the read-out writer read, as host arrays of the DETACHED tokens (a device sync)"""
    return {k: tk[k].detach().cpu().numpy() for k in ("cell", "valid", "pos", "dirv", "e_p", "flags_p", "s_p")}


def tokens_np(tn, b):
    """frame b of `host_tokens` as the read-out writer reads it (`lane_ptok_data.to_pred`): float64 pos / dirv / e_p / flags_p / s_p, int cell, bool valid; no GT field"""
    return dict(pos=tn["pos"][b].astype(np.float64), dirv=tn["dirv"][b].astype(np.float64), e_p=tn["e_p"][b].astype(np.float64), flags_p=tn["flags_p"][b].astype(np.float64), s_p=tn["s_p"][b].astype(np.float64),
                cell=tn["cell"][b].astype(np.int64), valid=tn["valid"][b].copy())


def label_batch(tk, tables, token, row, K, allowed=None):
    """the label side of a live batch from the DETACHED tokens `tk` (numpy: `label_frame`, `densify`) and the GT tables of the frames -> (the dict `lane_ptok_data.collate` makes of the items: valid, ytype, y, ylong, yscale, gt_out, token, row, tokens_np, label;
    the label dicts of `lane_tokcache.label_frame`; the seconds of the per-frame CPU loop). `allowed` = the pair gate (B, K, K) bool of the tokens (`lane_net.pair_gate`) or None; it only feeds statistics (`ypair_gate`, `n_reach` of the label
    dicts: y, ylong, yscale, valid, ytype and the rest are the same without it). A frame over PMAX target pairs raises `LabelOverflow`"""
    tn = host_tokens(tk)
    items, labs, t0 = [], [], time.perf_counter()
    for b in range(len(tables)):
        try:
            lab = LC.label_frame({k: tn[k][b] for k in ("pos", "dirv", "valid", "e_p")}, tables[b], K=K, allowed=None if allowed is None else allowed[b])
        except AssertionError as e:
            m = re.match(r"(\d+) target pairs exceed PMAX", str(e))
            if m is None:
                raise
            raise LabelOverflow(token[b], int(m.group(1))) from e
        y, _, ys = LP.densify(lab["ypair"], lab["ypair_long"], lab["ypair_scale"], K)
        tnp = tokens_np(tn, b)
        items.append(dict(valid=tn["valid"][b], ytype=lab["gt_type"].astype(np.int64), y=y, ylong=LP.chord_long(y, tnp["pos"]), yscale=ys,
                          gt_out=int(lab["gt_out"]), token=token[b], row=int(row[b]), tokens_np=tnp,
                          label=dict(gt_id=lab["gt_id"], gd=lab["gd"], gt_flags=lab["gt_flags"], gt_aux=lab["gt_aux"])))
        labs.append(lab)
    return LP.collate(items), labs, time.perf_counter() - t0


def head_inputs(tk):
    """the keys the token head reads (`jtoken.INPUTS`: GRAPH_KEYS + valid) of the live tokens `tk`, graph kept: the head's forward reads nothing else, so a trainer can queue `head(head_inputs(tk))` before the label step"""
    return {k: tk[k] for k in GRAPH_KEYS + ("valid",)}


def live_batch(tk, tables, token, row, K, allowed=None):
    """the batch of `lane_ptok_data` from the live tokens `tk` of a live B': the token inputs with the graph, the labels as constants (module docstring)"""
    lb, labs, secs = label_batch(tk, tables, token, row, K, allowed)
    batch = head_inputs(tk)
    batch.update({k: v.to(tk["pos"].device) if torch.is_tensor(v) else v for k, v in lb.items()})
    return dict(batch, lab=labs, label_s=secs)
