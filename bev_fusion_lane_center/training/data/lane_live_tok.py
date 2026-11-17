"""E372 (stage S4 of the token champion, D113): the LIVE token producer. B' (a `QGraph` run, `runs/e292_a_s<S>`: pyramid + node head + the Matrix-NMS layer) runs inside the training step with its node head TRAINABLE, and the tokens and the
labels that `lane_tokcache.py` stores for the frozen B' are made again from its live output with the graph kept, so that the loss of the token head reaches B''s node head. Card: `experiments/E372_s4-joint-finetune-b-in-the-loop/experiment.md`.
Derived data: none (nothing is cached here); the code, its selftest and the fixture are the record.

    ds, dl = loader("train" | "val", seed=0, bs=8, workers=4)                          # the frames of the E365 cache of B' seed `seed` (TokData(split, res=2.0)), asserted in the cache's order
    b = next(iter(dl))                                                                 # x (B, 39, 200, 200) f16, table (list of the GT tables), token (list), row (B,), targets (the dense node-loss maps of B)
    src = LiveSource("runs/e292_a_s0", K=512, train_nodes=True).to(dev)                # B' loaded, everything frozen but `model.nodes.*`
    out = src(b["x"], b["table"], b["token"], b["row"])                                # out["batch"]: the batch of `lane_ptok_data` (head input + labels); out["no"]: B''s node-stage outputs WITH the graph
    loss_h, info_h = jtoken.jtoken_loss(head(out["batch"]), out["batch"])              # the head's loss (the head must not cut its inputs: PLAN)
    loss_b, info_b = b_loss(out["no"], b["targets"])                                   # B''s own node loss (its E273 / E292 form, `qgraph.node_loss`)
    wsl bash tools/env/_py14.sh lane_live_tok.py check   [--split val] [--seed 0] [--n 20]                           # the guards, the frame order of the cache, node_targets against the maps of `lane_gt_data.QData`, bit for bit
    wsl bash tools/env/_py14.sh lane_live_tok.py fixture [--split val] [--n 40] [--seeds 0 1] [--gpu] [--out FILE]   # the live batch against the cache (`PTokData`), field by field; --gpu: batches of 4 as the cache, must be bit for bit (exit 1 otherwise)
    wsl bash tools/env/_py14.sh lane_live_tok.py selftest                                                            # no cache, no GPU (a toy `QGraph`); tests: grid/test_lane_live.py (the hand mutants)

Frames: TRAIN / VAL as the cache (`lane_tok_data.TokData(split, res=2.0)`); TEST guard: VAL is exactly `data/splits.json['val']` (`lane_past_data.check_rows`; the feature files also hold the TEST rows, never read).
LiveData item: x (39, 200, 200) f16 (normalised as QData / TokData), token, row, table (the E357 GT table of the frame on the 2 m lattice); NO `QData` item (it fits the long-link scales per frame: too slow). `collate` adds `targets`: the dense maps of B''s own node
loss, `node_targets(table)` = `qtargets.grid_targets` on the table: exists (50, 50), flags (8, 50, 50), offset / dirv (2, 50, 50), sstart, uout, uin (50, 50); the tier / vis / ztier maps of the E276 switches are not made (B' was trained without them: `NODE_CFG`).
LiveSource.forward(x, tables, token, row, token_drop=0.0, seed=0) -> dict(no, batch). B' runs in float32 also inside the trainer's bf16 autocast (the cache was made in float32). `no` = `QGraph.node_stage` outputs (e = the damped logit, e_raw = the head's own).
`batch` = the dict of `lane_ptok_data.collate` (tok (B, K, 64), pos, dirv, e, e_raw, s, uout, uin, flags, valid, ytype, y, ylong, yscale, gt_out, token, row, tokens_np, label) + `lab` (the per-frame dicts of `lane_tokcache.label_frame`: counts, pair gate) and
`label_s` (seconds of the CPU label loop): the TOKEN INPUTS (tok, pos, dirv, e, e_raw, s, uout, uin, flags) carry the graph to B''s node head; valid and every label are constants. The tokens are those of `lane_tokcache.select_tokens` (the K cells with the largest e, a stable
argsort = the named index budget: no gradient through the choice; probabilities from the whole map; K = 512 as the cache), `tok` rounded to float16 as the cache stores it with the gradient passed straight through (a plain .half() would round the gradient too). The labels
are `label_frame` + `densify` on the DETACHED live tokens (numpy: a graph there would raise), the gate `allowed` from `jtok.pair_geometry` on them, ylong on the TOKEN chord as `PTokData` defines it; the loader's `token_drop` (seeded by (seed, token)) as `PTokData` does it.

PLAN for the joint trainer (the head's trainer is `train/jtoken_train.py`; the joint loop is the lane chat's):
  per step    b = next(dl); out = src(b["x"], b["table"], b["token"], b["row"]); loss = jtoken_loss(head(out["batch"]), out["batch"]) + w * b_loss(out["no"], b["targets"])   (w = 1, the card); one backward, one clip, one optimiser step
  parameters  two groups: head.parameters() and src.trainable() (= the node head's 99 304 parameters, lr 1e-4 each); `src.model.state_dict()` is a complete `QGraph` state: `qpredict.load_model` reads a model.pt made of it (args as the run's)
  the head    needs an option that keeps its inputs: `JTokenHead.forward` cuts them with `_const` (`jtoken.py`); with `live=True` it must read `batch[k]` as it is. Until then the tests patch `jtoken._const` to the identity and say so
  autocast    `src` switches autocast off for B' (float32 = the cache's numbers); the head may stay in bf16 as in S1; under `torch.no_grad()` (read-out) the same code gives the live tokens without a graph
  a resume    the node head and the head are in the checkpoint; the pyramid is not trained, its weights come from the run (`src.state_dict()` holds them too: 8.4 MB)
  CPU / memory / speed: see the report of the build step (numbers measured on this box; the label step runs on the CPU in the main process, between the tokens and the loss)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtok as JT  # noqa: E402
import lane_past_data as LPD  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import qgraph as QG  # noqa: E402
import qpredict as QP  # noqa: E402
import qtargets as QT  # noqa: E402

NODE_KEYS = ("exists", "flags", "offset", "dirv", "sstart", "uout", "uin")                              # the maps `qgraph.node_loss` reads (B''s dense target)
NODE_CFG = dict(node_w_evt=0.0, node_w_inter=0.0, tier_w=None, vis_w=1.0, neg_w=None, node_loss="e273")  # `qgraph.CFG` of B''s training (args of runs/e292_a_s<S>/model.pt = the defaults): the node loss is the plain dense one
GRAPH_KEYS = tuple(k for k in LP.TOKEN_FIELDS if k != "valid")                                          # the token inputs of the head that carry the graph (valid is a constant)
STATS = ("n_gt", "n_gt_assigned", "n_links", "n_links_c", "gt_out", "n_nosrc", "n_reach", "n_pairs", "n_conflict", "n_valid", "n_deliv", "n_deliv_near", "n_gt_deliv_cov", "n_e50_in")      # per-frame counts of a label dict, as the cache stores them


# ------------------------------------------------------------------------------------------------------------------------------ B's own loss
def node_targets(table):
    """the dense maps of B''s node loss (NODE_KEYS, float32 numpy) from the GT table of a frame on the 2 m lattice: `qtargets.grid_targets` with no points (the tier maps are not needed, the node loss reads exists, flags, offset, dirv, sstart, uout, uin)"""
    M = len(table["pos"])
    g = QT.grid_targets(dict(table, pts=np.zeros((0, 2)), tier=np.zeros(M, np.uint8), vis=np.ones(M, bool)))
    return {k: g[k] for k in NODE_KEYS}


def b_loss(no, targets):
    """B''s own node loss at its E273 / E292 form (`qgraph.node_loss`, the weight is the caller's): no = node-stage outputs, targets = the `targets` of `collate` -> (loss, info). The loss switches of `qgraph.CFG` must be those of B''s training (NODE_CFG)"""
    bad = {k: QG.CFG[k] for k, v in NODE_CFG.items() if QG.CFG[k] != v}
    assert not bad, f"`qgraph.CFG` is not that of B''s training: {bad} (B' was trained with {NODE_CFG})"
    return QG.node_loss(no, targets)


# ------------------------------------------------------------------------------------------------------------------------------ frames
def check_order(cache_dir, rows, tokens):
    """the frames (rows, tokens) are those of the cache directory's frames.json, in its order"""
    fr = json.load(open(os.path.join(cache_dir, "frames.json")))
    assert [int(r) for r in fr["rows"]] == [int(r) for r in rows] and list(fr["tokens"]) == list(tokens), f"{cache_dir}: its frames are not the frames of this data (rows, tokens), in this order"


class LiveData(torch.utils.data.Dataset):
    """the frames of `lane_tok_data.TokData(split, res=2.0, feats=True)` in the order of the E365 cache of B' seed `seed` (asserted against its frames.json); the item is x, token, row, table. `td` is injected by the selftest only"""

    def __init__(self, split="train", seed=0, ident="e292_a", td=None):
        injected = td is not None
        if not injected:
            import lane_tok_data as LT
            td = LT.TokData(split, res=2.0, feats=True)
        self.td, self.split, self.seed = td, split, int(seed)
        self.rows, self.tokens = td.rows, td.tokens
        LPD.check_rows(split, self.rows, expect=None)                       # the TEST guard: VAL = data/splits.json['val'] exactly
        if not injected:
            check_order(LP.cache_dir(split, seed, ident), self.rows, self.tokens)

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


def label_batch(tk, tables, token, row, K, token_drop=0.0, seed=0):
    """the label side of a live batch from the DETACHED tokens `tk` (numpy: `label_frame`, `densify`) and the GT tables of the frames -> (the dict `lane_ptok_data.collate` makes of the items: valid, ytype, y, ylong, yscale, gt_out, token, row, tokens_np, label;
    the label dicts of `lane_tokcache.label_frame`; the seconds of the per-frame CPU loop). The gate `allowed` is that of `jtok.pair_geometry` on the live tokens"""
    with torch.no_grad():
        allowed = JT.pair_geometry(tk["pos"][:, :K].detach(), tk["dirv"][:, :K].detach(), tk["valid"][:, :K])[1].cpu().numpy()
    tn = {k: tk[k].detach().cpu().numpy() for k in ("cell", "valid", "pos", "dirv", "e_p", "flags_p", "s_p")}
    items, labs, t0 = [], [], time.perf_counter()
    for b in range(len(tables)):
        lab = LC.label_frame({k: tn[k][b] for k in ("pos", "dirv", "valid", "e_p")}, tables[b], K=K, allowed=allowed[b])
        keep = np.ones(K, bool)
        if token_drop > 0:
            from jtok_eval import keep_nodes
            keep = keep_nodes(token[b], K, token_drop, seed)                  # as PTokData: seeded by (seed, the frame's token)
        y, _, ys = LP.densify(lab["ypair"], lab["ypair_long"], lab["ypair_scale"], K, keep if token_drop > 0 else None)
        pos = tn["pos"][b].astype(np.float64)
        valid = tn["valid"][b] & keep
        items.append(dict(valid=valid, ytype=np.where(keep, lab["gt_type"], 0).astype(np.int64), y=y, ylong=(y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT), yscale=ys,
                          gt_out=int(lab["gt_out"]), token=token[b], row=int(row[b]),
                          tokens_np=dict(pos=pos, dirv=tn["dirv"][b].astype(np.float64), e_p=tn["e_p"][b].astype(np.float64), flags_p=tn["flags_p"][b].astype(np.float64), s_p=tn["s_p"][b].astype(np.float64),
                                         cell=tn["cell"][b].astype(np.int64), valid=valid.copy()),
                          label=dict(gt_id=lab["gt_id"], gd=lab["gd"], gt_flags=lab["gt_flags"], gt_aux=lab["gt_aux"])))
        labs.append(lab)
    return LP.collate(items), labs, time.perf_counter() - t0


def live_batch(no, tables, token, row, K, token_drop=0.0, seed=0):
    """the batch of `lane_ptok_data` from the node-stage outputs `no` of a live B': the token inputs with the graph, the labels as constants (module docstring)"""
    tk = live_tokens(no, K)
    lb, labs, secs = label_batch(tk, tables, token, row, K, token_drop, seed)
    batch = {k: tk[k] for k in GRAPH_KEYS}
    batch.update({k: v.to(tk["pos"].device) if torch.is_tensor(v) else v for k, v in lb.items()})
    return dict(batch, lab=labs, label_s=secs)


class LiveSource(nn.Module):
    """B' as a live token source. `run`: a run folder (`qpredict.load_model`, on the CPU: move the module with .to) or a `QGraph`. Everything is frozen except `model.nodes.*` (nothing when `train_nodes` is False); the pyramid is
    always in eval mode and under no_grad (`train()` of this module does not change that), the node head follows the module's mode. K = the tokens per frame (512 as the cache)"""

    def __init__(self, run, K=512, train_nodes=True):
        super().__init__()
        self.model = run if isinstance(run, nn.Module) else QP.load_model(run, "cpu")
        self.K = K
        for p in self.model.parameters():
            p.requires_grad_(False)
        if train_nodes:
            for p in self.model.nodes.parameters():
                p.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        self.model.nodes.train(mode)
        return self

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, tables, token, row, token_drop=0.0, seed=0):
        dev = next(self.model.parameters()).device
        with torch.autocast(device_type=dev.type, enabled=False):         # float32, as the cache was made, also inside the trainer's autocast
            with torch.no_grad():
                P = self.model.pyr(x.to(dev).float())
            no = self.model.node_stage(P)
            return dict(no=no, batch=live_batch(no, tables, token, row, self.K, token_drop, seed))


# ------------------------------------------------------------------------------------------------------------------------------ check: the frames and the node targets
def check(split="val", seed=0, ident="e292_a", n=20, log=print):
    """the guards and the order of the cache (LiveData asserts them), then `node_targets(table)` against the maps of the `lane_gt_data.QData` item of the same frame (the loader B' was trained with), bit for bit, on n evenly spaced frames -> dict"""
    ds = LiveData(split, seed, ident)
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    bad = {k: 0 for k in NODE_KEYS}
    for i in idx:
        want, got = ds.td.qd[int(i)], node_targets(ds.td.table(int(i)))
        assert want["token"] == ds.tokens[i] and int(want["row"]) == ds.rows[i], f"frame {i}: QData is on another frame"
        for k in NODE_KEYS:
            bad[k] += int(not (want[k].numpy().dtype == got[k].dtype and np.array_equal(want[k].numpy(), got[k])))
    res = dict(split=split, seed=seed, frames=len(ds), checked=len(idx), differing_frames=bad, ok=not any(bad.values()))
    log(f"{split}: {len(ds)} frames = the rows / tokens of the cache of B' seed {seed} in its order" + (", VAL = data/splits.json['val']" if split == "val" else ""))
    log(f"node targets vs QData on {len(idx)} frames: " + ", ".join(f"{k} {'equal' if not v else str(v) + ' DIFFER'}" for k, v in bad.items()))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ fixture: the live batch against the cache
def _worst(a, w, same):
    """largest absolute difference of two (B, ...) arrays over all frames and over the frames in `same` (B,) bool (NaN where both are NaN counts as equal)"""
    a, w = np.asarray(a, np.float64), np.asarray(w, np.float64)
    d = np.where(np.isnan(a) & np.isnan(w), 0.0, np.abs(a - w)).reshape(len(a), -1)
    return float(d.max()), float(d[same].max()) if same.any() else 0.0


def fixture(split="val", n=40, seeds=(0, 1), gpu=False, ident="e292_a", log=print):
    """the live batch of the first n frames of the split (B' of every seed in `seeds`, frozen, eval, no_grad, batches of 4 as the cache) against the cache items (`PTokData`): per field the largest absolute difference (all frames / the frames whose K cells
    are the same), the number of differing entries of the integer and boolean fields, the label counts. On the GPU it must be exactly zero (`ok`); on the CPU the forward of B' differs from the GPU-made cache by float noise -> dict"""
    dev = "cuda" if gpu else "cpu"
    assert not gpu or torch.cuda.is_available(), "--gpu: no GPU visible"
    cpu = lambda t: t.detach().cpu()  # noqa: E731
    res = {}
    for s in seeds:
        src = LiveSource(os.path.join(ROOT, "runs", f"{ident}_s{s}"), 512, train_nodes=False).to(dev).eval()
        ds, pt = LiveData(split, s, ident), LP.PTokData(split, s, ident)
        assert pt.rows[:n] == ds.rows[:n] and pt.tokens[:n] == ds.tokens[:n] and pt.K == src.K, "the cache is not the frames / K of the live source"
        diff = {k: [0.0, 0.0] for k in GRAPH_KEYS + ("yscale",)}
        cnt = {k: 0 for k in ("frames_other_cells", "valid", "ytype", "y", "ylong", "gt_out", "label", "stats")}
        same_n, secs = 0, []
        for i0 in range(0, n, 4):
            idx = list(range(i0, min(i0 + 4, n)))
            b = collate([ds[i] for i in idx])
            with torch.no_grad():
                got = src(b["x"], b["table"], b["token"], b["row"])["batch"]
            want = LP.collate([pt[i] for i in idx])
            secs.append(got["label_s"])
            same = np.array([np.array_equal(g["cell"], w["cell"]) for g, w in zip(got["tokens_np"], want["tokens_np"])])
            same_n += int(same.sum())
            cnt["frames_other_cells"] += int((~same).sum())
            for k in GRAPH_KEYS + ("yscale",):
                a, w = _worst(cpu(got[k]), want[k].float(), same)
                diff[k] = [max(diff[k][0], a), max(diff[k][1], w)]
            for k in ("valid", "ytype", "y", "ylong", "gt_out"):
                cnt[k] += int((cpu(got[k]) != want[k]).sum())
            cnt["label"] += sum(int(not np.array_equal(cpu(got["label"][k]).numpy(), want["label"][k].numpy(), equal_nan=k == "gd")) for k in want["label"])
            cnt["stats"] += sum(int(lab[k] != pt.c.f[k][i]) for lab, i in zip(got["lab"], idx) for k in STATS)
        exact = all(v == 0.0 for d_ in diff.values() for v in d_) and not any(cnt.values())
        res[s] = dict(frames=n, frames_same_cells=same_n, max_abs_diff_all_and_same_cells=diff, differing_entries=cnt, label_s_per_batch_of_4_median=float(np.median(secs)), exact=exact)
        log(f"seed {s} ({dev}): {n} frames, K cells identical on {same_n}; exact: {exact}")
        log("   largest |diff| (all frames / frames with the same cells): " + ", ".join(f"{k} {v[0]:.2e} / {v[1]:.2e}" for k, v in diff.items()))
        log("   differing entries: " + ", ".join(f"{k} {v}" for k, v in cnt.items()) + f"; label loop {np.median(secs) * 1000:.0f} ms per batch of 4")
    return dict(res, ok=all(r["exact"] for r in res.values()), device=dev)


# ------------------------------------------------------------------------------------------------------------------------------ selftest
class _Toy:
    """the stand-in of `TokData` for the selftest: frame i has one GT node at (-9, 2 i + 0.5) and x = i everywhere"""

    def __init__(self, rows, tokens):
        self.rows, self.tokens = list(rows), list(tokens)

    def __getitem__(self, i):
        return dict(toy_table([[-9.0, 2.0 * i + 0.5]], [], []), x=torch.full((39, 4, 4), float(i)).half(), token=self.tokens[i], row=self.rows[i])


def refuses(fn, text):
    """fn must fail with an AssertionError whose message holds `text` (a call that is accepted raises a RuntimeError, which no test catches)"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    raise RuntimeError(f"accepted although it must be refused ({text})")


def toy_table(pos, links, longs, flags=None):
    """a GT table (the E357 keys) of nodes at `pos`, all heading (1, 0); links = [(src, dst)], longs = their class"""
    pos, M = np.asarray(pos, float).reshape(-1, 2), len(pos)
    return dict(pos=pos, dirv=np.tile([1.0, 0.0], (M, 1)), cell=np.clip(np.floor((pos + QT.HALF) / QT.RES), 0, QT.N - 1).astype(np.int16), flags=np.zeros((M, 8), bool) if flags is None else np.asarray(flags, bool),
                piece=np.zeros(M, np.int32), sstart=np.eye(1, M, 0, dtype=bool)[0] if M else np.zeros(0, bool), src=np.array([a for a, _ in links], np.int32), dst=np.array([b for _, b in links], np.int32),
                long=np.array(longs, bool), scale=np.full((len(links), 2), np.nan), conn=np.zeros(len(links), bool))


def toy_no(B=2, seed=0, rank_cells=((20, 27), (21, 27), (22, 27)), grad=True):
    """node-stage outputs of B frames (random maps) in which the cells `rank_cells` have the largest existence (logits 5, 4, 3 in that order, all others -8), no offset, heading (1, 0): the tokens of a lane along x, the first at (-9, 5) m.
    The leaves carry a gradient when `grad`"""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    e = torch.full((B, 50, 50), -8.0)
    for k, (i, j) in enumerate(rank_cells):
        e[:, i, j] = 5.0 - k
    c = -QT.HALF + (torch.arange(50.0) + 0.5) * QT.RES
    dr = torch.zeros(B, 2, 50, 50)
    dr[:, 0] = 1.0 + 0.1 * r(B, 50, 50).abs()
    no = dict(e=e, e_raw=e + 0.5 * r(B, 50, 50), pos=torch.stack([c.view(50, 1).expand(50, 50), c.view(1, 50).expand(50, 50)])[None].repeat(B, 1, 1, 1), off=torch.zeros(B, 2, 50, 50), dir_raw=dr,
              flags=r(B, 8, 50, 50), s=r(B, 50, 50), uout=r(B, 50, 50), uin=r(B, 50, 50), tok=r(B, 6, 50, 50))
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
    # B's loss: a perfect prediction costs almost nothing, a wrong one costs more; the loss switches must be those of B''s training
    tg = {k: torch.from_numpy(v)[None] for k, v in g.items()}
    ex = tg["exists"]
    no = dict(e=ex * 24 - 12, flags=tg["flags"] * 24 - 12, off=tg["offset"], dir_raw=tg["dirv"] + (1 - ex)[:, None] * torch.tensor([0.0, 1.0]).view(1, 2, 1, 1), s=tg["sstart"] * 24 - 12, uout=tg["uout"] * 24 - 12, uin=tg["uin"] * 24 - 12)
    good, info = b_loss(no, tg)
    assert float(good) < 0.01 and sorted(info) == ["dir", "e", "flags", "off", "s", "u"], (float(good), info)
    assert float(b_loss(dict(no, e=-no["e"]), tg)[0]) > 1.0 and float(b_loss(no, {k: v.transpose(-1, -2) for k, v in tg.items()})[0]) > 0.1
    QG.CFG["node_w_evt"] = 2.0
    try:
        refuses(lambda: b_loss(no, tg), "not that of B''s training")
    finally:
        QG.CFG["node_w_evt"] = NODE_CFG["node_w_evt"]


def _t_tokens():
    K = 16
    no = toy_no()
    tk = live_tokens(no, K)
    ref = LC.select_tokens({k: v.detach() for k, v in no.items()}, None, K)
    assert sorted(tk) == sorted(ref) and all(torch.equal(tk[k], ref[k]) for k in ref if k != "tok"), "the live selection is `select_tokens`, bit for bit"
    assert torch.equal(tk["tok"], ref["tok"].float()) and tk["tok"].dtype == torch.float32, "tok = the float16 rounding of the cache, as float32"
    assert tk["cell"][0, :3].tolist() == [20 * 50 + 27, 21 * 50 + 27, 22 * 50 + 27] and tk["valid"][0].tolist() == [True] * 3 + [False] * 13
    batch = live_batch(no, [toy_table([[-9.0, 5.0]], [], [])] * 2, ["a", "b"], [1, 2], K)
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
    fl = np.zeros((3, 8), bool)
    fl[1, 0] = True                                                                                             # the middle node is a stop
    tabs = [toy_table([[-9.0, 5.0], [-7.0, 5.0], [-5.0, 5.0]], [(0, 1), (1, 2)], [False, False], fl), toy_table(np.zeros((0, 2)), [], [])]
    out = live_batch(no, tabs, ["f0", "f1"], [7, 8], K)
    assert out["y"].shape == (B, K, K) and out["y"].dtype == torch.uint8 and out["ylong"].shape == (B, K, K) and out["yscale"].shape == (B, K, K, 2) and out["ytype"].dtype == torch.int64 and out["row"].tolist() == [7, 8]
    assert out["label"]["gt_id"][0].tolist() == [0, 1, 2] + [-1] * 13 and (out["label"]["gt_id"][1] == -1).all(), "the three tokens take the three GT nodes in the order of their existence"
    assert out["ytype"][0].tolist() == [1, 4, 1] + [0] * 13 and out["ytype"][1].sum() == 0, "lane point, stop = entry (4), lane point; the padding and the empty frame are type 0"
    assert out["y"][0].nonzero().tolist() == [[0, 1], [1, 2]] and int(out["y"][1].sum()) == 0 and not out["ylong"].any() and torch.isnan(out["yscale"]).all()
    assert out["valid"][0].tolist() == [True] * 3 + [False] * 13 and out["gt_out"].tolist() == [0, 0] and out["token"] == ["f0", "f1"]
    assert out["lab"][0]["n_reach"] == 2 and out["lab"][0]["n_pairs"] == 2 and out["lab"][1]["n_gt"] == 0, "both pairs lie inside the gate of pair_geometry on the live tokens (forward along x)"
    tn = out["tokens_np"][0]
    assert tn["pos"].dtype == np.float64 and tn["cell"].dtype == np.int64 and tn["cell"][:3].tolist() == [1027, 1077, 1127] and np.allclose(tn["pos"][:3], [[-9, 5], [-7, 5], [-5, 5]]) and tn["valid"].sum() == 3
    assert sorted(tn) == ["cell", "dirv", "e_p", "flags_p", "pos", "s_p", "valid"], "no GT field in tokens_np"
    # the long chord class follows the TOKEN chord (12 m), not the table's class of the link
    o2 = live_batch(toy_no(1, rank_cells=((20, 27), (26, 27))), [toy_table([[-9.0, 5.0], [3.0, 5.0]], [(0, 1)], [False])], ["g"], [0], K)
    assert o2["y"][0].nonzero().tolist() == [[0, 1]] and bool(o2["ylong"][0, 0, 1]) and int(o2["ylong"].sum()) == 1
    # the loader's token drop: the draw of `jtok_eval.keep_nodes` seeded by (seed, token); a removed token is not valid, has type none and no pair
    from jtok_eval import keep_nodes
    seed = next(s for s in range(50) if not keep_nodes("f0", K, 0.5, s)[:3].all() and keep_nodes("f0", K, 0.5, s)[:3].any())
    keep = torch.from_numpy(keep_nodes("f0", K, 0.5, seed))
    od = live_batch(no, tabs[:1] * 2, ["f0", "f0"], [7, 7], K, token_drop=0.5, seed=seed)
    assert od["valid"][0].tolist() == (out["valid"][0] & keep).tolist() and od["valid"][0].tolist() != out["valid"][0].tolist() and od["tokens_np"][0]["valid"].tolist() == od["valid"][0].tolist()
    assert (od["ytype"][0][~keep] == 0).all() and not od["y"][0][~keep].any() and not od["y"][0][:, ~keep].any() and int(od["y"][0].sum()) < int(out["y"][0].sum())


def _t_data():
    rows = json.load(open(os.path.join(ROOT, "data", "splits.json")))["val"]
    toks = [f"t{i}" for i in range(len(rows))]
    ds = LiveData("val", 0, td=_Toy(rows, toks))
    assert len(ds) == len(rows) and ds[5]["token"] == "t5" and ds[5]["row"] == rows[5] and float(ds[5]["x"].float().mean()) == 5.0 and float(ds[5]["table"]["pos"][0, 1]) == 10.5, "item i is frame i"
    b = collate([ds[3], ds[4]])
    assert b["x"].shape == (2, 39, 4, 4) and b["row"].tolist() == [rows[3], rows[4]] and b["token"] == ["t3", "t4"] and len(b["table"]) == 2 and sorted(b["targets"]) == sorted(NODE_KEYS)
    assert b["targets"]["exists"].shape == (2, 50, 50) and b["targets"]["exists"][0].nonzero().tolist() == [[20, 28]] and b["targets"]["exists"][1].nonzero().tolist() == [[20, 29]], "targets[i] belong to frame i"
    refuses(lambda: LiveData("val", 0, td=_Toy(rows[::-1], toks)), "VAL cache are not data/splits.json['val']")                   # the TEST guard
    refuses(lambda: LiveData("val", 0, td=_Toy(rows[:-1], toks[:-1])), "VAL cache are not data/splits.json['val']")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        json.dump(dict(rows=[1, 2, 3], tokens=["a", "b", "c"]), open(os.path.join(d, "frames.json"), "w"))
        check_order(d, [1, 2, 3], ["a", "b", "c"])
        refuses(lambda: check_order(d, [1, 3, 2], ["a", "c", "b"]), "not the frames of this data")
        refuses(lambda: check_order(d, [1, 2, 3], ["a", "b", "x"]), "not the frames of this data")


def _toy_model(seed=0):
    torch.manual_seed(seed)
    m = QG.QGraph(width=16, f05=16, d=64, soft_nms=["mnms", 1.75, 0.1])                                         # d = 64: the node token of the real B' (the head reads 64 channels)
    with torch.no_grad():
        m.nodes.head1.bias[0] = 3.0                                                                              # existence p = 0.95 at the start: many valid tokens
    return m


def _t_source():
    import jtoken as JK
    from unittest import mock
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
    assert int(o0["batch"]["valid"].sum()) >= 8 and not o0["batch"]["tok"].requires_grad, "enough valid tokens for the tests (the toy bias)"
    assert all(torch.equal(o0["batch"][k], o1["batch"][k]) for k in GRAPH_KEYS), "B' runs in float32 inside an autocast"
    # tables made of the tokens themselves: two GT nodes at the first allowed pair of valid tokens, linked
    bt = o0["batch"]
    allowed = JT.pair_geometry(bt["pos"], bt["dirv"], bt["valid"])[1]
    tabs = []
    for b in range(B):
        i, j = [int(v) for v in allowed[b].nonzero()[0]]
        t = toy_table(bt["pos"][b][[i, j]].double().numpy(), [(0, 1)], [False])
        t["dirv"] = bt["dirv"][b][[i, j]].double().numpy()
        tabs.append(t)
    tg = {k: torch.from_numpy(np.stack([node_targets(t)[k] for t in tabs])) for k in NODE_KEYS}
    head = JK.JTokenHead(d=32, heads=4, layers=2, ffn=64)
    opt = torch.optim.AdamW([dict(params=list(head.parameters()), lr=1e-3), dict(params=src.trainable(), lr=1e-3)])
    states = {n: v.clone() for n, v in m.pyr.state_dict().items()}

    def step(live):
        opt.zero_grad(set_to_none=True)
        out = src(x, tabs, ["a", "b"], [0, 1])
        assert out["batch"]["tok"].requires_grad and int(out["batch"]["y"].sum()) > 0, "the toy tables give the head positive pairs"
        with mock.patch.object(JK, "_const", (lambda t: t) if live else JK._const):                              # until the head has its `live` option: the identity, in the test only
            loss, info = JK.jtoken_loss(head(out["batch"]), out["batch"])
        loss.backward()
        return out, info

    out, info = step(live=False)
    assert all(p.grad is None for p in src.trainable()), "the head's `_const` cuts the graph at its inputs: no gradient reaches B' (why the head needs its live option)"
    out, info = step(live=True)
    assert info["pos_pairs"] > 0
    dead = [n for n, p in src.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    assert not dead, f"the head's loss does not reach the node head's {dead}"
    assert not [n for n, p in src.named_parameters() if not n.startswith("model.nodes.") and (p.requires_grad or p.grad is not None)], "the frozen stages got a gradient"
    opt.zero_grad(set_to_none=True)
    lb, ib = b_loss(src(x, tabs, ["a", "b"], [0, 1])["no"], tg)                                                  # B''s own loss reaches the node head on its own
    lb.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in src.trainable()) and sorted(ib) == ["dir", "e", "flags", "off", "s", "u"]
    opt.step()
    assert all(torch.equal(states[n], v) for n, v in m.pyr.state_dict().items()), "the pyramid (weights and statistics) changed in a training step"
    with mock.patch.object(JK, "_const", lambda t: t):
        o = LiveSource(_toy_model(), K=K, train_nodes=False)(x, tabs, ["a", "b"], [0, 1])
    assert not o["batch"]["tok"].requires_grad and not o["batch"]["pos"].requires_grad, "train_nodes=False: no graph in the batch"


def selftest():
    torch.manual_seed(0)
    _t_targets()
    _t_tokens()
    _t_labels()
    _t_data()
    _t_source()
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
    s.add_argument("--gpu", action="store_true", help="on the GPU with batches of 4 as the cache: every difference must be exactly zero (exit code 1 otherwise)")
    s.add_argument("--out", default=None)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "check":
        sys.exit(0 if check(a.split, a.seed, n=a.n)["ok"] else 1)
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        r = fixture(a.split, a.n, tuple(a.seeds), a.gpu)
        if a.out:
            json.dump(r, open(a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out), "w"), indent=1)
        print("FIXTURE " + ("PASS (exactly equal)" if r["ok"] else ("FAIL" if a.gpu else "not exact on the CPU: see the numbers")))
        sys.exit(0 if (r["ok"] or not a.gpu) else 1)
