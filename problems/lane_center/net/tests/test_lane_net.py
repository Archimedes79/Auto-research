"""Tests of the lane network as one module (`net/model/lane_net.py`: `LaneNet`, `live_tokens`, `pair_gate`) and of the data of its live token path (`net/prep/lane_live_data.py`).

    python net/tests/test_lane_net.py                                                   # the selftest, the real-data commands, then the hand mutants (about 15 minutes)
    python net/tests/test_lane_net.py selftests                                         # the selftest and the real-data commands only
    python net/tests/test_lane_net.py selftest                                          # no cache, no GPU (a toy `QGraph`)
    python net/tests/test_lane_net.py check   [--split val] [--seed 0] [--n 20]         # the guards, the frame order and the checkpoint of the cache; node_targets against the maps of `lane_gt_data.QData`, bit for bit
    python net/tests/test_lane_net.py fixture [--split val] [--n 40] [--seeds 0 1] [--gpu] [--bs 4] [--out FILE]      # features -> tokens -> labels against the cache (`PTokData`), field by field; --gpu: must be bit for bit (exit 1 otherwise)
    python net/tests/test_lane_net.py grad    [--split val] [--seed 0] [--n 2] [--head e373_cb_s0]                    # the gradient check on real frames with the real B' (CPU)

selftest (a toy `QGraph` on the CPU): the node targets by hand, the live selection against `select_tokens`, the graph and its gradient at the chosen cells, the labels and the values of tokens_np by hand, the pair gate, the PMAX
overflow, the loader and its guards, `compare_frame` / `_vs_cache`, the freezing, tokens + labelled = live, the head reads only its inputs, the gradient check (the head's loss reaches every node-head tensor and no frozen
stage), the pyramid group, and features -> graph: `LaneNet.graph` is the read-out of the predict path's records, a checkpoint loads back strictly (`from_checkpoint`).
A mutant = one textual change of lane_net.py (N), lane_live_data.py (D) or of the comparison code of this file (T); the mutated copy stands in for its module (LANE_MUTANT) while a child runs `selftest`, four at a time. A mutant
counts as KILLED only when the child's own output shows an AssertionError (or `refuses`' "accepted although it must be refused"), or, for a mutant that is killed by a crash on purpose, the error text named in its last field; any
other non-zero exit is reported as CRASHED and fails the test, like a mutant that survives.
The real-data commands need the token caches (VAL and TRAIN of B' seeds 0 and 1) and the node-table cache; `fixture --gpu` is run by hand (the GPU): its rule is below.

GPU FIXTURE ACCEPTANCE (a DIAGNOSTIC: it feeds no verdict): PASS = on the first --n frames in batches of --bs 4 (as the cache build; n a multiple of 4), for EVERY seed given, in BOTH parts (`labels` = the label step on the
cached tokens, `live` = the whole path features -> tokens -> labels): the K cells are the same ORDERED list on every frame and every field of `compare_frame` is exactly zero. A difference in the `labels` part is a BUG on every
machine and batch size (no model is involved). A difference in `live` alone: look at the environment first (the json `env`: torch / CUDA / cuDNN / GPU, the checkpoint sha; another GPU model, memory pressure, --bs other than 4:
cuDNN picks its algorithms by shape, at bs 8 tok differs by two float16 ulps and a Matrix-NMS neighbour order may flip). On the CPU the forward of B' differs from the GPU-made cache by more than float noise (pos 2e-3 m);
only the `labels` part decides there."""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile

MUTANT = os.environ.get("LANE_MUTANT")                                                                  # a mutant run (`run_mutants`): this mutated copy of a module stands in for the module
if MUTANT:
    import importlib.util
    spec_ = importlib.util.spec_from_file_location(os.path.basename(MUTANT)[:-3], MUTANT)
    sys.modules[spec_.name] = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(sys.modules[spec_.name])

import numpy as np  # noqa: E402
import torch  # noqa: E402

import bflc_paths as BP  # noqa: E402
import jtok as JT  # noqa: E402
import jtoken as JK  # noqa: E402
import lane_live_data as LLD  # noqa: E402
import lane_net as LN  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
import lane_readout as RO  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import loss_jtoken as LJK  # noqa: E402
import loss_qgraph as LQG  # noqa: E402
import qgraph as QG  # noqa: E402
import qtargets as QT  # noqa: E402

NP_KEYS = ("pos", "dirv", "e_p", "flags_p", "s_p")                                                     # the float fields of `tokens_np` (what the read-out writer reads; plus cell and valid)
STATS = ("n_gt", "n_gt_assigned", "n_links", "n_links_c", "gt_out", "n_nosrc", "n_reach", "n_pairs", "n_conflict", "n_valid", "n_deliv", "n_deliv_near", "n_gt_deliv_cov", "n_e50_in")      # per-frame counts of a label dict, as the cache stores them


# The token path on the CPU against the GPU-made cache (values that do not come from this code): the same cells but for a few at the budget's edge, the fields of the common cells within
# 4 - 5 times the float gap of the CPU against the GPU's convolution rounding (measured 2026-11-17 and 2026-11-28 on 16 - 40 VAL frames of both seeds: 512 common cells, tok 1.56e-2 = two
# float16 ulps, pos 1.4e-3 m, dirv 1.5e-2, e_raw 2.9e-3, s 2.3e-2, uout / uin 2.0e-2, flags 1.8e-2). A wrong cell is 2 m off, a swapped field differs by about 1. The damped existence `e`
# has no bound: one Matrix-NMS neighbour order may flip (up to 1.2).
CPU_TOL = dict(tok=6.3e-2, pos=1e-2, dirv=0.1, e_raw=2e-2, s=0.1, uout=0.1, uin=0.1, flags=0.1)
CPU_COMMON = 508


def live_close(agg):
    """the `live` summary of a seed (`_summary`) within CPU_TOL of the cache"""
    return agg["common_min"] >= CPU_COMMON and all(agg["floats"][k] <= v for k, v in CPU_TOL.items())


def labelled_batch(tk, tables, token, row, K, gate=True):
    """`lane_live_data.live_batch` with the pair gate of the tokens, as `LaneNet.labelled` calls it"""
    return LLD.live_batch(tk, tables, token, row, K, LN.pair_gate(tk, K) if gate else None)


# ------------------------------------------------------------------------------------------------------------------------------ check: the frames and the node targets
def check(split="val", seed=0, ident="e292_a", n=20, log=print):
    """the guards and the order of the cache (LiveData asserts them), the cache's checkpoint (`check_run_sha`), then `node_targets(table)` against the maps of the `lane_gt_data.QData` item of the same frame (the loader B' was trained with), bit for bit, on n evenly
    spaced frames -> dict"""
    ds = LLD.LiveData(split, seed, ident)
    LLD.check_run_sha(LP.cache_dir(split, seed, ident), f"runs/{ident}_s{seed}")
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    bad = {k: 0 for k in LLD.NODE_KEYS}
    for i in idx:
        want, got = ds.td.qd[int(i)], LLD.node_targets(ds.td.table(int(i)))
        assert want["token"] == ds.tokens[i] and int(want["row"]) == ds.rows[i], f"frame {i}: QData is on another frame"
        for k in LLD.NODE_KEYS:
            bad[k] += int(not (want[k].numpy().dtype == got[k].dtype and np.array_equal(want[k].numpy(), got[k])))
    res = dict(split=split, seed=seed, frames=len(ds), checked=len(idx), differing_frames=bad, ok=not any(bad.values()))
    log(f"{split}: {len(ds)} frames = the rows / tokens of the cache of B' seed {seed} in its order, the cache was made from runs/{ident}_s{seed}/model.pt" + (", VAL = data/splits.json['val']" if split == "val" else ""))
    log(f"node targets vs QData on {len(idx)} frames: " + ", ".join(f"{k} {'equal' if not v else str(v) + ' DIFFER'}" for k, v in bad.items()))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ fixture: the live batch against the cache
def compare_frame(a, w):
    """two versions of one frame (numpy dicts: the LLD.GRAPH_KEYS (K, ...), valid, ytype (K,), y, ylong, yscale (K, K, ...), gt_out, cell (K,) = the flat cells of the tokens, label = dict gt_id, gd, gt_flags, gt_aux (K, ...), tnp = the `tokens_np` fields NP_KEYS and
    valid (K, ...) that the read-out writer reads) -> dict. The tokens are matched by their CELL (a float-noise swap of two neighbours in the ranking or at the budget's edge must not misalign the fields); same_order = the K cells are the same list; common /
    only_a / only_w = the cells in both lists / in one only; floats = the largest absolute difference per field over the common cells (NaN = NaN counts as equal), ints = the number of differing entries (the pair fields over pairs of common cells)"""
    ia, iw = {int(c): k for k, c in enumerate(a["cell"])}, {int(c): k for k, c in enumerate(w["cell"])}
    com = sorted(set(ia) & set(iw))
    ja, jw = np.array([ia[c] for c in com], int), np.array([iw[c] for c in com], int)
    tok = lambda d, k, j: np.asarray(d[k])[j]  # noqa: E731
    pair = lambda d, k, j: np.asarray(d[k])[np.ix_(j, j)]  # noqa: E731
    gap = lambda x, y: float(np.where(np.isnan(x) & np.isnan(y), 0.0, np.where(np.isnan(x) | np.isnan(y), np.inf, np.abs(x.astype(np.float64) - y.astype(np.float64)))).max(initial=0.0))  # noqa: E731   (one NaN: inf)
    floats = {k: gap(tok(a, k, ja), tok(w, k, jw)) for k in LLD.GRAPH_KEYS}
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
        live = {k: cpu(got[k])[j] for k in LLD.GRAPH_KEYS + ("valid", "ytype", "y", "ylong", "yscale")}
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
        shas[s] = LLD.check_run_sha(LP.cache_dir(split, s, ident), f"runs/{ident}_s{s}")
        src = LN.LaneNet(LN.load_b(BP.resolve(f"runs/{ident}_s{s}")), K=512).to(dev).eval()
        ds, pt = LLD.LiveData(split, s, ident), LP.PTokData(split, s, ident)
        assert n <= len(ds) and pt.rows[:n] == ds.rows[:n] and pt.tokens[:n] == ds.tokens[:n] and pt.K == src.K, "the cache is not the frames / K of the live source"
        cmp = dict(labels=[], live=[])
        stats, secs = dict(labels=0, live=0), []
        for i0 in range(0, n, bs):
            idx = list(range(i0, min(i0 + bs, n)))
            b = LLD.collate([ds[i] for i in idx])
            cached = {k: torch.from_numpy(np.stack([np.array(pt.c.f[k][i]) for i in idx])) for k in ("cell", "valid", "pos", "dirv", "e_p", "flags_p", "s_p")}
            lb, labs, _ = LLD.label_batch(cached, b["table"], b["token"], b["row"], pt.K, LN.pair_gate(cached, pt.K))                     # the label step on the cache's own tokens
            want = LP.collate([pt[i] for i in idx])
            r, st = _vs_cache(dict(want, **lb, lab=labs), pt, idx)
            cmp["labels"] += r
            stats["labels"] += st
            with torch.no_grad():
                got = src.live(b["x"], b["table"], b["token"], b["row"])["batch"]
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
    return dict(res, ok=all(a["exact"] for r in res.values() for a in r.values()), labels_ok=all(r["labels"]["exact"] for r in res.values()), live_close=all(live_close(r["live"]) for r in res.values()), device=dev, env=env, parity_at_cache_bs=bs == 4)


def gradient_check(src, head, b):
    """the gradient check of the card on one batch `b` of `collate` (x, table, token, row, targets); `src` a `LaneNet` without a head whose node head trains, `head` a `jtoken.JTokenHead` (it needs its `live_inputs` option). (1) With `live_inputs` False (the champion's default: `jtoken._const` cuts the
    inputs at the batch) the head's loss gives the node head no gradient. (2) With `live_inputs` True it reaches EVERY node-head parameter and no frozen one. (3) B''s own loss reaches the node head on its own. (4) An AdamW step on both groups leaves the pyramid (weights
    and buffers) as it was. -> dict(head_loss, b_loss, params, grad_norm_from_head, grad_norm_from_b: the norms of the node head's GRADIENT from each loss; the card's secondary number is the share of the parameter UPDATE norm, which Adam rescales per
    parameter: not the same). The step trains `src` and `head` and leaves `head.live_inputs` True: pass throw-away copies"""
    assert hasattr(head, "live_inputs"), "the head has no `live_inputs` option (jtoken.JTokenHead)"
    args = (b["x"], b["table"], b["token"], b["row"])
    opt = torch.optim.AdamW([dict(params=list(head.parameters()), lr=1e-3), dict(params=src.trainable(), lr=1e-3)])
    states = {n: v.clone() for n, v in src.b.pyr.state_dict().items()}
    norm = lambda: float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in src.trainable())))  # noqa: E731

    def head_step(live):
        opt.zero_grad(set_to_none=True)
        out = src.live(*args)
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
    assert not [n for n, p in src.named_parameters() if not n.startswith("b.nodes.") and (p.requires_grad or p.grad is not None)], "the frozen stages got a gradient"
    g_head = norm()
    opt.zero_grad(set_to_none=True)
    lb, ib = LQG.node_loss(src.live(*args)["no"], b["targets"])                                                              # B''s own loss reaches the node head on its own
    lb.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in src.trainable()) and sorted(ib) == ["dir", "e", "flags", "off", "s", "u"]
    g_b = norm()
    opt.step()
    assert all(torch.equal(states[n], v) for n, v in src.b.pyr.state_dict().items()), "the pyramid (weights and statistics) changed in a training step"
    return dict(head_loss=float(loss), b_loss=float(lb), params=len(src.trainable()), grad_norm_from_head=g_head, grad_norm_from_b=g_b)


def grad(split="val", seed=0, n=2, head=None, ident="e292_a", log=print):
    """`gradient_check` on the first n real frames: the real B' of `seed` on the CPU, the head = a fresh `JTokenHead` or the run `runs/<head>` (the champion's stage-1 head `e373_cb_s0`). It reports the head's share of the GRADIENT norm on the node head at step 0 on a few frames: a
    guide to the size of the head's pull, not the card's secondary number (the share of the parameter UPDATE norm: Adam rescales every parameter's step, so the two differ)"""
    ds = LLD.LiveData(split, seed, ident)
    if head:
        hd = LN.load_head(head, "cpu")[0]
    else:
        hd = JK.JTokenHead()
    r = gradient_check(LN.LaneNet(LN.load_b(BP.resolve(f"runs/{ident}_s{seed}")), K=512, train_nodes=True), hd, LLD.collate([ds[i] for i in range(n)]))
    share = r["grad_norm_from_head"] / (r["grad_norm_from_head"] + r["grad_norm_from_b"])
    log(f"{split} {n} frames, B' seed {seed}, head {head or 'fresh'}: head loss {r['head_loss']:.3f}, B's node loss {r['b_loss']:.3f}; all {r['params']} node-head tensors get a gradient from the head's loss, no frozen stage does, the pyramid is unchanged by a step; "
        f"gradient norm on the node head: from the head's loss {r['grad_norm_from_head']:.3f}, from B's loss {r['grad_norm_from_b']:.3f} (the head's share {share:.3f})")
    return dict(r, share=share)



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
    g = LLD.node_targets(t)
    assert sorted(g) == sorted(LLD.NODE_KEYS) and all(g[k].dtype == np.float32 for k in LLD.NODE_KEYS)
    assert g["exists"].shape == (50, 50) and g["flags"].shape == (8, 50, 50) and g["offset"].shape == (2, 50, 50) and g["dirv"].shape == (2, 50, 50)
    assert np.argwhere(g["exists"] > 0).tolist() == [[20, 27], [21, 27], [30, 27]], "the cell of a node is (floor((x + 50) / 2), floor((y + 50) / 2))"
    assert np.allclose(g["offset"][:, 20, 27], [-0.3, -0.8], atol=1e-6) and np.allclose(g["offset"][:, 21, 27], [0.0, 0.0]), "the offset is pos - the cell centre (x first)"
    assert g["flags"][2, 21, 27] == 1 and g["flags"][0, 30, 27] == 1 and g["flags"].sum() == 2, "split is channel 2, stop channel 0 (qtargets.FLAGS)"
    assert g["sstart"][20, 27] == 1 and g["sstart"].sum() == 1 and np.allclose(g["dirv"][:, 20, 27], [1.0, 0.0]) and g["dirv"][0].sum() == 3
    assert g["uout"][21, 27] == 1 and g["uin"][30, 27] == 1 and g["uout"].sum() == 1 and g["uin"].sum() == 1, "a long link marks its source (uout) and its destination (uin)"
    assert all(float(np.abs(v).sum()) == 0 for v in LLD.node_targets(toy_table(np.zeros((0, 2)), [], [])).values()), "a frame without GT nodes has empty maps"
    # B's loss: a perfect prediction costs almost nothing, a wrong one costs more
    tg = {k: torch.from_numpy(v)[None] for k, v in g.items()}
    ex = tg["exists"]
    no = dict(e=ex * 24 - 12, flags=tg["flags"] * 24 - 12, off=tg["offset"], dir_raw=tg["dirv"] + (1 - ex)[:, None] * torch.tensor([0.0, 1.0]).view(1, 2, 1, 1), s=tg["sstart"] * 24 - 12, uout=tg["uout"] * 24 - 12, uin=tg["uin"] * 24 - 12)
    good, info = LQG.node_loss(no, tg)
    assert float(good) < 0.01 and sorted(info) == ["dir", "e", "flags", "off", "s", "u"], (float(good), info)
    assert float(LQG.node_loss(dict(no, e=-no["e"]), tg)[0]) > 1.0 and float(LQG.node_loss(no, {k: v.transpose(-1, -2) for k, v in tg.items()})[0]) > 0.1


def _t_tokens():
    K = 16
    no = toy_no()
    tk = LN.live_tokens(no, K)
    ref = LC.select_tokens({k: v.detach() for k, v in no.items()}, None, K)
    assert sorted(tk) == sorted(ref) and all(torch.equal(tk[k], ref[k]) for k in ref if k != "tok"), "the live selection is `select_tokens`, bit for bit"
    assert torch.equal(tk["tok"], ref["tok"].float()) and tk["tok"].dtype == torch.float32, "tok = the float16 rounding of the cache, as float32"
    assert tk["cell"][0, :3].tolist() == [20 * 50 + 27, 21 * 50 + 27, 22 * 50 + 27] and tk["valid"][0].tolist() == [True] * 3 + [False] * 13
    assert sorted(LLD.GRAPH_KEYS) == ["dirv", "e", "e_raw", "flags", "pos", "s", "tok", "uin", "uout"] and sorted(LLD.head_inputs(tk)) == sorted(LLD.GRAPH_KEYS + ("valid",)), "the head's inputs: tok, pos, dirv, e, e_raw, s, uout, uin, flags + valid"
    batch = labelled_batch(tk, [toy_table([[-9.0, 5.0]], [], [])] * 2, ["a", "b"], [1, 2], K)
    assert sorted(k for k in batch if k not in ("lab", "label_s")) == sorted(LLD.GRAPH_KEYS + ("valid", "ytype", "y", "ylong", "yscale", "gt_out", "token", "row", "tokens_np", "label"))
    for k in LLD.GRAPH_KEYS:
        assert batch[k].requires_grad and torch.equal(batch[k], ref[k].float() if k == "tok" else ref[k]), k
    assert not any(torch.is_tensor(batch[k]) and batch[k].requires_grad for k in ("valid", "ytype", "y", "ylong", "yscale", "gt_out", "row")), "the labels are constants"
    # the graph reaches every map of B, at the K chosen cells only (the choice itself has no gradient); the gradient of tok is not rounded to float16 (1e-9 would underflow there)
    w = {k: 1.0 + torch.rand(batch[k].shape, generator=torch.Generator().manual_seed(1)) for k in LLD.GRAPH_KEYS}
    (sum((batch[k] * w[k]).sum() for k in LLD.GRAPH_KEYS if k != "tok") + (batch["tok"] * 1e-9).sum()).backward()
    for k_n in ("pos", "dir_raw", "e", "e_raw", "s", "uout", "uin", "flags", "tok"):
        gr = no[k_n].grad
        assert gr is not None and float(gr.abs().sum()) > 0, f"the graph does not reach no['{k_n}']"
        per_cell = gr.abs().flatten(2).amax(1) if gr.dim() == 4 else gr.abs().flatten(1)
        assert (per_cell > 0).sum(1).tolist() == [K, K], f"no['{k_n}']: the gradient is not at exactly the K = {K} chosen cells"
    assert abs(float(no["tok"].grad.abs().max()) / 1e-9 - 1) < 1e-6 and no["tok"].grad.dtype == torch.float32, "the gradient of tok must not be rounded to float16"


def _t_labels():
    B, K = 2, 16
    no = toy_no(B)
    tk = LN.live_tokens(no, K)
    fl = np.zeros((3, 8), bool)
    fl[1, 0], fl[2, 2] = True, True                                                                             # node 1 is a stop, node 2 a split
    tabs = [toy_table([[-9.0, 5.3], [-7.0, 5.0], [-5.0, 4.6]], [(0, 1), (1, 2)], [False, False], fl), toy_table(np.zeros((0, 2)), [], [])]      # the GT nodes lie 0.3 / 0 / 0.4 m from the tokens
    out = labelled_batch(tk, tabs, ["f0", "f1"], [7, 8], K)
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
    o2 = labelled_batch(LN.live_tokens(toy_no(1, rank_cells=((20, 27), (26, 27))), K), [toy_table([[-9.0, 5.0], [3.0, 5.0]], [(0, 1)], [False])], ["g"], [0], K)
    assert o2["y"][0].nonzero().tolist() == [[0, 1]] and bool(o2["ylong"][0, 0, 1]) and int(o2["ylong"].sum()) == 1
    o4 = labelled_batch(LN.live_tokens(toy_no(1, rank_cells=((20, 27), (24, 27))), K), [toy_table([[-9.0, 5.0], [-1.0, 5.0]], [(0, 1)], [False])], ["i"], [0], K)
    assert o4["y"][0].nonzero().tolist() == [[0, 1]] and not o4["ylong"].any(), "a chord of exactly 8 m is short"
    # a GT node that no token covers: the link into it is lost (gt_out counts it before the contraction)
    o3 = labelled_batch(LN.live_tokens(toy_no(1), K), [toy_table([[-9.0, 5.0], [-7.0, 5.0], [-7.0, 40.0]], [(0, 1), (1, 2)], [False, True])], ["h"], [0], K)
    assert o3["gt_out"].tolist() == [1] and o3["lab"][0]["gt_out"] == 1 and o3["lab"][0]["n_links"] == 2 and o3["lab"][0]["n_links_c"] == 1
    # the pair gate only feeds statistics: without it y, ylong, yscale, valid, ytype and gt_out are the same
    og = labelled_batch(tk, tabs, ["f0", "f1"], [7, 8], K, gate=False)
    for k in ("valid", "ytype", "y", "ylong", "yscale", "gt_out"):
        assert np.array_equal(og[k].numpy(), out[k].numpy(), equal_nan=True), f"{k} differs without the gate"
    assert og["lab"][0]["n_reach"] == 0 and og["lab"][0]["n_pairs"] == 2 and not og["lab"][0]["ypair_gate"].any() and out["lab"][0]["ypair_gate"][:2].all(), "gate=False: no pair is counted as reachable, nothing else changes"
    # a frame over PMAX target pairs is named, with its token and its count (the trainer decides)
    pmax, LC.PMAX = LC.PMAX, 1
    try:
        labelled_batch(tk, [tabs[1], tabs[0]], ["fa", "fb"], [8, 7], K)                                              # the second frame has two pairs, the first none
        raise RuntimeError("accepted although it must be refused (more target pairs than PMAX)")
    except LLD.LabelOverflow as e:
        assert e.token == "fb" and e.pairs == 2 and isinstance(e, RuntimeError) and "exceed PMAX" in str(e), str(e)
    finally:
        LC.PMAX = pmax


def _t_data():
    import tempfile
    with open(os.path.join(BP.ROOT, "data", "splits.json")) as fh:
        rows = json.load(fh)["val"]
    toks = [f"t{i}" for i in range(len(rows))]
    ds = LLD.LiveData("val", 0, td=_Toy(rows, toks))
    assert len(ds) == len(rows) and ds[5]["token"] == "t5" and ds[5]["row"] == rows[5] and float(ds[5]["x"].float().mean()) == 5.0 and float(ds[5]["table"]["pos"][0, 1]) == 10.5, "item i is frame i"
    b = LLD.collate([ds[3], ds[4]])
    assert b["x"].shape == (2, 39, 4, 4) and b["row"].tolist() == [rows[3], rows[4]] and b["token"] == ["t3", "t4"] and len(b["table"]) == 2 and sorted(b["targets"]) == sorted(LLD.NODE_KEYS)
    assert [float(t["pos"][0, 1]) for t in b["table"]] == [6.5, 8.5], "table[i] is the table of frame i"
    assert b["targets"]["exists"].shape == (2, 50, 50) and b["targets"]["exists"][0].nonzero().tolist() == [[20, 28]] and b["targets"]["exists"][1].nonzero().tolist() == [[20, 29]], "targets[i] belong to frame i"
    refuses(lambda: LLD.LiveData("val", 0, td=_Toy(rows[::-1], toks)), "VAL cache are not data/splits.json['val']")                   # the TEST guard
    refuses(lambda: LLD.LiveData("val", 0, td=_Toy(rows[:-1], toks[:-1])), "VAL cache are not data/splits.json['val']")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "frames.json"), "w") as fh:
            json.dump(dict(rows=[1, 2, 3], tokens=["a", "b", "c"]), fh)
        LLD.check_order(d, [1, 2, 3], ["a", "b", "c"])
        refuses(lambda: LLD.check_order(d, [1, 3, 2], ["a", "b", "c"]), "not the frames of this data")                  # the rows alone, the tokens alone
        refuses(lambda: LLD.check_order(d, [1, 2, 3], ["a", "c", "b"]), "not the frames of this data")
        refuses(lambda: LLD.check_order(d, [1, 2], ["a", "b"]), "not the frames of this data")
        with open(os.path.join(d, "frames.json"), "w") as fh:                                                      # LiveData itself runs the order check (here against an injected cache directory)
            json.dump(dict(rows=rows, tokens=toks), fh)
        LLD.LiveData("val", 0, td=_Toy(rows, toks), cache_dir=d)
        refuses(lambda: LLD.LiveData("val", 0, td=_Toy(rows, toks[::-1]), cache_dir=d), "not the frames of this data")
        os.makedirs(os.path.join(d, "run"))                                                                        # the checkpoint of the cache: sha256("abc") starts ba7816bf8f01cfea
        with open(os.path.join(d, "run", "model.pt"), "wb") as fh:
            fh.write(b"abc")
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(dict(run_sha="ba7816bf8f01cfea"), fh)
        assert LLD.check_run_sha(d, os.path.join(d, "run")) == "ba7816bf8f01cfea"
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(dict(run_sha="0123456789abcdef"), fh)
        refuses(lambda: LLD.check_run_sha(d, os.path.join(d, "run")), "is not the checkpoint that the cache")


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
    K, B = 64, 2
    x = torch.randn(B, 39, 200, 200).half()
    m = _toy_model()
    src = LN.LaneNet(m, K=K, train_nodes=True)
    names = [n for n, p in src.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("b.nodes.") for n in names) and len(names) == len(list(m.nodes.parameters())) and len(src.trainable()) == len(names), "only the node head trains"
    assert not any(p.requires_grad for n, p in src.named_parameters() if not n.startswith("b.nodes.")), "the pyramid, links, map encoder and assigner are frozen"
    src.train()
    assert not any(mm.training for mm in m.pyr.modules()) and m.nodes.training, "the pyramid stays in eval mode, the node head follows the module"
    src.eval()
    assert not m.nodes.training
    src.train()
    assert not any(p.requires_grad for p in LN.LaneNet(_toy_model(), K=K).parameters()), "train_nodes=False: nothing trains"
    # a live forward without a graph, and inside the trainer's autocast: B' stays in float32
    tab0 = [toy_table([[-9.0, 5.0]], [], [])] * B
    with torch.no_grad():
        o0 = src.live(x, tab0, ["a", "b"], [0, 1])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            o1 = src.live(x, tab0, ["a", "b"], [0, 1])
        no_, tk_ = src.tokens(x)                                                                                 # forward = tokens + labelled, bit for bit
        b_ = src.labelled(tk_, tab0, ["a", "b"], [0, 1])
    assert int(o0["batch"]["valid"].sum()) >= 8 and not o0["batch"]["tok"].requires_grad, "enough valid tokens for the tests (the toy bias)"
    assert all(torch.equal(o0["batch"][k], o1["batch"][k]) for k in LLD.GRAPH_KEYS), "B' runs in float32 inside an autocast"
    assert sorted(o0["no"]) == sorted(no_) and all(torch.equal(o0["no"][k], no_[k]) for k in no_), "forward = tokens + labelled: the node-stage outputs"
    assert sorted(o0["batch"]) == sorted(b_) and all(np.array_equal(o0["batch"][k].numpy(), b_[k].numpy(), equal_nan=True) for k in LLD.GRAPH_KEYS + ("valid", "ytype", "y", "ylong", "yscale", "gt_out", "row")), "forward = tokens + labelled: the batch"
    head = JK.JTokenHead(d=32, heads=4, layers=2, ffn=64).eval()                                                 # the head reads the INPUT keys only: the tokens alone give the same output as the whole batch
    with torch.no_grad():
        h1, h2 = head(LLD.head_inputs(tk_)), head(b_)
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
        og = src.live(x, tabs, ["a", "b"], [0, 1])["batch"]["lab"]
        on = src.live(x, tabs, ["a", "b"], [0, 1], gate=False)["batch"]["lab"]
    assert sum(lab["n_reach"] for lab in og) > 0 and all(lab["n_reach"] == 0 for lab in on), "gate=False reaches the labels through the source"
    tg = {k: torch.from_numpy(np.stack([LLD.node_targets(t)[k] for t in tabs])) for k in LLD.NODE_KEYS}
    r = gradient_check(src, JK.JTokenHead(d=32, heads=4, layers=2, ffn=64), dict(x=x, table=tabs, token=["a", "b"], row=[0, 1], targets=tg))        # the card's gradient check
    assert r["params"] == len(names) and r["grad_norm_from_head"] > 0 and r["grad_norm_from_b"] > 0
    o = LN.LaneNet(_toy_model(), K=K).live(x, tabs, ["a", "b"], [0, 1])
    assert not o["batch"]["tok"].requires_grad and not o["batch"]["pos"].requires_grad, "train_nodes=False: no graph in the batch"


def _t_vs_cache():
    K = 16
    got = labelled_batch(LN.live_tokens(toy_no(2), K), [toy_table([[-9.0, 5.0], [-7.0, 5.0]], [(0, 1)], [False]), toy_table(np.zeros((0, 2)), [], [])], ["a", "b"], [1, 2], K)
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
    sp, s0 = LN.LaneNet(_toy_model(), K=K, train_nodes=True, train_pyr=True), LN.LaneNet(_toy_model(), K=K, train_nodes=True)
    names = {n for n, p in sp.named_parameters() if p.requires_grad}
    assert names == {"b.pyr." + n for n, _ in sp.b.pyr.named_parameters()} | {"b.nodes." + n for n, _ in sp.b.nodes.named_parameters()}, "train_pyr: the pyramid and the node head train, nothing else"
    assert len(sp.pyr_params()) == len(list(sp.b.pyr.parameters())) and len(sp.node_params()) == len(list(sp.b.nodes.parameters())) and len(sp.trainable()) == len(sp.pyr_params()) + len(sp.node_params())
    assert not s0.pyr_params() and len(s0.node_params()) == len(s0.trainable())
    refuses(lambda: LN.LaneNet(_toy_model(), K=K, train_pyr=True), "train_pyr needs train_nodes")
    sp.train()
    assert not any(mm.training for mm in sp.b.pyr.modules()), "the pyramid stays in eval mode"
    got = {}
    for src in (sp, s0):
        no, tk = src.tokens(x)
        sum(tk[k].float().sum() for k in LLD.GRAPH_KEYS).backward()
        got[src is sp] = tk
    assert all(torch.equal(got[True][k].detach(), got[False][k].detach()) for k in LLD.GRAPH_KEYS + ("valid", "cell")), "train_pyr changes the tokens"
    held = {n for n, p in sp.b.pyr.named_parameters() if p.grad is not None and bool(p.grad.ne(0).any())}
    assert held == {n for n, _ in sp.b.pyr.named_parameters() if not n.startswith("lines05.")}, f"the loss on the tokens does not reach every pyramid tensor the node head reads: {sorted(set(n for n, _ in sp.b.pyr.named_parameters()) - held)}"
    cw, cb = sp.b.pyr.centre1.weight.grad.flatten(1), sp.b.pyr.centre1.bias.grad
    read = [0] + list(range(6, 12))
    assert bool(cw[1:6].eq(0).all()) and bool(cb[1:6].eq(0).all()) and bool(cw[read].ne(0).any(1).all()), "centre1: only the channels 0 and 6 - 11 are read"
    assert all(p.grad is None for p in s0.b.pyr.parameters()), "without train_pyr the pyramid holds no gradient"
    with torch.no_grad():
        assert not sp.tokens(x)[1]["tok"].requires_grad, "train_pyr under an outer no_grad: no graph"


def _t_guards():
    refuses(lambda: fixture(n=0), "nothing to compare")                                                          # no vacuous PASS
    refuses(lambda: fixture(seeds=()), "nothing to compare")
    refuses(lambda: fixture(n=6, gpu=True, bs=4), "not a multiple of bs")
    near = dict(common_min=512, floats=dict({k: 0.5 * v for k, v in CPU_TOL.items()}, e=1.2))                    # the CPU's distance to the cache: inside, a far position, lost cells
    assert live_close(near) and not live_close(dict(near, floats=dict(near["floats"], pos=2.0))) and not live_close(dict(near, common_min=500))                                        # a short last batch is not a batch of the cache build


def _t_graph():
    """features -> graph in one forward: `LaneNet.graph` gives the graphs that the predict path's records give (tokens -> labelled batch -> the head -> `batch_dump` -> `lane_readout.graph`), at the given decode;
    a stage-2 checkpoint loads back strictly and gives the same graphs (`from_checkpoint`: its own `b_state` in the architecture of `b_args`, or on top of the named B' run), a file without a B' needs its B' run"""
    K, B = 64, 2
    torch.manual_seed(3)
    b_args = dict(width=16, soft_nms=["mnms", 1.75, 0.1])
    b = QG.from_args(b_args)
    with torch.no_grad():
        b.nodes.head1.bias[0] = 3.0                                                                              # existence p = 0.95 at the start: many valid tokens
    head = LN.build_head({})
    head.set_step(JK.SWITCH)                                                                                     # the mode of a trained head: the key weight of its own p_none
    net = LN.LaneNet(b, head, K=K).eval()
    assert [n.split(".")[0] for n, _ in net.named_parameters()][0] == "head" and not any(p.requires_grad for p in net.b.parameters()), "the head's tensors come first; B' is frozen"
    x = torch.randn(B, 39, 200, 200).half()
    d = RO.Decode(link_thr=0.05, link_thr_long=0.05, keep_thr=0.9, none_gate=False)
    got, rec = net.graph(x, d), net.graph(x)
    with torch.no_grad():
        no, tk = net.tokens(x)
        lb = net.labelled(tk, [toy_table([[-9.0, 5.0]], [], [])] * B, ["a", "b"], [0, 1])
        o2 = net.head(lb)                                                                                        # the predict path: the head on the labelled batch
        recs = RO.batch_dump(o2, lb)
        out, tk2 = net(x)
    assert all(torch.equal(tk[k], tk2[k]) for k in tk) and sorted(out) == sorted(o2) and all(torch.equal(out[k], o2[k]) for k in out if torch.is_tensor(out[k])), "forward = tokens + the head on the head's inputs"
    want = [RO.graph(t, f, t["valid"], d) for t, f in zip(lb["tokens_np"], recs)]
    same = lambda p, q: all(np.array_equal(p[g][k], q[g][k]) for g in ("nodes", "links") for k in p[g] if p[g][k] is not None)  # noqa: E731
    assert len(got) == B and all(sorted(p) == ["assign", "links", "maps", "nodes"] and same(p, q) for p, q in zip(got, want)), "graph(x) is not the read-out of the predict path's records"
    n_nodes, n_links = sum(len(p["nodes"]["e"]) for p in got), sum(len(p["links"]["src"]) for p in got)
    assert n_nodes > 0 and n_links > 0 and not all(same(p, q) for p, q in zip(got, rec)), f"the toy delivers nothing to compare ({n_nodes} nodes, {n_links} links) or the decode changes nothing"
    refuses(lambda: net.train().graph(x), "in eval mode")
    net.eval()
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "b"))
        torch.save(dict(state=b.state_dict(), args=b_args), os.path.join(tmp, "b", "model.pt"))
        torch.save(dict(state=head.state_dict(), args={}, b_state=b.state_dict(), b_args=b_args), os.path.join(tmp, "s2.pt"))
        torch.save(dict(state=head.state_dict(), args={}), os.path.join(tmp, "s1.pt"))
        for path, b_run in (("s2.pt", None), ("s2.pt", os.path.join(tmp, "b")), ("s1.pt", os.path.join(tmp, "b"))):
            back, ck = LN.LaneNet.from_checkpoint(os.path.join(tmp, path), K=K, b_run=b_run)
            assert not back.training and not back.b.nodes.training and not any(p.requires_grad for p in back.b.parameters()), "a loaded network is frozen and in eval mode"
            assert all(same(p, q) for p, q in zip(back.graph(x, d), got)), f"{path} (b_run {b_run}) loaded back gives another graph"
        refuses(lambda: LN.LaneNet.from_checkpoint(os.path.join(tmp, "s1.pt"), K=K), "holds no B'")
        torch.save(dict(state=head.state_dict(), args=dict(rounds=2), b_state=b.state_dict(), b_args=b_args), os.path.join(tmp, "old.pt"))
        refuses(lambda: LN.LaneNet.from_checkpoint(os.path.join(tmp, "old.pt"), K=K), "options that no longer exist")


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
    _t_graph()
    print("test_lane_net selftest ok")


def commands(argv):
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
    a = ap.parse_args(argv)
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
            with open(a.out if os.path.isabs(a.out) else os.path.join(BP.ROOT, a.out), "w") as fh:
                json.dump(r, fh, indent=1)
        ok = r["ok"] if a.gpu else r["labels_ok"] and (r["live_close"] or a.bs != 4)                                                           # the CPU has no say on the model's numerics, only on the label step
        print(f"FIXTURE bs {a.bs} " + ("PASS" if ok else "FAIL") + (" (the whole path is exactly the cache" + ("" if r["parity_at_cache_bs"] else f", at bs {a.bs}, NOT the cache build's bs 4") + ")" if a.gpu else " (CPU: the label step on the cached tokens is exact; the token path within CPU_TOL of the cache at bs 4, exact only on the GPU)"))
        sys.exit(0 if ok else 1)


# ------------------------------------------------------------------------------------------------------------------------------ the mutants (the text below this line is not mutated)
HERE = os.path.abspath(__file__)
SOURCES = dict(N=os.path.join(BP.ROOT, "net", "model", "lane_net.py"), D=os.path.join(BP.ROOT, "net", "prep", "lane_live_data.py"), T=HERE)
ENV = dict(os.environ, BFLC_ROOT=BP.ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")                    # a shared box: one thread per test process
ENV.pop("LANE_MUTANT", None)
# `import numpy, torch` walks the project directories of PYTHONPATH on /mnt/c first: the children import the two with that path moved behind site-packages, then restore it
LAUNCH = ("import os, sys, runpy; o = list(sys.path); r = os.environ['BFLC_ROOT']; sys.path[:] = [p for p in o if not p.startswith(r)] + [p for p in o if p.startswith(r)]; import numpy, torch; sys.path[:] = o; "
          "p = sys.argv[1]; sys.argv = sys.argv[1:]; runpy.run_path(p, run_name='__main__')")
ASSERTED = re.compile(r"^(AssertionError|RuntimeError: accepted although)", re.M)                       # how a test names its own failure
MARK = "\n# " + "-" * 126 + " the mutants (the text below this line is not mutated)\n"

HEAD_IN = "return {k: tk[k] for k in GRAPH_KEYS + (\"valid\",)}"
LABEL_CALL = "tables[b], K=K, allowed=None if allowed is None else allowed[b])"
TNP = "floats.update({\"np_\" + k: gap(tok(a[\"tnp\"], k, ja), tok(w[\"tnp\"], k, jw)) for k in NP_KEYS})"
CHECK_ORDER = "check_order(cache_dir or LP.cache_dir(split, seed, ident), self.rows, self.tokens)"
ORDER_ASSERT = "assert [int(r) for r in fr[\"rows\"]] == [int(r) for r in rows] and list(fr[\"tokens\"]) == list(tokens)"

MUTANTS = [
    ("D", "the item of frame i is frame i + 1", "o = self.td[i]", "o = self.td[(i + 1) % len(self.rows)]"),
    ("D", "every frame of a batch gets the table of the first", "table=[o[\"table\"] for o in b]", "table=[b[0][\"table\"] for o in b]"),
    ("D", "the TEST guard is removed", "check_rows(split, self.rows)  ", "pass  "),
    ("D", "the TEST guard ignores the order of VAL", "assert [int(r) for r in rows] == sv, \"the rows of the VAL cache", "assert sorted(int(r) for r in rows) == sorted(sv), \"the rows of the VAL cache"),
    ("D", "LiveData skips the order check", CHECK_ORDER, "pass"),
    ("D", "the order check ignores the rows", ORDER_ASSERT, "assert list(fr[\"tokens\"]) == list(tokens)"),
    ("D", "the order check ignores the tokens", ORDER_ASSERT, "assert [int(r) for r in fr[\"rows\"]] == [int(r) for r in rows]"),
    ("D", "the checkpoint of the cache is not checked", "assert got == want, f\"{run}/model.pt", "assert True, f\"{run}/model.pt"),
    ("N", "tok is not rounded to float16", "return x.half().float()", "return x.clone()"),
    ("N", "the gradient of tok is rounded to float16", "    def backward(ctx, g):\n        return g", "    def backward(ctx, g):\n        return g.half().float()"),
    ("N", "tok is gathered from the wrong cells", "tk[\"cell\"].long()[:, None, :].expand(", "tk[\"cell\"].long().flip(1)[:, None, :].expand("),
    ("N", "the graph is cut in the gather of tok", "tok = no[\"tok\"].flatten(2).gather(", "tok = no[\"tok\"].detach().flatten(2).gather("),
    ("D", "the graph is cut in the token inputs", HEAD_IN, "return {k: tk[k].detach() for k in GRAPH_KEYS + (\"valid\",)}"),
    ("D", "e and e_raw are swapped", HEAD_IN, "return {k: tk[\"e_raw\" if k == \"e\" else \"e\" if k == \"e_raw\" else k] for k in GRAPH_KEYS + (\"valid\",)}"),
    ("D", "the head's inputs lack valid", HEAD_IN, "return {k: tk[k] for k in GRAPH_KEYS}"),
    ("D", "the head's input keys lose the flags", "GRAPH_KEYS = tuple(k for k in LP.TOKEN_FIELDS if k != \"valid\")", "GRAPH_KEYS = tuple(k for k in LP.TOKEN_FIELDS if k not in (\"valid\", \"flags\"))"),
    ("D", "the labels are those of another K", LABEL_CALL, "tables[b], K=K // 2, allowed=None if allowed is None else allowed[b])"),
    ("D", "the labels are made of tokens that carry the graph", "return {k: tk[k].detach().cpu().numpy() for k in", "return {k: tk[k].cpu().numpy() for k in", "requires grad"),
    ("D", "the pair gate is read from the wrong direction", "else allowed[b])", "else allowed[b].T)"),
    ("N", "the pair gate is computed although it is off", "pair_gate(tk, self.K) if gate else None", "pair_gate(tk, self.K) if True else None"),
    ("N", "the network does not pass the gate on", "batch=self.labelled(tk, tables, token, row, gate))", "batch=self.labelled(tk, tables, token, row))"),
    ("D", "a frame over PMAX is not named", "raise LabelOverflow(token[b], int(m.group(1))) from e", "raise"),
    ("D", "a PMAX overflow names the wrong frame", "raise LabelOverflow(token[b], int(m.group(1)))", "raise LabelOverflow(token[0], int(m.group(1)))"),
    ("D", "a PMAX overflow names the wrong count", "raise LabelOverflow(token[b], int(m.group(1)))", "raise LabelOverflow(token[b], 0)"),
    ("D", "ylong is classed by twice the chord", "ylong=LP.chord_long(y, tnp[\"pos\"])", "ylong=LP.chord_long(y, 2 * tnp[\"pos\"])"),
    ("D", "ylong marks every positive pair", "ylong=LP.chord_long(y, tnp[\"pos\"])", "ylong=(y > 0)"),
    ("D", "gt_out is lost", "gt_out=int(lab[\"gt_out\"])", "gt_out=0"),
    ("D", "the row of a frame is that of the first", "row=int(row[b])", "row=int(row[0])"),
    ("D", "the token of a frame is that of the first", "token=token[b], row=", "token=token[0], row="),
    ("D", "tokens_np s_p is the existence", "s_p=tn[\"s_p\"][b].astype(np.float64)", "s_p=tn[\"e_p\"][b].astype(np.float64)"),
    ("D", "tokens_np e_p is the section start", "e_p=tn[\"e_p\"][b].astype(np.float64)", "e_p=tn[\"s_p\"][b].astype(np.float64)"),
    ("D", "tokens_np flags_p is the existence", "flags_p=tn[\"flags_p\"][b].astype(np.float64)", "flags_p=tn[\"e_p\"][b][:, None].repeat(8, 1).astype(np.float64)"),
    ("D", "tokens_np dirv is the position", "dirv=tn[\"dirv\"][b].astype(np.float64)", "dirv=tn[\"pos\"][b].astype(np.float64)"),
    ("D", "the label gd is lost", "label=dict(gt_id=lab[\"gt_id\"], gd=lab[\"gd\"],", "label=dict(gt_id=lab[\"gt_id\"], gd=np.zeros_like(lab[\"gd\"]),"),
    ("D", "the label flags are those of the aux", "gt_flags=lab[\"gt_flags\"], gt_aux=lab[\"gt_aux\"]", "gt_flags=np.pad(lab[\"gt_aux\"], ((0, 0), (0, 5))), gt_aux=lab[\"gt_aux\"]"),
    ("D", "the node targets are transposed", "return {k: g[k] for k in NODE_KEYS}", "return {k: np.swapaxes(g[k], -1, -2) for k in NODE_KEYS}"),
    ("N", "the pyramid is left in train mode", "        super().train(mode)\n        self.b.eval()\n", "        super().train(mode)\n"),
    ("N", "the node head is frozen", "self.b.nodes.parameters():\n                p.requires_grad_(True)", "self.b.nodes.parameters():\n                p.requires_grad_(False)"),
    ("N", "train_pyr leaves the pyramid under no_grad", "with contextlib.nullcontext() if self.train_pyr else torch.no_grad():", "with torch.no_grad():"),
    ("N", "train_pyr does not unfreeze the pyramid", "        if self.train_pyr:\n            for p in self.b.pyr.parameters():", "        if False:\n            for p in self.b.pyr.parameters():"),
    ("N", "train_pyr unfreezes the whole of B'", "            for p in self.b.pyr.parameters():\n                p.requires_grad_(True)", "            for p in self.b.parameters():\n                p.requires_grad_(True)"),
    ("N", "train_pyr without train_nodes is accepted", "assert train_nodes or not train_pyr,", "assert True,"),
    ("N", "pyr_params lists the node head", "return [p for p in self.b.pyr.parameters() if p.requires_grad]", "return [p for p in self.b.nodes.parameters() if p.requires_grad]"),
    ("N", "node_params lists every trained tensor", "return [p for p in self.b.nodes.parameters() if p.requires_grad]", "return self.trainable()"),
    ("N", "the pyramid is not frozen", "        for p in self.b.parameters():\n            p.requires_grad_(False)", "        for p in self.b.nodes.parameters():\n            p.requires_grad_(False)"),
    ("N", "train_nodes=False is ignored", "        if train_nodes:\n", "        if True:\n"),
    ("N", "trainable() lists the frozen parameters", "return [p for p in self.b.parameters() if p.requires_grad]", "return [p for p in self.b.parameters()]"),
    ("N", "B' runs inside the trainer's autocast", "with torch.autocast(device_type=dev.type, enabled=False) if float32 else contextlib.nullcontext():", "with contextlib.nullcontext():"),
    ("N", "graph() ignores the decode it is given", "return [RO.graph(t, f, t[\"valid\"], decode) for", "return [RO.graph(t, f, t[\"valid\"]) for"),
    ("N", "graph() writes the nodes of the first frame", "frames = [LLD.tokens_np(tn, b) for b in range(len(x))]", "frames = [LLD.tokens_np(tn, 0) for b in range(len(x))]"),
    ("N", "a checkpoint's trained B' is not loaded", "        if \"b_state\" in ck:\n            b.load_state_dict(ck[\"b_state\"])\n", ""),
    ("N", "a checkpoint of a removed option loads", "        check_removed(ck[\"args\"], path)\n        head = build_head(ck[\"args\"])", "        head = JK.JTokenHead(d=ARCH[\"d\"], heads=ARCH[\"heads\"], layers=ARCH[\"layers\"], ffn=ARCH[\"ffn\"])"),
    ("N", "a loaded network stays in train mode", "return cls(b, head, K).to(device).eval(), ck", "return cls(b, head, K).to(device), ck"),
    ("T", "compare_frame aligns by rank, not by cell", "    ja, jw = np.array([ia[c] for c in com], int), np.array([iw[c] for c in com], int)", "    ja = jw = np.arange(len(a[\"cell\"]))"),
    ("T", "compare_frame ignores the pairs y", "for k in (\"y\", \"ylong\")})", "for k in (\"ylong\",)})"),
    ("T", "compare_frame counts a one-sided NaN as equal", "np.where(np.isnan(x) | np.isnan(y), np.inf,", "np.where(np.isnan(x) | np.isnan(y), 0.0,"),
    ("T", "compare_frame ignores the label counts of gt_id", "for k in (\"gt_id\", \"gt_flags\", \"gt_aux\"))", "for k in (\"gt_flags\", \"gt_aux\"))"),
    ("T", "compare_frame ignores s_p of tokens_np", TNP, TNP.replace("for k in NP_KEYS})", "for k in NP_KEYS if k != \"s_p\"})")),
    ("T", "compare_frame ignores the valid of tokens_np", "np_valid=int((tok(a[\"tnp\"], \"valid\", ja) != tok(w[\"tnp\"], \"valid\", jw)).sum())", "np_valid=0"),
    ("T", "the fixture compares tokens_np with itself", "tnp=o[\"tokens_np\"]", "tnp=live[\"tnp\"]"),
    ("T", "the fixture ignores the label counts", "stats += int(out[-1][\"same_order\"] and any(", "stats += int(False and any("),
    ("T", "the fixture accepts nothing to compare", "assert n > 0 and len(seeds) > 0 and bs > 0, f\"nothing to compare", "assert True, f\"nothing to compare"),
    ("T", "the CPU fixture accepts a far position", "all(agg[\"floats\"][k] <= v for k, v in CPU_TOL.items())", "True"),
    ("T", "the CPU fixture accepts lost cells", "agg[\"common_min\"] >= CPU_COMMON and", ""),
    ("N", "graph() runs in train mode", "        assert not self.training,", "        assert True,"),
    ("T", "the GPU fixture accepts a short last batch", "assert n % bs == 0 or not gpu, f\"--gpu: n", "assert True, f\"--gpu: n"),
]


def run_selftest():
    r = subprocess.run([sys.executable, "-c", LAUNCH, HERE, "selftest"], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, f"selftest failed:\n{r.stdout[-800:]}{r.stderr[-2500:]}"
    print(r.stdout.strip().splitlines()[-1], flush=True)


def run_real():
    """the real-data commands: the node targets against QData and the cache's checkpoint (VAL and TRAIN, 20 frames each), the fixture on 8 frames of both seeds at bs 4 and of one seed at bs 8 (exit code 0 = the label step on the cached tokens is exactly the cache at any
    batch size) and the gradient check on 2 real frames with the champion's stage-1 head e373_cb_s0 (the head's loss reaches every node-head parameter of the real B', none of the frozen stages, the pyramid is unchanged by a step)"""
    for args in (["check", "--split", "val", "--n", "20"], ["check", "--split", "train", "--n", "20"], ["fixture", "--n", "8", "--seeds", "0", "1"], ["fixture", "--n", "8", "--seeds", "1", "--bs", "8"], ["grad", "--n", "2", "--head", "e373_cb_s0"]):
        r = subprocess.run([sys.executable, "-c", LAUNCH, HERE] + args, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{' '.join(args)} failed:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"
        print(f"{' '.join(args)}: {r.stdout.strip().splitlines()[-1]}", flush=True)


def one(args):
    k, (where, name, a, b, *expect), tmp = args
    with open(SOURCES[where], encoding="utf-8") as fh:
        src = fh.read()
    body, rest = src.split(MARK) if where == "T" else (src, None)
    if body.count(a) != 1:
        return name, f"NOT APPLIED ({body.count(a)} matches)"
    d = os.path.join(tmp, str(k))
    os.makedirs(d)
    p = os.path.join(d, os.path.basename(SOURCES[where]))
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(body.replace(a, b) + (MARK + rest if where == "T" else ""))
    r = subprocess.run([sys.executable, "-c", LAUNCH, p if where == "T" else HERE, "selftest"], capture_output=True, text=True, env=ENV if where == "T" else dict(ENV, LANE_MUTANT=p))
    if r.returncode == 0:
        return name, "SURVIVED"
    if (re.compile(expect[0]) if expect else ASSERTED).search(r.stderr + r.stdout):
        return name, "killed"
    return name, "CRASHED: " + (r.stderr.strip().splitlines() or ["no output"])[-1][:140]


def run_mutants(workers=4):
    left = []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(workers) as ex:
        for name, res in ex.map(one, [(k, m, tmp) for k, m in enumerate(MUTANTS)]):
            print(f"{res:<12} {name}", flush=True)
            if res != "killed":
                left.append(name)
    print(f"{len(MUTANTS) - len(left)} of {len(MUTANTS)} killed", flush=True)
    return left


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "selftests":
        commands(sys.argv[1:])
        sys.exit(0)
    run_selftest()
    run_real()
    if len(sys.argv) < 2:
        left = run_mutants()
        assert not left, f"mutants survived or crashed: {left}"
    print("test_lane_net ok")
