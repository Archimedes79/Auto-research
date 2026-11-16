"""Read-out, scoring, controls and checks of E365 S1 (the token head on the predicted nodes of the frozen lane head B', `training/grid/jtoken.py`); the cache and the loader are `training/data/lane_tokcache.py` / `lane_ptok_data.py`.

    wsl bash tools/env/_py14.sh jtoken_eval.py dump  --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0] [--cpu] [--max N]   # GPU: the head on the cache tokens -> runs/R/token_<split>[_drop..].pkl
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0]                     # CPU -> runs/R/token_card_<decode>[_train][_drop..].json (rec, argmax; never overwritten)
    wsl bash tools/env/_py14.sh jtoken_eval.py ctl   --seed S --ident e292_a|e273_p [--arms hi,hi_y,rule,floor,bsec0_plain,bsec0_close] [--max N]   # CPU: the controls -> runs/eval/e365_ctl/<arm>_<ident>_s<S>.json
    wsl bash tools/env/_py14.sh jtoken_eval.py check-layer1 --seed S [--dump PATH]                                                       # CPU: this scorer fed B's stored dump reproduces runs/eval/e324s_val/card_s<S>.json (PASS / FAIL)
    wsl bash tools/env/_py14.sh jtoken_eval.py check-parity --run R --seed S [--b-run runs/e292_a_s<S>] [--n 20]                         # CPU: the head from the cache tokens (fp16 tok) against tokens recomputed live from B' (fp32)
    wsl bash tools/env/_py14.sh jtoken_eval.py timing --run R [--b-run runs/e292_a_s<S>] [--K 384] [--cpu]                               # parameters, GMACs, ms (GPU only when idle: < 2 GB used and no run.py job)

Read-out of record (named before the run, on the output only): a node is kept when p_none < 0.5 (e = 1 - p_none); decode `rec`: a link i -> j of class c when p^c(j|i) >= 0.3, p^c(j|i) >= p^c(none|i) and both ends are kept (a fork keeps both
branches); decode `argmax`: per source and class the argmax over {none} + the candidates (one link per source and class). The prediction is written by `lane_ptok_data.to_pred` from the cached TOKENS only (pos, dirv, B''s flags and s; every node in
section 0): no field of the GT table enters (`check_writer` asserts it on every frame) and the scorer is the standard one (`qgraph_eval.readout` -> `graph_counts` -> `graph_numbers` through `score_frames`, inside the `jtok.lattice(2.0)` block because
`LaneRef` builds a node table at score time; the headline graph numbers use the old graph GT and do not depend on it: `check-layer1` proves it). Handle scales: the head's for the long links, 1/3 for the short ones. p = 1 is written for every kept link (the
decode is the selection); the probability stays in the dump. A TRAIN read-out is 873 frames spaced evenly over the cache (`sample`), the degraded read-out (`--drop`) removes the share of the tokens that E357's `keep_nodes` removes (seeded by `--drop-seed` and
the frame token) before the head reads them.
Controls (no learning, the same tokens, the same writer and scorer): hi = the GT links of the frame's table between the ASSIGNED tokens (nodes = the assigned tokens), contracted over the GT nodes without an assigned token; hi_y = the cache's y
pairs between assigned tokens (diagnostic: y also holds tokens that are assigned to the successor's neighbours); rule = E357's `lo_near` on the nodes with
B''s e >= 0.5; floor = the untrained head (random init seed 0, the key weights of a trained head), decode of record; bsec0_plain / bsec0_close = B's own stored nodes and links with every section 0, read out as `plain@0.5` / `close:1.5@0.5`
(`--ident e292_a`: B' with the Matrix-NMS layer, `e273_p`: plain B). Tests: evaluation/tests/test_jtoken_eval.py
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtok as JT  # noqa: E402
import jtok_eval as JV  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402

RUNS = JV.RUNS
CTL = os.path.join(JV.CARDS, "e365_ctl")
SCRATCH = os.path.join(ROOT, "temp", "e365")
DECODES = ("rec", "argmax")
ARMS = ("hi", "hi_y", "rule", "floor", "bsec0_plain", "bsec0_close")
LATTICE = 2.0
NSAMPLE = 873                                                               # frames of a TRAIN read-out
KEEP_P = 0.05                                                               # candidate links below this probability are not written into a dump
KEYS = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "Q", "junction_F1", "dup_frac", "vert100_dp")
CHECK = ("topo_F1", "F1@1.0", "lanes_found", "dup_frac", "vert100_dp")      # the layer-1 numbers (to 1e-6)
BASE_CARD = os.path.join(JV.CARDS, "e324s_val", "card_s{seed}.json")        # variants.primary.numbers.base = plain B with the close-node step, B's own sections
B_DUMPS = dict(e273_p="temp/e343/val/predfull_e273_p_s{seed}.pkl", e292_a="temp/e292/predfull_e292_a_s{seed}.pkl")      # the fresh VAL dump of plain B made the E324 s card; B' = the dump of E292
CLOSE, PLAIN = "close:1.5@0.5", "plain@0.5"
NODE_KEYS = ["cell", "dirv", "e", "flags", "pos", "s", "section"]
THIRD = 1 / 3
MACS_NOTE = "matmul, linear and einsum only; softmax, norms, GELU, the pair-geometry features and the sort are not counted"


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x.item() if isinstance(x, (np.generic,)) else (x.tolist() if isinstance(x, np.ndarray) else x)


# ------------------------------------------------------------------------------------------------------------------------------ tokens and the head
def cache_kw(args):
    """the loader arguments of a run's recipe (`jtoken_train` writes ident / K / cache into the run's args; an older smoke run has none: B', K found, the default root)"""
    return dict(seed=args["seed"], ident=args.get("ident") or "e292_a", K=args.get("K"), **({"cache": args["cache"]} if args.get("cache") else {}))


def open_cache(split, seed, ident="e292_a", K=None, cache=None):
    return LP.PTokData(split, seed, ident, K, 0.0, **({"cache": cache} if cache else {}))


def sample(n, split):
    """cache indices of the frames that are read out: all of VAL, NSAMPLE evenly spaced frames of TRAIN"""
    return np.arange(n) if split == "val" else np.unique(np.linspace(0, n - 1, NSAMPLE).round().astype(int))


def batches(ds, idx, bs=8, workers=0):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [int(i) for i in idx]), batch_size=bs, shuffle=False, num_workers=workers, collate_fn=LP.collate)


def apply_drop(b, drop, drop_seed):
    """the degraded read-out: every token of a frame is removed with probability `drop` (`jtok_eval.keep_nodes`, seeded by (drop_seed, the frame token)) before the head reads it; the removed token is not valid (the loader's `token_drop` does the same with
    the cache seed as drop seed)"""
    if drop > 0:
        keep = np.stack([JV.keep_nodes(t, b["valid"].shape[1], drop, drop_seed) for t in b["token"]])
        b["valid"] = b["valid"] & torch.from_numpy(keep)
        for t, k in zip(b["tokens_np"], keep):
            t["valid"] = t["valid"] & k
    return b


def run_head(m, b, dev):
    with torch.no_grad():
        return m(JKT.to_device(b, dev))


def links_of(link, p, longm, scale):
    """the links of one frame from the boolean (K, K) link matrix of a decode, the probabilities, the chord class (> 8 m) and the head's handle scales (K, K, 2): the head's scales for the long links, 1/3 for the short ones"""
    src, dst = np.nonzero(link)
    lg = longm[src, dst]
    ab = np.where(lg[:, None], scale[src, dst], THIRD)
    return dict(src=src.astype(np.int16), dst=dst.astype(np.int16), p=p[src, dst].astype(np.float32), long=lg, a=ab[:, 0].astype(np.float32), b=ab[:, 1].astype(np.float32))


def batch_dump(out, b):
    """the head's output of a batch -> one dict per frame: token, row, cell (the tokens' cells: the score checks them against the cache), pnone / ptype (K,), keep (K,), none_cls (2, K) = p^c(none|i), cand = the candidate links with p >= KEEP_P
    (src, dst, p, long), rec / argmax = the links of the two decodes (src, dst, p, long, a, b)"""
    K = out["cand"].shape[-1]
    pr, rec, alt = JK.probs(out), JK.decode(out), JK.decode_argmax(out)
    npy = lambda t: t.detach().cpu().numpy()  # noqa: E731
    pt, p, cand, longm, scale, none_cls = (npy(x) for x in (pr["pt"], pr["p"], pr["cand"], pr["longm"], out["scale"].float(), out["logp"][..., K].exp()))
    keep, lrec, lalt = npy(rec["keep"]), npy(rec["link"]), npy(alt["link"])
    frames = []
    for i in range(len(b["token"])):
        cs, cd = np.nonzero((p[i] >= KEEP_P) & cand[i])
        frames.append(dict(token=b["token"][i], row=int(b["row"][i]), cell=np.asarray(b["tokens_np"][i]["cell"], np.int16), pnone=pt[i, :, 0].astype(np.float32), ptype=pt[i].astype(np.float16), keep=keep[i],
                           none_cls=none_cls[i].astype(np.float16), cand=dict(src=cs.astype(np.int16), dst=cd.astype(np.int16), p=p[i][cs, cd].astype(np.float32), long=longm[i][cs, cd]),
                           rec=links_of(lrec[i], p[i], longm[i], scale[i]), argmax=links_of(lalt[i], p[i], longm[i], scale[i])))
    return frames


# ------------------------------------------------------------------------------------------------------------------------------ the writer
def check_writer(pred, tokens, keep):
    """the prediction of a frame holds the kept TOKENS and nothing else: the node fields are the tokens' own (pos, dirv, B''s flags and s) at the kept indices, one section, no assignment, no maps, e >= 0.5 (a kept node), the links name nodes of it"""
    nd, lk, idx = pred["nodes"], pred["links"], np.flatnonzero(keep)
    assert sorted(nd) == NODE_KEYS and pred["assign"] is None and pred["maps"] is None and lk["curves"] is None, (sorted(nd), pred["assign"] is None, pred["maps"] is None)
    for k, v in (("pos", tokens["pos"]), ("dirv", tokens["dirv"]), ("flags", tokens["flags_p"]), ("s", tokens["s_p"])):
        assert np.array_equal(nd[k], np.asarray(v, float)[idx]), f"node field {k} is not the tokens' own"
    assert not np.any(nd["section"]) and len(nd["section"]) == len(idx) and bool((nd["e"] >= 0.5).all()), "section / e of the written nodes"
    assert all(len(lk[k]) == len(lk["src"]) for k in ("dst", "p", "long", "a", "b")) and (len(lk["src"]) == 0 or max(lk["src"].max(), lk["dst"].max()) < len(idx)), "links outside the nodes"
    return pred


def frame_prediction(tokens, f, decode):
    """the qformat prediction of one frame from its tokens (`tokens_np`) and its dump frame: kept = the head's p_none < 0.5, e = 1 - p_none, the links of the decode (p = 1)"""
    lk = dict(f[decode], p=np.ones(len(f[decode]["src"])))
    return check_writer(LP.to_pred(tokens, f["keep"], 1.0 - f["pnone"].astype(float), lk), tokens, f["keep"])


# ------------------------------------------------------------------------------------------------------------------------------ the controls
def hi_frame(tokens, gt_id, table):
    """c-hi, the ceiling of the token set: the GT links of the frame's table between the ASSIGNED tokens (`gt_id` >= 0: the nodes = the assigned tokens, e = 1), contracted over the GT nodes that have no assigned token (`jtok_eval.sub_table`);
    fitted handle scales for the long links that have one, 1/3 else -> (keep, e, links)"""
    a = np.asarray(gt_id) >= 0
    has = np.zeros(len(table["pos"]), bool)
    has[np.asarray(gt_id)[a]] = True
    tok_of = -np.ones(len(has), int)
    tok_of[np.asarray(gt_id)[a]] = np.flatnonzero(a)
    sub = JV.sub_table(table, has)
    nodes = np.flatnonzero(has)
    src, dst = tok_of[nodes[sub["src"]]], tok_of[nodes[sub["dst"]]]
    return a, np.ones(len(a)), _links(tokens, src, dst, np.asarray(sub["scale"], float).reshape(-1, 2))


def hi_y_frame(tokens, gt_id, y, yscale):
    """c-hi-y, a diagnostic: the cache's y pairs between assigned tokens (what a perfect head of the cache's targets would decode: y holds every token within 1.5 m of the GT successor, so also tokens that are assigned to the successor's
    neighbours) -> (keep, e, links)"""
    a = np.asarray(gt_id) >= 0
    ii, jj = np.nonzero(np.asarray(y).astype(bool) & a[:, None] & a[None, :])
    return a, np.ones(len(a)), _links(tokens, ii, jj, np.asarray(yscale)[ii, jj])


def _links(tokens, src, dst, scale):
    """the links of a control: p = 1, the long class by the tokens' chord, the fitted scales (NaN: none) for the long links, 1/3 else"""
    lg = np.linalg.norm(tokens["pos"][src] - tokens["pos"][dst], axis=1) > JT.SHORT
    ab = np.where(lg[:, None] & np.isfinite(scale), scale, THIRD)
    return dict(src=src, dst=dst, p=np.ones(len(src)), long=lg, a=ab[:, 0], b=ab[:, 1])


def rule_frame(tokens):
    """c-rule: E357's `lo_near` links on the nodes with B''s e >= 0.5 (valid tokens) -> (keep, e, links)"""
    keep = np.asarray(tokens["valid"], bool) & (np.asarray(tokens["e_p"]) >= 0.5)
    idx = np.flatnonzero(keep)
    pos, dirv = tokens["pos"][idx], tokens["dirv"][idx]
    src, dst = JV.rule_links(pos, dirv)
    s2, d2 = JV.open_links(pos, dirv, src, dst, every=False)
    src, dst = idx[np.concatenate([src, s2]).astype(int)], idx[np.concatenate([dst, d2]).astype(int)]
    return keep, np.asarray(tokens["e_p"], float), _links(tokens, src, dst, np.full((len(src), 2), np.nan))


def control_prediction(tokens, keep, e, links):
    return check_writer(LP.to_pred(tokens, keep, e, links), tokens, keep)


def floor_model():
    """c-floor: the untrained head, random init (seed 0), in the mode of a trained one (the key weight of its own p_none)"""
    torch.manual_seed(0)
    m = JK.JTokenHead()
    m.set_step(JK.SWITCH)
    return m.eval()


def pred_of_dump(d, spec, sec0):
    """a stored full dump frame of B (`qgraph_readout.dump_full`) -> the prediction as the read-out `spec` sees it (as `jtest_eval.score_dumps`); sec0: every node in section 0"""
    nodes = dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=np.zeros_like(d["section"]) if sec0 else d["section"], s=d.get("s"))
    return QR.apply(dict(nodes=nodes, links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), QR.parse(spec))


# ------------------------------------------------------------------------------------------------------------------------------ scoring
def frames_of(split, ds):
    """the scorer's Frame of every cache frame, in the cache order (VAL = `qgraph_eval.val_frames()`; TRAIN from the train files of the same layout)"""
    if split == "val":
        fr = QE.val_frames()
    else:
        tokens = json.load(open(f"{QE.LS.B123}/feats/train_tokens.json"))
        scene = np.load(f"{QE.LS.B123}/poses_train.npz")["scene"]
        gt = pickle.load(open(f"{QE.LS.B123}/graph_gt_train.pkl", "rb"))
        fr = [QE.Frame(k, int(r), tokens[r], int(scene[r]), gt[r]) for k, r in enumerate(ds.rows)]
    assert [f.token for f in fr] == list(ds.tokens), "the scorer's frames are not the frames of the cache, in order"
    return fr


def score_preds(preds, frames, log=print):
    """the standard scorer on a list of predictions (qformat dicts) inside the lattice block -> dict(graph, nodes: the flat numbers, rec: the per-frame records for the paired bootstrap, lane_missing)"""
    rec, miss = dict(graph=[], node=[], assign=[]), 0
    t0 = time.time()
    with JT.lattice(LATTICE):
        JT.check_lattice(LATTICE)
        for a in range(0, len(frames), 100):
            out, r = QE.score_frames(preds[a:a + 100], frames[a:a + 100], records=True)
            miss += out["lane_missing"]
            for k in rec:
                rec[k] += r[k]
            log(f"   {min(a + 100, len(frames))}/{len(frames)} frames, {time.time() - t0:.0f} s", flush=True)
    graph = {k: v for k, v in QE.graph_numbers(rec["graph"]).items() if k not in JV.SEC_VOID}
    return dict(graph=graph, nodes=QE.node_numbers(QE.pool_counts(rec["node"])) if rec["node"] else {}, rec=rec, lane_missing=miss)


def show(tag, m):
    print(f"{tag:<34}" + "  ".join(f"{k} {m.get(k, float('nan')):.4f}" for k in KEYS), flush=True)


def write_card(path, name, res, n_frames, split="val", **info):
    """a card of one scored arm: the graph numbers + the node numbers (`node_`), never overwritten"""
    assert not os.path.exists(path), f"a stored card is never overwritten: {path}"
    metrics = dict(res["graph"], **{f"node_{k}": v for k, v in res["nodes"].items()})
    card = JV.make_card(name, n_frames, metrics, "sec0", lane_missing=res["lane_missing"], **info)
    card["split"] = split
    card["protocol"]["scorer"] = "qgraph_eval.score_frames (readout -> graph_counts -> graph_numbers) inside the 2 m lattice block, against the old graph GT; nodes: node_numbers against the lane GT nodes (node_*)"
    JV.write_json(path, _jsonable(card))
    return card


def plain_b(seed, dump=None, cache=True):
    """plain B as the E324 s card delivers it (B's own sections, the close-node step, links at p >= 0.5, no connectors) scored by THIS scorer -> the result of `score_preds`; the per-frame records are kept in temp/e365/planb_s<S>.pkl
    (scratch) for the paired intervals"""
    path = os.path.join(SCRATCH, f"planb_s{seed}.pkl")
    if cache and dump is None and os.path.exists(path):
        return pickle.load(open(path, "rb"))
    dump_path = _abs(dump or B_DUMPS["e273_p"].format(seed=seed))
    d = pickle.load(open(dump_path, "rb"))
    frames = QE.val_frames()
    assert [x["row"] for x in d] == [f.row for f in frames], f"{dump_path} is not in the order of the VAL frames"
    res = score_preds([pred_of_dump(x, CLOSE, False) for x in d], frames)
    if cache and dump is None:
        os.makedirs(SCRATCH, exist_ok=True)
        pickle.dump(res, open(path, "wb"))
    return res


def compare_numbers(got, ref, keys=CHECK, tol=1e-6):
    """-> (ok, {key: got - ref}) for the keys of a stored card"""
    assert all(k in got and k in ref for k in keys), f"missing numbers: {[k for k in keys if k not in got or k not in ref]}"
    diff = {k: float(got[k]) - float(ref[k]) for k in keys}
    return all(abs(v) <= tol for v in diff.values()), diff


def g1(card_s, ctl_hi, ctl_rule):
    """the gap share (S1 - c-rule) / (c-hi - c-rule) on topo F1 (NaN when the controls do not differ)"""
    d = ctl_hi - ctl_rule
    return (card_s - ctl_rule) / d if abs(d) > 1e-12 else float("nan")


def read_ctl(arm, ident, seed):
    p = os.path.join(CTL, f"{arm}_{ident}_s{seed}.json")
    return json.load(open(p))["metrics"]["topo_F1"] if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------------------------------------------ commands
def dump(a, runs=RUNS):
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU; --cpu only for tests"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jtoken_eval dump {a.run}")
    m, args = JKT.load_model(a.run, dev, runs)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split)[: a.max or None]
    path = os.path.join(runs, a.run, f"token_{a.split}{JV.suffix(2.5, a.drop, a.drop_seed)}{f'_n{len(idx)}' if a.max else ''}.pkl")
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    frames = []
    for b in batches(ds, idx, workers=0 if a.cpu else 2):
        frames += batch_dump(run_head(m, apply_drop(b, a.drop, a.drop_seed), dev), b)
    meta = dict(run=a.run, seed=a.seed, split=a.split, drop=a.drop, drop_seed=a.drop_seed, idx=idx.tolist(), K=ds.K, ident=kw["ident"], cache=ds.dir, steps=args.get("steps"))
    with open(path, "wb") as fh:
        pickle.dump(dict(meta=meta, frames=frames), fh)
    kept = np.mean([f["keep"].sum() for f in frames])
    print(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames, kept nodes per frame {kept:.1f}, links per frame rec {np.mean([len(f['rec']['src']) for f in frames]):.1f} argmax {np.mean([len(f['argmax']['src']) for f in frames]):.1f}, device {dev}")
    return path


def score(a, runs=RUNS):
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    sfx = ("" if a.split == "val" else f"_{a.split}") + JV.suffix(2.5, a.drop, a.drop_seed)
    paths = {d: os.path.join(runs, a.run, f"token_card_{d}{sfx}.json") for d in DECODES}
    held = [p for p in paths.values() if os.path.exists(p)]
    assert not held, f"a stored card is never overwritten: {held}"
    d = pickle.load(open(os.path.join(runs, a.run, f"token_{a.split}{JV.suffix(2.5, a.drop, a.drop_seed)}.pkl"), "rb"))
    meta, fr = d["meta"], d["frames"]
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split)
    assert meta["idx"] == idx.tolist() and meta["drop"] == a.drop and meta["drop_seed"] == a.drop_seed and [f["token"] for f in fr] == [ds.tokens[i] for i in idx], "the dump is not the dump of this read-out (frames, drop)"
    toks = [ds[int(i)]["tokens_np"] for i in idx]
    assert all(np.array_equal(f["cell"], t["cell"]) for f, t in zip(fr, toks)), "the dump's tokens are not the cache's (a stale dump or cache)"
    every = frames_of(a.split, ds)
    frames = [every[int(i)] for i in idx]
    ref = plain_b(a.seed) if a.split == "val" and not a.drop else None
    params = JK.count_params(JK.JTokenHead())
    out = {}
    for dec in DECODES:
        res = score_preds([frame_prediction(t, f, dec) for t, f in zip(toks, fr)], frames)
        show(f"{a.run} {dec}{sfx}", res["graph"])
        extra = {}
        if ref is not None:
            extra["paired_vs_plain_b"] = QE.paired(res["rec"], ref["rec"], frames)
            extra["plain_b"] = {k: ref["graph"][k] for k in KEYS}
            lo, hi = read_ctl("hi", kw["ident"], a.seed), read_ctl("rule", kw["ident"], a.seed)
            extra["g1_topo_F1"] = None if lo is None or hi is None else g1(res["graph"]["topo_F1"], lo, hi)
        card = write_card(paths[dec], f"{a.run}_token_{dec}{sfx}", res, len(frames), a.split, run=a.run, seed=a.seed, decode=dec, drop=a.drop, drop_seed=a.drop_seed, steps=ck.get("steps"), head_params=params,
                          K=ds.K, ident=kw["ident"], decode_rule="rec: p >= 0.3 and p >= p(none), both ends kept (p_none < 0.5); argmax: one link per source and class", **extra)
        out[dec] = card
    return out


def ctl(a, runs=RUNS):
    ds = open_cache("val", a.seed, a.ident, a.K, a.cache)
    arms = a.arms.split(",") if a.arms else list(ARMS)
    assert all(x in ARMS for x in arms), arms
    n = a.max or len(ds)
    idx = np.arange(n)
    frames = QE.val_frames()[:n]
    assert [f.token for f in frames] == list(ds.tokens[:n])
    out_dir = os.path.join(SCRATCH, "out") if a.max else CTL
    paths = {x: os.path.join(out_dir, f"{x}_{a.ident}_s{a.seed}{f'_n{n}' if a.max else ''}.json") for x in arms}
    held = [p for p in paths.values() if os.path.exists(p) and not a.max]
    assert not held, f"a stored card is never overwritten: {held}"
    preds = {x: [] for x in arms if x in ("hi", "hi_y", "rule", "floor")}
    m = floor_model() if "floor" in arms else None
    if "hi" in preds:
        import lane_tok_data as TD
        td = TD.TokData("val", LATTICE, feats=False)
    if preds:
        k = 0
        for b in batches(ds, idx):
            fl = batch_dump(run_head(m, b, "cpu"), b) if m is not None else None
            for i, t in enumerate(b["tokens_np"]):
                gid = b["label"]["gt_id"][i].numpy()
                if "hi" in preds:
                    preds["hi"].append(control_prediction(t, *hi_frame(t, gid, td.table(k + i))))
                if "hi_y" in preds:
                    preds["hi_y"].append(control_prediction(t, *hi_y_frame(t, gid, b["y"][i].numpy(), b["yscale"][i].numpy())))
                if "rule" in preds:
                    preds["rule"].append(control_prediction(t, *rule_frame(t)))
                if "floor" in preds:
                    preds["floor"].append(frame_prediction(t, fl[i], "rec"))
            k += len(b["tokens_np"])
    for x in arms:
        if x.startswith("bsec0"):
            dump_path = _abs(a.dump or B_DUMPS[a.ident].format(seed=a.seed))
            d = pickle.load(open(dump_path, "rb"))[:n]
            assert [y["row"] for y in d] == [f.row for f in frames], f"{dump_path} is not in the order of the VAL frames"
            preds[x] = [pred_of_dump(y, PLAIN if x == "bsec0_plain" else CLOSE, True) for y in d]
    for x in arms:
        res = score_preds(preds[x], frames)
        show(f"ctl {x} {a.ident} s{a.seed}", res["graph"])
        write_card(paths[x], f"e365_ctl_{x}_{a.ident}_s{a.seed}", res, len(frames), "val", arm=x, ident=a.ident, seed=a.seed, K=ds.K, frames_scored=n)
    return paths


def check_layer1(a):
    """the scorer path fed B's stored predictions (the fresh VAL dump, B's own sections, the E275 close-node step, links at p >= 0.5, no connectors: what `jtest_eval.score_dumps` makes of the `plain` variant) against the stored base numbers of the
    E324 s card -> 0 (PASS: topo F1, F1@1.0, lanes found, dup_frac, vert100_dp within 1e-6) / 1 (FAIL); the other stored numbers are printed"""
    ref = json.load(open(BASE_CARD.format(seed=a.seed)))["variants"]["primary"]["numbers"]["base"]
    res = plain_b(a.seed, a.dump, cache=False)
    ok, diff = compare_numbers(res["graph"], ref)
    for k in ref:
        print(f"   {k}: {res['graph'][k]:.9f} stored {ref[k]:.9f} diff {res['graph'][k] - ref[k]:+.2e}" + ("" if k in CHECK else "   (not checked)"))
    print(f"{'PASS' if ok else 'FAIL'}: layer 1, seed {a.seed}: this scorer on B's stored predictions against {os.path.relpath(BASE_CARD.format(seed=a.seed), ROOT)}: " + ", ".join(f"{k} {v:+.2e}" for k, v in diff.items()))
    if a.dump is None and not os.path.exists(os.path.join(SCRATCH, f"planb_s{a.seed}.pkl")):
        os.makedirs(SCRATCH, exist_ok=True)
        pickle.dump(res, open(os.path.join(SCRATCH, f"planb_s{a.seed}.pkl"), "wb"))
    return int(not ok)


def live_tokens(mb, x, K, exact=False):
    """B' (`QP.load_model`) on the input features x (B, 39, 200, 200) -> the head's input batch exactly as `lane_tokcache.build` makes the cache (the same `select_tokens`); exact: tok in float32 (the cache stores float16)"""
    P = mb.pyr(x.float())
    no = mb.node_stage(P)
    t = LC.select_tokens(no, None, K)
    if exact:
        order = torch.argsort(no["e"].float().flatten(1), dim=1, descending=True, stable=True)[:, :K]
        t["tok"] = no["tok"].flatten(2).gather(2, order[:, None, :].expand(-1, no["tok"].shape[1], -1)).transpose(1, 2).float()
    return {k: t[k] for k in JK.INPUTS + ("cell",)}


def compare_outputs(o1, o2, tol):
    """two head outputs on the same frames -> (ok, numbers): the largest difference of p_none (valid tokens) and of the link probabilities (pairs of valid tokens), the kept nodes and the links of the decode of record that differ; PASS
    when both probability differences are within `tol` (a kept node or a link can only differ where its probability is within `tol` of its threshold)"""
    p1, p2, r1, r2 = JK.probs(o1), JK.probs(o2), JK.decode(o1), JK.decode(o2)
    v = o1["valid"] & o2["valid"]
    d_none = float((p1["pt"][..., 0] - p2["pt"][..., 0]).abs()[v].max())
    d_p = float((p1["p"] - p2["p"]).abs()[v[:, :, None] & v[:, None, :]].max())
    num = dict(max_dpnone=d_none, max_dp=d_p, kept_nodes=int(r1["keep"].sum()), kept_nodes_differ=int((r1["keep"] != r2["keep"]).sum()), links=int(r1["link"].sum()), links_differ=int((r1["link"] != r2["link"]).sum()), tol=tol)
    return d_none <= tol and d_p <= tol, num


def align(cells_a, cells_b):
    """(B, K) cell ids of two token sets -> perm (B, K) long: perm[b, j] = the index in set b of the token with the cell of set a's token j, -1 when the cell is not in set b (a near tie in B''s e can swap two tokens' ranks between a GPU and a
    CPU run: the head does not depend on the order of the tokens, so the two sets are compared by cell)"""
    perm = -np.ones(np.shape(cells_a), int)
    for b in range(len(perm)):
        where = {int(c): k for k, c in enumerate(np.asarray(cells_b[b]))}
        perm[b] = [where.get(int(c), -1) for c in np.asarray(cells_a[b])]
    return torch.from_numpy(perm)


def reorder(batch, perm):
    """the head's input batch of set b in the token order of set a (`align`); a token of a that set b lacks is not valid"""
    pm = perm.clamp(min=0)
    ar = torch.arange(len(pm))[:, None]
    out = {k: batch[k][ar, pm] for k in JK.INPUTS}
    out["valid"] = out["valid"] & (perm >= 0)
    return out


def check_parity(a, runs=RUNS):
    """the head on the first `--n` VAL frames, once from the cache tokens (fp16 `tok`) and once from tokens recomputed live from B' on the CPU in float32 (`live_tokens`, compared by cell: `align`): PASS when p_none and every link
    probability of the valid tokens agree within --tol; the input fields of the matched valid tokens and the tokens that only one side has are reported -> 0 / 1"""
    import lane_tok_data as TD
    import qpredict as QP
    m, args = JKT.load_model(a.run, "cpu", runs)
    kw = cache_kw(args)
    ds = open_cache("val", kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    n = a.n
    b_run = _abs(a.b_run or f"runs/{kw['ident']}_s{kw['seed']}")
    mb = QP.load_model(b_run, "cpu")
    td = TD.TokData("val", LATTICE, feats=True)
    assert list(td.tokens[:n]) == list(ds.tokens[:n])
    cb = next(iter(batches(ds, range(n), bs=n)))
    x = torch.stack([td[i]["x"] for i in range(n)])
    with torch.no_grad():
        live = live_tokens(mb, x, ds.K, exact=True)
    perm = align(np.stack([t["cell"] for t in cb["tokens_np"]]), live["cell"].long().numpy())
    live_c = reorder(live, perm)                                                   # the live tokens in the cache's order
    o_c, o_l = run_head(m, cb, "cpu"), run_head(m, live_c, "cpu")
    ok, num = compare_outputs(o_c, o_l, a.tol)
    v = cb["valid"] & live_c["valid"]
    fields = {k: float((cb[k].float() - live_c[k].float()).abs()[v].max()) for k in JK.INPUTS if k != "valid"}
    num.update(frames=n, b_run=os.path.relpath(b_run, ROOT), valid_cache=int(cb["valid"].sum()), valid_live=int(live["valid"].sum()), valid_cache_not_live=int((cb["valid"] & ~live_c["valid"]).sum()), valid_live_not_cache=int(live["valid"].sum() - live_c["valid"].sum()),
               input_max_abs_diff_matched_valid_tokens=fields, note="cache: tok fp16, built on the GPU; live: CPU, tok fp32; tokens are matched by cell, padding tokens are not compared")
    print(json.dumps(num, indent=1))
    path = os.path.join(runs, a.run, "token_parity.json")
    if not os.path.exists(path):
        JV.write_json(path, _jsonable(dict(num, ok=bool(ok))))
    print(f"{'PASS' if ok else 'FAIL'}: parity of the head on {n} VAL frames, cache against live B' ({a.run}), tolerance {a.tol}")
    return int(not ok)


def head_macs(m, K=384):
    """GMACs of the head for one frame of K tokens (torch FlopCounterMode; MACS_NOTE says what it counts)"""
    from torch.utils.flop_counter import FlopCounterMode
    m = m.cpu().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(JK.synthetic_batch(1, K, seed=0))
    return fc.get_total_flops() / 2e9


def gpu_free(limit_mib=2000):
    """(ok, why): the timing rule of the card: nvidia-smi shows less than `limit_mib` used and no run.py job is active"""
    try:
        used = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().split("\n")[0])
        jobs = subprocess.run(["pgrep", "-f", "autoresearch/tools/run.py"], capture_output=True, text=True, timeout=20).stdout.split()
    except Exception as e:  # noqa: BLE001
        return False, f"cannot read the GPU: {e}"
    if used >= limit_mib or jobs:
        return False, f"GPU memory used {used:.0f} MiB (limit {limit_mib}), run.py jobs {jobs}"
    return True, f"GPU memory used {used:.0f} MiB, no run.py job"


def timing(a, runs=RUNS):
    """parameters, GMACs and ms per frame of the head (batch 1, bf16 autocast on the GPU, CUDA events, `--frames` cached VAL frames, `--repeats` passes after one untimed pass) and, with --b-run, of B' alone (pyramid + node head + top-K gather) and of
    the whole path (B' + head in one timed region). The GPU only when `gpu_free`; --cpu times the code path on the CPU (not G-cost evidence) -> runs/R/token_timing[_cpu].json"""
    import jrecur_time as JRT
    import jtiming as JTG
    m, args = JKT.load_model(a.run, "cpu", runs)
    ok, why = (False, "--cpu") if a.cpu else gpu_free()
    if not a.cpu and not ok:
        sys.exit(f"STOP: no GPU timing ({why}); --cpu times the code path on the CPU only")
    dev = "cpu" if a.cpu else "cuda"
    K, gmacs = a.K, head_macs(m, a.K)                                                  # on the CPU, before the head moves
    m = m.to(dev)
    kw = cache_kw(args)
    ds = open_cache("val", kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    load0, gpu0 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    cut = lambda b: {k: v[:, :K] for k, v in b.items() if k in JK.INPUTS}  # noqa: E731
    items = [JKT.to_device(cut(LP.collate([ds[i]])), dev) for i in range(a.frames)]
    ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")  # noqa: E731
    mb = xs = None
    if a.b_run:
        import lane_tok_data as TD
        import qpredict as QP
        mb = QP.load_model(_abs(a.b_run), dev)
        td = TD.TokData("val", LATTICE, feats=True)
        xs = [td[i]["x"][None].to(dev) for i in range(a.frames)]
    ms = dict(head=[], bprime=[], whole=[])
    for rep in range(a.repeats + 1):
        row = dict(head=[], bprime=[], whole=[])
        for i, b in enumerate(items):
            with torch.no_grad(), ac():
                row["head"].append(JRT.timed(lambda: m(b), dev == "cuda")[1])
                if mb is not None:
                    row["bprime"].append(JRT.timed(lambda: live_tokens(mb, xs[i], K), dev == "cuda")[1])
                    row["whole"].append(JRT.timed(lambda: m(live_tokens(mb, xs[i], K)), dev == "cuda")[1])
        if rep:                                                                           # the first pass is untimed
            for k in ms:
                ms[k].append(row[k])
    load1, gpu1 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    quiet, line = JTG.quiet_verdict(load0, load1, dict(util_max=0.0) if a.cpu else gpu0, dict(util_max=0.0) if a.cpu else gpu1)
    res = dict(run=a.run, device=dev, K=K, frames=a.frames, repeats=a.repeats, params=JK.count_params(m), gmacs_head=gmacs, host=line, gpu_rule=why, bf16=dev == "cuda",
               median_ms={k: float(np.median(v)) for k, v in ms.items() if v}, p90_ms={k: float(np.percentile(v, 90)) for k, v in ms.items() if v}, macs_note=MACS_NOTE)
    if mb is not None:
        res["params_bprime"] = JK.count_params(mb.pyr) + JK.count_params(mb.nodes)           # the path of the tokens: pyramid + node head (the link, map and assign heads of the checkpoint are not on it)
        res["params_whole_path"] = res["params_bprime"] + res["params"]
    print(json.dumps(res, indent=1))
    path = os.path.join(runs, a.run, f"token_timing{'_cpu' if a.cpu else ''}.json")
    if not os.path.exists(path):
        JV.write_json(path, _jsonable(res))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ command line
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "score"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--split", default="val", choices=("val", "train"))
        s.add_argument("--drop", type=float, default=0.0, help="the degraded read-out: the share of the tokens removed (0.17)")
        s.add_argument("--drop-seed", type=int, default=0)
        if name == "dump":
            s.add_argument("--cpu", action="store_true", help="tests only: dump on the CPU")
            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; the score needs all of them)")
    s = sub.add_parser("ctl")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--ident", required=True, choices=("e292_a", "e273_p"))
    s.add_argument("--K", type=int, default=None, help="tokens per frame, only when the cache root holds more than one K for the seed")
    s.add_argument("--cache", default=None)
    s.add_argument("--arms", default=None, help="comma list of " + ",".join(ARMS) + " (default all)")
    s.add_argument("--dump", default=None, help="the stored B dump of the bsec0 arms (default per ident, see B_DUMPS)")
    s.add_argument("--max", type=int, default=0, help="the first N frames; cards go to temp/e365/out/")
    s = sub.add_parser("check-layer1")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--dump", default=None, help="the stored dump of plain B (default temp/e343/val/predfull_e273_p_s<seed>.pkl)")
    s = sub.add_parser("check-parity")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--b-run", default=None, help="the folder of B' (default runs/<ident>_s<seed>)")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--tol", type=float, default=0.01)
    s = sub.add_parser("timing")
    s.add_argument("--run", required=True)
    s.add_argument("--b-run", default=None, help="also time B' alone and the whole path")
    s.add_argument("--K", type=int, default=384)
    s.add_argument("--frames", type=int, default=24)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(2)
    r = {"dump": dump, "score": score, "ctl": ctl, "check-layer1": check_layer1, "check-parity": check_parity, "timing": timing}[a.cmd](a)
    sys.exit(r if isinstance(r, int) else 0)


if __name__ == "__main__":
    main()
