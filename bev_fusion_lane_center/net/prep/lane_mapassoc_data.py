"""E391 step 2 of the SD-map line (card experiments/E391_sd-map-fusion-association): the data of the ASSOCIATION of the champion's perceived tokens to the segments of the E390 road graph.

    wsl bash tools/env/_py14.sh lane_mapassoc_data.py build --split val|train --seed S     # runs/e391_assoc/<split>_s<S>.pkl: per frame and token the association label, the node-level route role, the delivered mask
    wsl bash tools/env/_py14.sh lane_mapassoc_data.py stats --split val|train --seed S     # the counts of the card (labelled, false, unlabelled, delivered, holdout)
    wsl bash tools/env/_py14.sh lane_mapassoc_data.py selftest

Tokens = the cached B' tokens of the E365 cache (`lane_ptok_data.PTokData`, K 512, cache seed S), their GT node `gt_id` (a row of the frame's E357 table: `lane_tok_data.TokData.table`). The ASSOCIATION LABEL of a token
(`assoc_labels`): its GT node's piece and position -> the E390 road-graph node (`lane_sdgraph_data.frame_graph`'s node order) of the stretch that holds it: the piece's segment at the node's nearest point lies in an
intersection -> the intersection node of the piece whose area contains the point (the only one if the piece lists one); outside -> the piece's only ROAD node; a piece on two or more road nodes outside an intersection
(a split / merge stretch) -> unlabelled (-3, counted). A FALSE token (valid, gt_id -1) -> -1 = 'unassigned'; an invalid token -2. The node-level ROUTE ROLE = E389's `on_lane` of the token's GT node (the node's
piece is a tier-1 route piece; `lane_route_data.TokenRoute.token_labels`). The DELIVERED mask = valid and the E373 stage-1 head's p_none < 0.6 (VAL: the stored dump `runs/e373_cb_s<S>/token_val.pkl`; TRAIN: computed
by the head on the CPU for the holdout frames only, `holdout`: every 10th TRAIN scene by rank as E390 v7; other TRAIN frames: no mask, training uses every labelled valid token as E389). The frame lists of the token
cache, the E389 route cache and the E390 graph cache are the same (asserted by token).
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "training/data/prep", "net/prep", "net/model", "net/loss", "training/train", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))

VERSION = "v1"
CACHE = os.path.join(ROOT, "runs", "e391_assoc")
KEEP_P = 0.6                      # the champion's decode keep: a token is delivered at p_none < 0.6
UNLAB, FALSE, INVALID = -3, -1, -2


def _inside(pt, poly):
    x, y, n, c = float(pt[0]), float(pt[1]), len(poly), False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            c = not c
    return c


def node_list(g):
    """the E390 node order of a frame (`frame_graph`: the roads of g with a piece in the frame) -> (nodes, piece index -> [node indices])"""
    pidx = {p["id"]: k for k, p in enumerate(g["pieces"])}
    nodes = [r for r in g["roads"] if any(p in pidx for p in r.get("pieces", []))]
    memb = {}
    for i, r in enumerate(nodes):
        for p in r["pieces"]:
            if p in pidx:
                memb.setdefault(pidx[p], []).append(i)
    return nodes, memb


def node_of_point(g, nodes, memb, k, pos):
    """the E390 node of the stretch of piece k that holds the point pos (rule of the module docstring); UNLAB when ambiguous or the piece is on no node"""
    cands = memb.get(k, [])
    if not cands:
        return UNLAB
    if len(cands) == 1:
        return cands[0]
    p = g["pieces"][k]
    q = np.asarray(p["pts"], float)
    j = int(np.argmin(np.linalg.norm(q - pos, axis=1)))
    segs = p.get("segments") or []
    inside = any(bool(s.get("in_intersection")) and int(s["i0"]) <= j <= int(s["i1"]) for s in segs)
    kinds = [nodes[c]["kind"] for c in cands]
    if inside:
        xs = [c for c, kd in zip(cands, kinds) if kd == "intersection"]
        if len(xs) == 1:
            return xs[0]
        areas = {a.get("id"): np.asarray(a.get("exterior", []), float) for a in g.get("areas", {}).get("intersection", [])}
        hit = [c for c in xs if nodes[c]["id"] in areas and len(areas[nodes[c]["id"]]) >= 3 and _inside(pos, areas[nodes[c]["id"]])]
        return hit[0] if len(hit) == 1 else UNLAB
    rs = [c for c, kd in zip(cands, kinds) if kd == "road"]
    return rs[0] if len(rs) == 1 else UNLAB


def assoc_labels(g, table, gt_id, valid):
    """(K,) int: the association label of every token (module docstring); table = the frame's E357 table (pos, piece), gt_id / valid of the cache"""
    nodes, memb = node_list(g)
    node_lab = np.array([node_of_point(g, nodes, memb, int(k), np.asarray(pos, float)) for k, pos in zip(table["piece"], table["pos"])], int) if len(table["piece"]) else np.zeros(0, int)
    gt_id, valid = np.asarray(gt_id), np.asarray(valid, bool)
    out = np.full(len(gt_id), INVALID, int)
    out[valid & (gt_id < 0)] = FALSE
    m = valid & (gt_id >= 0)
    out[m] = node_lab[gt_id[m]]
    return out, len(nodes)


def holdout_scenes(scenes):
    """the TRAIN holdout of E390 v7: every 10th TRAIN scene by rank among the sorted distinct scene ids"""
    u = np.unique(np.asarray(scenes))
    return set(u[np.arange(len(u)) % 10 == 0].tolist())


def e373_pnone(split, seed, idx, ds, log=print, head=None):
    """the E373 stage-1 head's p_none of every token of the frames idx: VAL from the stored dump, TRAIN by the head on the CPU (head = `champion_pnone.head_pnone`, passed in: preprocessing does not import the model or the trainer; the cached tokens of seed S) -> {frame index: (K,) float32}"""
    run = f"e373_cb_s{seed}"
    if split == "val":
        d = pickle.load(open(os.path.join(ROOT, "runs", run, "token_val.pkl"), "rb"))
        assert [f["token"] for f in d["frames"]] == [ds.tokens[i] for i in range(len(ds))], "the E373 VAL dump is not in the cache order"
        return {i: np.asarray(d["frames"][i]["pnone"], np.float32) for i in idx}
    assert head is not None, "TRAIN needs the stored head: pass head=champion_pnone.head_pnone (training/train; the command line of this module does)"
    import lane_ptok_data as LP
    pnone = head(seed)
    out = {}
    for s in range(0, len(idx), 8):
        ii = idx[s: s + 8]
        b = LP.collate([ds[i] for i in ii])
        pt = pnone(b)
        for r, i in enumerate(ii):
            out[i] = pt[r].astype(np.float32)
        if s % 400 == 0:
            log(f"  p_none {s}/{len(idx)}", flush=True)
    return out


def cache_path(split, seed, cache=CACHE):
    return os.path.join(cache, f"{split}_s{seed}.pkl")


def build(split, seed, cache=CACHE, log=print, head=None):
    """the label cache of a split and cache seed (module docstring) -> runs/e391_assoc/<split>_s<S>.pkl (never overwritten)"""
    import lane_gt_data as LD
    import lane_ptok_data as LP
    import lane_route_data as LR
    import lane_sdgraph_data as SG
    import lane_tok_data as LT
    path = cache_path(split, seed, cache)
    assert not os.path.exists(path), f"never overwritten: {path}"
    lg, _ = LD.lane_store()
    ds = LP.PTokData(split, seed, "e292_a", None, 0.0)
    td = LT.TokData(split, res=2.0, feats=False)
    rc = LR.TokenRoute(LR.route_dir(split))
    sg = SG.load(split)
    assert list(ds.tokens) == list(td.tokens) == list(rc.tokens) == list(sg["tokens"]), "the token cache, the E357 tables, the route cache and the E390 graphs are not the same frames in order"
    hold = holdout_scenes(sg["scene"]) if split == "train" else set()
    pidx = list(range(len(ds))) if split == "val" else [i for i, sc in enumerate(sg["scene"]) if sc in hold]
    pnone = e373_pnone(split, seed, pidx, ds, log, head)
    frames = []
    for i in range(len(ds)):
        it = ds[i]
        gt_id, valid = np.asarray(it["label"]["gt_id"]), np.asarray(it["valid"], bool)
        lab, n = assoc_labels(lg.load(ds.tokens[i]), td.table(i), gt_id, valid)
        assert n == len(sg["graphs"][i]["kind"]), f"frame {i}: the node list ({n}) is not the E390 graph's ({len(sg['graphs'][i]['kind'])})"
        node_lane = rc.labels(i)[1]                                          # E389's on_lane of the table nodes (as `RouteData.token_labels`)
        on_lane = -np.ones(len(gt_id), np.int8)
        got = gt_id >= 0
        if len(node_lane):
            on_lane[got] = np.asarray(node_lane)[gt_id[got]]
        rec = dict(seg=lab.astype(np.int16), on_lane=on_lane)
        if i in pnone:
            rec["kept"] = valid & (pnone[i] < KEEP_P)
        frames.append(rec)
        if i % 500 == 0:
            log(f"  {i}/{len(ds)}", flush=True)
    os.makedirs(cache, exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(dict(version=VERSION, split=split, seed=seed, tokens=list(ds.tokens), scene=list(sg["scene"]), holdout=sorted(hold), frames=frames), fh)
    os.replace(path + ".tmp", path)
    log(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames")
    return path


def load(split, seed, cache=CACHE):
    with open(cache_path(split, seed, cache), "rb") as fh:
        d = pickle.load(fh)
    assert d["version"] == VERSION, f"{split}_s{seed}: version {d['version']}, the code {VERSION}: rebuild"
    return d


def stats(split, seed, log=print):
    d = load(split, seed)
    seg = np.concatenate([f["seg"] for f in d["frames"]])
    kept_frames = [f for f in d["frames"] if "kept" in f]
    ks = np.concatenate([f["seg"][f["kept"]] for f in kept_frames]) if kept_frames else np.zeros(0)
    res = dict(frames=len(d["frames"]), tokens_valid=int((seg != INVALID).sum()), labelled=int((seg >= 0).sum()), false=int((seg == FALSE).sum()), unlabelled=int((seg == UNLAB).sum()),
               kept_frames=len(kept_frames), kept_tokens=int(len(ks)), kept_labelled=int((ks >= 0).sum()), kept_false=int((ks == FALSE).sum()), kept_unlabelled=int((ks == UNLAB).sum()),
               on_lane_pos=int(sum((f["on_lane"][f["seg"] >= 0] == 1).sum() for f in d["frames"])), holdout_scenes=len(d["holdout"]))
    log(json.dumps(res))
    return res


def selftest():
    sq = [[-5.0, 25.0], [5.0, 25.0], [5.0, 35.0], [-5.0, 35.0]]
    y = np.linspace(0, 60, 121)
    pts = np.stack([np.zeros_like(y), y], 1)
    piece = dict(id="a", pts=pts, segments=[dict(i0=0, i1=49, in_intersection=False), dict(i0=50, i1=70, in_intersection=True), dict(i0=71, i1=120, in_intersection=False)])
    split_piece = dict(id="b", pts=np.stack([np.full(21, 10.0), np.linspace(0, 10, 21)], 1), segments=[dict(i0=0, i1=20, in_intersection=False)])
    g = dict(pieces=[piece, split_piece], areas=dict(intersection=[dict(id="x1", exterior=sq)]),
             roads=[dict(id="r1", kind="road", pieces=["a"]), dict(id="x1", kind="intersection", pieces=["a"]), dict(id="r2", kind="road", pieces=["a", "b"]), dict(id="r3", kind="road", pieces=["b"])])
    nodes, memb = node_list(g)
    assert memb == {0: [0, 1, 2], 1: [2, 3]}
    assert node_of_point(g, nodes, memb, 0, np.array([0.0, 30.0])) == 1, "inside the intersection"
    assert node_of_point(g, nodes, memb, 0, np.array([0.0, 5.0])) == UNLAB, "a stretch on two roads is ambiguous"
    assert node_of_point(g, nodes, memb, 1, np.array([10.0, 5.0])) == UNLAB
    table = dict(piece=np.array([0, 0]), pos=np.array([[0.0, 30.0], [0.0, 5.0]]))
    lab, n = assoc_labels(g, table, np.array([0, 1, -1, -1]), np.array([True, True, True, False]))
    assert lab.tolist() == [1, UNLAB, FALSE, INVALID] and n == 4
    assert holdout_scenes([3, 1, 2, 1]) == {1}
    print("lane_mapassoc_data selftest ok")


def main(argv=None, head=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "stats", "selftest"))
    ap.add_argument("--split", choices=("val", "train"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "build":
        return build(a.split, a.seed, head=head)
    return stats(a.split, a.seed)


if __name__ == "__main__":
    import champion_pnone  # the command line runs the stored champion head (training/train)
    main(head=champion_pnone.head_pnone)
