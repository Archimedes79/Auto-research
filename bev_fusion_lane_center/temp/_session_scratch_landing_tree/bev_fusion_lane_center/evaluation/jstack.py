"""E363 (promoted from the scratch `temp/e363/make_run.py`, `faithful.py`, `surgery_check.py`, used for six run folders): the pair run of E324 s read on ANOTHER graph's node head, as a run folder, a dump and the checks that make the comparison faithful.

    wsl bash tools/env/_py14.sh jstack.py surgery <c2|a|b> <seed>             # CPU: runs/e363_<g>_s<S>/model.pt = the checkpoint of runs/e324_s_s<S> with the 38 `nodes.*` tensors of the node-head source (runs/e292_<g>_s<S>, b = B itself: runs/e273_p_s<S>, the FIXTURE
                                                                              #   with 0 tensors changed) + provenance.json (sha256 of source, node-head source and result); the pyramid of the source is checked equal; an existing folder is refused
    wsl bash tools/env/_pyr.sh jstack.py dump --run e363_<g>_s<S>             # GPU: `jends_eval.dump` (unchanged) on the pair model; for g = a the Matrix-NMS decay of B' is applied to the node outputs BEFORE `Attach` (the layer is not inside `nodes.*`: it lives in
                                                                              #   `QGraph.node_stage`, read from args['soft_nms'] = ['mnms', radius, sigma] of runs/e292_a_s<S>); writes runs/<run>/ends_val.pkl and ends_head.json; refuses to overwrite
    wsl bash tools/env/_pyr.sh jstack.py check-nodes --seed S --g a|c2 [--n 16]  # GPU: the pair model's node outputs e, pos, e_raw against `QGraph.node_stage` of B' (runs/e292_<g>_s<S>) on the first n val_clean frames: max |difference| <= 1e-5; exit 1 otherwise;
                                                                              #   writes temp/e363/nodecheck_<g>_s<S>.json
    wsl bash tools/env/_py14.sh jstack.py check-arrays                        # CPU: every runs/e363_* folder: `nodes.*` equal to their source, every other tensor equal to E324 s, the provenance sha256, the node-free arrays of its dump (kp_*, logit, valid, allowed, scale) bit for bit
                                                                              #   those of the stored runs/e324_s_s<S>/ends_val.pkl, and for b EVERY array of the dump (the fixture); writes temp/e363/surgery_check.json
    wsl bash tools/env/_py14.sh jstack.py selftest                            # CPU, toy tensors: the swap, the array comparison, the layer wrapper against `qmnms.matrix_nms_logit`, the layer parameters

The stored runs are never touched; the scoring of such a run folder is `jends_eval.py score --run e363_<g>_s<S> --full <predfull of B'> --base <runs/e292_<g>_s<S>/model.pt> --spec plain@0.5 --tag <tag> --attach --snap --snap-nogeom` (card E363).
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch.nn as nn

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qmnms as QMN  # noqa: E402

NODE_FREE = ("kp_pos", "kp_dir", "kp_p", "logit", "valid", "allowed", "scale")                              # the arrays of a pair-run dump that do not depend on the node head
GRAPHS = ("c2", "a", "b")


def node_source(g, seed):
    """the checkpoint that holds the node head: B' = runs/e292_<g>_s<S> (g = c2 | a), B itself for the fixture (g = b)"""
    return os.path.join(ROOT, "runs", f"e273_p_s{seed}" if g == "b" else f"e292_{g}_s{seed}", "model.pt")


def swap_nodes(state, node_state):
    """-> (new state, number of node tensors, number of them that changed): `state` with every `nodes.*` tensor replaced by that of `node_state`; the key sets and the shapes must be equal; nothing else changes"""
    mine = {k: v for k, v in state.items() if k.startswith("nodes.")}
    src = {k: v for k, v in node_state.items() if k.startswith("nodes.")}
    assert src and set(src) == set(mine), f"node-head keys differ: {sorted(set(src) ^ set(mine))[:6]}"
    assert all(src[k].shape == mine[k].shape for k in src), "node-head shapes differ"
    new = dict(state)
    new.update(src)
    return new, len(src), sum(not torch.equal(src[k], mine[k]) for k in src)


def pyramid_equal(node_state, state):
    """the pyramid of the node-head source equals the pair run's (`pyr.*` of B' against `jmodel.pyr.*` or `pyr.*` of the pair model): the connectors then sit on the same pixel features as the delivered graph"""
    pyr = [k for k in node_state if k.startswith("pyr.")]
    other = [("jmodel." + k) if ("jmodel." + k) in state else k for k in pyr]
    return bool(pyr) and all(k2 in state and torch.equal(node_state[k], state[k2]) for k, k2 in zip(pyr, other)), len(pyr)


def same(u, v):
    """bit for bit (NaN equal NaN); None only equals None"""
    if u is None or v is None:
        return (u is None) == (v is None)
    return np.array_equal(np.asarray(u), np.asarray(v), equal_nan=True)


def compare_dumps(old, new, keys):
    """-> list of (row, key) where a frame of `new` differs from the frame of `old` (both lists of dump frames, same order) in one of `keys`, or the rows differ"""
    return [(x["row"], k) for x, y in zip(old, new) for k in keys if x["row"] != y["row"] or not same(x.get(k), y.get(k))]


class LayerNodes(nn.Module):
    """the node head followed by the Matrix-NMS decay of B' (as `QGraph.node_stage`): `e` is the damped logit, `e_raw` the head's own, every other key (tok, flags, ...) is kept"""
    def __init__(self, nodes, radius, sigma):
        super().__init__()
        self.nodes, self.radius, self.sigma = nodes, float(radius), float(sigma)

    def forward(self, *a):
        no = dict(self.nodes(*a))
        no["e_raw"] = no["e"]
        no["e"] = QMN.matrix_nms_logit(no["e"], no["pos"], self.radius, self.sigma)
        return no


def layer_params(g, seed):
    """(radius, sigma) of the Matrix-NMS layer of runs/e292_<g>_s<S> (args['soft_nms'] = ['mnms', radius, sigma]) for g = a; None for g = c2 / b (asserted: no layer in c2)"""
    if g == "b":
        return None
    args = torch.load(os.path.join(ROOT, "runs", f"e292_{g}_s{seed}", "model.pt"), map_location="cpu")["args"]
    sn = args.get("soft_nms")
    if g == "a":
        assert sn and sn[0] == "mnms" and len(sn) == 3, f"runs/e292_a_s{seed}: soft_nms {sn}"
        return float(sn[1]), float(sn[2])
    assert not sn, f"runs/e292_{g}_s{seed} has a layer {sn}"
    return None


def model_for(run, dev, load_model):
    """the pair model of the run folder `e363_<g>_s<S>` (`load_model` = `jends_eval.load_model`) with the layer of B' in front of Attach (arm a only)"""
    g = run.split("_")[1]
    seed = int(run.rsplit("_s", 1)[1])
    assert g in GRAPHS, run
    m, args = load_model(run, dev)
    lp = layer_params(g, seed)
    if lp is not None:
        m.nodes = LayerNodes(m.nodes, *lp).to(dev).eval()
    return m, args


def node_diff(nd, nb, layered):
    """-> ({key: largest absolute difference}, ok) of the pair model's node outputs `nd` against `QGraph.node_stage` outputs `nb` of B': `e` and `pos` must be in both (a missing key is an error, not a pass), `e_raw` must be in both when B' has the layer
    (`layered`), the differences must be finite and at most 1e-5"""
    worst, ok = {}, True
    for k in ("e", "pos") + (("e_raw",) if layered else ()):
        assert k in nd and k in nb, f"node output {k!r} missing (pair model: {sorted(nd)}, B': {sorted(nb)}): nothing compared"
        v = float((nd[k].float() - nb[k].float()).abs().max())
        worst[k] = max(worst.get(k, 0.0), v) if v == v else float("nan")
        ok = ok and v == v and v <= 1e-5
    return worst, ok


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cmd_surgery(a):
    g, seed = a.g, a.seed
    assert g in GRAPHS and seed in (0, 1), (g, seed)
    src, bp = os.path.join(ROOT, "runs", f"e324_s_s{seed}", "model.pt"), node_source(g, seed)
    dst_dir = os.path.join(ROOT, "runs", f"e363_{g}_s{seed}")
    assert not os.path.exists(dst_dir), f"{dst_dir} exists: refused"
    ck = torch.load(src, map_location="cpu")
    bs = torch.load(bp, map_location="cpu")["state"]
    ok, n_pyr = pyramid_equal(bs, ck["state"])
    assert ok, "the pyramid of the node-head source is not the pair run's: the connectors would sit on other pixel features"
    new_state, n_nodes, n_changed = swap_nodes(ck["state"], bs)
    new = dict(ck)
    new["state"] = new_state
    os.makedirs(dst_dir)
    torch.save(new, os.path.join(dst_dir, "model.pt"))
    json.dump(dict(source_pair_run=f"e324_s_s{seed}", node_head_from=os.path.basename(os.path.dirname(bp)), node_tensors=n_nodes, node_tensors_changed=n_changed, pyramid_tensors_checked_equal=n_pyr, sha256_source=sha256(src), sha256_node_head_source=sha256(bp),
                   sha256_new=sha256(os.path.join(dst_dir, "model.pt")), note="E363: the pair run of E324 s with the node head of another graph (checkpoint surgery: only `nodes.*`); the stored runs are untouched"), open(os.path.join(dst_dir, "provenance.json"), "w"), indent=1)
    print(f"{dst_dir}: {n_nodes} node-head tensors, {n_changed} differ from the pair run's own; {n_pyr} pyramid tensors equal")


def cmd_dump(a):
    import jends_eval as JE
    p = os.path.join(ROOT, "runs", a.run, "ends_val.pkl")
    assert not os.path.exists(p), f"{p} exists: refused (move it away by hand first)"
    orig = JE.load_model
    assert not getattr(orig, "_jstack_patched", False), "jends_eval.load_model is already patched by jstack: wrapping it twice would damp the layer twice"
    patched = lambda run, dev: model_for(run, dev, orig)  # noqa: E731
    patched._jstack_patched = True
    JE.load_model = patched                                                                                  # `jends_eval.dump` calls `load_model(a.run, dev)` of its own module: an in-process patch, no frozen file is edited
    try:
        JE.dump(NS(run=a.run, thr=0.5, max=0))
    finally:
        JE.load_model = orig


def cmd_check_nodes(a):
    import jends_eval as JE
    import lane_gt_data as LD
    import qpredict as QP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = f"e363_{a.g}_s{a.seed}"
    m, _ = model_for(run, dev, JE.load_model)
    B = QP.load_model(os.path.join(ROOT, "runs", f"e292_{a.g}_s{a.seed}"), dev)
    LD.QData.DEFAULT_JUNCTION = True
    ds, dl = LD.loader("val_clean", bs=8, workers=0, shuffle=False, map_mode="error", pixels=False)
    lp = layer_params(a.g, a.seed)
    worst, ok_all = dict(e=0.0, pos=0.0, e_raw=0.0), True
    n = 0
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(dev).float()
            P, _ = m.jmodel(x, joint=False)
            nd = m.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
            nb = B.node_stage(B.pyr(x))
            w, ok_b = node_diff(nd, nb, lp is not None)
            ok_all = ok_all and ok_b
            for k, v in w.items():
                worst[k] = max(worst[k], v) if v == v else float("nan")
            n += x.shape[0]
            if n >= a.n:
                break
    ok = ok_all and n > 0
    print(f"NODE CHECK {run}: {n} frames, layer {'mnms %s' % (lp,) if lp else 'none'}, max |difference| e {worst['e']:.2e}, pos {worst['pos']:.2e}, e_raw {worst['e_raw']:.2e}: {'PASS' if ok else 'FAIL'}")
    os.makedirs(os.path.join(ROOT, "temp", "e363"), exist_ok=True)
    json.dump(dict(run=run, frames=n, layer=lp, max_abs=worst, passed=bool(ok)), open(os.path.join(ROOT, "temp", "e363", f"nodecheck_{a.g}_s{a.seed}.json"), "w"))
    sys.exit(0 if ok else 1)


def cmd_check_arrays(a):
    res = dict(runs={}, ok_all=True)
    for g in GRAPHS:
        for s in (0, 1):
            run = f"e363_{g}_s{s}"
            d = os.path.join(ROOT, "runs", run)
            if not os.path.exists(os.path.join(d, "model.pt")):
                continue
            r = dict(ok=True, notes=[])
            mine = torch.load(os.path.join(d, "model.pt"), map_location="cpu")["state"]
            src = torch.load(os.path.join(ROOT, "runs", f"e324_s_s{s}", "model.pt"), map_location="cpu")["state"]
            nsrc = torch.load(node_source(g, s), map_location="cpu")["state"]
            if set(mine) != set(src):
                r["ok"] = False
                r["notes"].append(f"tensor key sets differ from E324 s: {sorted(set(mine) ^ set(src))[:4]}")
            for k, v in mine.items():
                if k not in src or not torch.equal(v, nsrc[k] if k.startswith("nodes.") else src[k]):
                    r["ok"] = False
                    r["notes"].append(f"tensor {k} differs from its source")
            r["notes"].append(f"{sum(k.startswith('nodes.') for k in mine)} node tensors checked against the node-head source, {sum(not k.startswith('nodes.') for k in mine)} others against E324 s")
            pv = json.load(open(os.path.join(d, "provenance.json")))
            if pv["sha256_new"] != sha256(os.path.join(d, "model.pt")):
                r["ok"] = False
                r["notes"].append("provenance sha256 differs from model.pt")
            dp = os.path.join(d, "ends_val.pkl")
            if os.path.exists(dp):
                new = pickle.load(open(dp, "rb"))
                old = pickle.load(open(os.path.join(ROOT, "runs", f"e324_s_s{s}", "ends_val.pkl"), "rb"))
                keys = [k for k in old[0] if k not in ("row", "token")] if g == "b" else NODE_FREE
                bad = compare_dumps(old, new, keys)
                if len(new) != 873 or len(old) != 873 or bad:
                    r["ok"] = False
                    r["notes"].append(f"dump: {len(new)} frames, {len(bad)} differing arrays {bad[:3]}")
                r["notes"].append(f"dump compared on {len(keys)} fields over {len(new)} frames: {'bit for bit' if not bad else 'DIFFERENT'}")
            else:
                r["notes"].append("no dump yet")
                r["ok"] = False
            res["runs"][run] = r
            res["ok_all"] = res["ok_all"] and r["ok"]
            print(run, "OK" if r["ok"] else "FAIL", "; ".join(r["notes"])[:300])
    res["pass"] = bool(res.pop("ok_all")) and bool(res["runs"])
    os.makedirs(os.path.join(ROOT, "temp", "e363"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "temp", "e363", "surgery_check.json"), "w"), indent=1)
    print("SURGERY CHECK", "PASS" if res["pass"] else "FAIL")


def _selftest():
    import tempfile
    torch.manual_seed(0)
    # 1. the swap: only nodes.* changes, equal key sets and shapes are required
    pair = {"nodes.w": torch.zeros(3, 2), "nodes.b": torch.zeros(2), "jmodel.pyr.x": torch.ones(4), "je.y": torch.full((2,), 5.0)}
    src = {"nodes.w": torch.ones(3, 2), "nodes.b": torch.zeros(2), "pyr.x": torch.ones(4), "other": torch.zeros(1)}
    new, n, ch = swap_nodes(pair, src)
    assert (n, ch) == (2, 1) and torch.equal(new["nodes.w"], torch.ones(3, 2)) and torch.equal(new["je.y"], pair["je.y"]) and torch.equal(new["jmodel.pyr.x"], pair["jmodel.pyr.x"]) and "other" not in new and torch.equal(pair["nodes.w"], torch.zeros(3, 2)), (n, ch)
    for bad, what in ((dict(src, **{"nodes.extra": torch.zeros(1)}), "keys"), ({k: v for k, v in src.items() if k != "nodes.b"}, "keys"), (dict(src, **{"nodes.w": torch.ones(2, 2)}), "shapes"), ({"pyr.x": torch.ones(4)}, "no node tensors")):
        try:
            swap_nodes(pair, bad)
        except AssertionError:
            continue
        raise AssertionError(f"swap_nodes accepted {what}")
    assert swap_nodes(pair, dict(src, **{"nodes.w": torch.zeros(3, 2)}))[2] == 0                               # B itself: nothing changes (the fixture)
    # 2. the pyramid check: `pyr.*` of the source against `jmodel.pyr.*` (or `pyr.*`) of the pair model
    assert pyramid_equal(src, pair) == (True, 1) and not pyramid_equal(dict(src, **{"pyr.x": torch.zeros(4)}), pair)[0] and not pyramid_equal({"nodes.w": torch.ones(1)}, pair)[0]
    assert pyramid_equal(src, {"pyr.x": torch.ones(4)})[0]
    # 3. the array comparison: bit for bit, NaN equals NaN, None only equals None, a shifted row counts
    f = lambda row, **k: dict(row=row, token=str(row), **k)  # noqa: E731
    old = [f(1, a=np.array([1.0, np.nan]), b=None), f(2, a=np.array([2.0, 3.0]), b=None)]
    assert compare_dumps(old, [f(1, a=np.array([1.0, np.nan]), b=None), f(2, a=np.array([2.0, 3.0]), b=None)], ("a", "b")) == []
    assert compare_dumps(old, [f(1, a=np.array([1.0, np.nan]), b=None), f(2, a=np.array([2.0, 3.0000001]), b=None)], ("a", "b")) == [(2, "a")]
    assert compare_dumps(old, [f(1, a=np.array([1.0, np.nan]), b=np.zeros(1)), f(2, a=np.array([2.0, 3.0]), b=None)], ("a", "b")) == [(1, "b")]
    assert compare_dumps(old, [f(1, a=np.array([1.0, np.nan]), b=None), f(3, a=np.array([2.0, 3.0]), b=None)], ("a",)) == [(2, "a")]
    assert compare_dumps(old, [f(1, a=np.array([1.0]), b=None), f(2, a=np.array([2.0, 3.0]), b=None)], ("a",)) == [(1, "a")]
    # 4. the layer wrapper: e is the damped logit of `matrix_nms_logit`, e_raw the head's own, every other key kept; two close strong cells are damped, a lone cell is not
    H = W = 6
    cell = torch.stack(torch.meshgrid(2.0 * torch.arange(H).float(), 2.0 * torch.arange(W).float(), indexing="ij"), 0)[None]
    e = torch.full((1, H, W), -4.0)
    e[0, 2, 2], e[0, 2, 3], e[0, 5, 5] = 3.0, 2.0, 3.0
    pos = cell.clone()
    pos[:, :, 2, 3] = pos[:, :, 2, 2] + torch.tensor([0.5, 0.0])[None, :]                                    # the weaker neighbour lies 0.5 m from the stronger cell
    head = lambda *a: dict(e=e.clone(), pos=pos.clone(), tok=torch.arange(3.0), flags=torch.ones(1))  # noqa: E731
    out = LayerNodes(head, 1.75, 0.1)(None)
    want = QMN.matrix_nms_logit(e, pos, 1.75, 0.1)
    assert torch.equal(out["e"], want) and torch.equal(out["e_raw"], e) and set(out) == {"e", "pos", "tok", "flags", "e_raw"}, sorted(out)
    assert out["e"][0, 2, 3] < e[0, 2, 3] - 1.0 and torch.equal(out["e"][0, 2, 2], e[0, 2, 2]) and torch.equal(out["e"][0, 5, 5], e[0, 5, 5]), "the weaker close cell is damped, the stronger and the lone one are not"
    # 5. the layer parameters come from the checkpoint args of B' (mnms for a, none for c2 / b); a wrong kind is refused
    global ROOT
    old_root = ROOT
    try:
        with tempfile.TemporaryDirectory() as d:
            ROOT = d
            for name, args in (("e292_a_s0", dict(soft_nms=["mnms", 1.75, 0.1])), ("e292_c2_s0", dict()), ("e292_a_s1", dict(soft_nms=None)), ("e292_c2_s1", dict(soft_nms=["mnms", 1.0, 0.1])), ("e292_a_s2", dict(soft_nms=[1.5, 0.1, 3]))):
                os.makedirs(os.path.join(d, "runs", name))
                torch.save(dict(args=args), os.path.join(d, "runs", name, "model.pt"))
            assert layer_params("a", 0) == (1.75, 0.1) and layer_params("c2", 0) is None and layer_params("b", 0) is None
            for g, s in (("a", 1), ("c2", 1), ("a", 2)):
                try:
                    layer_params(g, s)
                except AssertionError:
                    continue
                raise AssertionError(f"layer_params accepted {g} {s}")
            # 6. model_for wraps `nodes` for a and only for a (a stub of `load_model`)
            for run, wrapped in (("e363_a_s0", True), ("e363_c2_s0", False), ("e363_b_s0", False)):
                holder = NS(nodes=head)
                m, _ = model_for(run, "cpu", lambda r, dv, h=holder: (h, dict(run=r)))
                assert isinstance(m.nodes, LayerNodes) == wrapped, run
            try:
                model_for("e363_x_s0", "cpu", lambda r, dv: (NS(nodes=head), {}))
                raise RuntimeError("model_for accepted an unknown graph")
            except AssertionError:
                pass
    finally:
        ROOT = old_root
    # 7. the node comparison: equal passes, a difference above 1e-5 fails, a NaN fails, a missing key is an error (not a pass), e_raw only needed with the layer
    t = lambda v: torch.full((1, 2, 2), float(v))  # noqa: E731
    good = dict(e=t(1.0), pos=t(2.0))
    assert node_diff(good, dict(good), False) == (dict(e=0.0, pos=0.0), True)
    assert node_diff(dict(good, e_raw=t(3.0)), dict(good, e_raw=t(3.0)), True)[1]
    assert not node_diff(dict(good, e=t(1.001)), good, False)[1] and node_diff(dict(good, e=t(1.000001)), good, False)[1] is True
    assert not node_diff(dict(good, e=t(1.0005)), good, False)[1] and node_diff(dict(good, e=t(1.5)), good, False)[0] == dict(e=0.5, pos=0.0), "a difference of 5e-4 fails, the largest difference is returned"
    assert not node_diff(dict(good, pos=t(float("nan"))), good, False)[1] and not node_diff(good, dict(good, e=t(float("nan"))), False)[1]
    for nd_, nb_, lay in ((dict(e=t(1.0)), good, False), (good, dict(e=t(1.0)), False), (good, good, True), (dict(good, e_raw=t(1.0)), good, True)):
        try:
            node_diff(nd_, nb_, lay)
        except AssertionError:
            continue
        raise AssertionError("node_diff accepted a missing key")
    print("jstack selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("surgery")
    s.add_argument("g", choices=GRAPHS)
    s.add_argument("seed", type=int)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    c = sub.add_parser("check-nodes")
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--g", choices=("a", "c2"), required=True)
    c.add_argument("--n", type=int, default=16)
    sub.add_parser("check-arrays")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"surgery": cmd_surgery, "dump": cmd_dump, "check-nodes": cmd_check_nodes, "check-arrays": cmd_check_arrays, "selftest": lambda _: _selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
