"""The gates of the lane head (E273): after every phase a quick reading on VAL frames (card section 6 and the kill criterion of section 0). VAL only, never TEST.

    gate(model, phase, run, n=150) -> dict(pass, why, metrics...)

Phase 2  centre-pixel F1 (recall of the cells within 0.75 m of a lane, precision against 1.5 m) of the 1 m centre channel against the E271 dense head's own (`runs/e271_a_s<seed>`), must not be
         more than 0.02 below; road / intersection IoU and line-cell F1 are reported.
Phase 3  node F1 against the lane points (a node counts when a GT lane point lies within 2 m, a GT lane point (every 4 m) is found when a node lies within 2 m) not more than 0.05 below the
         E271 read-out's at the same tolerance (its card threshold `thr`), median position error of the nodes in cells that have a GT node <= 0.5 m; event recall per flag is reported.
Phase 4  link precision / recall on the GT-linked cells (reported), and, when `evaluation/qgraph_eval.py` is importable, the lane suite numbers of the delivered graph: junction F1 must exceed the
         reference graph's 0.273 and F1@1.0 may not be more than 0.05 below its 0.878.
Phase 5  the most probable section start is the right node for >= 95 % of the GT nodes.
Phase 6  assignment accuracy on the nodes (map with an error of 1-2 m) not below the geometric start assigner's on the same nodes; with the exact map not more than 0.01 below it.
Phase 7  the numbers of all of the above (the verdict itself is made from the full scoring, not here).
"""
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qassign as QA  # noqa: E402
import qnodes as QN  # noqa: E402
import sections as SE  # noqa: E402
import targets as TG  # noqa: E402

REF = {}
GATE_CFG = dict(node_prec_min=None, node_rec_min=None, tier_mode=False)          # tier_mode (E276): the F1-against-E271 condition of the node gate is dropped (it counts recall against ALL GT points, an arm that does not demand unseen nodes fails it by design); the floors stay          # E274: floors on the node precision / recall within 2 m of the node gate (set by `qgraph_train.py --gate-node-prec / --gate-node-rec`)


def _batches(map_mode, n, bs=4):
    ds = LD.QData("val_clean", map_mode=map_mode, pixels=True)
    ks = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    sub = torch.utils.data.Subset(ds, ks.tolist())
    return torch.utils.data.DataLoader(sub, batch_size=bs, shuffle=False, num_workers=2, collate_fn=LD.collate)


def _f1(p, r):
    return 2 * p * r / max(p + r, 1e-9)


def _dev(model):
    return next(model.parameters()).device


def e271_head(seed=0):
    import head as GH
    path = os.path.join(ROOT, "runs", f"e271_a_s{seed}", "model.pt")
    h = GH.GridHead()
    h.load_state_dict(torch.load(path, map_location="cpu")["state"])
    return h.eval()


def centre_f1(p_logit, g05, t05):
    with torch.no_grad():
        d, _, _, _ = TG.cell_targets(g05, t05, n=p_logit.shape[-1])
        pred = p_logit.sigmoid() > 0.5
        tgt = d <= 0.75
        tp_r = (pred & tgt).sum().item()
        tp_p = (pred & (d <= 1.5)).sum().item()
        return _f1(tp_p / max(pred.sum().item(), 1), tp_r / max(tgt.sum().item(), 1))


def gate_pixels(model, run, n):
    dev = _dev(model)
    ref = e271_head(0).to(dev) if "e271" not in REF else REF["e271"]
    REF["e271"] = ref
    f_m, f_r, iou_road, iou_int = [], [], [], []
    model.eval()
    with torch.no_grad():
        for b in _batches("exact", n):
            x = b["x"].to(dev).float()
            P = model.pyr(x)
            g05, t05 = b["g05"].to(dev), b["t05"].to(dev)
            f_m.append(centre_f1(P["centre1"][:, 0], g05, t05))
            f_r.append(centre_f1(ref(x)[:, 0], g05, t05))
            for c, store in ((0, iou_road), (1, iou_int)):
                pr, gt = P["road2"][:, c].sigmoid() > 0.5, b["road2"][:, c].to(dev) > 0.5
                store.append(((pr & gt).sum() / ((pr | gt).sum().clamp_min(1))).item())
    model.train()
    fm, fr = float(np.mean(f_m)), float(np.mean(f_r))
    return dict(centre_f1=fm, e271_centre_f1=fr, road_iou=float(np.mean(iou_road)), inter_iou=float(np.mean(iou_int)), **{"pass": fm >= fr - 0.02, "why": f"centre F1 {fm:.3f} vs E271 {fr:.3f} - 0.02"})


def _node_stats(model, b, dev):
    x = b["x"].to(dev).float()
    P = model.pyr(x)
    no = model.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
    return P, no


def gate_nodes(model, run, n):
    dev = _dev(model)
    prec, rec, err, rec_flag, n_flag = [], [], [], np.zeros(8), np.zeros(8)
    prec2, rec2 = [], []
    from scipy.spatial import cKDTree
    model.eval()
    with torch.no_grad():
        for b in _batches("exact", n):
            P, no = _node_stats(model, b, dev)
            for i in range(b["x"].shape[0]):
                m = no["e"][i].sigmoid() > 0.5
                pos = no["pos"][i].permute(1, 2, 0)[m].cpu().numpy()
                gm = b["g05"][i][b["m05"][i] > 0].numpy()
                if len(gm) == 0:
                    continue
                gpts = gm[::8]                                                                  # a GT lane point every 4 m
                tg = cKDTree(gm)
                prec.append((tg.query(pos)[0] <= 2.0).mean() if len(pos) else 0.0)
                rec.append((cKDTree(pos).query(gpts)[0] <= 2.0).mean() if len(pos) else 0.0)
                ex = b["exists"][i] > 0.5
                if ex.any() and len(pos):                                                       # E274: the scorer's definition: delivered nodes (e > 0.5, predicted position) against the true node positions (cell centre + offset), 2 m
                    tp = (b["offset"][i] + torch.from_numpy(QN.cell_centres("cpu")[0].numpy()))[:, ex].T.numpy()
                    prec2.append((cKDTree(tp).query(pos)[0] <= 2.0).mean())
                    rec2.append((cKDTree(pos).query(tp)[0] <= 2.0).mean())
                both = ex & m.cpu()
                if both.any():
                    gp = (b["offset"][i] + torch.from_numpy(QN.cell_centres("cpu")[0].numpy()))
                    pp = no["pos"][i].cpu()
                    err.append(float((pp - gp)[:, both].norm(dim=0).median()))
                fl = no["flags"][i].sigmoid().cpu() > 0.5
                for f in range(8):
                    t = (b["flags"][i, f] > 0.5) & ex
                    n_flag[f] += int(t.sum())
                    rec_flag[f] += int((t & fl[f] & m.cpu()).sum())
    model.train()
    p, r = float(np.mean(prec)), float(np.mean(rec))
    f = _f1(p, r)
    if "e271_nodes" not in REF:
        REF["e271_nodes"] = e271_vertex_f1(n)
    ref = REF["e271_nodes"]
    pos_err = float(np.median(err)) if err else float("nan")
    p2, r2 = (float(np.mean(prec2)) if prec2 else float("nan")), (float(np.mean(rec2)) if rec2 else float("nan"))
    ok = ((f >= ref - 0.05) or GATE_CFG.get("tier_mode", False)) and (pos_err <= 0.5)
    why_extra = "; tier mode: no F1 condition (unseen nodes are not demanded)" if GATE_CFG.get("tier_mode", False) else ""
    if GATE_CFG["node_prec_min"] is not None:
        ok = ok and p2 >= GATE_CFG["node_prec_min"]
        why_extra += f"; node precision within 2 m {p2:.3f} >= {GATE_CFG['node_prec_min']}"
    if GATE_CFG["node_rec_min"] is not None:
        ok = ok and r2 >= GATE_CFG["node_rec_min"]
        why_extra += f"; node recall within 2 m {r2:.3f} >= {GATE_CFG['node_rec_min']}"
    return dict(node_f1=f, node_prec=p, node_rec=r, node_prec_2m=p2, node_rec_2m=r2, e271_f1=ref, pos_err_med=pos_err, flag_recall={f"{k}": float(rec_flag[i] / max(n_flag[i], 1)) for i, k in enumerate(("stop", "exit", "split", "merge", "start", "end", "win_in", "win_out"))},
                **{"pass": bool(ok), "why": f"node F1 {f:.3f} vs E271 {ref:.3f} - 0.05; median position error {pos_err:.2f} m <= 0.5" + why_extra})


def e271_vertex_f1(n):
    """the same two numbers for the E271 read-out (card threshold) on the same frames"""
    import json
    import soft_lines as SL
    import vertex_eval as VE
    from scipy.spatial import cKDTree
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = e271_head(0).to(dev)
    card = json.load(open(os.path.join(ROOT, "runs", "e271", "e271_a_s0", "card.json")))
    thr = float(card["thr"])
    prec, rec = [], []
    with torch.no_grad():
        for b in _batches("exact", n):
            sp = SL.sparsify(head(b["x"].to(dev).float()))
            for i in range(b["x"].shape[0]):
                V, H, w = VE.read_out(sp[i].cpu(), thr)
                V = np.asarray(V)
                gm = b["g05"][i][b["m05"][i] > 0].numpy()
                if len(gm) == 0:
                    continue
                prec.append((cKDTree(gm).query(V)[0] <= 2.0).mean() if len(V) else 0.0)
                rec.append((cKDTree(V).query(gm[::8])[0] <= 2.0).mean() if len(V) else 0.0)
    return _f1(float(np.mean(prec)), float(np.mean(rec)))


def component_sections(p, thr_e=0.5, thr_link=0.5):
    """sections = the connected components of the kept links (used where the section head is not trained yet)"""
    nd, lk = p["nodes"], p["links"]
    M = len(nd["e"])
    par = list(range(M))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    keep = (np.asarray(nd["e"]) >= thr_e)
    for a, b, pr in zip(lk["src"], lk["dst"], lk["p"]):
        if pr >= thr_link and keep[a] and keep[b]:
            par[f(int(a))] = f(int(b))
    nd["section"] = np.array([f(i) for i in range(M)], int)
    return p


def graph_numbers(model, n, own_sections=True):
    """the lane suite numbers of the delivered graph on n evenly spaced VAL frames: the scorer's read-out (e >= 0.5, p >= 0.5, one piece per section) of the model's prediction"""
    import qgraph_eval as QE
    import qpredict as QP
    frames = QE.val_frames()
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    fr = [frames[i] for i in idx]
    preds = []
    for b in _batches("exact", n):
        preds += QP.predict(model, b)
    model.train()
    assert len(preds) == len(fr), (len(preds), len(fr))
    if not own_sections:
        preds = [component_sections(p) for p in preds]
    out = QE.score_frames(preds, fr)
    g = out["graph"]
    return dict(junction_F1=g.get("junction_F1"), F1_1=g.get("F1@1.0"), topo_F1=g.get("topo_F1"), lanes_found=g.get("lanes_found"), Q=g.get("Q"), pieces_per_gt_piece=g.get("pieces_per_gt_piece"),
                id_switches=g.get("id_switches_per_100m"), vert100_dp=g.get("vert100_dp"), node_f1_scorer=out["nodes"].get("node_F1") if out.get("nodes") else None)


def gate_links(model, run, n):
    dev = _dev(model)
    tp = fp = fn = 0
    model.eval()
    with torch.no_grad():
        for b in _batches("exact", n):
            out = model(b, upto="links", frozen=("pyr", "nodes", "links"), gt=False)
            L = out["L"]
            pr = L["logit"].sigmoid() > 0.5
            cells = L["act"]
            key = (L["pb"] * 2500 + cells[L["pb"], L["pa"]]) * 2500 + cells[L["pb"], L["pc"]]
            lk = b["lk"].to(dev)
            okl = lk[..., 0] >= 0
            bb = torch.arange(lk.shape[0], device=dev)[:, None].expand_as(okl)
            gk = ((bb[okl] * 2500 + lk[..., 0][okl] * 50 + lk[..., 1][okl]) * 2500 + lk[..., 2][okl] * 50 + lk[..., 3][okl])
            hit = torch.isin(key, gk)
            tp += int((pr & hit).sum())
            fp += int((pr & ~hit).sum())
            fn += int(len(gk)) - int((pr & hit).sum())
    model.train()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    res = dict(link_prec=p, link_rec=r, link_f1=_f1(p, r))
    try:
        gn = graph_numbers(model, n, own_sections=False)                                   # the section head is trained in phase 5: sections = link components here
        res.update(gn)
        ok = (gn["junction_F1"] is not None and gn["junction_F1"] > 0.273) and (gn["F1_1"] is not None and gn["F1_1"] >= 0.878 - 0.05)
        res.update({"pass": bool(ok), "why": f"junction F1 {gn['junction_F1']} > 0.273 and F1@1.0 {gn['F1_1']} >= 0.828 (reference graph d: 0.273 / 0.878)"})
    except Exception as e:                                                                   # the scorer is not importable: the graph gate cannot be read
        import traceback
        run.log("graph gate error", traceback.format_exc()[-600:])
        res.update({"pass": False, "why": f"graph gate could not be read: {e!r}"})
    return res


def gate_sections(model, run, n):
    dev = _dev(model)
    ok = tot = 0
    model.eval()
    with torch.no_grad():
        for b in _batches("exact", n):
            out = model(b, upto="sections", frozen=("pyr", "nodes", "links", "sections"), gt=True)
            gtd = dict(piece=b["piece"].to(dev), sstart=b["sstart"].to(dev))
            tgt, pa = SE.start_targets(gtd, out["L"])
            v = out["L"]["valid"] & (pa >= 0) & (tgt >= 0)
            ok += int(((out["M"].argmax(-1) == tgt) & v).sum())
            tot += int(v.sum())
    model.train()
    acc = ok / max(tot, 1)
    res = dict(section_start_acc=acc)
    ppg = None
    try:
        gn = graph_numbers(model, n)
        res.update(gn)
        ppg = gn.get("pieces_per_gt_piece")
    except Exception as e:
        run.log("graph numbers for the section gate not read:", repr(e))
    ok_all = acc >= 0.95 and ppg is not None and ppg == ppg and ppg <= 2.0
    return dict(res, **{"pass": bool(ok_all), "why": f"section start accuracy {acc:.3f} >= 0.95 and delivered pieces per GT piece {ppg} <= 2"})


def lane_acc(model, loader, dev, geometric=False):
    asg0 = QA.Assigner(d=model.asg.d).to(dev) if geometric else None
    if asg0 is not None:
        for m_ in (asg0.wq, asg0.wk, asg0.dust):                                                 # the geometric start assigner has no learned part (E272's start values)
            torch.nn.init.zeros_(m_.weight)
            torch.nn.init.zeros_(m_.bias)
    hit = tot = 0
    with torch.no_grad():
        for b in loader:
            out = model(b, upto="assign", frozen=("pyr", "nodes", "links", "sections", "assign"), gt=True)
            L = out["L"]
            if geometric:
                with torch.autocast(device_type="cuda", enabled=False):
                    A = asg0(L["tok"].float(), L["pos"].float(), L["dirv"].float(), torch.ones_like(L["valid"], dtype=torch.float), L["valid"], out["mp"], out["enc"], rounds=0)
            else:
                A = out["A"]
            piece = b["piece"].to(dev).flatten(1).gather(1, L["act"])
            valid = L["valid"] & (piece >= 0)
            pm = b["piece_map"].to(dev)
            K = pm.shape[-1]
            tgt = pm.gather(1, piece.clamp_min(0)[..., None].expand(-1, -1, K)) & valid[..., None]
            has = tgt.any(-1) & valid
            pred = A["P_piece"].argmax(-1)
            h = tgt.gather(2, pred.clamp_max(K - 1)[..., None])[..., 0] & (pred < K)
            hit += int((h & has).sum())
            tot += int(has.sum())
    return hit / max(tot, 1)


def gate_assign(model, run, n):
    dev = _dev(model)
    model.eval()
    res = {}
    for mode in ("error", "exact"):
        dl = _batches(mode, n)
        res[f"lane_acc_{mode}"] = lane_acc(model, dl, dev)
        res[f"geo_acc_{mode}"] = lane_acc(model, dl, dev, geometric=True)
    model.train()
    ok = res["lane_acc_error"] >= res["geo_acc_error"] and res["lane_acc_exact"] >= res["geo_acc_exact"] - 0.03
    return dict(res, **{"pass": bool(ok), "why": "lane_acc(error) >= geometric start; lane_acc(exact) >= geometric start - 0.03 (the card's guard of 0.01 is read on the full scoring)"})


def gate(model, phase, run, n=150):
    fn = {2: gate_pixels, 3: gate_nodes, 4: gate_links, 5: gate_sections, 6: gate_assign}.get(phase)
    if phase == 7:
        res = {}
        for k in (2, 3, 4, 5, 6):
            try:
                res[f"p{k}"] = {kk: vv for kk, vv in gate(model, k, run, n).items() if kk not in ("pass", "why")}
            except Exception as e:
                res[f"p{k}"] = dict(error=repr(e))
        return dict(res, **{"pass": True, "why": "phase 7: numbers only"})
    try:
        return fn(model, run, n)
    except Exception as e:                                                                   # fail CLOSED: a gate that cannot be read stops the night (the kill criterion is not enforced otherwise)
        import traceback
        run.log("gate error", traceback.format_exc()[-800:])
        return {"pass": False, "why": f"gate could not be read: {e!r}"}
