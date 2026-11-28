"""The gates of stage 0 (`qgraph_train.py`): after every phase a reading on `n` evenly spaced VAL frames (150; a smoke run 8), never TEST.

    gate(model, phase, ref, n=150, prec_min=None, rec_min=None, log=print, workers=2) -> dict(pass, why, numbers...)

ref = the E271 grid head the stage started from (a model.pt: state + args; in a run of stage 0 the run's own phase 1, by hand by default the stored
runs/e271_a_s<seed>), the reference of the gates of phases 2 and 3; its read-out threshold is fitted here (`fit_threshold`; v1 always read the head of
e271_a_s0 and the threshold of its card, 0.06347190588712692, which the fit gives exactly for that head).
Phase 1  (the E271 grid head itself) numbers only: its centre-pixel F1.
Phase 2  centre-pixel F1 of the 1 m centre channel (recall of the cells within 0.75 m of a lane, precision against 1.5 m) not more than 0.02 below the E271
         head's own; road / intersection IoU reported.
Phase 3  node F1 against the lane points (a node counts when a GT lane point lies within 2 m, a GT lane point every 4 m is found when a node lies within 2 m)
         not more than 0.05 below that of the E271 read-out (`read_out` at the threshold of 11.2 vertices per 100 m of GT lane, fitted on 300 TRAIN frames:
         `fit_threshold`), median position error of the nodes in cells with a GT node <= 0.5 m, and, when given, node precision >= prec_min and recall >=
         rec_min within 2 m of the true node positions (E292 a: 0.75 / 0.5); flag recall reported.
Phase 4  link precision / recall / F1 on the GT-linked cells: numbers only.
Phase 5  the most probable section start is the right node for >= 95 % of the GT nodes.
Phase 6  assignment accuracy on the nodes with a map 1-2 m off not below the geometric start assigner's; with the exact map not more than 0.03 below.
Phase 7 / 9  the numbers of phases 2 - 6 / 3 - 6 (phase 9 has a frozen pyramid), no condition.
A gate that cannot be read fails (closed; its reason starts with UNREAD). The gates of phases 4 and 5 hold no condition on the delivered graph (junction
F1, F1@1.0, pieces per GT piece: the tag v1 has them): that needs the scorer, which lives in evaluation/ (protected), and training/ does not import
evaluation/ (DESIGN R13), nor may a second copy of a metric exist; the champion's B' comes from runs that went on past those gates (`--gate-phases`).
The node part of the network runs through the forward path (`model(batch, upto=...)`), never through the node head alone. Test: net/tests/test_qgraph_train.py"""
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from lane_center.net.loss import loss_sections as LSE
from lane_center.net.model import head as GH
from lane_center.net.model import lane_net as LN
from lane_center.net.model import qassign as QA
from lane_center.net.model import qnodes as QN
from lane_center.net.model import soft_lines as SL
from lane_center.net.prep import grid_data as GD
from lane_center.net.prep import lane_gt_data as LD
from lane_center.net.prep import targets as TG

E271_BUDGET = 11.2                 # vertices per 100 m of GT lane: the read-out threshold of the E271 card (the reference of the node gate)
FIT_FRAMES = 300                   # TRAIN frames the threshold is fitted on (in-sample, as the E271 card)
_REF = {}                          # path -> (the E271 head, its threshold or None): read once per process
UNREAD = "gate could not be read"  # the start of the reason of a gate that failed closed (a restart reads it again)


def val_batches(map_mode, n, bs=4, workers=2):
    ds = LD.QData("val_clean", map_mode=map_mode, pixels=True)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT: the gates would read another set of frames"
    ks = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, ks.tolist()), batch_size=bs, shuffle=False, num_workers=workers, collate_fn=LD.collate)


def _f1(p, r):
    return 2 * p * r / max(p + r, 1e-9)


def _dev(model):
    return next(model.parameters()).device


# ------------------------------------------------------------------------------------------------------------------------------ the E271 reference
def read_out(sparse, thr, res=1.0, origin=(-50.0, -50.0)):
    """the read-out of a grid head: sparse map (5, X, Y) [w, cos, sin, ox, oy] -> vertices (N, 2) metres: the cells with w >= thr that are the maximum of
    their 3 x 3 neighbourhood (rows = x, columns = y)"""
    s = torch.as_tensor(sparse)
    w = s[0][None, None]
    peak = (w == F.max_pool2d(w, 3, 1, 1))[0, 0] & (s[0] >= thr)
    ix, iy = torch.nonzero(peak, as_tuple=True)
    return torch.stack([(ix.float() + 0.5 + s[3][ix, iy]) * res + origin[0], (iy.float() + 0.5 + s[4][ix, iy]) * res + origin[1]], 1).numpy()


def peak_weights(sparse, floor=0.01, box=48.0):
    """the weights of the 3 x 3 maxima of a sparse map (5, X, Y) with a cell centre inside |x|, |y| < box: the candidates of `read_out`"""
    w = torch.as_tensor(sparse)[0][None, None]
    peak = (w == F.max_pool2d(w, 3, 1, 1))[0, 0] & (w[0, 0] >= floor)
    n = peak.shape[0]
    c = (torch.arange(n, dtype=torch.float32) + 0.5) - n / 2
    inb = (c.abs() < box)[:, None] & (c.abs() < box)[None, :]
    return w[0, 0][peak & inb].numpy()


def fit_threshold(head, budget=E271_BUDGET, n=FIT_FRAMES, seed=0, workers=2):
    """the read-out threshold of a grid head at `budget` vertices per 100 m of GT lane on n TRAIN frames (drawn with seed): the k-th largest peak weight,
    k = budget x the GT length / 100 -> (threshold, the vertices per 100 m it reaches there)"""
    ds = GD.GridData("train")
    ks = sorted(np.random.RandomState(seed).choice(len(ds), n, replace=False).tolist())
    dev, ws, glen = _dev(head), [], 0.0
    with torch.no_grad():
        for b in GD.loader(GD.GridData("train", ks=ks), bs=8, workers=workers):
            sp = SL.sparsify(head(b["x"].to(dev).float())).cpu()
            ws += [peak_weights(sp[i]) for i in range(sp.shape[0])]
            glen += float(b["m05"].sum()) * 0.5
    ws = np.sort(np.concatenate(ws))[::-1]
    k = min(max(int(round(budget * glen / 100.0)), 1), len(ws))
    thr = float(ws[k - 1])
    return thr, 100.0 * float((ws >= thr).sum()) / glen


def reference(path, dev, with_thr, fit_n=FIT_FRAMES, workers=2):
    """-> (the E271 grid head of `path` in eval mode on dev, its read-out threshold (`fit_threshold`; None until a gate needs it)), each made once per process"""
    if path not in _REF:
        ck = LN.load_checkpoint(path)
        h = GH.GridHead(width=int((ck.get("args") or {}).get("width", 48)))
        h.load_state_dict(ck["state"])
        _REF[path] = (h.to(dev).eval(), None)
    if with_thr and _REF[path][1] is None:
        _REF[path] = (_REF[path][0], fit_threshold(_REF[path][0], n=fit_n, workers=workers)[0])
    return _REF[path]


# ------------------------------------------------------------------------------------------------------------------------------ the gates
def centre_f1(p_logit, g05, t05):
    with torch.no_grad():
        d, _, _, _ = TG.cell_targets(g05, t05, n=p_logit.shape[-1])
        pred, tgt = p_logit.sigmoid() > 0.5, d <= 0.75
        tp_r, tp_p = (pred & tgt).sum().item(), (pred & (d <= 1.5)).sum().item()
        return _f1(tp_p / max(pred.sum().item(), 1), tp_r / max(tgt.sum().item(), 1))


def gate_grid(model, ref, n, workers):
    dev = _dev(model)
    with torch.no_grad():
        f = [centre_f1(model(b["x"].to(dev).float())[:, 0], b["g05"].to(dev), b["t05"].to(dev)) for b in val_batches("exact", n, workers=workers)]
    return {"centre_f1": float(np.mean(f)), "pass": True, "why": "phase 1: numbers only"}


def gate_pixels(model, ref, n, workers):
    dev = _dev(model)
    head = ref[0]
    f_m, f_r, iou_road, iou_int = [], [], [], []
    with torch.no_grad():
        for b in val_batches("exact", n, workers=workers):
            x = b["x"].to(dev).float()
            P = model(b, upto="pyr")["P"]
            g05, t05 = b["g05"].to(dev), b["t05"].to(dev)
            f_m.append(centre_f1(P["centre1"][:, 0], g05, t05))
            f_r.append(centre_f1(head(x)[:, 0], g05, t05))
            for c, store in ((0, iou_road), (1, iou_int)):
                pr, gt = P["road2"][:, c].sigmoid() > 0.5, b["road2"][:, c].to(dev) > 0.5
                store.append(((pr & gt).sum() / ((pr | gt).sum().clamp_min(1))).item())
    fm, fr = float(np.mean(f_m)), float(np.mean(f_r))
    return dict(centre_f1=fm, e271_centre_f1=fr, road_iou=float(np.mean(iou_road)), inter_iou=float(np.mean(iou_int)), **{"pass": fm >= fr - 0.02, "why": f"centre F1 {fm:.3f} vs E271 {fr:.3f} - 0.02"})


def e271_vertex_f1(ref, n, workers):
    """node F1 (2 m) of the E271 read-out on the frames of the node gate"""
    head, thr = ref
    dev = _dev(head)
    prec, rec = [], []
    with torch.no_grad():
        for b in val_batches("exact", n, workers=workers):
            sp = SL.sparsify(head(b["x"].to(dev).float())).cpu()
            for i in range(sp.shape[0]):
                V = read_out(sp[i], thr)
                gm = b["g05"][i][b["m05"][i] > 0].numpy()
                if len(gm):
                    prec.append((cKDTree(gm).query(V)[0] <= 2.0).mean() if len(V) else 0.0)
                    rec.append((cKDTree(V).query(gm[::8])[0] <= 2.0).mean() if len(V) else 0.0)
    return _f1(float(np.mean(prec)), float(np.mean(rec)))


def gate_nodes(model, ref, n, workers, prec_min=None, rec_min=None):
    dev = _dev(model)
    prec, rec, prec2, rec2, err, rec_flag, n_flag = [], [], [], [], [], np.zeros(8), np.zeros(8)
    centres = QN.cell_centres("cpu")[0]
    with torch.no_grad():
        for b in val_batches("exact", n, workers=workers):
            no = model(b, upto="nodes")["no"]                                                   # the forward path: the node head and the Matrix-NMS layer when the model has it
            for i in range(b["x"].shape[0]):
                m = no["e"][i].sigmoid() > 0.5
                pos = no["pos"][i].permute(1, 2, 0)[m].cpu().numpy()
                gm = b["g05"][i][b["m05"][i] > 0].numpy()
                if len(gm) == 0:
                    continue
                prec.append((cKDTree(gm).query(pos)[0] <= 2.0).mean() if len(pos) else 0.0)
                rec.append((cKDTree(pos).query(gm[::8])[0] <= 2.0).mean() if len(pos) else 0.0)
                ex = b["exists"][i] > 0.5
                tp = (b["offset"][i] + centres)[:, ex].T.numpy()                               # the true node positions (cell centre + offset)
                if ex.any():                                                                   # nothing delivered on a frame with nodes: 0, not skipped
                    prec2.append((cKDTree(tp).query(pos)[0] <= 2.0).mean() if len(pos) else 0.0)
                    rec2.append((cKDTree(pos).query(tp)[0] <= 2.0).mean() if len(pos) else 0.0)
                both = ex & m.cpu()
                if both.any():
                    err.append(float((no["pos"][i].cpu() - (b["offset"][i] + centres))[:, both].norm(dim=0).median()))
                fl = no["flags"][i].sigmoid().cpu() > 0.5
                for f in range(8):
                    t = (b["flags"][i, f] > 0.5) & ex
                    n_flag[f] += int(t.sum())
                    rec_flag[f] += int((t & fl[f] & m.cpu()).sum())
    p, r = float(np.mean(prec)), float(np.mean(rec))
    f, f_ref = _f1(p, r), e271_vertex_f1(ref, n, workers)
    pos_err = float(np.median(err)) if err else float("nan")
    p2, r2 = (float(np.mean(prec2)) if prec2 else float("nan")), (float(np.mean(rec2)) if rec2 else float("nan"))
    ok = f >= f_ref - 0.05 and pos_err <= 0.5
    why = f"node F1 {f:.3f} vs E271 {f_ref:.3f} - 0.05; median position error {pos_err:.2f} m <= 0.5"
    if prec_min is not None:
        ok, why = ok and p2 >= prec_min, why + f"; node precision within 2 m {p2:.3f} >= {prec_min}"
    if rec_min is not None:
        ok, why = ok and r2 >= rec_min, why + f"; node recall within 2 m {r2:.3f} >= {rec_min}"
    flags = {k: float(rec_flag[i] / max(n_flag[i], 1)) for i, k in enumerate(("stop", "exit", "split", "merge", "start", "end", "win_in", "win_out"))}
    return dict(node_f1=f, node_prec=p, node_rec=r, node_prec_2m=p2, node_rec_2m=r2, e271_f1=f_ref, pos_err_med=pos_err, flag_recall=flags, **{"pass": bool(ok), "why": why})


def gate_links(model, ref, n, workers):
    dev = _dev(model)
    tp = fp = fn = 0
    with torch.no_grad():
        for b in val_batches("exact", n, workers=workers):
            L = model(b, upto="links", gt=False)["L"]
            pr, cells = L["logit"].sigmoid() > 0.5, L["act"]
            key = (L["pb"] * 2500 + cells[L["pb"], L["pa"]]) * 2500 + cells[L["pb"], L["pc"]]
            lk = b["lk"].to(dev)
            okl = lk[..., 0] >= 0
            bb = torch.arange(lk.shape[0], device=dev)[:, None].expand_as(okl)
            gk = (bb[okl] * 2500 + lk[..., 0][okl] * 50 + lk[..., 1][okl]) * 2500 + lk[..., 2][okl] * 50 + lk[..., 3][okl]
            hit = torch.isin(key, gk)
            tp, fp, fn = tp + int((pr & hit).sum()), fp + int((pr & ~hit).sum()), fn + int(len(gk)) - int((pr & hit).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"link_prec": p, "link_rec": r, "link_f1": _f1(p, r), "pass": True, "why": "phase 4: numbers only (the graph conditions need the scorer of evaluation/)"}


def gate_sections(model, ref, n, workers):
    dev = _dev(model)
    ok = tot = 0
    with torch.no_grad():
        for b in val_batches("exact", n, workers=workers):
            out = model(b, upto="sections", gt=True)
            tgt, pa = LSE.start_targets(dict(piece=b["piece"].to(dev), sstart=b["sstart"].to(dev)), out["L"])
            v = out["L"]["valid"] & (pa >= 0) & (tgt >= 0)
            ok, tot = ok + int(((out["M"].argmax(-1) == tgt) & v).sum()), tot + int(v.sum())
    acc = ok / max(tot, 1)
    return {"section_start_acc": acc, "pass": acc >= 0.95, "why": f"section start accuracy {acc:.3f} >= 0.95"}


def lane_acc(model, loader, geometric=False):
    """assignment accuracy on the active nodes with a GT piece; geometric: the start assigner without a learned part (E272's start values)"""
    dev = _dev(model)
    asg0 = QA.Assigner(d=model.asg.d).to(dev) if geometric else None
    if asg0 is not None:
        for m_ in (asg0.wq, asg0.wk, asg0.dust):
            torch.nn.init.zeros_(m_.weight)
            torch.nn.init.zeros_(m_.bias)
    hit = tot = 0
    with torch.no_grad():
        for b in loader:
            out = model(b, upto="assign", gt=True)
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
            hit, tot = hit + int((h & has).sum()), tot + int(has.sum())
    return hit / max(tot, 1)


def gate_assign(model, ref, n, workers):
    res = {}
    for mode in ("error", "exact"):
        dl = val_batches(mode, n, workers=workers)
        res[f"lane_acc_{mode}"], res[f"geo_acc_{mode}"] = lane_acc(model, dl), lane_acc(model, dl, geometric=True)
    ok = res["lane_acc_error"] >= res["geo_acc_error"] and res["lane_acc_exact"] >= res["geo_acc_exact"] - 0.03
    return dict(res, **{"pass": bool(ok), "why": "lane_acc(error) >= geometric start; lane_acc(exact) >= geometric start - 0.03"})


GATES = {1: gate_grid, 2: gate_pixels, 3: gate_nodes, 4: gate_links, 5: gate_sections, 6: gate_assign}


def gate(model, phase, ref, n=150, prec_min=None, rec_min=None, log=print, workers=2, fit_n=FIT_FRAMES):
    """the gate of `phase` on the model in eval mode (its mode is restored afterwards) -> dict(pass, why, numbers); ref = the path of the E271 head, fit_n = the TRAIN
    frames of its threshold"""
    was = model.training
    model.eval()
    try:
        if phase in (7, 9):
            res = {f"p{k}": {kk: vv for kk, vv in gate(model, k, ref, n, log=log, workers=workers, fit_n=fit_n).items() if kk != "pass"}
                   for k in ((2, 3, 4, 5, 6) if phase == 7 else (3, 4, 5, 6))}
            return dict(res, **{"pass": True, "why": f"phase {phase}: numbers only"})
        r = reference(ref, _dev(model), phase == 3, fit_n, workers) if phase in (2, 3) else None
        extra = dict(prec_min=prec_min, rec_min=rec_min) if phase == 3 else {}
        return GATES[phase](model, r, n, workers, **extra)
    except Exception as e:                                                                 # fail CLOSED: a gate that cannot be read stops the stage
        log("gate error", traceback.format_exc()[-800:])
        return {"pass": False, "why": f"{UNREAD}: {e!r}"}
    finally:
        model.train(was)
