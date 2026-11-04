"""From the lane head's batched outputs to the per-frame prediction of the contract (`qformat`): numpy dicts for the scorer and the pictures.

    pred = predict(model, batch)            # list of B dicts, the format of qformat.py ("nodes", "links", "assign", "maps")
    pred = predict_from_out(out, batch)     # the same from an `out` of QGraph.forward(batch, upto="assign", gt=False)

Nodes = the active nodes (existence above 0.05 are kept: the read-out threshold e >= 0.5 is the scorer's, an OUTPUT step); `section` = the argmax of the section membership M (the active index of the
first node of the section), links = the candidate pairs with their probability and handle scales, `assign.P` = the probabilities over the map pieces (batch `map_ids`) and the dustbin.
The model runs under no_grad with gt=False (no ground truth enters the forward path).
`make_pred_fn(run)` gives a token -> pred callable for the pictures. Run the self-test: python qpredict.py (needs a trained or smoke model: `--run runs/e273_smoke2`)
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/grid", "training/data", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph as QG  # noqa: E402


def predict_from_out(out, batch, keep_maps=False, min_p=0.02):
    L, no, A, M = out["L"], out["no"], out["A"], out["M"]
    B, K = L["valid"].shape
    ex = no["e"].float().flatten(1).sigmoid().gather(1, L["act"])
    flags = no["flags"].float().flatten(2).sigmoid()                                                   # (B, 8, 2500)
    sprob = no["s"].float().flatten(1).sigmoid()
    preds = []
    for b in range(B):
        v = L["valid"][b]
        idx = v.nonzero(as_tuple=True)[0]
        new = -torch.ones(K, dtype=torch.long, device=v.device)
        new[idx] = torch.arange(len(idx), device=v.device)
        cell = L["act"][b][idx]
        sec = new[M[b].argmax(-1)[idx]]                                                                  # the section = the node index (in the kept list) of its start
        sec = torch.where(sec >= 0, sec, torch.arange(len(idx), device=v.device))
        nodes = dict(pos=L["pos"][b][idx].float().cpu().numpy(), dirv=L["dirv"][b][idx].float().cpu().numpy(), e=ex[b][idx].cpu().numpy(), flags=flags[b][:, cell].T.cpu().numpy(),
                     s=sprob[b][cell].cpu().numpy(), cell=torch.stack([cell // 50, cell % 50], -1).cpu().numpy(), section=sec.cpu().numpy())
        sel = (L["pb"] == b) & v[L["pa"]] & v[L["pc"]] & (L["logit"].float().sigmoid() >= min_p)          # links below min_p are dropped (memory); the read-out threshold is 0.5
        links = dict(src=new[L["pa"][sel]].cpu().numpy(), dst=new[L["pc"][sel]].cpu().numpy(), p=L["logit"][sel].float().sigmoid().cpu().numpy(), long=L["is_long"][sel].cpu().numpy(),
                     a=L["sa"][sel].float().cpu().numpy(), b=L["sb"][sel].float().cpu().numpy(), curves=None)
        k = len(batch["map_ids"][b])
        R = len(batch["map_roads"][b])
        assign = dict(P=A["P_piece"][b][idx][:, list(range(k)) + [A["P_piece"].shape[-1] - 1]].float().cpu().numpy(), map_ids=list(batch["map_ids"][b]),
                      road=A["P_road"][b][idx][:, list(range(R)) + [A["P_road"].shape[-1] - 1]].float().cpu().numpy(), road_ids=list(batch["map_roads"][b]))
        maps = None
        if keep_maps:
            P = out["P"]
            lines = P["lines05"][b].float()
            c1 = P["centre1"][b].float()
            maps = dict(lines05=torch.cat([lines[:3].sigmoid(), 2.0 * lines[3:].sigmoid()]).cpu().numpy(), centre1=torch.cat([c1[:1].sigmoid(), c1[1:5], c1[5:6].sigmoid()]).cpu().numpy(),
                        events1=c1[6:].sigmoid().cpu().numpy(), road2=P["road2"][b].float().sigmoid().cpu().numpy())            # probabilities (sigmoid applied), distances in metres
        preds.append(dict(nodes=nodes, links=links, assign=assign, maps=maps, token=batch["token"][b]))
    return preds


@torch.no_grad()
def predict(model, batch, keep_maps=False, min_p=0.02):
    model.eval()
    out = model(batch, upto="assign", frozen=("pyr", "nodes", "links", "sections", "assign"), gt=False)
    return predict_from_out(out, batch, keep_maps=keep_maps, min_p=min_p)


def load_model(run_dir, device="cuda"):
    sd = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    a = sd.get("args") or {}
    m = QG.QGraph(width=int(a.get("width", 48)))
    m.load_state_dict(sd["state"])
    return m.to(device).eval()


def _selftest(run):
    import lane_gt_data as LD
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_model(run, dev)
    ds, dl = LD.loader("val_clean", bs=2, workers=0, shuffle=False, map_mode="error")
    b = next(iter(dl))
    pr = predict(m, b, keep_maps=True)
    p = pr[0]
    n = len(p["nodes"]["e"])
    assert p["nodes"]["pos"].shape == (n, 2) and p["nodes"]["flags"].shape == (n, 8) and p["assign"]["P"].shape == (n, len(p["assign"]["map_ids"]) + 1)
    assert p["links"]["src"].max(initial=-1) < n and abs(p["assign"]["P"].sum(1).mean() - 1) < 1e-3
    print("qpredict selftest ok: nodes", n, "links", len(p["links"]["src"]), "map pieces", len(p["assign"]["map_ids"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/e273_smoke2")
    _selftest(os.path.join(ROOT, ap.parse_args().run))


def make_pred_fn(run, map_mode="error", device="cuda", keep_maps=True):
    """a callable token -> pred dict (with the dense maps for the pictures) for the VAL frames, from a trained run (`runs/<run>/model.pt`): `qgraph_view.py` takes it as its prediction source"""
    import lane_gt_data as LD
    m = load_model(os.path.join(ROOT, "runs", run) if not os.path.isabs(run) else run, device)
    ds = LD.QData("val_clean", map_mode=map_mode, pixels=False)
    by_tok = {ds.tokens[int(ds.bd.rows[k])]: i for i, k in enumerate(ds.ks)}

    def fn(token):
        b = LD.collate([ds[by_tok[token]]])
        return predict(m, b, keep_maps=keep_maps)[0]
    return fn
