"""Decodes, dump, scoring and attribution of E359 (the token head with the node-type head and the per-source softmax, `training/grid/jtok2.py`). The controls (rules, GT links) are E357's stored cards and are NOT re-run;
the writer, the read-outs, the link tables and the cards are `jtok_eval.py`'s.

    wsl bash tools/env/_py14.sh jtok2_eval.py dump  --run R --seed S [--drop 0.17 --drop-seed 0] [--cpu] [--max N]   # GPU: the 873 VAL frames -> runs/R/tok2_val[_drop..].pkl
    wsl bash tools/env/_py14.sh jtok2_eval.py score --run R --seed S [--drop 0.17 --drop-seed 0]                      # CPU -> runs/R/tok2_card_<decode>[_drop..]_<readout>.json (decode rec, cond03; readout sec0, gtsec); never overwrites
    wsl bash tools/env/_py14.sh jtok2_eval.py diag  --run R --seed S [--drop 0.17 --drop-seed 0]                      # CPU: the attribution numbers on the stored dump -> prints and writes runs/R/tok2_diag[_drop..].json

Decodes (per frame, numpy; s = the conditional softmax of `jtok2.probs`, tp = the type probabilities (long_out, long_in, short_out), a candidate = an allowed pair of the class: short chord <= 8 m / long chord > 8 m):
  rec     OF RECORD: per source and class the candidate with the highest s (ties: the LOWER node index), kept when tp of the class (short_out for short, long_out for long) is at least 0.5. At most one short and one long link
          per source (a fork gives one of its links: a named cap).
  cond03  per gated source (tp of the class at least 0.5) every candidate with s >= 0.3 (the multi-successor decode; IN-SAMPLE threshold, never the verdict).
  Both write p = 1 (the decode is the selection: the scorer's p >= 0.5 passes every kept link), the predicted handle scales for long links and 1/3 for short ones; the edge probability tp x s is kept in `pred["p_edge"]`.
Dump (one list, a dict per frame): token, row, rec, cond03 (predictions of `jtok_eval.edge_prediction`), types (M, 3) float32 = tp of EVERY node, cand = the top-3 candidates by s of every source and class (src, dst, s, long;
sorted by s, ties by the lower node index). A source that has a GT link of the class and no stored candidate counts as a miss in `diag`.
Diag (IN-SAMPLE on VAL: a diagnosis next to the verdict): type head per node type: precision / recall / F1 at 0.3 / 0.5 / 0.7 and ROC AUC over all nodes of the frames (labels from the in-gate GT links: long_out = the node is the
source of a long one, long_in its destination, short_out the source of a short one); oracle-source top-1 accuracy per class (is the highest-s candidate a GT link, among the sources that truly have one in the gate);
composition of the delivered long links of `rec` (P_link, R_link over the GT long links, P_src / R_src / F1_src at the source level: R_src = sources whose delivered link is a GT link / sources with a GT long link; with one
link per source P_src = P_link); the GT links outside the gate; a GEOMETRY-ONLY NULL (no training): the distance to the nearest node ahead inside the gate with chord <= 8 m (inf when none) as a score for long_out and for
'has no short successor' (AUC), and the rule 'no allowed short candidate' for long_out (precision / recall). Tests: training/grid/test_jtok2.py
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtok as JT  # noqa: E402
import jtok2 as J2  # noqa: E402
import jtok_eval as JV  # noqa: E402
import lane_tok_data as TD  # noqa: E402

RUNS = JV.RUNS
DECODES = ("rec", "cond03")
THRS = (0.3, 0.5, 0.7)
TOPK = 3


# ------------------------------------------------------------------------------------------------------------------------------ decodes of one frame
def _gate(tp, longm):
    """(M, M) the type probability of the source's class for every pair (long_out for the long pairs, short_out for the short ones)"""
    return np.where(longm, tp[:, 0][:, None], tp[:, 2][:, None])


def decode_record(s, cand, longm, tp):
    """s, cand, longm (M, M), tp (M, 3) -> (src, dst): the decode `rec` (module docstring)"""
    src, dst = [], []
    for lg in (False, True):
        cm = cand & (longm == lg)
        i = np.flatnonzero(cm.any(1) & (tp[:, 0 if lg else 2] >= 0.5))
        src.append(i)
        dst.append(np.where(cm, s, -1.0).argmax(1)[i])                      # the first maximum = the lower node index
    return np.concatenate(src), np.concatenate(dst)


def decode_cond(s, cand, longm, tp, thr=0.3):
    """-> (src, dst): the decode `cond03`: every candidate with s >= thr of a source whose type probability of the class is at least 0.5"""
    return np.nonzero(cand & (_gate(tp, longm) >= 0.5) & (s >= thr))


def top_candidates(s, cand, longm, k=TOPK):
    """-> dict(src, dst, s, long): per source and class the k candidates with the highest s (ties: the lower node index)"""
    out = dict(src=[], dst=[], s=[], long=[])
    for lg in (False, True):
        sc = np.where(cand & (longm == lg), s, -1.0)
        order = np.argsort(-sc, axis=1, kind="stable")[:, :k]
        val = np.take_along_axis(sc, order, 1)
        i, c = np.nonzero(val >= 0)
        out["src"].append(i)
        out["dst"].append(order[i, c])
        out["s"].append(val[i, c])
        out["long"].append(np.full(len(i), lg))
    return {key: np.concatenate(v) for key, v in out.items()}


def to_prediction(t, src, dst, scale, p_edge):
    """the links (src, dst) of a decode through the writer: p = 1, predicted handle scales for the chords above 8 m and 1/3 below (as `jtok_eval.pair_prediction`), `p_edge` kept next to the links"""
    lg = np.linalg.norm(t["pos"][src] - t["pos"][dst], axis=1) > JT.SHORT
    a, b = (np.where(lg, scale[src, dst, i], 1 / 3) for i in (0, 1))
    return dict(JV.edge_prediction(t, src, dst, np.ones(len(src)), a, b), p_edge=np.asarray(p_edge, float))


def frame_dump(t, s, cand, longm, tp, scale):
    """one frame of the dump (module docstring): t = the table, s, cand, longm (N, N), tp (N, 3), scale (N, N, 2) of the padded batch"""
    M = len(t["pos"])
    s, cand, longm, tp, scale = s[:M, :M], cand[:M, :M], longm[:M, :M], tp[:M], scale[:M, :M]
    pe = _gate(tp, longm) * s
    out = dict(types=tp.astype(np.float32), cand=top_candidates(s, cand, longm))
    for name, (src, dst) in (("rec", decode_record(s, cand, longm, tp)), ("cond03", decode_cond(s, cand, longm, tp))):
        out[name] = to_prediction(t, src, dst, scale, pe[src, dst])
    return out


# ------------------------------------------------------------------------------------------------------------------------------ attribution
def prf(pred, label):
    """precision, recall, F1 of a boolean prediction against a boolean label (an empty side gives 0.0, the scorer's convention, NaN only when there is neither a label nor a prediction)"""
    tp, npred, nlab = float((pred & label).sum()), float(pred.sum()), float(label.sum())
    return JV.QE._prf(tp, npred, nlab)


def auc(score, label):
    """ROC AUC of `score` for the boolean `label` over all items (ties count half; NaN when one class is empty)"""
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    u, inv, cnt = np.unique(np.asarray(score, float), return_inverse=True, return_counts=True)
    rank = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]                          # the average 1-based rank of each score value
    return float((rank[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _frame_truth(t):
    """the in-gate GT of one table -> dict(GS, GL, allowed) (numpy, M x M; the gate as the head computes it, float32), dist (M,) = the distance to the nearest node ahead inside the gate with chord <= 8 m (inf: none)"""
    M = len(t["pos"])
    g, allowed = JT.pair_geometry(torch.from_numpy(t["pos"]).float()[None], torch.from_numpy(t["dirv"]).float()[None])
    chord = g[..., 2] * 20.0
    y = torch.zeros(1, M, M)
    y[0, torch.from_numpy(np.asarray(t["src"], np.int64)), torch.from_numpy(np.asarray(t["dst"], np.int64))] = 1.0
    GS, GL = J2.gt_classes(y, allowed, chord)
    near = torch.where(allowed & (chord <= JT.SHORT), chord, torch.full_like(chord, float("inf"))).min(2)[0][0]
    return dict(GS=GS[0].numpy(), GL=GL[0].numpy(), allowed=allowed[0].numpy(), dist=near.numpy())


def _top1(cand, G, M):
    """oracle-source top-1 of one class: (hits, sources) over the sources that have an in-gate GT link of the class; a source without a stored candidate is a miss"""
    sc = np.full((M, M), -1.0)
    sc[cand[0], cand[1]] = cand[2]
    top = sc.argmax(1)
    src = G.any(1)
    return int((src & (sc.max(1) >= 0) & G[np.arange(M), top]).sum()), int(src.sum())


def diagnose(preds, tables):
    """the attribution numbers of a stored dump (module docstring): preds = the dump list, tables = the GT table of every frame (in order) -> a flat-ish json-able dict"""
    names = J2.TYPES
    tp, lab, dist = [], [], []
    top = {"short": [0, 0], "long": [0, 0]}
    comp = dict(deliv=0, tp=0, gt=0, src_gt=0)
    gate = dict(short=[0, 0], long=[0, 0])
    for p, t in zip(preds, tables):
        M = len(t["pos"])
        assert len(p["types"]) == M, "the dump and the tables are not the same nodes"
        f = _frame_truth(t)
        tp.append(p["types"])
        lab.append(np.stack([f["GL"].any(1), f["GL"].any(0), f["GS"].any(1)], 1))
        dist.append(f["dist"])
        c = p["cand"]
        for name, G in (("short", f["GS"]), ("long", f["GL"])):
            m = c["long"] == (name == "long")
            h, n = _top1((c["src"][m], c["dst"][m], c["s"][m]), G, M)
            top[name][0] += h
            top[name][1] += n
        out = ~f["allowed"][t["src"], t["dst"]]
        for name, m in (("short", ~t["long"]), ("long", t["long"])):
            gate[name][0] += int(out[m].sum())
            gate[name][1] += int(m.sum())
        lk = p["rec"]["links"]
        GTL = np.zeros((M, M), bool)
        GTL[t["src"][t["long"]], t["dst"][t["long"]]] = True
        sel = lk["long"]
        comp["deliv"] += int(sel.sum())
        comp["tp"] += int(GTL[lk["src"][sel], lk["dst"][sel]].sum())
        comp["gt"] += int(t["long"].sum())
        comp["src_gt"] += int(GTL.any(1).sum())
    tp, lab, dist = np.concatenate(tp), np.concatenate(lab), np.concatenate(dist)
    out = dict(frames=len(preds), nodes=len(tp), type={})
    for k, name in enumerate(names):
        e = dict(n_pos=int(lab[:, k].sum()), auc=auc(tp[:, k], lab[:, k]))
        for thr in THRS:
            e.update({f"{m}@{thr:g}": v for m, v in zip(("precision", "recall", "f1"), prf(tp[:, k] >= thr, lab[:, k]))})
        out["type"][name] = e
    out["top1"] = {n: dict(sources=v[1], hits=v[0], accuracy=v[0] / v[1] if v[1] else float("nan")) for n, v in top.items()}
    P_, R_, F_ = JV.QE._prf(comp["tp"], comp["deliv"], comp["gt"])
    Rs = comp["tp"] / comp["src_gt"] if comp["src_gt"] else float("nan")
    out["composition"] = dict(delivered_long=comp["deliv"], true_long=comp["tp"], gt_long=comp["gt"], gt_long_sources=comp["src_gt"], P_link=P_, R_link=R_, F1_link=F_, P_src=P_, R_src=Rs,
                              F1_src=2 * P_ * Rs / (P_ + Rs) if P_ + Rs > 0 else 0.0)
    out["gate"] = dict(gt_links=gate["short"][1] + gate["long"][1], outside=gate["short"][0] + gate["long"][0], share=(gate["short"][0] + gate["long"][0]) / max(gate["short"][1] + gate["long"][1], 1),
                       short_share=gate["short"][0] / max(gate["short"][1], 1), long_share=gate["long"][0] / max(gate["long"][1], 1))
    nosucc = np.isinf(dist)
    rp, rr, rf = prf(nosucc, lab[:, 0])
    out["geometry_null"] = dict(auc_long_out=auc(dist, lab[:, 0]), auc_no_short_successor=auc(dist, ~lab[:, 2]), rule_no_short_candidate_precision=rp, rule_no_short_candidate_recall=rr, rule_f1=rf,
                                nodes_without_short_candidate=int(nosucc.sum()))
    lo = out["type"]["long_out"]
    out["attribution"] = dict(type_head_works=bool(lo["f1@0.5"] >= 0.90 and lo["auc"] >= 0.97), softmax_works=bool(out["top1"]["long"]["accuracy"] >= 0.85),
                              rule="the card: type head works when long_out F1 >= 0.90 and AUC >= 0.97 at 0.5; softmax works when the oracle-source long top-1 accuracy >= 0.85 (one run: the card needs both seeds)")
    return out


def show_diag(d):
    ty = d["type"]
    for name in J2.TYPES:
        e = ty[name]
        print(f"type {name:<9} n_pos {e['n_pos']:>6}  AUC {e['auc']:.4f}  " + "  ".join(f"@{t:g} P {e[f'precision@{t:g}']:.3f} R {e[f'recall@{t:g}']:.3f} F1 {e[f'f1@{t:g}']:.3f}" for t in THRS), flush=True)
    g = d["geometry_null"]
    print(f"geometry null: AUC long_out {g['auc_long_out']:.4f}, AUC no short successor {g['auc_no_short_successor']:.4f}; rule 'no short candidate' for long_out: P {g['rule_no_short_candidate_precision']:.3f} "
          f"R {g['rule_no_short_candidate_recall']:.3f} F1 {g['rule_f1']:.3f}")
    print(f"oracle-source top-1: short {d['top1']['short']['accuracy']:.4f} ({d['top1']['short']['sources']} sources), long {d['top1']['long']['accuracy']:.4f} ({d['top1']['long']['sources']})")
    c = d["composition"]
    print(f"delivered long links (rec): {c['delivered_long']} for {c['gt_long']} GT; P_link {c['P_link']:.3f} R_link {c['R_link']:.3f} F1_link {c['F1_link']:.3f}; source level P {c['P_src']:.3f} R {c['R_src']:.3f} F1 {c['F1_src']:.3f}")
    print(f"GT links outside the gate: {d['gate']['outside']} of {d['gate']['gt_links']} = {100 * d['gate']['share']:.3f} % (short {100 * d['gate']['short_share']:.3f} %, long {100 * d['gate']['long_share']:.3f} %)")
    print(f"attribution: type head {'works' if d['attribution']['type_head_works'] else 'FAILED'}, softmax {'works' if d['attribution']['softmax_works'] else 'FAILED'}")


# ------------------------------------------------------------------------------------------------------------------------------ commands
def _sfx(a, args):
    return JV.suffix(args["lattice"], a.drop, a.drop_seed)


def dump(a, runs=RUNS):
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU; --cpu only for tests"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jtok2_eval dump {a.run}")
    m, args = J2.load_model(a.run, dev, runs)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    ds = TD.TokData("val", args["lattice"])
    data = JV.Degraded(ds, a.drop, a.drop_seed) if a.drop > 0 else ds
    n = min(a.max or len(ds), len(ds))
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(n)), batch_size=8, shuffle=False, num_workers=0 if a.cpu else 2, collate_fn=TD.collate)
    path = os.path.join(runs, a.run, f"tok2_val{_sfx(a, args)}{f'_n{n}' if n < len(ds) else ''}.pkl")
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    out_all = []
    with torch.no_grad():
        for b in dl:
            out = J2.forward_batch(m, b)
            pr = J2.probs(out)
            s, tp, lm, sc, al = (x.cpu().numpy() for x in (pr["s"], pr["tp"], pr["longm"], out["scale"].float(), out["allowed"]))
            out_all += [dict(frame_dump(t, s[i], al[i], lm[i], tp[i], sc[i]), token=b["token"][i], row=int(b["row"][i])) for i, t in enumerate(b["table"])]
    with open(path, "wb") as fh:
        pickle.dump(out_all, fh)
    print(f"written {os.path.relpath(path, ROOT)}: {len(out_all)} frames, links per frame rec {sum(len(p['rec']['links']['src']) for p in out_all) / max(len(out_all), 1):.1f}, "
          f"cond03 {sum(len(p['cond03']['links']['src']) for p in out_all) / max(len(out_all), 1):.1f}, device {dev}")
    return path


def _load(a, runs):
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu")
    args = ck["args"]
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    with open(os.path.join(runs, a.run, f"tok2_val{_sfx(a, args)}.pkl"), "rb") as fh:
        return ck, args, pickle.load(fh)


def score(a, runs=RUNS):
    ck, args, preds = _load(a, runs)
    sfx = _sfx(a, args)
    path = {(d, r): os.path.join(runs, a.run, f"tok2_card_{d}{sfx}_{r}.json") for d in DECODES for r in JV.READOUTS}
    held = [p for p in path.values() if os.path.exists(p)]
    assert not held, f"a stored card is never overwritten: {held}"
    frames, table = JV.val_tables(args["lattice"], a.drop, a.drop_seed)
    assert len(preds) == len(frames) == 873 and [p["token"] for p in preds] == [f.token for f in frames], f"the dump holds {len(preds)} frames: the score needs the 873 VAL frames in order"
    res = JV.score_arms(frames, lambda f: (table(f.k), {d: preds[f.k][d] for d in DECODES}), args["lattice"], 0.5)
    head = sum(v.numel() for k, v in ck["state"].items() if k.startswith("tok.")) / 1e6
    for (d, r), p in path.items():
        JV.write_json(p, JV.make_card(f"{a.run}_tok2_{d}{sfx}_{r}", len(frames), res[d][r], r, run=a.run, seed=a.seed, lattice=args["lattice"], decode=d, drop=a.drop, drop_seed=a.drop_seed, steps=ck.get("steps"),
                                      head_params_m=head, decode_rule="rec: argmax s per source and class kept at the type probability >= 0.5; cond03: every candidate with s >= 0.3 of a gated source (IN-SAMPLE threshold)"))
        JV.show(f"{a.run} {d} {r}", res[d][r])
    return res


def diag(a, runs=RUNS):
    _, args, preds = _load(a, runs)
    path = os.path.join(runs, a.run, f"tok2_diag{_sfx(a, args)}.json")
    assert not os.path.exists(path), f"a stored diagnosis is never overwritten: {path}"
    ds = TD.TokData("val", args["lattice"], feats=False)
    assert len(preds) == len(ds) == 873 and [p["token"] for p in preds] == ds.tokens, "the diagnosis needs the dump of the 873 VAL frames in order"
    tabs = [ds.table(k) for k in range(len(ds))]
    if a.drop > 0:
        tabs = [JV.sub_table(t, JV.keep_nodes(ds.tokens[k], len(t["pos"]), a.drop, a.drop_seed)) for k, t in enumerate(tabs)]
    d = diagnose(preds, tabs)
    d.update(run=a.run, seed=a.seed, drop=a.drop, note="IN-SAMPLE on VAL: a diagnosis next to the verdict; thresholds other than 0.5 are not chosen on anything")
    JV.write_json(path, d)
    show_diag(d)
    print(f"written {os.path.relpath(path, ROOT)}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name_ in ("dump", "score", "diag"):
        s = sub.add_parser(name_)
        s.add_argument("--run", required=True)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--drop", type=float, default=0.0, help="the degraded-node read-out: the share of the nodes removed (0.17), the same removal as E357")
        s.add_argument("--drop-seed", type=int, default=0)
        if name_ == "dump":
            s.add_argument("--cpu", action="store_true", help="tests only: dump on the CPU")
            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; the score needs all 873)")
    a_ = ap.parse_args()
    torch.set_num_threads(2)
    {"dump": dump, "score": score, "diag": diag}[a_.cmd](a_)
