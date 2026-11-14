"""E325 step 2: training and diagnostics of the wiring head `Wire` (`training/grid/jwire_head.py`) on records of `evaluation/jwire_data.py`.

    wsl bash tools/env/_py14.sh jwire_train.py cv --seed 0 [--folds 5] [--epochs 60]            # DIAGNOSTIC (card section 1, step 2 of the guard): a scene-wise cross-validation on the VAL records: the head is
                                                                                                 # trained on the labelled ends of a part of the VAL scenes and judged on the rest: IN-SAMPLE on VAL, never a claim; it only
                                                                                                 # decides whether the TRAIN dumps are built. Also the distance-only ablation (features d, along, across only).
    wsl bash tools/env/_py14.sh jwire_train.py run --seed 0 [--name e325_w_s0] [--heads 3] [--epochs 60] [--max-train N]   # the head of the card: through run.py (E325.w_s<S>): TRAIN records (a live pass, GPU) -> the pilot, the jitter ->
                                                                                                 # the heads (3 head seeds + the distance-only ablation) -> `runs/<name>/model.pt` (head seed 0 = the arm), `model_h<k>.pt`, `wire_agreement.json`
    wsl bash tools/env/_py14.sh jwire_train.py selftest

The agreement = the share of the ends with an ORACLE label (a matched GT end that finds a piece in reach and whose place is among the keypoint's places) on which the argmax place equals the label place; `real choices` = the ends
with two or more places; the rule's agreement is the share of label place 0 (the twin). Loss: cross entropy over the places of the labelled ends only.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jwire_data as JD  # noqa: E402
import jwire_head as JH  # noqa: E402

DIST_ONLY = (0, 1, 2)                                                                                  # the ablation: distance and the two offsets


def fit(feat, mask, label, epochs=60, lr=3e-3, bs=512, seed=0, cols=None, wd=1e-4):
    """-> a trained `Wire` (float32, CPU); cols = the feature columns the head may see (the others are zeroed: the ablation)"""
    torch.manual_seed(seed)
    if cols is not None:
        keep = torch.zeros(JH.FEAT)
        keep[list(cols)] = 1.0
        feat = feat * keep
    w = JH.Wire()
    opt = torch.optim.AdamW(w.parameters(), lr=lr, weight_decay=wd)
    sel = (label >= 0).nonzero().squeeze(-1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = sel[torch.randperm(len(sel), generator=g)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = JH.wire_loss(w(feat[b], mask[b]), label[b])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("E325 kill: the loss is not finite")                              # the run fails (exit != 0): the kill criterion of the card
            loss.backward()
            opt.step()
    w.cols = cols
    return w


@torch.no_grad()
def predict(w, feat, mask):
    if getattr(w, "cols", None) is not None:
        keep = torch.zeros(JH.FEAT)
        keep[list(w.cols)] = 1.0
        feat = feat * keep
    return w(feat, mask).argmax(-1)


def agree(pred, label, kind_oracle, n_places):
    """-> dict(oracle, head_same, rule_same, multi, head_multi_same, rule_multi_same) over the ends with an oracle label (place 0 = the rule)"""
    o = kind_oracle & (label >= 0)
    m = o & (n_places >= 2)
    return dict(oracle=int(o.sum()), head_same=int((pred[o] == label[o]).sum()), rule_same=int((label[o] == 0).sum()), multi=int(m.sum()), head_multi_same=int((pred[m] == label[m]).sum()), rule_multi_same=int((label[m] == 0).sum()))


def kill_fired(val, train_jitter):
    """the kill of the card on the numbers of head seed 0: the place-level agreement on the VAL real choices below the rule's by more than 0.02, or on the TRAIN ends (same jitter draws) below the rule's by more than 0.02;
    `val` / `train_jitter` = the dicts of `agree`. A kill caused by the shift between TRAIN and VAL is not a failure to learn (the in-sample VAL cross-validation is above the rule): the log says which of the two fired"""
    v = val["head_multi_same"] / max(val["multi"], 1) < val["rule_multi_same"] / max(val["multi"], 1) - 0.02
    t = train_jitter["head_multi_same"] / max(train_jitter["multi"], 1) < train_jitter["rule_multi_same"] / max(train_jitter["multi"], 1) - 0.02
    return bool(v or t), bool(v), bool(t)


@torch.no_grad()
def calibration(w, feat, mask, label, orc, bins=10):
    """the calibration of the head's weights on LABELLED ends (oracle label): bins of the largest softmax weight -> [(lo, hi, ends, share the argmax place is the label)]"""
    sel = orc & (label >= 0)
    pr = w(feat[sel], mask[sel]).softmax(-1)
    top, arg = pr.max(-1)
    hit = (arg == label[sel]).float()
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (top >= lo) & (top < hi if b < bins - 1 else top <= hi)
        out.append((lo, hi, int(m.sum()), float(hit[m].mean()) if int(m.sum()) else None))
    return out


def tensors(records, jitter=None):
    """-> feat, mask, label, oracle (bool), n_places, scene (int) over the instances of all records"""
    inst, scene = [], []
    for r in records:
        ii = JD.instances_of(r, jitter)
        inst += ii
        scene += [r["scene"]] * len(ii)
    feat, mask, label = JH.pack(inst)
    return feat, mask, label, torch.tensor([i["kind"] == "oracle" for i in inst]), torch.tensor([len(i["piece"]) for i in inst]), torch.tensor(scene)


def cv(records, folds=5, epochs=60, seed=0):
    feat, mask, label, orc, npl, scene = tensors(records)
    scenes = sorted(set(scene.tolist()))
    fold_of = {s: i % folds for i, s in enumerate(scenes)}
    fo = torch.tensor([fold_of[int(s)] for s in scene])
    res = {k: dict(oracle=0, head_same=0, rule_same=0, multi=0, head_multi_same=0, rule_multi_same=0) for k in ("full", "dist_only")}
    for f in range(folds):
        tr, te = fo != f, fo == f
        for name, cols in (("full", None), ("dist_only", DIST_ONLY)):
            w = fit(feat[tr], mask[tr], label[tr], epochs, seed=seed, cols=cols)
            a = agree(predict(w, feat[te], mask[te]), label[te], orc[te], npl[te])
            for k, v in a.items():
                res[name][k] += v
    return res


def run_pipeline(seed, name, heads=3, epochs=60, workers=3, max_train=0, draws=3):
    """the training run of the card (E325.w_s<seed>): see the module docstring; the kill numbers of the card (agreement against the rule on the VAL real choices and on the TRAIN ends with the jitter) are written
    to `wire_agreement.json` and logged"""
    import json
    import trainer as T
    run = T.Run(name)
    run.log(f"E325 head, seed {seed}: records of the TRAIN frames (live pass of B and the pair run), jitter calibrated on the keypoint error, {heads} head seeds + the distance-only ablation")
    vpath = JD.cache_path("val", seed)
    if not os.path.exists(vpath):
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
        pickle.dump(JD.val_records(seed, f"e312_p_s{seed}"), open(vpath, "wb"))
    val = pickle.load(open(vpath, "rb"))
    assert len(val) == 873 and len({r["row"] for r in val}) == 873, f"{vpath}: {len(val)} VAL records (873 needed, unique rows)"
    tpath = JD.cache_path("train", seed, max_train)
    if os.path.exists(tpath):
        train = pickle.load(open(tpath, "rb"))
        assert len({r["row"] for r in train}) == len(train) and (max_train or len(train) >= 8000), f"{tpath}: {len(train)} TRAIN records"
    else:
        train = JD.live_records(seed, "train", 0.1, max_train, workers)
        pickle.dump(train, open(tpath, "wb"))
    pt, pv = JD.pilot_stats(train), JD.pilot_stats(val)
    scale, med = JD.calibrate_jitter(train, pv["err_median"])
    info = dict(seed=seed, train_frames=len(train), pilot_train={k: v for k, v in pt.items() if k != "counts"}, pilot_val={k: v for k, v in pv.items() if k != "counts"}, jitter_scale=scale, jitter_median_error=med)
    run.log("pilot TRAIN", json.dumps(info["pilot_train"]), "VAL", json.dumps(info["pilot_val"]), f"-> jitter scale {scale:.3f} m (median error {med:.3f} m, VAL {pv['err_median']:.3f} m); the keypoint probabilities are blind columns of the head (jwire_head.BLIND)")
    parts = [tensors(train, JD.make_jitter(scale, d)) for d in range(draws)]
    feat, mask, label, orc, npl = (torch.cat([p[i] for p in parts]) for i in range(5))
    vf, vm, vl, vo, vn, _ = tensors(val)
    run.log(f"training ends {len(label)} ({int((label >= 0).sum())} labelled), VAL ends {len(vl)} ({int(vo.sum())} with an oracle label)")
    out = dict(info, heads=[])
    for h in range(heads + 1):
        cols = DIST_ONLY if h == heads else None                                                         # the last one is the ablation
        w = fit(feat, mask, label, epochs, seed=h, cols=cols)
        av = agree(predict(w, vf, vm), vl, vo, vn)
        at = agree(predict(w, feat, mask), label, orc, npl)
        rec = dict(head=h, ablation=cols is not None, val=av, train_jitter=at)
        out["heads"].append(rec)
        run.log(f"head {h}{' (distance-only)' if cols is not None else ''}: VAL overall {av['head_same'] / av['oracle']:.3f} (rule {av['rule_same'] / av['oracle']:.3f}) real choices {av['head_multi_same'] / max(av['multi'], 1):.3f} "
                f"(rule {av['rule_multi_same'] / max(av['multi'], 1):.3f}); TRAIN+jitter real choices {at['head_multi_same'] / max(at['multi'], 1):.3f} (rule {at['rule_multi_same'] / max(at['multi'], 1):.3f})")
        if h == 0:
            out["calibration_val_labelled_ends"] = calibration(w, vf, vm, vl, vo)
        if cols is None:
            torch.save(dict(state=w.state_dict(), args=dict(seed=seed, head_seed=h, jitter_scale=scale, epochs=epochs)), run.path("model.pt" if h == 0 else f"model_h{h}.pt"))
    h0 = out["heads"][0]
    kill, k_val, k_train = kill_fired(h0["val"], h0["train_jitter"])
    out["kill_fired"], out["kill_on_val"], out["kill_on_train"] = kill, k_val, k_train
    json.dump(out, open(run.path("wire_agreement.json"), "w"), indent=1, default=float)
    run.log(f"KILL (agreement below the rule's by more than 0.02 on the VAL real choices or on the TRAIN ends): {kill} (VAL {k_val}, TRAIN {k_train}); parameters 545")
    run.mark_done()
    return out


def _selftest():
    torch.manual_seed(0)
    N = 800
    feat = torch.randn(N, JH.P_MAX, JH.FEAT)
    mask = torch.zeros(N, JH.P_MAX, dtype=torch.bool)
    mask[:, :3] = True
    label = feat[:, :3, 5].argmax(-1)
    label[:50] = -1                                                                                      # unlabelled ends do not enter
    w = fit(feat, mask, label, epochs=40)
    acc = float((predict(w, feat, mask)[50:] == label[50:]).float().mean())
    assert acc > 0.8, acc
    wd = fit(feat, mask, label, epochs=40, cols=DIST_ONLY)                                               # the distance-only head cannot see feature 5: it stays near chance (1/3)
    accd = float((predict(wd, feat, mask)[50:] == label[50:]).float().mean())
    assert accd < 0.6, accd
    a = agree(predict(w, feat, mask), label, torch.ones(N, dtype=torch.bool), torch.full((N,), 3))
    assert a["oracle"] == N - 50 and a["multi"] == N - 50 and a["head_same"] > 0.8 * a["oracle"], a
    # the kill: fires when the VAL real-choice agreement is more than 0.02 below the rule's, or the TRAIN one; not at a tie
    tie = dict(multi=100, head_multi_same=70, rule_multi_same=71)
    low = dict(multi=100, head_multi_same=60, rule_multi_same=71)
    assert kill_fired(tie, tie)[0] is False and kill_fired(low, tie) == (True, True, False) and kill_fired(tie, low) == (True, False, True)
    # a non-finite loss stops the run (the kill 'the loss is not finite')
    bad = feat.clone()
    bad[100, 0, 0] = float("nan")                                                                        # a labelled end (the first 50 are unlabelled)
    try:
        fit(bad, mask, label, epochs=1, bs=800)
        raise AssertionError("a NaN loss must stop the fit")
    except FloatingPointError:
        pass
    # calibration: bins over the labelled ends, counts add up
    cal = calibration(w, feat, mask, label, torch.ones(N, dtype=torch.bool))
    assert sum(c[2] for c in cal) == N - 50
    # the jitter calibration reaches the median error it is asked for (records without keypoint error: the error is the jitter itself)
    toy = [dict(pairs=np.zeros((3, 8)), gts=np.zeros((3, 2, 4))) for _ in range(40)]
    sc, md = JD.calibrate_jitter(toy, 1.0)
    assert abs(md - 1.0) < 0.1 and sc > 0, (sc, md)
    print(f"jwire_train selftest ok (a head sees feature 5: {acc:.2f}; the distance-only ablation does not: {accd:.2f})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cv")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--epochs", type=int, default=60)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--name", default=None)
    r.add_argument("--heads", type=int, default=3)
    r.add_argument("--epochs", type=int, default=60)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--max-train", type=int, default=0)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "run":
        run_pipeline(a.seed, a.name or f"e325_w_s{a.seed}", a.heads, a.epochs, a.workers, a.max_train)
        return
    if a.cmd == "selftest":
        _selftest()
        return
    recs = pickle.load(open(JD.cache_path("val", a.seed), "rb"))
    res = cv(recs, a.folds, a.epochs, seed=0)
    print(f"seed {a.seed}: scene-wise {a.folds}-fold cross-validation on VAL (IN-SAMPLE, diagnostic)")
    for name, r in res.items():
        print(f"  {name:9s} ends with an oracle label {r['oracle']}: head {r['head_same'] / r['oracle']:.3f}, rule {r['rule_same'] / r['oracle']:.3f} | real choices {r['multi']}: head {r['head_multi_same'] / max(r['multi'], 1):.3f}, rule {r['rule_multi_same'] / max(r['multi'], 1):.3f}")


if __name__ == "__main__":
    main()
