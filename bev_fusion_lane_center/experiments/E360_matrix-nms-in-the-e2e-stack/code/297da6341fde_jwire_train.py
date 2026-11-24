"""E325 step 2: training and diagnostics of the wiring head `Wire` (`training/grid/jwire_head.py`) on records of `evaluation/jwire_data.py`.

    wsl bash tools/env/_py14.sh jwire_train.py cv --seed 0 [--pair-run e312_p_s0] [--ends kp|att] [--folds 5] [--epochs 60]   # DIAGNOSTIC (card section 1, step 2 of the guard): a scene-wise cross-validation on the VAL records: the head is
                                                                                                 # trained on the labelled ends of a part of the VAL scenes and judged on the rest: IN-SAMPLE on VAL, never a claim; it only
                                                                                                 # decides whether the TRAIN dumps are built. Also the distance-only ablation (features d, along, across only).
    wsl bash tools/env/_py14.sh jwire_train.py run --seed 0 [--name e325_w_s0 | --pair-run e324_s_s0 [--ends att] --name e358_w_s0] [--heads 3] [--epochs 60] [--max-train N]   # the head of the card: through run.py (E325.w_s<S>): TRAIN records (a live pass, GPU) -> the pilot, the jitter ->
                                                                                                 # the heads (3 head seeds + the distance-only ablation + the no-jitter head `wn`) -> `runs/<name>/model.pt` (head seed 0 = the arm), `model_h<k>.pt`, `model_wn.pt`, `wire_agreement.json`
    wsl bash tools/env/_py14.sh jwire_train.py selftest

The PAIR RUN (default E312's `e312_p_s<seed>`; E350: `--pair-run e324_s_s<seed>`) is the run whose keypoints the records hold: it reaches the VAL records, the live pass and the cache names (`jwire_data.cache_path`), the caches are
checked against their meta, and the head's checkpoint args and `wire_agreement.json` carry `pair_run` and `seed`. A run folder that already holds a `wire_agreement.json` or a
`model.pt` is refused in EVERY case (`trainer.Run` has no overwrite guard; the default name e325_w_s<seed> is E325's run); with `--pair-run` the run needs its own `--name`; a name is a folder name.
The END KIND (E358, `--ends kp|att`, default kp): the records hold the keypoints (E325, E350) or the ATTACHED ends of the pair run (`jwire_data`: att needs `--pair-run` of an attach model), it reaches the VAL records, the live pass, the cache
names and metas, and the head's checkpoint args and `wire_agreement.json` (`ends`). The jitter is calibrated on the error of the pair ends of the records (the attached ends under att). The no-jitter head `wn` (E358, reported only) is the
same training on the records without any perturbation of the pair ends: `model_wn.pt` and `wire_agreement.json['no_jitter_head']`; it never replaces a head file.

The agreement = the share of the ends with an ORACLE label (a matched GT end that finds a piece in reach and whose place is among the keypoint's places) on which the argmax place equals the label place; `real choices` = the ends
with two or more places; the rule's agreement is the share of label place 0 (the twin). Loss: cross entropy over the places of the labelled ends only.
"""
import argparse
import os
import re
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


def run_target(seed, name, pair_run=None, ends="kp"):
    """-> (run name, pair run) of a training run. Default (pair_run None): E312's records and the name e325_w_s<seed> (E325); with a pair run `name` is mandatory. A name is a folder name, and a run folder that already
    holds a `wire_agreement.json` or a `model.pt` is refused in EVERY case (`trainer.Run` has no overwrite guard: a second call with the default name would overwrite E325's run); ends att needs a pair run with attached ends"""
    given = pair_run is not None
    pair_run = JD.resolve_pair_run(seed, pair_run)
    JD.check_ends(ends, seed, pair_run)
    assert name is not None or not given, f"--name is mandatory with --pair-run (the default name e325_w_s{seed} would overwrite E325's run)"
    name = f"e325_w_s{seed}" if name is None else name
    assert re.fullmatch(r"[A-Za-z0-9_-]+", name), f"--name {name!r}: a run folder name"
    for f in ("wire_agreement.json", "model.pt"):
        assert not os.path.exists(os.path.join(ROOT, "runs", name, f)), f"runs/{name}/{f} exists: a finished run is never overwritten (give a new --name)"
    return name, pair_run


def run_pipeline(seed, name, heads=3, epochs=60, workers=3, max_train=0, draws=3, pair_run=None, ends="kp"):
    """the training run of the card (E325.w_s<seed>, E350.w_s<seed> with another pair run, E358.w_s<seed> with the attached ends of it): see the module docstring; the kill numbers of the card (agreement against the rule on the
    VAL real choices and on the TRAIN ends with the jitter) are written to `wire_agreement.json` and logged"""
    import json
    import trainer as T
    name, pair_run = run_target(seed, name, pair_run, ends)                                              # before anything is created or computed
    run = T.Run(name, root=os.path.join(ROOT, "runs"))
    run.log(f"E325 head, seed {seed}, pair run {pair_run}, {ends} ends: records of the TRAIN frames (live pass of B and the pair run), jitter calibrated on the error of the pair ends, {heads} head seeds + the distance-only ablation + the no-jitter head")
    vpath = JD.cache_path("val", seed, pair_run=pair_run, ends=ends)
    val = JD.cached_records(vpath, seed, pair_run, lambda: JD.val_records(seed, pair_run, ends=ends), ends)
    assert len(val) == 873 and len({r["row"] for r in val}) == 873, f"{vpath}: {len(val)} VAL records (873 needed, unique rows)"
    tpath = JD.cache_path("train", seed, max_train, pair_run=pair_run, ends=ends)
    train = JD.cached_records(tpath, seed, pair_run, lambda: JD.live_records(seed, "train", 0.1, max_train, workers, pair_run=pair_run, ends=ends), ends)
    assert len({r["row"] for r in train}) == len(train) and (max_train or len(train) >= 8000), f"{tpath}: {len(train)} TRAIN records"
    pt, pv = JD.pilot_stats(train), JD.pilot_stats(val)
    scale, med = JD.calibrate_jitter(train, pv["err_median"])
    info = dict(seed=seed, pair_run=pair_run, ends=ends, train_frames=len(train), pilot_train={k: v for k, v in pt.items() if k != "counts"}, pilot_val={k: v for k, v in pv.items() if k != "counts"}, jitter_scale=scale, jitter_median_error=med)
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
            torch.save(dict(state=w.state_dict(), args=dict(seed=seed, pair_run=pair_run, ends=ends, head_seed=h, jitter_scale=scale, epochs=epochs)), run.path("model.pt" if h == 0 else f"model_h{h}.pt"))
    nf, nm, nl, no, nn, _ = tensors(train)                                                               # the no-jitter ablation (reported only): head seed 0 on the unperturbed records, as many rows as the jittered draws
    w = fit(nf.repeat(draws, 1, 1), nm.repeat(draws, 1), nl.repeat(draws), epochs, seed=0)
    av, at = agree(predict(w, vf, vm), vl, vo, vn), agree(predict(w, nf, nm), nl, no, nn)
    out["no_jitter_head"] = dict(val=av, train=at)
    torch.save(dict(state=w.state_dict(), args=dict(seed=seed, pair_run=pair_run, ends=ends, head_seed="wn", jitter_scale=0.0, epochs=epochs, no_jitter=True)), run.path("model_wn.pt"))
    run.log(f"no-jitter head (wn): VAL overall {av['head_same'] / av['oracle']:.3f} (rule {av['rule_same'] / av['oracle']:.3f}) real choices {av['head_multi_same'] / max(av['multi'], 1):.3f} (rule {av['rule_multi_same'] / max(av['multi'], 1):.3f}); TRAIN real choices {at['head_multi_same'] / max(at['multi'], 1):.3f} (rule {at['rule_multi_same'] / max(at['multi'], 1):.3f})")
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
    _test_pair_run()
    _test_ends()
    print(f"jwire_train selftest ok (a head sees feature 5: {acc:.2f}; the distance-only ablation does not: {accd:.2f})")


def _record_stubs(seen):
    """-> (val_stub, live_stub): stand-ins for `JD.val_records` / `JD.live_records` that note their arguments (seed, pair run, end kind ...) in `seen` and return toy records"""
    def val_stub(seed, run, thr=0.1, nmax=0, ends="kp"):
        seen["val"].append((seed, run, ends))
        return [JD.toy_record(i, i % 22) for i in range(873)]

    def live_stub(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp"):
        seen["live"].append((seed, split, nmax, pair_run, ends))
        return [JD.toy_record(i, i % 22) for i in range(nmax)]
    return val_stub, live_stub


def _test_pair_run():
    """E350 plumbing, with stubs on `JD.val_records` / `JD.live_records` (toy records, a temp ROOT: no real cache, run folder or model is touched): the pair run of the command line reaches the VAL records, the live pass, the cache
    names and the provenance of the run, E325's default path keeps its names, and the refusals fire before anything is computed (the hard-coded `e312_p_s<seed>` and the missing `pair_run=` of E325's pipeline are what this catches)"""
    import contextlib
    import hashlib
    import io
    import json
    import pickle
    global ROOT
    seen = dict(val=[], live=[])
    val_stub, live_stub = _record_stubs(seen)

    def refused(f, *a, **k):
        try:
            f(*a, **k)
        except AssertionError as e:
            return str(e)
        return None
    old_root, old_stubs = ROOT, (JD.val_records, JD.live_records)
    with JD.tmp_root() as d, contextlib.redirect_stdout(io.StringIO()):
        ROOT = d
        JD.val_records, JD.live_records = val_stub, live_stub
        cache = lambda f: os.path.join(d, "temp", "jwire", f)  # noqa: E731
        try:
            # the command line, with a pair run: the whole pipeline on toy records (1 head + the ablation, 1 epoch)
            assert "mandatory" in (refused(main, ["run", "--seed", "0", "--pair-run", "e324_s_s0"]) or ""), "--pair-run without --name must be refused"
            assert seen == dict(val=[], live=[]), "the refusal must come before anything is computed"
            main(["run", "--seed", "0", "--pair-run", "e324_s_s0", "--name", "e350_w_s0", "--heads", "1", "--epochs", "1", "--max-train", "30"])
            assert seen == dict(val=[(0, "e324_s_s0", "kp")], live=[(0, "train", 30, "e324_s_s0", "kp")]), seen   # the pair run arrives at val_records AND at live_records
            names = sorted(os.listdir(cache("")))
            assert names == ["train_records_v1_s0_max30_pair_e324_s_s0.pkl", "val_records_v1_s0_pair_e324_s_s0.pkl"], names     # and into both cache names (E325's names are not used)
            for f in names:
                assert JD.load_records(cache(f), 0, "e324_s_s0") and pickle.load(open(cache(f), "rb"))["meta"]["pair_run"] == "e324_s_s0"
            ck = torch.load(os.path.join(d, "runs", "e350_w_s0", "model.pt"), map_location="cpu")
            wa = json.load(open(os.path.join(d, "runs", "e350_w_s0", "wire_agreement.json")))
            assert ck["args"]["pair_run"] == "e324_s_s0" and ck["args"]["seed"] == 0 and wa["pair_run"] == "e324_s_s0" and wa["seed"] == 0, (ck["args"], wa["pair_run"])
            assert ck["args"]["ends"] == wa["ends"] == "kp" and os.path.exists(os.path.join(d, "runs", "e350_w_s0", "model_wn.pt")) and "no_jitter_head" in wa, "the end kind and the no-jitter head of every run"
            assert "kp ends" in open(os.path.join(d, "runs", "e350_w_s0", "log.txt")).read() and "no-jitter head (wn)" in open(os.path.join(d, "runs", "e350_w_s0", "log.txt")).read()
            assert "pair run e324_s_s0" in open(os.path.join(d, "runs", "e350_w_s0", "log.txt")).read(), "the run log names the pair run"
            # a finished run is never overwritten; a folder with only a model.pt too; nothing is computed
            assert "exists" in (refused(run_pipeline, 0, "e350_w_s0", pair_run="e324_s_s0") or "")
            os.makedirs(os.path.join(d, "runs", "e350_w_s5"))
            open(os.path.join(d, "runs", "e350_w_s5", "model.pt"), "wb").write(b"x")
            assert "exists" in (refused(run_pipeline, 5, "e350_w_s5", pair_run="e324_s_s5") or "")
            os.makedirs(os.path.join(d, "runs", "e350_w_s6"))
            open(os.path.join(d, "runs", "e350_w_s6", "wire_agreement.json"), "w").write("{}")                  # ... and a folder with only a wire_agreement.json
            assert "exists" in (refused(run_pipeline, 6, "e350_w_s6", pair_run="e324_s_s6") or "")
            assert len(seen["val"]) == 1, "a refused run computed something"
            # the caches are checked against their meta, also the TRAIN cache that is loaded silently when it exists: another pair run's cache under this name, or a file without meta, is refused
            for bad_meta_run in ("e312_p_s0", None):
                os.remove(cache("train_records_v1_s0_max30_pair_e324_s_s0.pkl"))
                if bad_meta_run:
                    JD.save_records(cache("train_records_v1_s0_max30_pair_e324_s_s0.pkl"), [JD.toy_record()], 0, bad_meta_run)
                else:
                    pickle.dump([JD.toy_record()], open(cache("train_records_v1_s0_max30_pair_e324_s_s0.pkl"), "wb"))
                assert "meta" in (refused(run_pipeline, 0, "e350_w_s9", 1, 1, 3, 30, 1, pair_run="e324_s_s0") or ""), bad_meta_run
            os.remove(cache("val_records_v1_s0_pair_e324_s_s0.pkl"))
            JD.save_records(cache("val_records_v1_s0_pair_e324_s_s0.pkl"), [JD.toy_record()], 1, "e324_s_s0")     # a VAL cache of the other seed
            assert "meta" in (refused(main, ["cv", "--seed", "0", "--pair-run", "e324_s_s0"]) or ""), "cv must refuse a cache of another seed"
            # E325's default path: E312's run, E325's cache names, the default run name; a SECOND call with the same (default) name is refused and the first run's files stay byte for byte
            run_pipeline(0, None, heads=1, epochs=1, max_train=30, draws=1)
            assert seen["val"][-1] == (0, "e312_p_s0", "kp") and seen["live"][-1] == (0, "train", 30, "e312_p_s0", "kp"), seen
            assert os.path.exists(cache("val_records_v1_s0.pkl")) and JD.load_records(cache("val_records_v1_s0.pkl"), 0) and os.path.exists(cache("train_records_v1_s0_max30.pkl")), "E325's cache names"
            rd = os.path.join(d, "runs", "e325_w_s0")
            files = lambda: {f: hashlib.sha256(open(os.path.join(rd, f), "rb").read()).hexdigest() for f in sorted(os.listdir(rd))}  # noqa: E731
            first, n_calls = files(), (len(seen["val"]), len(seen["live"]))
            assert {"model.pt", "wire_agreement.json", "log.txt"} <= set(first), first
            for argv in (["run", "--seed", "0", "--heads", "1", "--epochs", "1", "--max-train", "30"], ["run", "--seed", "0", "--name", "e325_w_s0", "--heads", "1", "--epochs", "1", "--max-train", "30"]):
                assert "exists" in (refused(main, argv) or ""), argv
            assert "exists" in (refused(run_pipeline, 0, None, 1, 1, 3, 30, 1) or "") and files() == first and (len(seen["val"]), len(seen["live"])) == n_calls, "a second call rewrote the run or computed something"
            os.makedirs(os.path.join(d, "runs", "e312_p_s0"))                                                # a run folder of another kind (the pair run's model.pt) is not a place for a head either
            open(os.path.join(d, "runs", "e312_p_s0", "model.pt"), "wb").write(b"pair head")
            assert "exists" in (refused(run_pipeline, 0, "e312_p_s0", 1, 1, 3, 30, 1) or "") and open(os.path.join(d, "runs", "e312_p_s0", "model.pt"), "rb").read() == b"pair head"
            for bad in ("../x", "a/b", "", "a b", ".."):                                                       # a name is a folder name, with and without a pair run
                for kw in (dict(), dict(pair_run="e324_s_s0")):
                    assert "run folder name" in (refused(run_pipeline, 0, bad, 1, 1, 3, 30, 1, **kw) or ""), (bad, kw)
            assert not os.path.exists(os.path.join(d, "x")) and not os.path.exists(os.path.join(d, "runs", "x")), "a bad name created a folder"
            # cv with the pair run reads the VAL cache of the pair run (E324 s here, rebuilt with the right meta) and not E325's
            os.remove(cache("val_records_v1_s0_pair_e324_s_s0.pkl"))
            JD.save_records(cache("val_records_v1_s0_pair_e324_s_s0.pkl"), [JD.toy_record(i, i % 22) for i in range(60)], 0, "e324_s_s0")
            os.remove(cache("val_records_v1_s0.pkl"))                                                       # E325's VAL cache is gone: a cv that reads it would fail
            main(["cv", "--seed", "0", "--pair-run", "e324_s_s0", "--folds", "2", "--epochs", "1"])
        finally:
            JD.val_records, JD.live_records = old_stubs
            ROOT = old_root


def _test_ends():
    """E358 plumbing (stubs on the records, a temp ROOT, as `_test_pair_run`): `--ends att` reaches the VAL records, the live pass, the cache names and metas, the checkpoint args and `wire_agreement.json`; att without an attach
    pair run is refused before anything is computed; a cache of the other kind is refused on load; `cv` reads the cache of its kind; the no-jitter head `wn` is trained on the unperturbed records and never replaces a head file"""
    import contextlib
    import io
    import json
    import pickle
    global ROOT
    seen = dict(val=[], live=[])
    val_stub, live_stub = _record_stubs(seen)
    old_root, old_stubs, old_fns = ROOT, (JD.val_records, JD.live_records), (globals()["tensors"], globals()["fit"])
    with JD.tmp_root() as d, contextlib.redirect_stdout(io.StringIO()):
        ROOT = d
        JD.val_records, JD.live_records = val_stub, live_stub
        cache = lambda f: os.path.join(d, "temp", "jwire", f)  # noqa: E731
        att_run = ["run", "--seed", "0", "--pair-run", "e324_s_s0", "--ends", "att", "--name", "e358_w_s0", "--heads", "1", "--epochs", "1", "--max-train", "30"]
        try:
            # att without an attach pair run (no --pair-run, or E312's own) and without its own name: refused before anything is computed or created
            for argv in (["run", "--seed", "0", "--ends", "att", "--name", "e358_w_s0"], ["run", "--seed", "0", "--pair-run", "e312_p_s0", "--ends", "att", "--name", "e358_w_s0"], ["cv", "--seed", "0", "--ends", "att"]):
                assert "attached ends" in (JD.refusal(main, argv) or ""), argv
            assert "mandatory" in (JD.refusal(main, ["run", "--seed", "0", "--pair-run", "e324_s_s0", "--ends", "att"]) or "")
            assert seen == dict(val=[], live=[]) and not os.path.exists(os.path.join(d, "runs")) and not os.path.exists(cache("")), "a refused call computed or created something"
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    main(["run", "--seed", "0", "--pair-run", "e324_s_s0", "--ends", "attached", "--name", "x"])
                    raise AssertionError("a bad end kind must be refused")
                except SystemExit as e:
                    assert e.code == 2
            # the att run: the kind reaches val_records, live_records, both cache names and metas, the checkpoint args, the json and the log; wn is trained on the unperturbed records (the same number of rows as the draws)
            calls = dict(tensors=[], fit=[])
            real_tensors, real_fit = old_fns
            globals()["tensors"] = lambda records, jitter=None: calls["tensors"].append((len(records), jitter is not None)) or real_tensors(records, jitter)
            globals()["fit"] = lambda feat, mask, label, epochs=60, lr=3e-3, bs=512, seed=0, cols=None, wd=1e-4: calls["fit"].append((len(label), seed, cols)) or real_fit(feat, mask, label, epochs, lr, bs, seed, cols, wd)
            main(att_run)
            globals()["tensors"], globals()["fit"] = old_fns
            assert seen == dict(val=[(0, "e324_s_s0", "att")], live=[(0, "train", 30, "e324_s_s0", "att")]), seen
            assert sorted(os.listdir(cache(""))) == ["train_records_v1_s0_max30_pair_e324_s_s0_ends_att.pkl", "val_records_v1_s0_pair_e324_s_s0_ends_att.pkl"], os.listdir(cache(""))
            for f in os.listdir(cache("")):
                assert pickle.load(open(cache(f), "rb"))["meta"] == JD.meta_of(0, "e324_s_s0", "att") and JD.load_records(cache(f), 0, "e324_s_s0", "att")
            rd = os.path.join(d, "runs", "e358_w_s0")
            wa = json.load(open(os.path.join(rd, "wire_agreement.json")))
            ck, wn = (torch.load(os.path.join(rd, f), map_location="cpu") for f in ("model.pt", "model_wn.pt"))
            assert wa["ends"] == ck["args"]["ends"] == wn["args"]["ends"] == "att" and ck["args"]["pair_run"] == wn["args"]["pair_run"] == "e324_s_s0" and "att ends" in open(os.path.join(rd, "log.txt")).read()
            assert ck["args"]["head_seed"] == 0 and wn["args"]["head_seed"] == "wn" and wn["args"]["no_jitter"] and wn["args"]["jitter_scale"] == 0.0 and "no_jitter" not in ck["args"], (ck["args"], wn["args"])
            assert set(wa["no_jitter_head"]) == {"val", "train"} and wa["no_jitter_head"]["val"]["oracle"] == wa["heads"][0]["val"]["oracle"] and {"log.txt", "model.pt", "model_wn.pt", "wire_agreement.json"} <= set(os.listdir(rd)), os.listdir(rd)
            assert not os.path.exists(os.path.join(rd, "model_h1.pt")), "heads = 1: the second one is the ablation and is not saved"
            unjit = real_tensors([JD.toy_record(i, i % 22) for i in range(30)])
            n_unjit = len(unjit[2])
            assert wa["no_jitter_head"]["train"]["oracle"] == int((unjit[3] & (unjit[2] >= 0)).sum()) > 0, "wn is judged on the unperturbed TRAIN ends"
            assert calls["tensors"] == [(30, True)] * 3 + [(873, False), (30, False)], calls["tensors"]            # three jittered draws, VAL, then the unperturbed TRAIN records of wn
            assert [c[1:] for c in calls["fit"]] == [(0, None), (1, DIST_ONLY), (0, None)] and calls["fit"][2][0] == 3 * n_unjit, calls["fit"]      # head 0, the ablation, wn: head seed 0, all columns, the unperturbed rows once per draw
            # a cache of the other kind is refused: a kp cache under the att name, an att cache under the kp name (val and train), also by cv
            for kind, fname in (("train", "train_records_v1_s0_max30_pair_e324_s_s0_ends_att.pkl"), ("val", "val_records_v1_s0_pair_e324_s_s0_ends_att.pkl")):
                os.remove(cache(fname))
                JD.save_records(cache(fname), [JD.toy_record(i, i % 22) for i in range(873 if kind == "val" else 30)], 0, "e324_s_s0")      # the meta of kp
                assert "meta" in (JD.refusal(run_pipeline, 0, "e358_w_s9", 1, 1, 3, 30, 1, pair_run="e324_s_s0", ends="att") or ""), kind
                if kind == "val":
                    assert "meta" in (JD.refusal(main, ["cv", "--seed", "0", "--pair-run", "e324_s_s0", "--ends", "att"]) or "")
            kpv = cache("val_records_v1_s0_pair_e324_s_s0.pkl")
            pickle.dump(dict(meta=JD.meta_of(0, "e324_s_s0", "att"), records=[JD.toy_record(1, 1)]), open(kpv, "wb"))
            assert "meta" in (JD.refusal(main, ["cv", "--seed", "0", "--pair-run", "e324_s_s0"]) or ""), "cv: an att cache under the kp name"
            # cv reads the cache of its kind: the att cache (rebuilt with its meta) and not the kp one
            os.remove(cache("val_records_v1_s0_pair_e324_s_s0_ends_att.pkl"))
            JD.save_records(cache("val_records_v1_s0_pair_e324_s_s0_ends_att.pkl"), [JD.toy_record(i, i % 22) for i in range(60)], 0, "e324_s_s0", "att")
            os.remove(kpv)
            main(["cv", "--seed", "0", "--pair-run", "e324_s_s0", "--ends", "att", "--folds", "2", "--epochs", "1"])
            try:
                main(["cv", "--seed", "0", "--pair-run", "e324_s_s0", "--folds", "2", "--epochs", "1"])
                raise AssertionError("cv without --ends must not read the att cache")
            except FileNotFoundError:
                pass
        finally:
            JD.val_records, JD.live_records = old_stubs
            globals()["tensors"], globals()["fit"] = old_fns
            ROOT = old_root


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cv")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--pair-run", default=None, help="the pair run of the VAL records (default e312_p_s<seed>)")
    s.add_argument("--ends", default="kp", choices=JD.ENDS, help="the pair ends of the records: the keypoints (default) or the ATTACHED ends of the pair run (needs --pair-run of an attach model)")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--epochs", type=int, default=60)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--pair-run", default=None, help="the pair run whose keypoints the records hold (default e312_p_s<seed>, E325); another one needs its own --name")
    r.add_argument("--ends", default="kp", choices=JD.ENDS, help="the pair ends of the records: the keypoints (default) or the ATTACHED ends of the pair run (E358: needs --pair-run of an attach model)")
    r.add_argument("--name", default=None, help="the run folder (default e325_w_s<seed>; mandatory with --pair-run)")
    r.add_argument("--heads", type=int, default=3)
    r.add_argument("--epochs", type=int, default=60)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--max-train", type=int, default=0)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run_pipeline(a.seed, a.name, a.heads, a.epochs, a.workers, a.max_train, pair_run=a.pair_run, ends=a.ends)
        return
    if a.cmd == "selftest":
        _selftest()
        return
    recs = JD.load_records(JD.cache_path("val", a.seed, pair_run=a.pair_run, ends=a.ends), a.seed, a.pair_run, a.ends)
    res = cv(recs, a.folds, a.epochs, seed=0)
    print(f"seed {a.seed}, pair run {a.pair_run or JD.default_pair_run(a.seed)}, {a.ends} ends: scene-wise {a.folds}-fold cross-validation on VAL (IN-SAMPLE, diagnostic)")
    for name, r in res.items():
        print(f"  {name:9s} ends with an oracle label {r['oracle']}: head {r['head_same'] / r['oracle']:.3f}, rule {r['rule_same'] / r['oracle']:.3f} | real choices {r['multi']}: head {r['head_multi_same'] / max(r['multi'], 1):.3f}, rule {r['rule_multi_same'] / max(r['multi'], 1):.3f}")


if __name__ == "__main__":
    main()
