def rd(p):
    return open(p, encoding="utf-8").read()


def wr(p, s):
    open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)


def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)


t = rd("training/train/jwire_train.py")
# finite loss
t = sub(t, "            opt.zero_grad()\n            JH.wire_loss(w(feat[b], mask[b]), label[b]).backward()\n            opt.step()\n",
        "            opt.zero_grad()\n            loss = JH.wire_loss(w(feat[b], mask[b]), label[b])\n            if not bool(torch.isfinite(loss)):\n                raise FloatingPointError(\"E325 kill: the loss is not finite\")                              # the run fails (exit != 0): the kill criterion of the card\n            loss.backward()\n            opt.step()\n")
# kill function + calibration bins
t = sub(t, "def tensors(records, jitter=None):", '''def kill_fired(val, train_jitter):
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


def tensors(records, jitter=None):''')
# pipeline: caches, pilot extras, kill, calibration
t = sub(t, '''    vpath = os.path.join(ROOT, "temp", "jwire", f"val_records_s{seed}.pkl")
    if not os.path.exists(vpath):
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
        pickle.dump(JD.val_records(seed, f"e312_p_s{seed}"), open(vpath, "wb"))
    val = pickle.load(open(vpath, "rb"))
    tpath = os.path.join(ROOT, "temp", "jwire", f"train_records_s{seed}{'_max' + str(max_train) if max_train else ''}.pkl")
    if os.path.exists(tpath):
        train = pickle.load(open(tpath, "rb"))
    else:
        train = JD.live_records(seed, "train", 0.1, max_train, workers)
        pickle.dump(train, open(tpath, "wb"))
''', '''    vpath = JD.cache_path("val", seed)
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
''')
t = sub(t, '''    run.log("pilot TRAIN", json.dumps(info["pilot_train"]), "VAL", json.dumps(info["pilot_val"]), f"-> jitter scale {scale:.3f} m (median error {med:.3f} m, VAL {pv['err_median']:.3f} m)")''',
        '''    run.log("pilot TRAIN", json.dumps(info["pilot_train"]), "VAL", json.dumps(info["pilot_val"]), f"-> jitter scale {scale:.3f} m (median error {med:.3f} m, VAL {pv['err_median']:.3f} m); the keypoint probabilities are blind columns of the head (jwire_head.BLIND)")''')
t = sub(t, '''        if cols is None:
            torch.save(''', '''        if h == 0:
            out["calibration_val_labelled_ends"] = calibration(w, vf, vm, vl, vo)
        if cols is None:
            torch.save(''')
t = sub(t, '''    h0 = out["heads"][0]
    kill = (h0["val"]["head_multi_same"] / max(h0["val"]["multi"], 1) < h0["val"]["rule_multi_same"] / max(h0["val"]["multi"], 1) - 0.02
            or h0["train_jitter"]["head_multi_same"] / max(h0["train_jitter"]["multi"], 1) < h0["train_jitter"]["rule_multi_same"] / max(h0["train_jitter"]["multi"], 1) - 0.02)
    out["kill_fired"] = bool(kill)
    json.dump(out, open(run.path("wire_agreement.json"), "w"), indent=1, default=float)
    run.log("KILL (the head does not learn: agreement below the rule's by more than 0.02 on the VAL real choices or on the TRAIN ends):", kill)
''', '''    h0 = out["heads"][0]
    kill, k_val, k_train = kill_fired(h0["val"], h0["train_jitter"])
    out["kill_fired"], out["kill_on_val"], out["kill_on_train"] = kill, k_val, k_train
    json.dump(out, open(run.path("wire_agreement.json"), "w"), indent=1, default=float)
    run.log(f"KILL (agreement below the rule's by more than 0.02 on the VAL real choices or on the TRAIN ends): {kill} (VAL {k_val}, TRAIN {k_train}); parameters 545")
''')
t = sub(t, '''    recs = pickle.load(open(os.path.join(ROOT, "temp", "jwire", f"val_records_s{a.seed}.pkl"), "rb"))''', '''    recs = pickle.load(open(JD.cache_path("val", a.seed), "rb"))''')
# selftest additions
t = sub(t, '''    print(f"jwire_train selftest ok (a head sees feature 5''', '''    # the kill: fires when the VAL real-choice agreement is more than 0.02 below the rule's, or the TRAIN one; not at a tie
    tie = dict(multi=100, head_multi_same=70, rule_multi_same=71)
    low = dict(multi=100, head_multi_same=60, rule_multi_same=71)
    assert kill_fired(tie, tie)[0] is False and kill_fired(low, tie) == (True, True, False) and kill_fired(tie, low) == (True, False, True)
    # a non-finite loss stops the run (the kill 'the loss is not finite')
    bad = feat.clone()
    bad[0, 0, 0] = float("nan")
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
    print(f"jwire_train selftest ok (a head sees feature 5''')
wr("training/train/jwire_train.py", t)
print("ok")
