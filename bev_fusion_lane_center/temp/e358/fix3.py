import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/training/train/jwire_train.py", [
("""def run_pipeline(seed, name, heads=3, epochs=60, workers=3, max_train=0, draws=3, pair_run=None):
    \"\"\"the training run of the card (E325.w_s<seed>, E350.w_s<seed> with another pair run): see the module docstring; the kill numbers of the card (agreement against the rule on the VAL real choices and on the TRAIN
    ends with the jitter) are written to `wire_agreement.json` and logged\"\"\"
    import json
    import trainer as T
    name, pair_run = run_target(seed, name, pair_run)                                                    # before anything is created or computed
    run = T.Run(name, root=os.path.join(ROOT, "runs"))
    run.log(f"E325 head, seed {seed}, pair run {pair_run}: records of the TRAIN frames (live pass of B and the pair run), jitter calibrated on the keypoint error, {heads} head seeds + the distance-only ablation")
    vpath = JD.cache_path("val", seed, pair_run=pair_run)
    val = JD.cached_records(vpath, seed, pair_run, lambda: JD.val_records(seed, pair_run))
    assert len(val) == 873 and len({r["row"] for r in val}) == 873, f"{vpath}: {len(val)} VAL records (873 needed, unique rows)"
    tpath = JD.cache_path("train", seed, max_train, pair_run=pair_run)
    train = JD.cached_records(tpath, seed, pair_run, lambda: JD.live_records(seed, "train", 0.1, max_train, workers, pair_run=pair_run))
    assert len({r["row"] for r in train}) == len(train) and (max_train or len(train) >= 8000), f"{tpath}: {len(train)} TRAIN records"
    pt, pv = JD.pilot_stats(train), JD.pilot_stats(val)
    scale, med = JD.calibrate_jitter(train, pv["err_median"])
    info = dict(seed=seed, pair_run=pair_run, train_frames=len(train), pilot_train={k: v for k, v in pt.items() if k != "counts"}, pilot_val={k: v for k, v in pv.items() if k != "counts"}, jitter_scale=scale, jitter_median_error=med)
    run.log("pilot TRAIN", json.dumps(info["pilot_train"]), "VAL", json.dumps(info["pilot_val"]), f"-> jitter scale {scale:.3f} m (median error {med:.3f} m, VAL {pv['err_median']:.3f} m); the keypoint probabilities are blind columns of the head (jwire_head.BLIND)")""",
"""def run_pipeline(seed, name, heads=3, epochs=60, workers=3, max_train=0, draws=3, pair_run=None, ends="kp"):
    \"\"\"the training run of the card (E325.w_s<seed>, E350.w_s<seed> with another pair run, E358.w_s<seed> with the attached ends of it): see the module docstring; the kill numbers of the card (agreement against the rule on the
    VAL real choices and on the TRAIN ends with the jitter) are written to `wire_agreement.json` and logged\"\"\"
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
    run.log("pilot TRAIN", json.dumps(info["pilot_train"]), "VAL", json.dumps(info["pilot_val"]), f"-> jitter scale {scale:.3f} m (median error {med:.3f} m, VAL {pv['err_median']:.3f} m); the keypoint probabilities are blind columns of the head (jwire_head.BLIND)")"""),
("""            torch.save(dict(state=w.state_dict(), args=dict(seed=seed, pair_run=pair_run, head_seed=h, jitter_scale=scale, epochs=epochs)), run.path("model.pt" if h == 0 else f"model_h{h}.pt"))
    h0 = out["heads"][0]""", """            torch.save(dict(state=w.state_dict(), args=dict(seed=seed, pair_run=pair_run, ends=ends, head_seed=h, jitter_scale=scale, epochs=epochs)), run.path("model.pt" if h == 0 else f"model_h{h}.pt"))
    nf, nm, nl, no, nn, _ = tensors(train)                                                               # the no-jitter ablation (reported only): head seed 0 on the unperturbed records, as many rows as the jittered draws
    w = fit(nf.repeat(draws, 1, 1), nm.repeat(draws, 1), nl.repeat(draws), epochs, seed=0)
    av, at = agree(predict(w, vf, vm), vl, vo, vn), agree(predict(w, nf, nm), nl, no, nn)
    out["no_jitter_head"] = dict(val=av, train=at)
    torch.save(dict(state=w.state_dict(), args=dict(seed=seed, pair_run=pair_run, ends=ends, head_seed="wn", jitter_scale=0.0, epochs=epochs, no_jitter=True)), run.path("model_wn.pt"))
    run.log(f"no-jitter head (wn): VAL overall {av['head_same'] / av['oracle']:.3f} (rule {av['rule_same'] / av['oracle']:.3f}) real choices {av['head_multi_same'] / max(av['multi'], 1):.3f} (rule {av['rule_multi_same'] / max(av['multi'], 1):.3f}); TRAIN real choices {at['head_multi_same'] / max(at['multi'], 1):.3f} (rule {at['rule_multi_same'] / max(at['multi'], 1):.3f})")
    h0 = out["heads"][0]"""),
])
print("ok")
