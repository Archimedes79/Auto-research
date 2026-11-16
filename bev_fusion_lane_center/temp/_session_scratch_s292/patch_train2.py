def rd(p):
    return open(p, encoding="utf-8").read()
def wr(p, s):
    open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)
def sub(s, a, b):
    assert s.count(a) == 1, a[:70]
    return s.replace(a, b)

# ---- jwire_head: load_wire
h = rd("training/grid/jwire_head.py")
h = sub(h, "@torch.no_grad()\ndef wire_choice_head(", '''def load_wire(run, name="model.pt"):
    """the head of a run (`runs/<run>/<name>`, written by `jwire_train.py run`) -> `Wire` in eval mode"""
    import os
    root = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ck = torch.load(os.path.join(root, "runs", run, name), map_location="cpu")
    w = Wire()
    w.load_state_dict(ck["state"])
    return w.eval()


@torch.no_grad()
def wire_choice_head(''')
wr("training/grid/jwire_head.py", h)

# ---- jwire_train: the run pipeline
t = rd("training/train/jwire_train.py")
t = sub(t, '''    wsl bash tools/env/_py14.sh jwire_train.py selftest
''', '''    wsl bash tools/env/_py14.sh jwire_train.py run --seed 0 [--name e325_w_s0] [--heads 3] [--epochs 60] [--max-train N]   # the head of the card: through run.py (E325.w_s<S>): TRAIN records (a live pass, GPU) -> the pilot, the jitter ->
                                                                                                 # the heads (3 head seeds + the distance-only ablation) -> `runs/<name>/model.pt` (head seed 0 = the arm), `model_h<k>.pt`, `wire_agreement.json`
    wsl bash tools/env/_py14.sh jwire_train.py selftest
''')
t = sub(t, '''def _selftest():''', '''def run_pipeline(seed, name, heads=3, epochs=60, workers=3, max_train=0, draws=3):
    """the training run of the card (E325.w_s<seed>): see the module docstring; the kill numbers of the card (agreement against the rule on the VAL real choices and on the TRAIN ends with the jitter) are written
    to `wire_agreement.json` and logged"""
    import json
    import trainer as T
    run = T.Run(name)
    run.log(f"E325 head, seed {seed}: records of the TRAIN frames (live pass of B and the pair run), jitter calibrated on the keypoint error, {heads} head seeds + the distance-only ablation")
    vpath = os.path.join(ROOT, "temp", "jwire", f"val_records_s{seed}.pkl")
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
    pt, pv = JD.pilot_stats(train), JD.pilot_stats(val)
    scale, med = JD.calibrate_jitter(train, pv["err_median"])
    info = dict(seed=seed, train_frames=len(train), pilot_train={k: v for k, v in pt.items() if k != "counts"}, pilot_val={k: v for k, v in pv.items() if k != "counts"}, jitter_scale=scale, jitter_median_error=med)
    run.log("pilot TRAIN", json.dumps(info["pilot_train"]), "VAL", json.dumps(info["pilot_val"]), f"-> jitter scale {scale:.3f} m (median error {med:.3f} m, VAL {pv['err_median']:.3f} m)")
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
        if cols is None:
            torch.save(dict(state=w.state_dict(), args=dict(seed=seed, head_seed=h, jitter_scale=scale, epochs=epochs)), run.path("model.pt" if h == 0 else f"model_h{h}.pt"))
    h0 = out["heads"][0]
    kill = (h0["val"]["head_multi_same"] / max(h0["val"]["multi"], 1) < h0["val"]["rule_multi_same"] / max(h0["val"]["multi"], 1) - 0.02
            or h0["train_jitter"]["head_multi_same"] / max(h0["train_jitter"]["multi"], 1) < h0["train_jitter"]["rule_multi_same"] / max(h0["train_jitter"]["multi"], 1) - 0.02)
    out["kill_fired"] = bool(kill)
    json.dump(out, open(run.path("wire_agreement.json"), "w"), indent=1, default=float)
    run.log("KILL (the head does not learn: agreement below the rule's by more than 0.02 on the VAL real choices or on the TRAIN ends):", kill)
    run.mark_done()
    return out


def _selftest():''')
t = sub(t, '''    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":''', '''    r = sub.add_parser("run")
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
    if a.cmd == "selftest":''')
wr("training/train/jwire_train.py", t)
print("ok")
