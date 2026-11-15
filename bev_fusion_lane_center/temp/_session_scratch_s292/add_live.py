p = "evaluation/jwire_data.py"
s = open(p, encoding="utf-8").read()
a = "def _selftest():\n    import jwire_head as H\n"
assert s.count(a) == 1
new = '''def live_records(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None):
    """the records of the frames of a split from a LIVE pass of B (`runs/e273_p_s<seed>`, exact-map pass, the graph read out with close:1.5@0.5 as the card of record does) and the pair run (`runs/<pair_run>`, default
    e312_p_s<seed>) on the loader of `split` ('train' for the training of the head, 'val_clean' to check that this pass reproduces the records of the stored dumps): GPU, about 10 minutes for the 8414 TRAIN frames"""
    import torch
    import jends_eval as JE
    import lane_gt_data as LD
    import qgraph_eval as QE
    import qgraph_readout as QR
    import qpredict as QP
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mb = QP.load_model(os.path.join(ROOT, "runs", f"e273_p_s{seed}"), dev)
    mp, _ = JE.load_model(pair_run or f"e312_p_s{seed}", dev)
    LD.QData.DEFAULT_ENDS = True
    ds, dl = LD.loader(split, bs=bs, workers=workers, shuffle=False, map_mode="exact", pixels=False, strict=True, ends=True)
    assert ds.missing == 0, f"{ds.missing} frames of {split} have no stored lane GT"
    spec = QR.parse("close:1.5@0.5")
    out, t0 = [], time.time()
    for b in dl:
        preds = QP.predict(mb, b)
        with torch.no_grad():
            _, _, o = mp(b["x"].to(dev).float(), gt=None)
        for i, pr in enumerate(preds):
            e = dict(kp_pos=o["kp_pos"][i].cpu().numpy(), kp_dir=o["kp_dir"][i].cpu().numpy(), kp_p=o["kp_p"][i].cpu().numpy(), valid=o["valid"][i].cpu().numpy(), logit=o["logit"][i].float().cpu().numpy(),
                     allowed=o["allowed"][i].cpu().numpy(), scale=o["scale"][i].float().cpu().numpy() if "scale" in o else None)
            g = dict(ent=b["ent"][i].numpy(), ent_m=b["ent_m"][i].numpy(), ext=b["ext"][i].numpy(), ext_m=b["ext_m"][i].numpy())
            pred = QR.apply(dict(nodes=dict(pos=pr["nodes"]["pos"], dirv=pr["nodes"]["dirv"], e=pr["nodes"]["e"], flags=pr["nodes"]["flags"], section=pr["nodes"]["section"], s=pr["nodes"].get("s")),
                                 links=dict(src=pr["links"]["src"], dst=pr["links"]["dst"], p=pr["links"]["p"], long=pr["links"]["long"], a=pr["links"]["a"], b=pr["links"]["b"])), spec)
            graph, bk = QE.readout(pred, 0.5, 0.5)
            out.append(record_of(pred, graph, bk, e, g, thr, int(b["row"][i]), -1))
        if nmax and len(out) >= nmax:
            break
        if len(out) % 400 < bs:
            print(f"  {len(out)}/{len(ds)}  {time.time() - t0:.0f} s", flush=True)
    return out[: nmax or None]


def pilot_stats(records):
    """the numbers of the pilot (card section 1, in-sample guard): the keypoint error of the matched ends (distance of the pair end to its GT end: mean / median / 90 %) and the rule's agreement with the oracle label"""
    err = []
    for r in records:
        for n in range(len(r["pairs"])):
            for side, sl in ((0, slice(0, 2)), (1, slice(4, 6))):
                g = r["gts"][n, side]
                if not np.isnan(g).any():
                    err.append(float(np.linalg.norm(r["pairs"][n, sl] - g[:2])))
    err = np.array(err)
    tot = {}
    for r in records:
        for k, v in agreement(instances_of(r)).items():
            tot[k] = tot.get(k, 0) + v
    return dict(matched_ends=len(err), err_mean=float(err.mean()), err_median=float(np.median(err)), err_p90=float(np.percentile(err, 90)), overall=tot["same"] / max(tot["oracle"], 1), multi=tot["multi_same"] / max(tot["multi"], 1), counts=tot)


def make_jitter(scale, seed=0, heading_deg=4.0):
    """a heavy-tailed keypoint jitter (Student t with 3 degrees of freedom per coordinate, times `scale` m, clipped at 8 m; a Gaussian heading rotation of `heading_deg` degrees): emulates the keypoint error of frames the networks have not been trained on"""
    rng = np.random.default_rng(seed)

    def jit(pos, tan):
        d = np.clip(scale * rng.standard_t(3, 2), -8.0, 8.0)
        a = np.radians(rng.normal(0.0, heading_deg))
        c, s = np.cos(a), np.sin(a)
        return pos + d, np.array([c * tan[0] - s * tan[1], s * tan[0] + c * tan[1]])
    return jit


def calibrate_jitter(records, target, seed=0, iters=10):
    """the jitter scale (m) at which the rule's agreement with the oracle label on `records` + jitter reproduces `target` = (overall, real choices) of the VAL records (the in-sample guard: TRAIN keypoints are
    better than VAL ones); bisection on the scale (the agreement falls with it), the mean of the two agreements is matched -> (scale, agreement at that scale)"""
    def ag(scale):
        jit = make_jitter(scale, seed)
        tot = {}
        for r in records:
            for k, v in agreement(instances_of(r, jit)).items():
                tot[k] = tot.get(k, 0) + v
        return tot["same"] / max(tot["oracle"], 1), tot["multi_same"] / max(tot["multi"], 1)
    lo, hi = 0.0, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        o, m = ag(mid)
        if 0.5 * (o + m) > 0.5 * (target[0] + target[1]):
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return s, ag(s)


'''
s = s.replace(a, new + a)
# CLI: live + pilot
b = '    sub.add_parser("selftest")\n    a = ap.parse_args()\n'
assert s.count(b) == 1
s = s.replace(b, '''    t = sub.add_parser("live")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--split", default="train")
    t.add_argument("--max", type=int, default=0)
    t.add_argument("--workers", type=int, default=3)
    t.add_argument("--device", default=None)
    t.add_argument("--out", default=None)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "live":
        recs = live_records(a.seed, a.split, 0.1, a.max, a.workers, device=a.device)
        path = a.out or os.path.join(ROOT, "temp", "jwire", f"{a.split}_records_s{a.seed}{'_max' + str(a.max) if a.max else ''}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickle.dump(recs, open(path, "wb"))
        print(f"{len(recs)} records -> {os.path.relpath(path, ROOT)}")
        print(pilot_stats(recs))
        return
''')
s = s.replace('    wsl bash tools/env/_py14.sh jwire_data.py selftest\n', '    wsl bash tools/env/_py14.sh jwire_data.py live --seed 0 [--split train|val_clean] [--max N]               # the same records from a LIVE pass of B and the pair run (GPU) + the pilot numbers\n    wsl bash tools/env/_py14.sh jwire_data.py selftest\n', 1)
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)
print("ok")
