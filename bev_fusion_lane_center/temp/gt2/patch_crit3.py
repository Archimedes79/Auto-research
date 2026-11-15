def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


# ---- jrule_control: strict one-to-one matching, the same keypoint threshold for the rule and the head, the ceiling
p = "evaluation/jrule_control.py"
s = open(p, encoding="utf-8").read()
a = s.index("def _match(")
b = s.index("def selftest():")
new = '''def _match(kp, kd, valid, G, pos_tol=4.0, ang=30.0):
    """STRICT one-to-one matching (critic 2026-11-11): greedy by distance over all (keypoint, GT end) pairs within pos_tol and ang degrees, every keypoint and every GT end used at most once -> (n_GT,) index of the keypoint, -1 none"""
    out = -np.ones(len(G), int)
    if not len(G) or not valid.any():
        return out
    ki = np.where(valid)[0]
    d = np.linalg.norm(kp[ki][:, None, :2] - np.asarray(G)[None, :, :2], axis=-1)
    a = np.degrees(np.arccos(np.clip(kd[ki][:, None, 0] * np.asarray(G)[None, :, 2] + kd[ki][:, None, 1] * np.asarray(G)[None, :, 3], -1, 1)))
    ok = (d <= pos_tol) & (a <= ang)
    used = set()
    for dist, i, j in sorted((d[i, j], i, j) for i, j in zip(*np.where(ok))):
        if i in used or out[j] >= 0:
            continue
        used.add(i)
        out[j] = int(ki[i])
    return out


def score(a):
    import qgraph_view as QV
    res = {}
    for run in a.run:
        frames = {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
        tot = dict(gt=0, ceiling=0, rule_hit=0, rule_n=0, rule_ok=0, head_hit=0, head_n=0, head_ok=0)
        for sid, (name, toks) in QV.val_scenes().items():
            for t in toks:
                if t not in frames:
                    continue
                jg = J.junction_gt(QV.load_gt(t))
                f = frames[t]
                K = f["logit"].shape[0]
                gt_pairs = set() if jg is None else {(int(p[0]), int(p[1])) for p in jg["pairs"]}
                tot["gt"] += len(gt_pairs)
                kp, kd = f["kp_pos"], f["kp_dir"]
                ve = f["valid"][0] & (f["kp_p"][:K] >= a.kp_thr)                                      # the same keypoint threshold for the rule and for the head
                vx = f["valid"][1] & (f["kp_p"][K:] >= a.kp_thr)
                ie, ix = np.where(ve)[0], np.where(vx)[0]
                R = rule_pairs(np.concatenate([kp[:K][ie], kd[:K][ie]], 1), np.concatenate([kp[K:][ix], kd[K:][ix]], 1))
                Hd = ((1 / (1 + np.exp(-f["logit"]))) * f["allowed"] >= 0.5) & ve[:, None] & vx[None, :]
                if jg is not None:
                    ma = _match(kp[:K], kd[:K], ve, jg["ent"])
                    mb = _match(kp[K:], kd[K:], vx, jg["ext"])
                    inv_a = {int(k): g_ for g_, k in enumerate(ma) if k >= 0}
                    inv_b = {int(k): g_ for g_, k in enumerate(mb) if k >= 0}
                    tot["ceiling"] += sum(1 for (x, y) in gt_pairs if ma[x] >= 0 and mb[y] >= 0)
                else:
                    inv_a = inv_b = {}
                for name_, M, rows, cols in (("rule", R, ie, ix), ("head", Hd, np.arange(K), np.arange(K))):
                    got = set()
                    for ii, jj in zip(*np.where(M)):
                        i, j = int(rows[ii]), int(cols[jj])
                        tot[f"{name_}_n"] += 1
                        key = (inv_a.get(i, -1), inv_b.get(j, -1))
                        if key in gt_pairs:
                            tot[f"{name_}_ok"] += 1
                            got.add(key)
                    tot[f"{name_}_hit"] += len(got)
        r = lambda x, y: x / max(y, 1)  # noqa: E731
        res[run] = dict(keypoint_thr=a.kp_thr, matching="strict one-to-one, 4 m, 30 degrees", gt_pairs=tot["gt"], ceiling_both_ends_matched=r(tot["ceiling"], tot["gt"]),
                        rule=dict(recall=r(tot["rule_hit"], tot["gt"]), precision=r(tot["rule_ok"], tot["rule_n"]), delivered=tot["rule_n"]),
                        head=dict(recall=r(tot["head_hit"], tot["gt"]), precision=r(tot["head_ok"], tot["head_n"]), delivered=tot["head_n"]))
        print(run, json.dumps(res[run]))
    os.makedirs(os.path.join(ROOT, "runs", "e287"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "runs", "e287", f"rule_control_thr{a.kp_thr:g}_" + "_".join(a.run) + ".json"), "w"), indent=1)


'''
s = s[:a] + new + s[b:]
s = s.replace('    s.add_argument("--run", nargs="+", required=True)\n    sub.add_parser("selftest")', '    s.add_argument("--run", nargs="+", required=True)\n    s.add_argument("--kp-thr", type=float, default=0.5, help="keypoints below this probability are dropped for the rule AND the head")\n    sub.add_parser("selftest")')
s = s.replace("    assert not rule_pairs(ent[:0], ext).any() and not rule_pairs(ent, ext[:0]).any()\n", "    assert not rule_pairs(ent[:0], ext).any() and not rule_pairs(ent, ext[:0]).any()\n    # strict matching: one keypoint between two GT ends 3.5 m apart serves only one of them\n    kpp = np.array([[0.0, 0.0]]); kdd = np.array([[0.0, 1.0]]); G = np.array([[-1.75, 0.0, 0.0, 1.0], [1.75, 0.0, 0.0, 1.0]])\n    mm = _match(kpp, kdd, np.array([True]), G)\n    assert (mm >= 0).sum() == 1, mm\n")
open(p, "w", encoding="utf-8").write(s)

# ---- jsplits: strict matching with a keypoint threshold, a clearer label, driven x range
p = "evaluation/jsplits.py"
s = open(p, encoding="utf-8").read()
s = s.replace('                ia = JR._match(kp[:K], kd[:K], f["valid"][0], jg["ent"])\n                ib = JR._match(kp[K:], kd[K:], f["valid"][1], jg["ext"])\n', '                ia = JR._match(kp[:K], kd[:K], f["valid"][0] & (f["kp_p"][:K] >= a.kp_thr), jg["ent"])\n                ib = JR._match(kp[K:], kd[K:], f["valid"][1] & (f["kp_p"][K:] >= a.kp_thr), jg["ext"])\n')
s = s.replace('                    hit = ia[i] >= 0 and ib[j] >= 0 and bool(f["allowed"][ia[i], ib[j]]) and f["logit"][ia[i], ib[j]] >= 0\n', '                    hit = ia[i] >= 0 and ib[j] >= 0 and bool(f["allowed"][ia[i], ib[j]]) and f["logit"][ia[i], ib[j]] >= 0\n                    rng = rng_class(float(np.linalg.norm(jg["ent"][i, :2])))\n')
s = s.replace('                    add(P, ("lanes of the entry arm", str(min(int(r[6]), 3)) if r[6] < 3 else "3+"), hit)\n', '                    add(P, ("entries in use in the maneuver", str(int(r[6])) if r[6] < 3 else "3+"), hit)\n                    add(P, ("driven x range", ("driven " if r[9] else "other ") + rng), hit)\n                    add(P, ("range of the entry", rng), hit)\n')
s = s.replace("visible (camera sees it, no box hides it: the l4 label `vis`) or not", "visible (the l4 label `vis`: a camera sees it, no box hides it; the near blind zone of about 5 m counts as hidden) or not")
s = s.replace('    s.add_argument("--run", nargs="+", required=True)\n    sub.add_parser("selftest")', '    s.add_argument("--run", nargs="+", required=True)\n    s.add_argument("--kp-thr", type=float, default=0.5, help="keypoints below this probability do not count (strict one-to-one matching, as the head\'s own counter)")\n    sub.add_parser("selftest")')
s = s.replace('json.dump(out, open(os.path.join(ROOT, "runs", "e300", "splits_" + "_".join(a.run) + ".json"), "w"), indent=1)', 'json.dump(out, open(os.path.join(ROOT, "runs", "e300", f"splits_thr{a.kp_thr:g}_" + "_".join(a.run) + ".json"), "w"), indent=1)')
open(p, "w", encoding="utf-8").write(s)
print("ok")
