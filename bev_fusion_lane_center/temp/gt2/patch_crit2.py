def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


# ---- D: the group / rank features come from the REAL peaks only (critic finding 4: teacher forcing completed the groups in training, at evaluation they are partial)
sub("training/grid/jends.py", "        idxs, valids = [], []\n        for kk in (0, 1):\n            p = heat[:, kk].detach().sigmoid()\n            mx = F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]\n            score = torch.where((p >= mx) & (p > PEAK_MIN), p, torch.zeros_like(p)).flatten(1)\n",
    "        idxs, valids, reals = [], [], []\n        for kk in (0, 1):\n            p = heat[:, kk].detach().sigmoid()\n            mx = F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]\n            score = torch.where((p >= mx) & (p > PEAK_MIN), p, torch.zeros_like(p)).flatten(1)\n            peak_score = score\n")
sub("training/grid/jends.py", "            v, ix = score.topk(self.k, dim=1)\n            idxs.append(ix)\n            valids.append(v > 0)\n        idx, valid = torch.stack(idxs, 1), torch.stack(valids, 1)                # (B, 2, K)\n",
    "            v, ix = score.topk(self.k, dim=1)\n            idxs.append(ix)\n            valids.append(v > 0)\n            reals.append(peak_score.gather(1, ix) > 0)                                    # a keypoint that is a real peak of the heat (not only an injected GT cell)\n        idx, valid, real = torch.stack(idxs, 1), torch.stack(valids, 1), torch.stack(reals, 1)                # (B, 2, K)\n")
sub("training/grid/jends.py", "            rk, ga, ra, gb, rb = self.rank_matrix(pos, D, valid)\n", "            rk, ga, ra, gb, rb = self.rank_matrix(pos, D, valid & real if gt is not None else valid)         # groups and ranks from the real peaks only: the same distribution in training and at evaluation\n")
sub("training/grid/jends.py", "    # E303: the scale head (two handle scales per pair) trains against the fitted scales of the true pairs\n", '''    # critic 2026-11-11: with no real peak at all (heat far below the peak floor) every injected GT keypoint is its own group: the rank features cannot see the injected ends
    jq = dict(jout, j1=jout["j1"].clone())
    jq["j1"][:, 2:4] = -9.0
    mq = JEnds(d=d, order=True, evidence=False)
    oq = mq(jq, P, gt=gt)
    ga = oq["ga"]
    assert all(len(set(ga[b].tolist())) == K for b in range(B)), "injected keypoints must not form groups"
    # E303: the scale head (two handle scales per pair) trains against the fitted scales of the true pairs
''')

# ---- E: jends_eval.score: the seed must match the maps run, the pyramid must be B's (non-joint) or the result is labelled maps-only
sub("evaluation/jends_eval.py", "def score(a):\n    import qends_eval as QE\n", '''def score(a):
    import qends_eval as QE
    ck = torch.load(os.path.join(ROOT, "runs", a.run, "model.pt"), map_location="cpu")
    args = ck["args"]
    mp = str(args.get("maps", ""))
    import re
    m = re.search(r"_s(\\d+)[/\\\\]", mp.replace("\\\\", "/") + "/")
    assert args.get("seed", a.seed) == a.seed and (m is None or int(m.group(1)) == a.seed), f"--seed {a.seed} does not match the run (seed {args.get('seed')}, maps {mp}): its delivered graph would be another seed's"
    base = torch.load(os.path.join(ROOT, "runs", f"e273_p_s{a.seed}", "model.pt"), map_location="cpu")["state"]
    pyr_same = all(torch.equal(ck["state"]["jmodel." + k] if "jmodel." + k in ck["state"] else ck["state"][k], v) for k, v in base.items() if k.startswith("pyr.") and (("jmodel." + k) in ck["state"] or k in ck["state"]))
    print("pyramid of the maps run == the base run's:", pyr_same, "(False = a joint pyramid: the delivered graph is B's, the connectors come from other pixel features: label the result 'maps-only')")
''')
print("ok")
