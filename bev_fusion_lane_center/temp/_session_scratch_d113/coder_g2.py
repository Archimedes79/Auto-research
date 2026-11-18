p = "training/grid/jpast.py"
s = open(p, encoding="utf-8").read()
i0 = s.index("    # u, the kill measure: 0 without a past")
i1 = s.index("    # a padding-only batch row and K = 512")
new = '''    # u, the kill measures (module docstring): u_none (the none key only) is 0 without a past and at zero init; u (the past of ANOTHER SAMPLE) needs B >= 2 or a donor
    ysub = lambda blk, xx, pp, pa: blk(xx, pp[0], pp[1], pp[2], pa) - xx  # noqa: E731
    xs = (x, pos, dirv, valid)
    u0 = b0.use(x, pos, dirv, valid, ysub(b0, x, xs[1:], past), past)
    assert u0["d2_none"] == 0.0 and u0["d2_other"] == 0.0 and u0["n_other"] == int(valid.sum()) and u0["n"] == int(valid.sum()), "at zero init the block changes nothing"
    nopast = dict(past, valid=torch.zeros_like(past["valid"]))
    assert b.use(x, pos, dirv, valid, ysub(b, x, xs[1:], None), None)["d2_none"] < 1e-9 and b.use(x, pos, dirv, valid, ysub(b, x, xs[1:], nopast), nopast)["d2_none"] < 1e-9 and b.use(x, pos, dirv, valid, ysub(b, x, xs[1:], None), None)["n_other"] == 0
    y = ysub(b, x, xs[1:], past)
    su = b.use(x, pos, dirv, valid, y, past)
    y0, y1 = ysub(b, x, xs[1:], None), ysub(b, x, xs[1:], roll_past(past))                              # the references by hand: none key only, the other sample's past
    mm = valid[..., None].double()
    d2n, d2o = (float((((y - r_).double() * mm) ** 2).sum()) for r_ in (y0, y1))
    assert abs(su["d2_none"] - d2n) < 1e-6 * max(d2n, 1) and abs(su["d2_other"] - d2o) < 1e-6 * max(d2o, 1) and abs(su["x2"] - float(((x.double() * mm) ** 2).sum())) < 1e-6 * su["x2"] and su["x2_other"] == su["x2"] and su["n"] == su["n_other"] == int(valid.sum()) and d2n > 0 and d2o > 0
    rp = roll_past(past)                                                                                    # the other sample's past: every tensor rolled by one along the batch axis, e_cur stays, a missing w stays
    assert all(torch.equal(rp[k][0], past[k][1]) and torch.equal(rp[k][1], past[k][0]) for k in ("x", "pos", "dirv", "valid", "e", "w")) and torch.equal(rp["e_cur"], past["e_cur"]) and roll_past(dict(past, w=None))["w"] is None and not torch.equal(rp["x"], past["x"])
    one = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in past.items()}                                 # a batch of ONE sample: no other sample, n_other 0; with a donor (sample 1's past) = what the batch gave sample 0
    y_one = b(x[:1], pos[:1], dirv[:1], valid[:1], one) - x[:1]
    s1 = b.use(x[:1], pos[:1], dirv[:1], valid[:1], y_one, one)
    assert s1["n_other"] == 0 and s1["d2_other"] == 0.0 and s1["x2_other"] == 0.0 and s1["d2_none"] > 0 and s1["n"] == int(valid[0].sum())
    donor = dict({k: (v[1:2] if torch.is_tensor(v) else v) for k, v in past.items()}, e_cur=one["e_cur"])
    s2 = b.use(x[:1], pos[:1], dirv[:1], valid[:1], y_one, one, donor)
    assert s2["n_other"] == int(valid[0].sum()) and abs(s2["d2_other"] - float((((y - y1)[:1].double() * mm[:1]) ** 2).sum())) < 1e-6 * max(d2o, 1), "a donor is the other sample"
    # the critic's case by hand: a block that ignores the CONTENT of the past. q = 0, geometry bias 0, key weights 1, v = the constant 1 for every key, v_none = 0, o = identity: with n valid past keys every valid query token gets
    # y = t on all D channels, t = 1.001 n / (1 + 1.001 n): u_none = t / RMS(x) > 0, u = 0. A block that READS the content (v = LayerNorm of the past state): y = a S, S = sum of the valid keys' LN(x), a = 1.001 / (1 + 1.001 n)
    def plain(content):
        blk = PastBlock()
        with torch.no_grad():
            for p_ in (blk.q.weight, blk.q.bias, blk.bias[2].weight, blk.bias[2].bias, blk.v_none, blk.kv.weight):
                p_.zero_()
            blk.kv.bias.zero_()
            blk.o.weight.copy_(torch.eye(D))
            if content:
                blk.kv.weight[D:].copy_(torch.eye(D))
            else:
                blk.kv.bias[D:] = 1.0
        return blk
    fq, fk = _frame(2, 7, 11, n_invalid=2), _frame(2, 6, 12, n_invalid=2)                                  # 5 valid query tokens per sample, 4 valid past keys per sample (n = 4), different contents
    pk = dict(_past_of(fk, fq["e"]), w=torch.ones(2, 6))
    qa = (fq["x"], fq["pos"], fq["dirv"], fq["valid"])
    n_, a_ = 4, 1.001 / (1 + 1.001 * 4)
    x2q = float((fq["x"].double()[fq["valid"]] ** 2).sum())
    blind = plain(False)
    sb = blind.use(*qa, blind(*qa, pk) - fq["x"], pk)
    t_ = n_ * a_
    assert abs(sb["d2_none"] - 10 * D * t_ ** 2) < 1e-6 and sb["d2_other"] < 1e-9 and sb["n"] == sb["n_other"] == 10 and abs(math.sqrt(sb["d2_none"] / sb["x2"]) - t_ / math.sqrt(x2q / (10 * D))) < 1e-6, f"a block that ignores the past content: {sb}"
    reader = plain(True)
    ln = lambda t: (t - t.mean(-1, keepdim=True)) / (t.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()  # noqa: E731
    S = [ln(fk["x"].double()[i][:4]).sum(0) for i in range(2)]
    sr = reader.use(*qa, reader(*qa, pk) - fq["x"], pk)
    assert abs(sr["d2_none"] - 5 * a_ ** 2 * float((S[0] ** 2).sum() + (S[1] ** 2).sum())) < 1e-6 and abs(sr["d2_other"] - 5 * a_ ** 2 * 2 * float(((S[0] - S[1]) ** 2).sum())) < 1e-6 and sr["d2_other"] > 1.0, f"a block that reads the past content: {sr}"
    um = UseMeter(blind)                                                                                    # the hook on these blocks: u = 0 and u_none > 0 for the blind one, a batch of one + a donor
    blind(*qa, pk)
    tk = um.take()
    assert tk["u"] < 1e-6 and abs(tk["u_none"] - math.sqrt(sb["d2_none"] / sb["x2"])) < 1e-9 and tk["n"] == tk["n_other"] == 10 and um.take() is None
    pk1 = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in pk.items()}
    blind(*(t[:1] for t in qa), pk1)
    tk = um.take()
    assert tk["u"] is None and tk["u_none"] > 0 and tk["n_other"] == 0 and tk["n"] == 5, "a batch of one sample without a donor has no u"
    um.detach()
    um = UseMeter(reader)
    um.donor = dict({k: (v[1:2] if torch.is_tensor(v) else v) for k, v in pk.items()}, e_cur=pk1["e_cur"])
    reader(*(t[:1] for t in qa), pk1)
    tk = um.take()
    x2_0 = float((fq["x"].double()[0][fq["valid"][0]] ** 2).sum())
    assert abs(tk["u"] - math.sqrt(5 * a_ ** 2 * float(((S[0] - S[1]) ** 2).sum()) / x2_0)) < 1e-6 and tk["n_other"] == 5, "a donor for a batch of one"
    um.detach()
    um = UseMeter(b)                                                                                       # the hook: calls without a past are not counted, the sums are those of `use`, `take` clears, two calls add up
    b(x, pos, dirv, valid, None), b(x, pos, dirv, valid, {})
    assert um.take() is None
    yh = b(x, pos, dirv, valid, past)
    s3 = um.take_sums()
    assert s3 == b.use(x, pos, dirv, valid, yh - x, past) and um.take() is None and set(s3) == set(SUMS)
    b(x, pos, dirv, valid, past), b(x, pos, dirv, valid, past)
    tk = um.take()
    assert abs(tk["u"] - math.sqrt(s3["d2_other"] / s3["x2_other"])) < 1e-9 and abs(tk["u_none"] - math.sqrt(s3["d2_none"] / s3["x2"])) < 1e-9 and tk["n"] == 2 * s3["n"] and tk["n_other"] == 2 * s3["n_other"], "u over two calls"
    um.detach()
    b(x, pos, dirv, valid, past)
    assert um.take() is None, "a detached meter still counts"
'''
s = s[:i0] + new + s[i1:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
