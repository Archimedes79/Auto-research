import os
p = os.path.join(os.environ["BFLC_ROOT"], "training/grid/test_jmap_train.py")
s = open(p, encoding="utf-8").read()
old = '''        T.train(oa, runs=tmp, ckpt_every=10, smoke_k=K)
        mo, ao = T.load_model("o", "cpu", tmp)'''
new = '''        seen_b, real_fit = [], T.T.fit

        def fit_spy(model, loss_fn, run, **kw):                                                         # the smoke batch of step 3 as the loop asks for it
            seen_b.append(kw["batch_fn"](3))
            return real_fit(model, loss_fn, run, **kw)
        T.T.fit = fit_spy
        try:
            T.train(oa, runs=tmp, ckpt_every=10, smoke_k=K)
        finally:
            T.T.fit = real_fit
        want_b, plain_b = T.synthetic_batch(2, K, 3, False, True), T.synthetic_batch(2, K, 3, False, False)
        assert all(torch.equal(seen_b[0]["maps"][k], want_b["maps"][k]) for k in LMD.KEYS) and not torch.equal(seen_b[0]["maps"]["pos"], plain_b["maps"]["pos"]), f"{label}: the smoke batches of --map-mode other"
        mo, ao = T.load_model("o", "cpu", tmp)'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
