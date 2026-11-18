import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e380", "chord_probe.py")
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
rep('    e, valid = np.array(c.f["e"][i], np.float64), np.array(c.f["valid"][i], bool)\n', '    e, valid = np.array(c.f["e"][i], np.float64), np.array(c.f["valid"][i], bool)\n    tk = np.array(c.f["tok"][i], np.float32)\n')
rep('    X.append(np.concatenate([base, ev], 1))', '    X.append(np.concatenate([base, tk[ii], tk[jj], ev], 1))')
rep('nb = X.shape[1] - 8\n', 'nb = X.shape[1] - 8\nng = nb - 128                                      # the geometry block: 13 pair features + 2 existence logits; then the two tokens\' 64-channel node-head features, then the 8 evidence numbers\n')
rep('''    for name, cols in (("baseline", slice(0, nb)), ("evidence", slice(0, nb + 8)), ("evidence_only", slice(nb, nb + 8))):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X[tr][:, cols], Y[tr])''', '''    G, T, E = np.arange(0, ng), np.arange(ng, nb), np.arange(nb, nb + 8)
    for name, cols in (("geo", G), ("geo+ev", np.r_[G, E]), ("geo+tok", np.r_[G, T]), ("geo+tok+ev", np.r_[G, T, E]), ("ev_only", E)):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X[tr][:, cols], Y[tr])''')
rep('''    print("fold %d: pairs %d pos %d | AUC baseline %.4f  + evidence %.4f (%+.4f)  evidence only %.4f | AP baseline %.4f  + evidence %.4f (%+.4f)" % (f, te.sum(), Y[te].sum(), out["baseline"][0], out["evidence"][0], out["evidence"][0] - out["baseline"][0], out["evidence_only"][0], out["baseline"][1], out["evidence"][1], out["evidence"][1] - out["baseline"][1]), flush=True)
gain = [r["evidence"]["auc"] - r["baseline"]["auc"] for r in res]
print("AUC gain per fold:", " ".join("%+.4f" % x for x in gain), "| mean %+.4f | folds with gain >= 0.03: %d of 5" % (np.mean(gain), sum(x >= 0.03 for x in gain)))
json.dump(dict(args=vars(a), pairs=int(len(Y)), positives=int(Y.sum()), folds=res, auc_gain=gain), open(os.path.join(R, a.out), "w"), indent=1)''', '''    print("fold %d: pairs %d pos %d | AUC geo %.4f geo+ev %.4f (%+.4f) | geo+tok %.4f geo+tok+ev %.4f (%+.4f) | ev only %.4f | AP geo+tok %.4f geo+tok+ev %.4f (%+.4f)" % (f, te.sum(), Y[te].sum(), out["geo"][0], out["geo+ev"][0], out["geo+ev"][0] - out["geo"][0], out["geo+tok"][0], out["geo+tok+ev"][0], out["geo+tok+ev"][0] - out["geo+tok"][0], out["ev_only"][0], out["geo+tok"][1], out["geo+tok+ev"][1], out["geo+tok+ev"][1] - out["geo+tok"][1]), flush=True)
g1 = [r["geo+ev"]["auc"] - r["geo"]["auc"] for r in res]
g2 = [r["geo+tok+ev"]["auc"] - r["geo+tok"]["auc"] for r in res]
print("AUC gain of the evidence per fold: over geometry only", " ".join("%+.4f" % x for x in g1), "| mean %+.4f | folds >= 0.03: %d of 5" % (np.mean(g1), sum(x >= 0.03 for x in g1)))
print("AUC gain of the evidence per fold: over geometry + the tokens' node-head features (the stronger baseline, the card's criterion)", " ".join("%+.4f" % x for x in g2), "| mean %+.4f | folds >= 0.03: %d of 5" % (np.mean(g2), sum(x >= 0.03 for x in g2)))
json.dump(dict(args=vars(a), pairs=int(len(Y)), positives=int(Y.sum()), folds=res, auc_gain_geo=g1, auc_gain_tok=g2), open(os.path.join(R, a.out), "w"), indent=1)''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
