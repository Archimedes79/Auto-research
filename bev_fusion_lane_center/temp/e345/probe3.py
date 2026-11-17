import os, sys, json, pickle, numpy as np
R = os.environ["BFLC_ROOT"]
c = pickle.load(open(os.path.join(R, "runs/e312_p_s0/ends_card_snap_thr0.1.counts.pkl"), "rb"))
sc = np.asarray(c["scenes"])
u = sorted(set(sc.tolist()))
cnt = {s: int((sc == s).sum()) for s in u}
print(u, [cnt[s] for s in u], type(sc[0]))
for name, order in (("numeric", u), ("string", sorted(u, key=str))):
    ev = sum(cnt[s] for k, s in enumerate(order) if k % 2 == 0); od = sum(cnt[s] for k, s in enumerate(order) if k % 2 == 1)
    print(name, "even", ev, "odd", od)
# first-appearance order (the order of the VAL list)
seen = []
for s in sc.tolist():
    if s not in seen: seen.append(s)
print("first appearance", seen, sum(cnt[s] for k, s in enumerate(seen) if k % 2 == 0), sum(cnt[s] for k, s in enumerate(seen) if k % 2 == 1))
