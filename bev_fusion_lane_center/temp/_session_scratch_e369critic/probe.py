"""critic probe (scratch, not evidence): the E369 prior as the card writes it vs c-rule on VAL tokens"""
import math, os, sys
import numpy as np, torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", ".."):
    sys.path.insert(0, os.path.join(ROOT, s))
import jtok as JT, jtok_eval as JV, lane_ptok_data as LP
N = int(sys.argv[1]); var = sys.argv[2]
ds = LP.PTokData("val", 0, "e292_a")
idx = np.linspace(0, len(ds) - 1, N).round().astype(int)
F = lambda x: torch.from_numpy(np.asarray(x, np.float32))
st = dict(rule=0, rule_lo=0, lo_out_cand=0, lo_minchord=0, lo_backj=0, rep=0, extra=0, keep_diff=0, keep_n=0, nearzero=0, valid=0, rule_long=0, rep_long=0, pr_long=0, free_in_wrong=0)
cosc, cost = math.cos(math.radians(45)), math.cos(math.radians(60))
for i in idx:
    o = ds[int(i)]
    v = o["valid"].astype(bool); e = o["e"].astype(np.float64); pos = o["pos"].astype(np.float64); dirv = o["dirv"].astype(np.float64)
    tk = dict(valid=v, e_p=o["tokens_np"]["e_p"], pos=pos, dirv=dirv)
    keep_r = v & (tk["e_p"] >= 0.5)
    kidx = np.flatnonzero(keep_r)
    s1, d1 = JV.rule_links(pos[kidx], dirv[kidx]); s2, d2 = JV.open_links(pos[kidx], dirv[kidx], s1, d1, every=False)
    rule_lo = set(zip(kidx[s1].tolist(), kidx[d1].tolist())); rule_long = set(zip(kidx[s2].tolist(), kidx[d2].tolist()))
    g, allowed = JT.pair_geometry(F(pos)[None], F(dirv)[None], torch.from_numpy(v)[None], e=F(e).sigmoid()[None])
    allowed = allowed[0].numpy(); L = g[0, ..., 2].numpy() * 20.0
    cs, cl = allowed & (L <= 8.0), allowed & (L > 8.0)
    for a, b in rule_lo:
        if not cs[a, b]:
            st["lo_out_cand"] += 1
            ch = pos[b] - pos[a]
            st["lo_minchord"] += int(np.linalg.norm(ch) < JT.MIN_CHORD); st["lo_backj"] += int((ch * dirv[b]).sum() <= 0)
    # the prior as written
    sS, sL, P, kap, beta = 0.5, 1.0, 20.0, 5.0, 4.0
    ch = pos[None] - pos[:, None]; Lr = np.linalg.norm(ch, axis=2)
    cone = (Lr > 1e-9) & (Lr <= 8) & ((ch * dirv[:, None]).sum(2) >= cosc * Lr) & (dirv @ dirv.T >= cost)
    lse = lambda x: np.logaddexp.reduce(x, axis=-1, keepdims=True)
    logsig = lambda x: -np.logaddexp(0, -x)
    keybias = np.log(1 / (1 + np.exp(-e)) + 1e-3)                     # the existing wiring key bias at step 0 (SWITCH not reached)
    if var == "card":
        zs = np.where(cone, -(Lr / sS) ** 2, -(Lr / sS) ** 2 - P) + kap * logsig(beta * e)[None]
    else:                                                              # a stronger variant: separations above the distance range
        zs = np.where(cone, -(Lr / sS) ** 2, -1e4) + 50 * logsig(40 * e)[None]
    zs = np.where(cs, zs, -1e4); zns = np.full((len(e), 1), -(8 / sS) ** 2 - 1)
    lp = np.concatenate([zs, zns], 1); lp = lp - lse(lp); ps = np.exp(lp)
    zs2 = np.where(cs, zs + keybias[None], -1e4); lp2 = np.concatenate([zs2, zns], 1); lp2 = lp2 - lse(lp2); ps2 = np.exp(lp2)
    free_out = ps[:, -1]
    if var == "card":
        free_in = np.exp(-(ps[:, :-1] * v[:, None]).sum(0))
        zl = np.where(Lr <= 40, -(Lr / sL) ** 2 + np.log(free_in + 1e-3)[None], -(Lr / sL) ** 2 - P)
        znl = np.log(1 - free_out + 1e-3) - np.log(free_out + 1e-3)
    else:
        w = 1 / (1 + np.exp(-40 * e))
        free_in = np.exp(-(ps[:, :-1] * w[:, None]).sum(0))
        zl = np.where(Lr <= 40, -(Lr / sL) ** 2 + 1e4 * (free_in[None] - 1) + 50 * logsig(40 * e)[None], -1e4)
        znl = np.where(free_out > 0.5, -(41 / sL) ** 2, 1e4) + 0 * free_out
    zl = np.where(cl, zl + keybias[None], -1e4); lpl = np.concatenate([zl, znl[:, None]], 1); lpl = lpl - lse(lpl); pl = np.exp(lpl)
    keep_h = v & (1 / (1 + np.exp(e)) < 0.5)
    link = set()
    for P_, Z in ((ps2, cs), (pl, cl)):
        a_, b_ = np.nonzero(Z & (P_[:, :-1] >= 0.3) & (P_[:, :-1] >= P_[:, -1:]) & keep_h[:, None] & keep_h[None])
        link |= set(zip(a_.tolist(), b_.tolist()))
    rl = rule_lo | rule_long
    st["rule"] += len(rl); st["rule_lo"] += len(rule_lo); st["rep"] += len(rl & link); st["extra"] += len(link - rl)
    st["rule_long"] += len(rule_long); st["rep_long"] += len(rule_long & link); st["pr_long"] += sum(1 for a, b in link if Lr[a, b] > 8)
    st["keep_diff"] += int((keep_h != keep_r).sum()); st["keep_n"] += int(keep_r.sum()); st["valid"] += int(v.sum()); st["nearzero"] += int((v & (np.abs(e) < 0.25)).sum())
print(var, N, st)
print("rule links reproduced %.3f, extra/rule %.3f, long reproduced %.3f, rule lo links outside the head's short candidates %.4f" % (st["rep"] / st["rule"], st["extra"] / st["rule"], st["rep_long"] / max(1, st["rule_long"]), st["lo_out_cand"] / st["rule_lo"]))
