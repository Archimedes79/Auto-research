"""E365 S1 diagnosis 3 (scratch, IN-SAMPLE on VAL): p_none of the assigned tokens that are dropped, and what B's own existence says about them"""
import os, sys, pickle
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_ptok_data as LP
S = int(sys.argv[1])
d = pickle.load(open(os.path.join(ROOT, f"runs/e365_s1_s{S}/token_val.pkl"), "rb")); fr = d["frames"]
ds, dl = LP.loader("val", seed=S, bs=8, workers=0, ident="e292_a")
pn_ass, e_ass, drop_pn, drop_e, unass_kept_e, all_ass_e = [], [], [], [], [], []
i = 0
for b in dl:
    for k in range(b["valid"].shape[0]):
        f = fr[i]; i += 1
        valid = b["valid"][k].numpy(); ass = (b["ytype"][k].numpy() > 0) & valid
        e = b["e"][k].numpy(); pn = f["pnone"]; keep = f["keep"].astype(bool)
        pn_ass.append(pn[ass]); e_ass.append(e[ass])
        dr = ass & ~keep; drop_pn.append(pn[dr]); drop_e.append(e[dr])
        unass_kept_e.append(e[keep & ~ass & valid])
pn_ass, e_ass, drop_pn, drop_e, uk = map(np.concatenate, (pn_ass, e_ass, drop_pn, drop_e, unass_kept_e))
q = lambda a: np.round(np.percentile(a, [10, 25, 50, 75, 90]), 2)
print("assigned tokens", len(pn_ass), "p_none pct 10/25/50/75/90", q(pn_ass), "| B' e logit pct", q(e_ass))
print("dropped assigned", len(drop_pn), "p_none pct", q(drop_pn), "share in [0.5, 0.7]: %.3f" % ((drop_pn < 0.7).mean()), "| their B' e logit pct", q(drop_e), "share with B' e >= 0 (kept by B'): %.3f" % (drop_e >= 0).mean())
print("kept but unassigned", len(uk), "B' e logit pct", q(uk), "share e >= 0: %.3f" % (uk >= 0).mean())
print("assigned with B' e >= 0: %.3f of assigned" % (e_ass >= 0).mean())
