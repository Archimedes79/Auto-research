"""E357 T0: a ranking diagnostic of a stored dump, NOT a verdict and NOT a delivered rule (a top-1 choice per source would be a new output selection, D88): does the head RANK the right link first?
For every node with a GT short link: is the highest-p short candidate (chord <= 8 m) a GT link?  For every node with a GT junction link: is the highest-p long candidate (chord > 8 m) a GT junction link?
Also: the mean number of candidates with p >= 0.5 per source node, and the share of GT links that are ranked first among the candidates of their source.
usage (from the problem folder): wsl bash tools/env/_py14.sh temp/e357/rank_diag.py <run>"""
import os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
import jtok_eval as JE  # noqa: E402

run = sys.argv[1]
with open(os.path.join(JE.ROOT, "runs", run, "tok_val.pkl"), "rb") as fh:
    preds = pickle.load(fh)
frames, table = JE.val_tables(2.5)
assert len(preds) == 873
n_src = {"short": 0, "junction": 0}
hit = {"short": 0, "junction": 0}
cand50 = {"short": [], "long": []}
with JE.JT.lattice(2.5):
    for k, p in enumerate(preds):
        t = table(k)
        lk = p["links"]
        src, dst, pp = np.asarray(lk["src"], int), np.asarray(lk["dst"], int), np.asarray(lk["p"], float)
        pos = t["pos"].astype(float)
        ch = np.linalg.norm(pos[src] - pos[dst], axis=1)
        gt = {}
        for s, d, lg, cn in zip(t["src"], t["dst"], t["long"], t["conn"]):
            gt[(int(s), int(d))] = "junction" if (lg and cn) else ("short" if not lg else "lane_long")
        for kind, sel in (("short", ch <= JE.JT.SHORT), ("junction", ch > JE.JT.SHORT)):
            s_, d_, p_ = src[sel], dst[sel], pp[sel]
            have = {int(s) for (s, d), c in gt.items() if c == kind}
            best = {}
            for s, d, q in zip(s_, d_, p_):
                if s not in best or q > best[s][1]:
                    best[s] = (int(d), float(q))
            for s in have:
                n_src[kind] += 1
                if s in best and gt.get((int(s), best[s][0])) == kind:
                    hit[kind] += 1
            cand50["short" if kind == "short" else "long"].append(((p_ >= 0.5).sum(), len(have) if len(have) else 0, len(set(s_[p_ >= 0.5].tolist()))))
print(f"run {run}: top-1 ranking among the candidates of a source (IN-SAMPLE on VAL; a diagnostic, not a verdict)")
for kind in ("short", "junction"):
    print(f"  {kind:9s} sources with a GT link {n_src[kind]}: the highest-p candidate is a GT link for {hit[kind] / max(n_src[kind], 1):.3f}")
for kind in ("short", "long"):
    a = np.array(cand50[kind], float)
    print(f"  {kind:9s} candidates with p >= 0.5: {a[:, 0].sum():.0f} from {a[:, 2].sum():.0f} source nodes ({a[:, 0].sum() / max(a[:, 2].sum(), 1):.2f} per source); GT sources {a[:, 1].sum():.0f}")
